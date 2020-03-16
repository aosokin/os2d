import os
import random
import time, datetime
import math
import copy
import logging
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from PIL import Image

from os2d.utils import add_to_meters_in_dict, log_meters, print_meters, time_since, get_trainable_parameters, checkpoint_model, init_log
from .evaluate import evaluate, make_iterator_extract_scores_from_images_batched
from .optimization import setup_lr, get_learning_rate, set_learning_rate
from os2d.structures.bounding_box import nms, cat_boxlist
from os2d.structures.feature_map import FeatureMapSize
import os2d.utils.visualization as visualizer


def prepare_batch_data(batch_data, is_cuda, logger):
    """Helper function to parse batch_data and put tensors on a GPU.
    Used in train_one_batch
    """
    images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
        batch_box_inverse_transform, batch_boxes, batch_img_size = \
        batch_data
    if is_cuda:
        images = images.cuda()
        class_images = [im.cuda() for im in class_images]
        loc_targets = loc_targets.cuda()
        class_targets = class_targets.cuda()

    logger.info("{0} imgs, {1} classes".format(images.size(0), len(class_images)))

    return images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
           batch_box_inverse_transform, batch_boxes, batch_img_size


def train_one_batch(batch_data, net, cfg, criterion, optimizer, dataloader, logger):
    """One training iteration
        
    Args:
        batch_data - input data, will be parsed with prepare_batch_data
        net - the network to use
        cfg - config with all the parameters
        criterion - criterion to optimize
        optimizer - optimizer to use
        dataloader - need it to call box_coder.remap_anchor_targets and pass to visualizations
        logger - logger to use

    Returns:
        meters (OrderedDict) - all computed metrics including meters["loss"], which is the loss optimized
    """
    t_start_batch = time.time()
    net.train(freeze_bn_in_extractor=cfg.train.model.freeze_bn,
              freeze_transform_params=cfg.train.model.freeze_transform,
              freeze_bn_transform=cfg.train.model.freeze_bn_transform)
    
    optimizer.zero_grad()

    # # Debugging NaNs int he gradients: load saved batch dat and investigate
    # dump_file = os.path.join("output/exp1/exp1.2.lossRLL_remap_seed0_ResNet50_init_imageNetCaffe2",
    #                          "error_nan_appeared-2020-02-22-08:43:48.pth")
    # data_nan = torch.load(dump_file)
    # batch_data = data_nan["batch_data"]
    # net.load_state_dict(data_nan["state_dict"])
    # optimizer.load_state_dict(data_nan["optimizer"])
    # grad = data_nan["grad"]

    images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
        batch_box_inverse_transform, batch_boxes, batch_img_size = \
        prepare_batch_data(batch_data, cfg.is_cuda, logger)

    loc_scores, class_scores, class_scores_transform_detached, fm_sizes, corners = \
        net(images, class_images,
            train_mode=True,
            fine_tune_features=cfg.train.model.train_features)

    cls_targets_remapped, ious_anchor, ious_anchor_corrected = \
        dataloader.box_coder.remap_anchor_targets(loc_scores, batch_img_size, class_image_sizes, batch_boxes)

    # compute the losses
    losses = criterion(loc_scores, loc_targets,
                       class_scores, class_targets,
                       cls_targets_remapped=cls_targets_remapped,
                       cls_preds_for_neg=class_scores_transform_detached if not cfg.train.model.train_transform_on_negs else None)

    if cfg.visualization.train.show_target_remapping:
        visualizer.show_target_remapping(images, class_targets, cls_targets_remapped,
                                         losses, class_scores, class_scores_transform_detached,
                                         ious_anchor, ious_anchor_corrected)

    if cfg.visualization.train.show_detections:
        visualizer.decode_scores_show_detections(dataloader, images, class_ids,
                                                 class_scores, loc_scores, corners)

    main_loss = losses["loss"]
    main_loss.backward()

    # save full grad
    grad = OrderedDict()
    for name, param in net.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad[name] = param.grad.clone().cpu()

    grad_norm = torch.nn.utils.clip_grad_norm_(get_trainable_parameters(net), cfg.train.optim.max_grad_norm, norm_type=2)
    # save error state if grad appears to be nan
    if math.isnan(grad_norm):
        # remove some unsavable objects
        batch_data = [b for b in batch_data]
        batch_data[6] = None

        data_nan = {"batch_data":batch_data, "state_dict":net.state_dict(), "optimizer": optimizer.state_dict(),
                    "cfg":cfg,  "grad": grad}
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        dump_file = "error_nan_appeared-"+time_stamp+".pth"
        if cfg.output.path:
            dump_file = os.path.join(cfg.output.path, dump_file)

        logger.error("gradient is NaN. Saving dump to {}".format(dump_file))
        torch.save(data_nan, dump_file)
    else:
        optimizer.step()

    # convert everything to numbers
    meters = OrderedDict()
    for l in losses:
        meters[l] = losses[l].mean().item()
    meters["grad_norm"] = grad_norm
    meters["batch_time"] = time.time() - t_start_batch
    return meters


@torch.no_grad() # mine patches in forward mode (for speed and memory)
def mine_hard_patches(dataloader, net, cfg, criterion):
    """Mine patches that are hard: classification false positives and negative, localization errors
    At each level of sampled image pyramid, we need to cut out a piece of size appropriate for training
    (levels are defined by cfg.train.mining.num_random_pyramid_scales, cfg.train.mining.num_random_negative_classes)

    Args:
        dataloader - dataloader to use (often the same as the one for training)
        net - the network to use
        cfg - config with all the parameters
        criterion - criterion (usually the same one as used for training)

    Returns:
        hardnegdata_per_imageid (OrderedDict) - mined data, keys are the image ids;
            further used in dataloader.set_hard_negative_data(hardnegdata_per_imageid) when preparing batches
    """
    logger = logging.getLogger("OS2D.mining_hard_patches")
    logger.info("Starting to mine hard patches")
    t_start_mining = time.time()
    net.eval()
    num_batches = len(dataloader)
    hardnegdata_per_imageid = OrderedDict()

    iterator = make_iterator_extract_scores_from_images_batched(dataloader, net, logger,
                                                                image_batch_size=cfg.eval.batch_size,
                                                                is_cuda=cfg.is_cuda,
                                                                num_random_pyramid_scales=cfg.train.mining.num_random_pyramid_scales,
                                                                num_random_negative_labels=cfg.train.mining.num_random_negative_classes)

    boxes = []
    gt_boxes = []
    losses = OrderedDict()

    # loop over all dataset images
    for data in iterator:
        t_item_start = time.time()

        image_id, image_loc_scores_pyramid, image_class_scores_pyramid, \
                    image_pyramid, query_img_sizes, \
                    batch_class_ids, box_reverse_transform_pyramid, image_fm_sizes_p, transform_corners_pyramid \
                = data

        img_size_pyramid = [FeatureMapSize(img=image) for image in image_pyramid]

        gt_boxes_one_image = dataloader.get_image_annotation_for_imageid(image_id)
        gt_boxes.append(gt_boxes_one_image)

        # compute losses
        # change labels to the ones local to the current image
        dataloader.update_box_labels_to_local(gt_boxes_one_image, batch_class_ids)
        num_labels = len(batch_class_ids)

        loc_targets_pyramid, class_targets_pyramid = \
                dataloader.box_coder.encode_pyramid(gt_boxes_one_image, img_size_pyramid, num_labels,
                                                    default_box_transform_pyramid=box_reverse_transform_pyramid)

        # vizualize GT for debug
        if cfg.visualization.mining.show_gt_boxes:
            visualizer.show_gt_boxes(image_id, gt_boxes_one_image, batch_class_ids, dataloader)

        # compute losses
        if cfg.is_cuda:
            loc_targets_pyramid = [loc_targets.cuda() for loc_targets in loc_targets_pyramid]
            class_targets_pyramid = [class_targets.cuda() for class_targets in class_targets_pyramid]

        add_batch_dim = lambda list_of_tensors: [t.unsqueeze(0) for t in list_of_tensors]
        loc_scores_pyramid = add_batch_dim(image_loc_scores_pyramid)
        
        cls_targets_remapped_pyramid = []
        for loc_scores, img_size, box_reverse_transform in zip(loc_scores_pyramid, img_size_pyramid, box_reverse_transform_pyramid):
            # loop over the pyramid levels
            cls_targets_remapped, ious_anchor, ious_anchor_corrected = \
                dataloader.box_coder.remap_anchor_targets(loc_scores, [img_size], query_img_sizes, [gt_boxes_one_image],
                                                          box_reverse_transform=[box_reverse_transform])
            cls_targets_remapped_pyramid.append(cls_targets_remapped)

        losses_iter, losses_per_anchor = criterion(loc_scores_pyramid,
                                                    add_batch_dim(loc_targets_pyramid),
                                                    add_batch_dim(image_class_scores_pyramid),
                                                    add_batch_dim(class_targets_pyramid),
                                                    cls_targets_remapped=cls_targets_remapped_pyramid,
                                                    patch_mining_mode=True)

        if cfg.visualization.mining.show_class_heatmaps:
            visualizer.show_class_heatmaps(image_id, batch_class_ids, image_fm_sizes_p, class_targets_pyramid, image_class_scores_pyramid,
                                            cfg_local=cfg.visualization.mining)

        assert dataloader.data_augmentation is not None, "Can mine hard patches only through data augmentation"
        crop_size = dataloader.data_augmentation.random_crop_size

        # convert to floats
        for l in losses_iter:
            losses_iter[l] = losses_iter[l].mean().item()
        # printing
        print_meters(losses_iter, logger)
        # update logs
        add_to_meters_in_dict(losses_iter, losses)

        # construct crop boxes for all the anchors and NMS them - NMS pos ang neg anchors separately
        query_fm_sizes = [dataloader.box_coder._get_feature_map_size_per_image_size(sz) for sz in query_img_sizes]
        
        crops = []
        achors = []
        labels_of_anchors = []
        pyramid_level_of_anchors = []
        losses_of_anchors = []
        corners_of_anchors = []
        losses_loc_of_anchors = []
        pos_mask_of_anchors = []
        pos_loc_mask_of_anchors = []
        neg_mask_of_anchors = []
        anchor_indices = []
        i_image_in_batch = 0 # only one image comes here
        for i_p, img_size in enumerate(img_size_pyramid):
            for i_label, query_fm_size in enumerate(query_fm_sizes):
                crop_position, anchor_position, anchor_index = \
                    dataloader.box_coder.output_box_grid_generator.get_box_to_cut_anchor(img_size,
                                                                                         crop_size,
                                                                                         image_fm_sizes_p[i_p],
                                                                                         box_reverse_transform_pyramid[i_p])
                cur_corners = transform_corners_pyramid[i_p][i_label].transpose(0,1)
                cur_corners = dataloader.box_coder.apply_transform_to_corners(cur_corners, box_reverse_transform_pyramid[i_p], img_size)
                if cfg.is_cuda:
                    crop_position, anchor_position = crop_position.cuda(), anchor_position.cuda()
                crops.append(crop_position)
                achors.append(anchor_position)
                device = crop_position.bbox_xyxy.device
                losses_of_anchors.append(losses_per_anchor["cls_loss"][i_p][i_image_in_batch, i_label].to(crop_position.bbox_xyxy))
                pos_mask_of_anchors.append(losses_per_anchor["pos_mask"][i_p][i_image_in_batch, i_label].to(device=device))
                neg_mask_of_anchors.append(losses_per_anchor["neg_mask"][i_p][i_image_in_batch, i_label].to(device=device))
                losses_loc_of_anchors.append(losses_per_anchor["loc_loss"][i_p][i_image_in_batch, i_label].to(crop_position.bbox_xyxy))
                pos_loc_mask_of_anchors.append(losses_per_anchor["pos_for_regression"][i_p][i_image_in_batch, i_label].to(device=device))
                corners_of_anchors.append(cur_corners.to(crop_position.bbox_xyxy))

                num_anchors = len(crop_position)
                labels_of_anchors.append(torch.full([num_anchors], i_label, dtype=torch.long))
                pyramid_level_of_anchors.append(torch.full([num_anchors], i_p, dtype=torch.long))
                anchor_indices.append(anchor_index)

        # stack all
        crops = cat_boxlist(crops)
        achors = cat_boxlist(achors)
        labels_of_anchors  = torch.cat(labels_of_anchors, 0)
        pyramid_level_of_anchors = torch.cat(pyramid_level_of_anchors, 0)
        losses_of_anchors = torch.cat(losses_of_anchors, 0)
        losses_loc_of_anchors = torch.cat(losses_loc_of_anchors, 0)
        pos_mask_of_anchors = torch.cat(pos_mask_of_anchors, 0)
        pos_loc_mask_of_anchors = torch.cat(pos_loc_mask_of_anchors, 0)
        neg_mask_of_anchors = torch.cat(neg_mask_of_anchors, 0)
        anchor_indices = torch.cat(anchor_indices, 0)
        corners_of_anchors = torch.cat(corners_of_anchors, 0)

        def nms_masked_and_collect_data(mask, crops_xyxy, scores, nms_iou_threshold_in_mining, max_etries=None):
            mask_ids = torch.nonzero(mask).squeeze(1)
            boxes_selected = copy.deepcopy(crops_xyxy[mask])
            boxes_selected.add_field("scores", scores[mask])
            remaining_boxes = nms(boxes_selected, nms_iou_threshold_in_mining)
            remaining_boxes = mask_ids[remaining_boxes]

            # sort and take the topk, because NMS is not sorting by default
            ids = torch.argsort(scores[remaining_boxes], descending=True)
            if max_etries is not None:
                ids = ids[:max_etries]
            remaining_boxes = remaining_boxes[ids]

            return remaining_boxes

        nms_iou_threshold_in_mining = cfg.train.mining.nms_iou_threshold_in_mining
        num_hard_patches_per_image = cfg.train.mining.num_hard_patches_per_image

        # hard negatives
        hard_negs = nms_masked_and_collect_data(neg_mask_of_anchors, crops, losses_of_anchors,
                                                nms_iou_threshold_in_mining,
                                                num_hard_patches_per_image)

        # hard positives for classification
        hard_pos  = nms_masked_and_collect_data(pos_mask_of_anchors, crops, losses_of_anchors,
                                                nms_iou_threshold_in_mining,
                                                num_hard_patches_per_image)

        # hard positives for localization
        hard_pos_loc  = nms_masked_and_collect_data(pos_loc_mask_of_anchors, crops, losses_loc_of_anchors,
                                                    nms_iou_threshold_in_mining,
                                                    num_hard_patches_per_image)

        # merge all together
        def standardize(v):
            return v.item() if type(v) == torch.Tensor else v
        def add_item(data, role, pyramid_level, label_local, anchor_index, crop_position_xyxy, anchor_position_xyxy, transform_corners):
            new_item = OrderedDict()
            new_item["pyramid_level"] = standardize(pyramid_level)
            new_item["label_local"] = standardize(label_local)
            new_item["anchor_index"] = standardize(anchor_index)
            new_item["role"] = role
            new_item["crop_position_xyxy"] = crop_position_xyxy
            new_item["anchor_position_xyxy"] = anchor_position_xyxy
            new_item["transform_corners"] = transform_corners
            data.append(new_item)

        hardnegdata = []
        for i in hard_negs:
            add_item(hardnegdata, "neg", pyramid_level_of_anchors[i],
                        labels_of_anchors[i], anchor_indices[i],
                        crops[i].cpu(), achors[i].cpu(), corners_of_anchors[i].cpu())
        for i in hard_pos:
            add_item(hardnegdata, "pos", pyramid_level_of_anchors[i],
                        labels_of_anchors[i], anchor_indices[i],
                        crops[i].cpu(), achors[i].cpu(), corners_of_anchors[i].cpu())
        for i in hard_pos_loc:
            add_item(hardnegdata, "pos_loc", pyramid_level_of_anchors[i],
                        labels_of_anchors[i], anchor_indices[i],
                        crops[i].cpu(), achors[i].cpu(), corners_of_anchors[i].cpu())

        # extract loss values and compute the box positions to crop
        for a in hardnegdata:
            a["label_global"] = standardize(batch_class_ids[ a["label_local"] ])
            a["loss"] = standardize(losses_per_anchor["cls_loss"][a["pyramid_level"]][i_image_in_batch, a["label_local"], a["anchor_index"]])
            a["loss_loc"] = standardize(losses_per_anchor["loc_loss"][a["pyramid_level"]][i_image_in_batch, a["label_local"], a["anchor_index"]])
            a["score"] = standardize(image_class_scores_pyramid[a["pyramid_level"]][a["label_local"], a["anchor_index"]])
            a["image_id"] = standardize(image_id)

        hardnegdata_per_imageid[image_id] = hardnegdata

        if cfg.visualization.mining.show_mined_patches:
            visualizer.show_mined_patches(image_id, batch_class_ids, dataloader, hardnegdata)

        logger.info("Item time: {0}, since mining start: {1}".format(time_since(t_item_start), time_since(t_start_mining)))
    logger.info("Hard negative mining finished in {0}".format(time_since(t_start_mining)))
    return hardnegdata_per_imageid


def evaluate_model(dataloaders, net, cfg, criterion=None, print_per_class_results=False):
    """Evaluation of the provided model at all validation datasets
  
    Args:
        dataloaders - a list of dataloaders to use for validation, at each validation stage all of them will be used sequentially
        net - the network to use
        cfg - config with all the parameters
        criterion - criterion (usually the same one as used for training), can be None
        print_per_class_results - flag showing whether to printout extra data (per class AP) - usually used at the final evaluation

    Returns:
        meters_all (OrderedDict) - all computed metrics: one entry for each dataloader
            meters_all.keys() - list of dataloader names
            meters_all[d] (OrderedDict) - all metrics for dataloader with name d, e.g., meters_all[d]["mAP@0.50"] - mAP at IoU threshold 0.5
    """
    meters_all = OrderedDict()
    for dataloader in dataloaders:
        # evaluate on validation
        if dataloader is not None:
            meters_val = evaluate(dataloader, net, cfg, criterion=criterion, print_per_class_results=print_per_class_results)
            meters_all[dataloader.get_name()] = meters_val
        else:
            meters_val = None
        
    return meters_all


def trainval_loop(dataloader_train, net, cfg, criterion, optimizer, dataloaders_eval=[]):
    """Main train+val loop
  
    Args:
        dataloader_train -dataloader to get training batches
        net - the network to use
        cfg - config with all the parameters
        criterion - criterion to optimize
        optimizer - optimization to use
        dataloaders_eval - a list of dataloaders to use for validation, at each validation stage all of them will be used sequentially

    Returns nothing
    """
    # init plotting and logging
    logger = logging.getLogger("OS2D.train")
    t_start = time.time()
    num_steps_for_logging, meters_running = 0, {}
    full_log = init_log()
   
    if cfg.train.optim.max_iter > 0 and cfg.train.do_training:
        logger.info("Start training")

        # setup the learning rate schedule
        _, anneal_lr_func = setup_lr(optimizer, full_log, cfg.train.optim.anneal_lr, cfg.eval.iter)

        # evaluate the initial model
        meters_eval = evaluate_model(dataloaders_eval, net, cfg, criterion)
        
        if cfg.output.best_model.do_get_best_model:
            assert (cfg.output.best_model.dataset and cfg.output.best_model.dataset in meters_eval) \
                or (len(cfg.eval.dataset_names) > 0 and cfg.eval.dataset_names[0] in meters_eval), \
                "Cannot determine which dataset to use for the best model"
            best_model_dataset_name = cfg.output.best_model.dataset if cfg.output.best_model.dataset else cfg.eval.dataset_names[0]
            best_model_metric = meters_eval[best_model_dataset_name][cfg.output.best_model.metric]

            logger.info(f"Init model is the current best on {best_model_dataset_name} by {cfg.output.best_model.metric}, value {best_model_metric:.4f}")
            if cfg.output.path:
                checkpoint_best_model_name = f"best_model_{best_model_dataset_name}_{cfg.output.best_model.metric}"
                checkpoint_best_model_path = \
                    checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, model_name=checkpoint_best_model_name,
                                          extra_fields={"criterion_dataset": best_model_dataset_name,
                                                        "criterion_metric": cfg.output.best_model.metric,
                                                        "criterion_mode": cfg.output.best_model.mode,
                                                        "criterion_value": best_model_metric,
                                                        "criterion_value_old": None})
            else:
                raise RuntimeError("cfg.output.best_model.do_get_best_model i set to True, but cfg.output.path is not provided, so cannot save best models")

        if cfg.train.optim.anneal_lr.reload_best_model_after_anneal_lr and\
           cfg.train.optim.anneal_lr.type != "none":
                assert cfg.output.best_model.do_get_best_model, "cfg.train.optim.anneal_lr.reload_best_model_after_anneal_lr was set to True, but cfg.output.best_model.do_get_best_model is False, so there is no best model to reload from"

        # add the initial point
        log_meters(full_log, t_start, -1, cfg.output.path,
                meters_eval=meters_eval,
                anneal_lr=anneal_lr_func)

        # save initial model
        if cfg.output.path:
            checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=0)

        # start training
        i_epoch = 0
        i_batch = len(dataloader_train)  # to start a new epoch at the first iteration
        for i_iter in range(cfg.train.optim.max_iter):
            # restart dataloader if needed
            if i_batch >= len(dataloader_train):
                i_epoch += 1
                i_batch = 0
                # shuffle dataset
                dataloader_train.shuffle()

            # mine hard negative classes
            if cfg.train.mining.do_mining and i_iter % cfg.train.mining.mine_hard_patches_iter == 0:
                hardnegdata_per_imageid = mine_hard_patches(dataloader_train, net, cfg, criterion)
                dataloader_train.set_hard_negative_data(hardnegdata_per_imageid)
            
            # print iter info
            logger.info(f"Iter {i_iter} ({cfg.train.optim.max_iter}), epoch {i_epoch}, time {time_since(t_start)}")

            # get data for training
            t_start_loading = time.time()
            batch_data = dataloader_train.get_batch(i_batch)
            t_data_loading = time.time() - t_start_loading

            i_batch += 1
            num_steps_for_logging += 1

            # train on one batch
            meters = train_one_batch(batch_data, net, cfg, criterion, optimizer, dataloader_train, logger)
            meters["loading_time"] = t_data_loading

            # print meters
            if i_iter % cfg.output.print_iter == 0:
                print_meters(meters, logger)

            # update logs
            add_to_meters_in_dict(meters, meters_running)

            # evaluation
            if (i_iter + 1) % cfg.eval.iter == 0:
                meters_eval = evaluate_model(dataloaders_eval, net, cfg, criterion)

                # checkpoint the best model 
                if cfg.output.best_model.do_get_best_model:
                    cur_metric = meters_eval[best_model_dataset_name][cfg.output.best_model.metric]
                    assert cfg.output.best_model.mode in ["max", "min"], f"cfg.output.best_model.mode should be 'max' or 'min', but have {cfg.output.best_model.mode}"
                    if (cfg.output.best_model.mode=="max" and cur_metric > best_model_metric) or \
                       (cfg.output.best_model.mode=="min" and cur_metric < best_model_metric):
                        # overwrite the best model
                        logger.info(f"New best model on {best_model_dataset_name} by {cfg.output.best_model.metric}, value {cur_metric:.4f}")

                        if cfg.output.path:
                            checkpoint_best_model_path = \
                                checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, model_name=checkpoint_best_model_name,
                                                 extra_fields={"criterion_dataset": best_model_dataset_name,
                                                               "criterion_metric": cfg.output.best_model.metric,
                                                               "criterion_mode": cfg.output.best_model.mode,
                                                               "criterion_value": cur_metric,
                                                               "criterion_value_old": best_model_metric})
                        best_model_metric = cur_metric

                # normalize by number of steps
                for k in meters_running:
                    meters_running[k] /= num_steps_for_logging

                # anneal learning rate
                meters_running["lr"] = get_learning_rate(optimizer)
                if anneal_lr_func:
                    lr = anneal_lr_func(i_iter + 1, anneal_now=i_iter > cfg.train.optim.anneal_lr.initial_patience)
                    flag_changed_lr = lr != meters_running["lr"]
                else:
                    lr = meters_running["lr"]
                    flag_changed_lr = False

                # if lr was annealed load the best up to now model and set it up
                if cfg.train.optim.anneal_lr.reload_best_model_after_anneal_lr and flag_changed_lr:
                    if cfg.output.best_model.do_get_best_model: # if have the best model at all
                        optimizer_state = net.init_model_from_file(checkpoint_best_model_path)
                        if optimizer_state is not None:
                            optimizer.load_state_dict(optimizer_state)
                        set_learning_rate(optimizer, lr)

                # eval and log
                log_meters(full_log, t_start, i_iter, cfg.output.path,
                        meters_running=meters_running,
                        meters_eval=meters_eval)

                # init for the next 
                num_steps_for_logging, meters_running = 0, {}

            # save intermediate model
            if cfg.output.path and cfg.output.save_iter and i_iter % cfg.output.save_iter == 0:
                checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=i_iter)


    # evaluate the final model
    logger.info("Final evaluation")
    meters_eval = evaluate_model(dataloaders_eval, net, cfg, criterion, print_per_class_results=True)

    # add the final point
    if cfg.train.optim.max_iter > 0 and cfg.train.do_training:
        log_meters(full_log, t_start, cfg.train.optim.max_iter, cfg.output.path,
                   meters_eval=meters_eval)

        # save the final model
        if cfg.output.path:
            checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=cfg.train.optim.max_iter)
