import os
import random
import time
import math
from collections import OrderedDict
import numpy as np
import copy
import logging

import torch
import torch.nn.functional as F
from torchvision import transforms

from os2d.data.voc_eval import do_voc_evaluation
import os2d.utils.visualization as visualizer
from os2d.utils import add_to_meters_in_dict, print_meters, get_image_size_after_resize_preserving_aspect_ratio
from os2d.structures.feature_map import FeatureMapSize
from os2d.structures.bounding_box import nms, BoxList, cat_boxlist

from cirtorch.networks.imageretrievalnet import init_network

from utils_maskrcnn import run_maskrcnn_on_images


def evaluate(dataloader, detector, cfg_maskrcnn, retrievalnet, opt, cfg_eval, cfg_visualization, is_cuda=False, logger_prefix="detector-retrieval"):
    logger = logging.getLogger(f"{logger_prefix}.evaluate")
    dataset_name = dataloader.get_name()
    dataset_scale = dataloader.get_eval_scale()
    logger.info("Starting to eval on {0}, scale {1}".format(dataset_name, dataset_scale))

    t_start_eval = time.time()
    detector.eval()
    retrievalnet.eval()

    ## setup retrievalnet
    # setting up the multi-scale parameters
    ms = [1]
    msp = 1
    if opt.retrieval_multiscale:
        ms = [1, 1./math.sqrt(2), 1./2]
        if retrievalnet.meta["pooling"] == "gem" and retrievalnet.whiten is None:
            msp = retrievalnet.pool.p.data.tolist()[0]
    #setup whitening
    if opt.retrieval_whitening_path is not None:
        logger.info("Whitening is precomputed, loading it from {0}".format(opt.retrieval_whitening_path))
        whitening_data = torch.load(opt.retrieval_whitening_path)
        
        if ( (opt.retrieval_multiscale and "ms" in whitening_data) or \
             (not opt.retrieval_multiscale and "ss" in whitening_data ) ):

            if opt.retrieval_multiscale:
                Lw = copy.deepcopy(whitening_data["ms"])
            else:
                Lw = copy.deepcopy(whitening_data["ss"])
        else:
            raise RuntimeError("Whitening should be precomputed with the network")

        # convert whitening data to torch tensors
        Lw["m"], Lw["P"] = torch.from_numpy(Lw["m"]), torch.from_numpy(Lw["P"])
        if is_cuda:
            Lw["m"], Lw["P"] = Lw["m"].cuda(), Lw["P"].cuda()
    else:
        Lw = None

    with torch.no_grad():  # do evaluation in forward mode only (for speed and memory)
        # extract features from query images
        query_images, _, _  = dataloader.get_all_class_images(do_resize=False)
        if is_cuda:
            query_images = [img.cuda() for img in query_images]
        query_images = [img[0] for img in query_images] # get rid of the batch dimension
        query_images = [resize_image_tensor(img, opt.retrieval_image_size) for img in query_images]
        query_images = [dataloader.unnorm_image(img) for img in query_images]

        query_images_with_aug = []
        for im in query_images:
            query_images_with_aug.append(im)
            if not cfg_eval.class_image_augmentation:
                num_class_views = 1
            elif cfg_eval.class_image_augmentation == "rotation90":
                im90 = im.rot90(1, [1, 2])
                im180 = im90.rot90(1, [1, 2])
                im270 = im180.rot90(1, [1, 2])
                query_images_with_aug.append(im90)
                query_images_with_aug.append(im180)
                query_images_with_aug.append(im270)
                num_class_views = 4
            elif cfg_eval.class_image_augmentation == "horflip":
                im_flipped = im.flip(2)
                query_images_with_aug.append(im_flipped)
                num_class_views = 2
            else:
                raise RuntimeError(f"Unknown value of class_image_augmentation: {cfg_eval.class_image_augmentation}")
        query_images = query_images_with_aug

        query_vectors = extract_vectors_from_images(retrievalnet, query_images, ms=ms, msp=msp)
        # apply whitening if defined
        if  Lw is not None:
            query_vectors = whitenapply(query_vectors, Lw["m"], Lw["P"])
        query_vectors = torch.transpose(query_vectors, 0, 1)

        # prepare looping over all iamges
        iterator = make_iterator_extract_scores_from_images_batched(dataloader, detector, cfg_maskrcnn, logger,
                                                                    image_batch_size=cfg_eval.batch_size,
                                                                    is_cuda=is_cuda)

        boxes, labels, scores = [], [], []
        gt_boxes = []
        image_ids = []
        losses = OrderedDict()
         
        # loop over all dataset images
        num_evaluted_images = 0
        for data in iterator:
            image_id, boxes_one_image, image_pyramid, query_img_sizes, class_ids, initial_img_size = data
            image_ids.append(image_id)
            logger.info(f"Image {num_evaluted_images}: id {image_id}")

            num_evaluted_images += 1
            img_size_pyramid = [FeatureMapSize(img=img) for img in image_pyramid]

            gt_boxes_one_image = dataloader.get_image_annotation_for_imageid(image_id)
            gt_boxes.append(gt_boxes_one_image)
            
            # vizualize GT for debug
            if cfg_visualization.show_gt_boxes:
                visualizer.show_gt_boxes(image_id, gt_boxes_one_image, class_ids, dataloader)
            
            # decode image predictions
            # merge boxes_one_image, labels_one_image, scores_one_image from different pyramid layers
            boxes_one_image = cat_boxlist(boxes_one_image)
            # do NMS
            good_indices = nms(boxes_one_image, opt.nms_iou_threshold_detector_score,
                               nms_score_threshold=opt.nms_score_threshold_detector_score)
            boxes_one_image = boxes_one_image[good_indices]

            # extract feature vectors from the predictions
            image_original = dataloader._transform_image(image_id, do_augmentation=True, hflip=False, vflip=False)[0]
            if is_cuda:
                image_original = image_original.cuda()
            image_patches = crop_resize_image_patches(image_original, boxes_one_image, opt.retrieval_image_size, logger,
                                                        unnorm_image=dataloader.unnorm_image,
                                                        is_cuda=is_cuda)
            # filter out cases when failed to crop a box: outside of the image
            good_indices = [i for i, p in enumerate(image_patches) if p is not None]
            if good_indices:
                # non empty
                image_patches = [p for p in image_patches if p is not None]
                boxes_one_image = boxes_one_image[good_indices]
                
                image_vectors = extract_vectors_from_images(retrievalnet, image_patches, ms=ms, msp=msp)

                # compute class scores from image_vectors and query_vectors (already transposed)
                if  Lw is not None:
                    # apply whitening if defined
                    image_vectors  = whitenapply(image_vectors, Lw["m"], Lw["P"])
                scores_retrieval = torch.mm(query_vectors, image_vectors)

                num_queries = scores_retrieval.size(0)
                num_detections = scores_retrieval.size(1)
                list_of_active_label = torch.LongTensor(class_ids)
                if cfg_eval.class_image_augmentation:
                    list_of_active_label = torch.stack( [list_of_active_label] * num_class_views, 1).view(-1)

                # take all labels for all boxes - will sort them by scores at eval
                scores_one_image = scores_retrieval.view(-1)
                boxes_one_image = cat_boxlist([boxes_one_image] * num_queries)
                labels_one_image = torch.stack( [list_of_active_label] * num_detections, 1 ).contiguous().view(-1)
                # add scores and labels: overwrite if existed
                boxes_one_image.add_field("labels", labels_one_image)
                boxes_one_image.add_field("scores", scores_one_image)

                # NMS using the retrieval scores
                good_indices = nms(boxes_one_image,
                                cfg_eval.nms_iou_threshold,
                                nms_score_threshold=cfg_eval.nms_score_threshold,
                                do_separate_per_label=not cfg_eval.nms_across_classes)
                boxes_one_image = boxes_one_image[good_indices]
            else:
                boxes_one_image.add_field("labels", torch.zeros(0, dtype=torch.long, device=boxes_one_image.bbox_xyxy.device))
                boxes_one_image.add_field("scores", torch.zeros(0, dtype=torch.float, device=boxes_one_image.bbox_xyxy.device))

            boxes.append(boxes_one_image.cpu())

            if cfg_visualization.show_detections:
                # do not pass class_ids - this is already taken care of
                visualizer.show_detections(boxes_one_image, image_id, dataloader, cfg_visualization, class_ids=None)

    # normalize by number of steps
    for k in losses:
        losses[k] /= num_evaluted_images

    # Save detection if requested 
    if cfg_visualization.path_to_save_detections:
        data = {"image_ids" : image_ids,
                "boxes_xyxy" : [bb.bbox_xyxy for bb in boxes], 
                "labels" : [bb.get_field("labels") for bb in boxes],
                "scores" : [bb.get_field("scores") for bb in boxes],
                "gt_boxes_xyxy" : [bb.bbox_xyxy for bb in gt_boxes],
                "gt_labels" : [bb.get_field("labels") for bb in gt_boxes],
                "gt_difficults" : [bb.get_field("difficult") for bb in gt_boxes]
        }

        dataset_name = dataloader.get_name()
        os.makedirs(cfg_visualization.path_to_save_detections, exist_ok=True)
        save_path = os.path.join(cfg_visualization.path_to_save_detections, dataset_name + "_detections.pth")
        torch.save(data, save_path) 

    # compute mAP
    for mAP_iou_threshold in cfg_eval.mAP_iou_thresholds:
        logger.info("Evaluating at IoU th {:0.2f}".format(mAP_iou_threshold))
        ap_data = do_voc_evaluation(boxes, gt_boxes, iou_thresh=mAP_iou_threshold, use_07_metric=False)
        losses["mAP@{:0.2f}".format(mAP_iou_threshold)] = ap_data["map"]
        losses["mAPw@{:0.2f}".format(mAP_iou_threshold)] = ap_data["map_weighted"]
        losses["recall@{:0.2f}".format(mAP_iou_threshold)] = ap_data["recall"]
        losses["AP_joint_classes@{:0.2f}".format(mAP_iou_threshold)] = ap_data["ap_joint_classes"]

        # per class AP information
        for i_class, (ap, recall, n_pos) in enumerate(zip(ap_data["ap_per_class"], ap_data["recall_per_class"], ap_data["n_pos"])):
            if not np.isnan(ap):
                assert i_class in class_ids, "Could not find class_id in the list of ids"
                logger.info("Class {0} (local {3}), AP {1:0.4f}, #obj {2}, recall {4:0.4f}".format(i_class,
                                                                                            ap,
                                                                                            n_pos,
                                                                                            class_ids.index(i_class),
                                                                                            recall
                                                                                            ))

    # save timing
    losses["eval_time"] = (time.time() - t_start_eval)
    logger.info("Evaluated on {0}, scale {1}".format(dataset_name, dataset_scale))
    print_meters(losses, logger)
    return losses


def whitenapply(X, m, P):
    # # numpy code
    # X = np.dot(P, X-m)
    # X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)
    X = torch.mm( P.to(X), X - m.to(X) )
    X = X / (X.norm(p=2, dim=0, keepdim=True) + 1e-6)
    return X


def extract_ss(net, input_var):
    return net(input_var.unsqueeze(0)).squeeze(1)


def extract_ms(net, input_var, ms, msp):
    v = torch.zeros(net.meta["outputdim"], device=input_var.device)
    
    for s in ms: 
        if s == 1:
            input_var_t = input_var.clone().unsqueeze(0)
        else:    
            size = [int(input_var.size(-2) * s), int(input_var.size(-1) * s)]
            # support image with width or height of 1 pixel
            if size[0] < 1 :
                size[0] = 1
            if size[1] < 1 :
                size[1] = 1
            if input_var.size(-2) == 1:
                input_var = torch.cat([input_var] * size[0] , dim=-2)
                size[0] = 1
            if input_var.size(-1) == 1:
                input_var = torch.cat([input_var] * size[1] , dim=-1)
                size[1] = 1
            input_var_t = F.interpolate(input_var.unsqueeze(0), size=size, mode="bilinear")
        v += net(input_var_t).pow(msp).squeeze(1)
        
    v /= len(ms)
    v = v.pow(1./msp)
    v /= v.norm()

    return v


def extract_vectors_from_images(net, images, ms=[1], msp=1):
    normalize_retrievalnet = transforms.Normalize(
        mean=net.meta["mean"],
        std=net.meta["std"]
    )
    net.eval()

    # extracting vectors
    vecs = []
    for i, img in enumerate(images):
        device = img.device
        img = normalize_retrievalnet(img.cpu()).to(device=device)
        if len(ms) == 1:
            vecs.append(extract_ss(net, img))
        else:
            vecs.append(extract_ms(net, img, ms, msp))
    vecs = torch.stack(vecs, 1)
    return vecs


def crop_image_tensor(img, crop_position_xyxy):
    assert img.dim() == 3, "img should be three dimensional to represent one image in C x H x W format"

    # get the good crop position
    crop_position_xyxy = crop_position_xyxy.cpu()
    min_x = max(int(crop_position_xyxy[0].item()), 0)
    min_y = max(int(crop_position_xyxy[1].item()), 0)
    max_x = min(int(crop_position_xyxy[2].item()), img.size(2))
    max_y = min(int(crop_position_xyxy[3].item()), img.size(1))

    # crop the image
    crop = img[:, min_y : max_y, min_x : max_x]

    return crop


def resize_image_tensor(img, target_size):
    h, w = img.size(1), img.size(2)
    size_hw = get_image_size_after_resize_preserving_aspect_ratio(h, w, target_size)
    img = F.interpolate(img.unsqueeze(0), size_hw, mode="bilinear").squeeze(0)
    return img


def crop_resize_image_patches(img, boxes, target_size, logger, unnorm_image=None, is_cuda=False):
    # revert normalization if provided
    if unnorm_image is not None:
        img = unnorm_image(img)
    image_patches = []
    for crop_position_xyxy in boxes.bbox_xyxy:
        crop = crop_image_tensor(img, crop_position_xyxy)
        if crop.dim() == 3 and crop.size(1) > 0 and crop.size(2) > 0:
            crop = resize_image_tensor(crop, target_size)
            image_patches.append(crop)
        else:
            logger.warning("Cropping box {0} from image of size {1} resulted in crop of bad size {2}".format(crop_position_xyxy, img.size(), crop.size()))
            image_patches.append(None)
    return image_patches


def build_retrievalnet_from_options(opt, is_cuda=False):
    logger = logging.getLogger("detector-retrieval.build_retrievalnet")
    logger.info("Building the retrieval model...")

    logger.info("Loading weights from {}".format(opt.retrieval_network_path))
    state = torch.load(opt.retrieval_network_path)

    # parsing net params from meta
    # architecture, pooling, mean, std required
    # the rest has default values, in case that is doesnt exist
    net_params = {}
    net_params["architecture"] = state["meta"]["architecture"]
    net_params["pooling"] = state["meta"]["pooling"]
    net_params["local_whitening"] = state["meta"].get("local_whitening", False)
    net_params["regional"] = state["meta"].get("regional", False)
    net_params["whitening"] = state["meta"].get("whitening", False)
    net_params["mean"] = state["meta"]["mean"]
    net_params["std"] = state["meta"]["std"]
    net_params["pretrained"] = False

    # load network
    net = init_network(net_params)
    net.load_state_dict(state["state_dict"])
    
    # if whitening is precomputed
    if "Lw" in state["meta"]:
        net.meta["Lw"] = state["meta"]["Lw"]
    
    logger.info("Loaded network: ")
    if "epoch" in state:
        logger.info("Model after {} epochs".format(state["epoch"]))
    logger.info(net.meta_repr())
    
    if is_cuda:
        net.cuda()
    else:
        net.cpu()
    net.eval()

    return net


def make_iterator_extract_scores_from_images_batched(dataloader, maskrcnn_model, maskrcnn_config, logger, image_batch_size=None, is_cuda=False):
    logger.info("Starting iterations over images")

    # get images of all classes
    class_images, class_aspect_ratios, class_ids = dataloader.get_all_class_images()
    num_classes = len(class_images)
    assert len(class_aspect_ratios) == num_classes
    assert len(class_ids) == num_classes
    query_img_sizes = [FeatureMapSize(img=img) for img in class_images]

    # loop over all images
    iterator_batches = dataloader.make_iterator_for_all_images(image_batch_size)
    for batch_ids, pyramids_batch, box_transforms_batch, initial_img_size_batch in iterator_batches:
        t_start_batch = time.time()
        # extract features at all pyramid levels
        batch_images_pyramid = []
        bboxes_xyxy = []
        labels = []
        scores = []
        num_pyramid_levels = len(pyramids_batch)
        for batch_images in pyramids_batch:
            if is_cuda:
                batch_images = batch_images.cuda()

            # print("Image size:", images_b.size())

            batch_images = [dataloader.unnorm_image(img) for img in batch_images]
            batch_images = torch.stack(batch_images, 0)

            bboxes_xyxy_, labels_, scores_ = run_maskrcnn_on_images(maskrcnn_model, maskrcnn_config, batch_images)

            bboxes_xyxy.append(bboxes_xyxy_)
            labels.append(labels_)
            scores.append(scores_)
            batch_images_pyramid.append(batch_images)
            
        for i_image_in_batch, image_id in enumerate(batch_ids):
            # get data from all pyramid levels
            bboxes_xyxy_p = []
            labels_p = []
            scores_p = []
            for i_p in range(num_pyramid_levels):
                bboxes_xyxy_p.append( bboxes_xyxy[i_p][i_image_in_batch] )
                labels_p.append( labels[i_p][i_image_in_batch] )
                scores_p.append( scores[i_p][i_image_in_batch] )

            # get a pyramid of one image[i_p]
            one_image_pyramid = [p[i_image_in_batch] for p in batch_images_pyramid]

            # extract the box transformations
            box_reverse_transforms = box_transforms_batch[i_image_in_batch]

            # get the boxes in the correct format
            bboxes_xyxy_p = [BoxList(bbox, FeatureMapSize(img=img), mode="xyxy") for bbox, img in zip(bboxes_xyxy_p, one_image_pyramid)]
            bboxes_xyxy_p = [t(bb) for t, bb in zip(box_reverse_transforms, bboxes_xyxy_p)]

            # add labels and scores into the box structure
            for bb, l, s in zip(bboxes_xyxy_p, labels_p, scores_p):
                bb.add_field("labels", l)
                bb.add_field("scores", s)

            # get the size of the initial image
            initial_img_size = initial_img_size_batch[i_image_in_batch]

            yield image_id, bboxes_xyxy_p, one_image_pyramid, query_img_sizes, class_ids, initial_img_size
