"""Script to evaluate a network on a dataset.
This script is a refactored version of
https://github.com/timy90022/One-Shot-Object-Detection/blob/master/test_net.py
"""
import os
import sys
import argparse
import time
import pprint
import numpy as np

import cv2

import torch

import _init_paths

from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='coco', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        default=True)
    parser.add_argument('--weights', dest='weights',
                        help='load this checkpoint for evaluation',
                        default=None, type=str)
    parser.add_argument('--class_image_augmentation', default=None, type=str,
                        help='augment images at test time: None or "rotation90"')
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


def test(args, model=None):
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Load dataset
    imdb_vu, roidb_vu, ratio_list_vu, ratio_index_vu, query_vu = combined_roidb(args.imdbval_name, False)
    imdb_vu.competition_mode(on=True)
    dataset_vu = roibatchLoader(roidb_vu, ratio_list_vu, ratio_index_vu, query_vu, 1, imdb_vu._classes, training=False)

    # initilize the network here.
    if not model:
        if args.net == 'vgg16':
            fasterRCNN = vgg16(imdb_vu.classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
           fasterRCNN = resnet(imdb_vu.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN = resnet(imdb_vu.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            fasterRCNN = resnet(imdb_vu.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
        fasterRCNN.create_architecture()

        # Load checkpoint
        print("load checkpoint %s" % (args.weights))
        checkpoint = torch.load(args.weights)
        fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print('load model successfully!')
    else:
        # evaluate constructed model
        fasterRCNN = model

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    query   = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    catgory = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()
        im_data = im_data.cuda()
        query = query.cuda()
        im_info = im_info.cuda()
        catgory = catgory.cuda()
        gt_boxes = gt_boxes.cuda()

    # record time
    start = time.time()

    # visiualization
    vis = args.vis if hasattr(args, 'vis') else None
    if vis:
        thresh = 0.05
    else:
        thresh = 0.0
    max_per_image = 100

    fasterRCNN.eval()
    dataset_vu.query_position = 0
    test_scales = cfg.TEST.SCALES
    multiscale_iterators = []
    for i_scale, test_scale in enumerate(test_scales):
        cur_dataloader_vu = torch.utils.data.DataLoader(dataset_vu, batch_size=1,shuffle=False, num_workers=0,pin_memory=True)
        cur_data_iter_vu = iter(cur_dataloader_vu)
        multiscale_iterators.append(cur_data_iter_vu)

    # total quantity of testing images, each images include multiple detect class
    num_images_vu = len(imdb_vu.image_index)
    num_detect = len(ratio_index_vu[0])

    all_boxes = [[[] for _ in range(num_images_vu)]
                for _ in range(imdb_vu.num_classes)]

    _t = {'im_detect': time.time(), 'misc': time.time()}

    for i,index in enumerate(ratio_index_vu[0]):
        det_tic = time.time()
        multiscale_boxes = []
        multiscale_scores = []
        for i_scale, (data_iter_vu, test_scale) in enumerate(zip(multiscale_iterators, test_scales)):
            # need to rewrite cfg.TRAIN.SCALES - very hacky!
            BACKUP_TRAIN_SCALES = cfg.TRAIN.SCALES
            cfg.TRAIN.SCALES = [test_scale]
            data = next(data_iter_vu)
            cfg.TRAIN.SCALES = BACKUP_TRAIN_SCALES

            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                query.resize_(data[1].size()).copy_(data[1])
                im_info.resize_(data[2].size()).copy_(data[2])
                gt_boxes.resize_(data[3].size()).copy_(data[3])
                catgory.data.resize_(data[4].size()).copy_(data[4])

                # Run Testing
                if not hasattr(args, "class_image_augmentation") or not args.class_image_augmentation:
                    queries = [query]
                elif args.class_image_augmentation.lower() == "rotation90":
                    queries = [query]
                    for _ in range(3):
                        queries.append(queries[-1].rot90(1, [2, 3]))
                else:
                    raise RuntimeError("Unknown class_image_augmentation: {}".format(args.class_image_augmentation))

                for q in queries:
                    rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, _, RCNN_loss_bbox, \
                    rois_label, weight = fasterRCNN(im_data, q, im_info, gt_boxes, catgory)

                    scores = cls_prob.data
                    boxes = rois.data[:, :, 1:5]

                    # Apply bounding-box regression
                    if cfg.TEST.BBOX_REG:
                        # Apply bounding-box regression deltas
                        box_deltas = bbox_pred.data
                        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                            # Optionally normalize targets by a precomputed mean and stdev
                            if args.class_agnostic:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                box_deltas = box_deltas.view(1, -1, 4)
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                box_deltas = box_deltas.view(1, -1, 4 * len(imdb_vu.classes))

                        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                    else:
                        # Simply repeat the boxes, once for each class
                        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                    # Resize to original ratio
                    pred_boxes /= data[2][0][2].item()

                    # Remove batch_size dimension
                    scores = scores.squeeze()
                    pred_boxes = pred_boxes.squeeze()

                    multiscale_scores.append(scores)
                    multiscale_boxes.append(pred_boxes)

        scores = torch.cat(multiscale_scores, dim=0)
        pred_boxes = torch.cat(multiscale_boxes, dim=0)

        # Record time
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        # Post processing
        inds = torch.nonzero(scores>thresh).view(-1)
        if inds.numel() > 0:
            # remove useless indices
            cls_scores = scores[inds]
            cls_boxes = pred_boxes[inds, :]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

            # rearrange order
            _, order = torch.sort(cls_scores, 0, True)
            cls_dets = cls_dets[order]

            # NMS
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            all_boxes[catgory][index] = cls_dets.cpu().numpy()

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                try:
                    image_scores = all_boxes[catgory][index][:,-1]
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]

                        keep = np.where(all_boxes[catgory][index][:,-1] >= image_thresh)[0]
                        all_boxes[catgory][index] = all_boxes[catgory][index][keep, :]
                except:
                    pass

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                .format(i + 1, num_detect, detect_time, nms_time))
            sys.stdout.flush()

            # save test image
            if vis and i%1==0:
                im2show = cv2.imread(dataset_vu._roidb[dataset_vu.ratio_index[i]]['image'])
                im2show = vis_detections(im2show, 'shot', cls_dets.cpu().numpy(), 0.3)

                o_query = data[1][0].permute(1, 2,0).contiguous().cpu().numpy()
                o_query *= [0.229, 0.224, 0.225]
                o_query += [0.485, 0.456, 0.406]
                o_query *= 255
                o_query = o_query[:,:,::-1]

                (h,w,c) = im2show.shape
                o_query = cv2.resize(o_query, (h, h),interpolation=cv2.INTER_LINEAR)
                im2show = np.concatenate((im2show, o_query), axis=1)

                vis_path = "./test_img"
                if not os.path.isdir(vis_path):
                    os.makedirs(vis_path)
                cv2.imwrite( os.path.join(vis_path, "%d_d.png"%(i)), im2show)

    print('Evaluating detections')
    mAP = imdb_vu.evaluate_detections(all_boxes, None)

    end = time.time()
    print("test time: %0.4fs" % (end - start))
    return mAP


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    np.random.seed(cfg.RNG_SEED)
    args.imdbval_name = args.dataset

    args.cfg_file = "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    test(args)
