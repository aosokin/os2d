# A modification version from mashrcnn-benchmark repository.
# (See https://github.com/facebookresearch/maskrcnn-benchmark/blob/v0.1/maskrcnn_benchmark/data/datasets/evaluation/voc/voc_eval.py)
# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)

import os
import copy
from collections import defaultdict
import numpy as np

from os2d.structures.bounding_box import BoxList, boxlist_iou


def do_voc_evaluation(predictions, gt_boxes, iou_thresh=0.5, use_07_metric=False):
    """Runs VOC evaluation on a full dataset.
    Resizes detection box lists to the same size as the corresponding GT boxlist and calls eval_detection_voc.
    Args:
        predictions (list of BoxList): prediction boxlists, each BoxList has fields "labels" and "scores".
            len(predictions) = num_images
        gt_boxes (list of BoxList): ground truth boxlists, eahc BoxList has field "labels" and optionally "difficult"
            len(gt_boxes) = num_images
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    pred_boxlists = []
    for prediction, gt in zip(predictions, gt_boxes):
        prediction = prediction.resize(gt.image_size)
        pred_boxlists.append(prediction)

    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxes,
        iou_thresh=iou_thresh,
        use_07_metric=use_07_metric)
    return result

def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Runs VOC evaluation on a full dataset.
    Detection and ground-truth boxes are assume to be resized to the same image size.
    Args:
        predictions (list of BoxList): prediction boxlists, each BoxList has fields "labels" and "scores".
            len(predictions) = num_images
        gt_boxes (list of BoxList): ground truth boxlists, eahc BoxList has field "labels" and optionally "difficult"
            len(gt_boxes) = num_images
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec, n_pos = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh)
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    recall, recall_per_class, n_pos = calc_detection_recall(rec, n_pos)

    prec_one_class, rec_one_class, n_pos_one_class = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh,
        merge_classes_together=True)
    ap_one_class = calc_detection_voc_ap(prec_one_class, rec_one_class, use_07_metric=use_07_metric)

    return {'ap_per_class': ap, 'map': np.nanmean(ap), 'map_weighted': np.nansum(ap * n_pos / n_pos.sum()),
            'recall_per_class': recall_per_class, 'recall': recall,
            'n_pos': n_pos, 'prec': prec, 'rec' : rec,
            'ap_joint_classes': ap_one_class[0]}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5, merge_classes_together=False):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox_xyxy.cpu().numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox_xyxy.cpu().numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        if gt_boxlist.has_field("difficult"):
            gt_difficult = gt_boxlist.get_field("difficult").numpy()
        else:
            gt_difficult = np.zeros_like(gt_label)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.image_size),
                BoxList(gt_bbox_l, gt_boxlist.image_size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    if merge_classes_together:
        n_pos = {0: sum(n_pos[i] for i in n_pos)}
        # merge lists together, copy to avoid rewriting the old lists
        old_score = copy.deepcopy(score)
        score = {0: sum((old_score[i] for i in old_score), [])}
        old_match = copy.deepcopy(match)
        match = {0: sum((old_match[i] for i in old_match), [])}

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec, n_pos


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def calc_detection_recall(rec, n_pos):
    n_fg_class = len(rec)
    recall_per_class = np.empty(n_fg_class)
    n_pos_np = np.empty(n_fg_class)
    n_pos_total = 0.0
    n_good_detections_total = 0.0
    for l in range(n_fg_class):
        n_pos_np[l] = n_pos[l]
        if rec[l] is None  or  n_pos[l] is None or n_pos[l] == 0:
            recall_per_class[l] = np.nan
        else:
            if len(rec[l]) > 0:
                recall_per_class[l] = rec[l][-1]
            else:
                recall_per_class[l] = 0.0
            n_pos_total += n_pos[l]
            n_good_detections_total += n_pos[l] * recall_per_class[l]
    if n_pos_total == 0:
        recall = float('nan')
    else:
        recall = n_good_detections_total / n_pos_total
    return recall, recall_per_class, n_pos_np
