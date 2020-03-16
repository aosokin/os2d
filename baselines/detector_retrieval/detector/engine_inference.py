# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
from collections import OrderedDict
from tqdm import tqdm

import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size
from maskrcnn_benchmark.utils.comm import all_gather
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.engine.bbox_aug import im_detect_bbox_aug

from os2d.data.voc_eval import do_voc_evaluation as eval_detection_voc
from detector_data import convert_boxlist_maskrcnn_to_os2d

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        device="cuda"
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    results =  do_voc_evaluation(dataset=dataset,
                                 predictions=predictions,
                                 logger=logger)
    return results


def do_voc_evaluation(dataset, predictions, logger):
    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlist = convert_boxlist_maskrcnn_to_os2d(gt_boxlist)
        gt_boxlists.append(gt_boxlist)
        prediction = convert_boxlist_maskrcnn_to_os2d(prediction)
        pred_boxlists.append(prediction)

    result = OrderedDict()
    for iou_thresh in [0.5]:
        ap_data = eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=iou_thresh, use_07_metric=False)
        result[f"mAP@{iou_thresh}"] = ap_data['map']
        result[f"recall@{iou_thresh}"] = ap_data['recall']

    result_strs = ["{0}: {1:.4f}".format(k,v) for k, v in result.items()]
    result_str = ', '.join(result_strs)

    logger.info(result_str)
    return result
