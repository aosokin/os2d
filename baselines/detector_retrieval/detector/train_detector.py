# Started the script from the template of maskrcnn-banchmark:
# https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/tools/train_net.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import os
import sys
import argparse
from functools import partial
from collections import OrderedDict

import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from detector_data import make_data_loader

from engine_trainer import do_train
from engine_inference import inference


def adjust_config_to_num_gpus(cfg, num_gpus_assumed_in_config=8):
    # use the number fo GPUs to configure the run
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    rescaling = num_gpus / num_gpus_assumed_in_config
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * rescaling
    cfg.SOLVER.IMS_PER_BATCH = int(cfg.SOLVER.IMS_PER_BATCH * rescaling)
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER / rescaling)
    cfg.SOLVER.STEPS = [int(step / rescaling) for step in cfg.SOLVER.STEPS]
    cfg.TEST.IMS_PER_BATCH = int(cfg.TEST.IMS_PER_BATCH * rescaling)


def train_with_validation(cfg, local_rank, distributed, test_weights=None):
    arguments = {}
    arguments["iteration"] = 0

    if test_weights:
        cfg.MODEL.WEIGHT = test_weights
        cfg.SOLVER.MAX_ITER = 0

    ignore_labels = (cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES == 0)

    # prepare training data
    root_path = os.path.expanduser(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")))
    data_loader, class_ids = make_data_loader(
        root_path,
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        ignore_labels=ignore_labels,
    )

    # overwrite the number of classes by considering the training set
    if cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES <= 0: # if we have binary classification or unknown number of classes
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(class_ids)

    # prepare model
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # prepare optimizer
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    # prepare validation
    run_validation_for_model = partial(run_validation, root_path=root_path, cfg=cfg.clone(), class_ids=class_ids, ignore_labels=ignore_labels, distributed=distributed)

    # setup checkpointer
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, use_latest=False if test_weights else True)
    arguments.update(extra_checkpoint_data)
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    validation_period = cfg.SOLVER.VALIDATION_PERIOD

    # start training
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        validation_period,
        checkpoint_period,
        arguments,
        run_validation_for_model)

    return model


def run_validation(model, iteration, root_path, cfg, class_ids, ignore_labels, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_names = cfg.DATASETS.TEST

    data_loaders_val = make_data_loader(root_path, cfg, is_train=False, is_distributed=distributed, class_ids=class_ids, ignore_labels=ignore_labels)
    results = OrderedDict()
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        results[dataset_name] = \
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                device=cfg.MODEL.DEVICE
            )
        synchronize()
    return results


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--validation_period', default=500, type=int,
                        help='Do validation after this number of training iterations')
    parser.add_argument('--test_weights', default=None, type=str,
                        help="Evaluate this weight file")
    args = parser.parse_args()

    print(args)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.SOLVER.VALIDATION_PERIOD = args.validation_period

    adjust_config_to_num_gpus(cfg, num_gpus_assumed_in_config=8)

    # cfg.freeze() # do not freeze, will change later based on the datasets

    if args.test_weights:
        cfg.OUTPUT_DIR = ""

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train_with_validation(cfg, args.local_rank, args.distributed, test_weights=args.test_weights)


if __name__ == "__main__":
    main()
