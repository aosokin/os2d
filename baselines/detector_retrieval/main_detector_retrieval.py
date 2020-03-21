import os
import random
import argparse
import numpy as np
import warnings

import torch

from os2d.data.dataloader import build_eval_dataloaders_from_cfg
from os2d.utils import  set_random_seed, print_meters, get_data_path, setup_logger, mkdir
from os2d.config import cfg

from evaluate_detector_retrieval import evaluate, build_retrievalnet_from_options
from utils_maskrcnn import build_maskrcnn_model


def parse_opts():
    parser = argparse.ArgumentParser(description="Evaluate detector-retrieval baseline")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--output_dir", default=None, type=str,
                        help="path where to save all the outputs")
    # retrieval opts
    # network
    parser.add_argument('--retrieval_network_path', type=str,
                        help='network path, destination where network is saved')
    parser.add_argument('--retrieval_image_size', default=240, type=int,
                        help='maximum size of longer image side used for testing (default: 240)')
    parser.add_argument('--retrieval_multiscale', action='store_true',
                        help='use multiscale vectors for testing')
    parser.add_argument('--retrieval_whitening_path', default=None, type=str,
                        help='path to add the whitening (default: None)')
    # maskrcnn opts
    # config
    parser.add_argument('--maskrcnn_config_file', type=str,
                        help='network path, destination where network is saved')
    # weights
    parser.add_argument('--maskrcnn_weight_file', type=str,
                        help='network path, destination where network is saved')
    parser.add_argument("--nms_iou_threshold_detector_score", default=0.3, type=float,
                        help='first round of nms done w.r.t. the detector scores: IoU threshold')
    parser.add_argument("--nms_score_threshold_detector_score", default=0.1, type=float,
                        help='first round of nms done w.r.t. the detector scores: score threshold')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg, args

def init_logger(args, logger_prefix="detector-retrieval"):
    if args.output_dir:
        mkdir(args.output_dir)

    logger = setup_logger(logger_prefix, args.output_dir if args.output_dir else None)

    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with the default eval section:\n{}".format(cfg.eval))
    else:
        logger.info("Launched with no OS2D config file")
    logger.info("Running args:\n{}".format(args))
    return logger


def main():
    cfg, args = parse_opts()
    logger_prefix="detector-retrieval"
    logger = init_logger(args, logger_prefix)

    # set this to use faster convolutions
    if cfg.is_cuda:
        assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
        torch.backends.cudnn.benchmark = True

    # random seed
    set_random_seed(cfg.random_seed, cfg.is_cuda)

    # Load the detector
    maskrcnn_model, maskrcnn_config = build_maskrcnn_model(args.maskrcnn_config_file, args.maskrcnn_weight_file)

    # Load the retrieval network
    retrievalnet = build_retrievalnet_from_options(args, is_cuda=cfg.is_cuda)

    # load the dataset
    data_path = get_data_path()
    img_normalization = {"mean":cfg.model.normalization_mean, "std": cfg.model.normalization_std} # do not actually use this - will use normalization encoded in the config of maskrcnn-benchmark
    box_coder = None
    dataloaders_eval = build_eval_dataloaders_from_cfg(cfg, box_coder, img_normalization,
                                                       data_path=data_path,
                                                       logger_prefix=logger_prefix)

    # start evaluation
    for dataloader in dataloaders_eval:
        losses = evaluate(dataloader, maskrcnn_model, maskrcnn_config, retrievalnet, args,
                          cfg_eval=cfg.eval, cfg_visualization=cfg.visualization.eval, is_cuda=cfg.is_cuda,
                          logger_prefix=logger_prefix)


if __name__ == "__main__":
    main()