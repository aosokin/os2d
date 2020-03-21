import os
import argparse

import torch

from os2d.modeling.model import build_os2d_from_config

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.engine.train import trainval_loop
from os2d.utils import set_random_seed, get_trainable_parameters, mkdir, save_config, setup_logger, get_data_path
from os2d.engine.optimization import create_optimizer
from os2d.config import cfg


def parse_opts():
    parser = argparse.ArgumentParser(description="Training and evaluation of the OS2D model")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
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

    return cfg, args.config_file


def init_logger(cfg, config_file):
    output_dir = cfg.output.path
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("OS2D", output_dir if cfg.output.save_log_to_file else None)

    if config_file:
        logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    else:
        logger.info("Config file was not provided")

    logger.info("Running with config:\n{}".format(cfg))

    # save config file only when training (to run multiple evaluations in the same folder)
    if output_dir and cfg.train.do_training:
        output_config_path = os.path.join(output_dir, "config.yml")
        logger.info("Saving config into: {}".format(output_config_path))
        # save overloaded model config in the output directory
        save_config(cfg, output_config_path)


def main():
    cfg, config_file = parse_opts()
    init_logger(cfg, config_file)

    # set this to use faster convolutions
    if cfg.is_cuda:
        assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
        torch.backends.cudnn.benchmark = True

    # random seed
    set_random_seed(cfg.random_seed, cfg.is_cuda)

    # Model
    net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)

    # Optimizer
    parameters = get_trainable_parameters(net)
    optimizer = create_optimizer(parameters, cfg.train.optim, optimizer_state)

    # load the dataset
    data_path = get_data_path()
    dataloader_train, datasets_train_for_eval = build_train_dataloader_from_config(cfg, box_coder, img_normalization,
                                                                                   data_path=data_path)

    dataloaders_eval = build_eval_dataloaders_from_cfg(cfg, box_coder, img_normalization,
                                                       datasets_for_eval=datasets_train_for_eval,
                                                       data_path=data_path)

    # start training (validation is inside)
    trainval_loop(dataloader_train, net, cfg, criterion, optimizer, dataloaders_eval=dataloaders_eval)


if __name__ == "__main__":
    main()
