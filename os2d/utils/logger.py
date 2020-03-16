import os
import sys
import time
import pickle
import math
import logging
import re

import torch


def init_log():
    return {}


def add_to_meters_in_dict(meters_next_step, meters_history):
    for k in meters_next_step:
        if k in meters_history:
            meters_history[k] += meters_next_step[k]
        else:
            meters_history[k] = meters_next_step[k]


def update_meter(log, name, num_log_steps, value):
    # create entry if needed
    if name not in log:
        log[name] = []
    meter = log[name]
    # add missing values if any
    while len(meter) < num_log_steps - 1:
        meter.append(float("nan"))
    # add the new value
    meter.append(value)


def print_meters(meters, logger):
    meters_str = ""
    for k, v in meters.items():
        meters_str += "%s %.4f, "%(k, v)
    logger.info(meters_str)


def log_meters(log, t_start, i_iter, log_path,
               meters_running=None, meters_eval=None, anneal_lr=None):
    logger = logging.getLogger("OS2D.logger")
    
    # guess how many log points where done before
    num_log_steps = 0
    if "time" in log:
        num_log_steps = max(num_log_steps, len(log["time"]))
    if "iter" in log:
        num_log_steps = max(num_log_steps, len(log["iter"]))

    # adding extra point
    num_log_steps += 1
    update_meter(log, "time", num_log_steps, (time.time() - t_start) / 3600)
    update_meter(log, "iter", num_log_steps, i_iter + 1)

    # add meters saved during training
    if meters_running is not None:
        for name, meter in meters_running.items():
            update_meter(log, name + "_running", num_log_steps, meter)


    # add meters computed at the evaluation stage
    if meters_eval is not None:
        for subset_name, subset_data in meters_eval.items():
            for meter_name, meter in subset_data.items():
                update_meter(log, meter_name + "_" + subset_name, num_log_steps, meter)

    # update other meters with NaNs to make sure all the meters are of equal length
    for name, meter in log.items():
        while len(meter) < num_log_steps:
            meter.append(float("nan"))

    # save plots
    if log_path:
        try:
            if not os.path.isdir(log_path):
                os.makedirs(log_path)
            pickle.dump(log, open(os.path.join(log_path, "train_log.pkl"), "wb"))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.error("could not save the log file for some reason: {}".format(str(e)))


def time_since(since):
    now = time.time()
    s = now - since
    return "%s" % (time_for_printing(s))


def time_for_printing(s, mode="hms"):
    if mode.lower() == "hms":
        h = math.floor(s / 3600)
        s -= h * 3600
        m = math.floor(s / 60)
        s -= m * 60
        return "%dh %dm %ds" % (h, m, s)
    elif mode.lower() == "s":
        return "%.2fs" % (s)
    else:
       raise(RuntimeError("Unknown time printing mode: {0}".format(mode)))


def save_config(cfg, path):
    """
    From https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/miscellaneous.py
    """
    with open(path, "w") as f:
        f.write(cfg.dump())


def setup_logger(name, save_dir=None, filename="log.txt"):
    """
    From https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/logger.py
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def checkpoint_model(net, optimizer, log_path, is_cuda, model_name=None, i_iter=None, extra_fields=None):
    logger = logging.getLogger("OS2D.checkpoint")
    net.cpu()
    try:
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        checkpoint = {}
        checkpoint["net"] = net.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict()
        if extra_fields:
            checkpoint.update(extra_fields)
        checkpoint_file_name = f"checkpoint_{model_name}.pth" if model_name else f"checkpoint_iter_{i_iter}.pth"
        checkpoint_file = os.path.join(log_path, checkpoint_file_name)
        logger.info("Saving checkpoint {0}".format(checkpoint_file))
        torch.save(checkpoint, checkpoint_file)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        logger.error("Could not save the checkpoint model for some reason: {}".format(str(e)))

    if is_cuda:
        net.cuda()
    
    return checkpoint_file


numeric_const_pattern = r"""
    [-+]? # optional sign
    (?:
        (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
        |
        (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
    )
    # followed by optional exponent part if desired
    (?: [Ee] [+-]? \d+ ) ?
    """


def mAP_percent_to_points(v):
    if v is not None:
        return f"{float(v)*100:0.1f}"
    else:
        return "None"


def extract_pattern_after_marked_line(result_file, dataset_pattern, eval_pattern):
    try:
        mAP_value = None
        with open(result_file, 'r') as f:
            for line in f:
                if dataset_pattern in line:
                    # found the dataset match
                    line = next(f, "")

                    matches = re.findall(eval_pattern, line, re.VERBOSE)
                    if matches:
                        mAP_value = matches[-1]
        return mAP_value
    except Exception as e:
        print(e)
        return None


def extract_map_value_from_os2d_log(result_file, eval_dataset):
    dataset_search_pattern = "Evaluated on {0}"
    dataset_pattern = dataset_search_pattern.format(eval_dataset)
    eval_pattern = f"mAP@0.50\s({numeric_const_pattern})"

    mAP_value = extract_pattern_after_marked_line(result_file, dataset_pattern, eval_pattern)
    return mAP_percent_to_points(mAP_value)


def extract_value_from_os2d_binary_log(data_file, value, reduce):
    try:
        logs = pickle.load(open(data_file, "rb"))
        val_plot = logs[value]
        if reduce == "max":
            val = max(val_plot)
        elif reduce == "min":
            val = min(val_plot)
        elif reduce == "first":
            val = val_plot[0]
        else:
            raise RuntimeError("Unknown reduce value: {}".format(reduce))
        return val
    except Exception as e:
        print(e)
        return None
