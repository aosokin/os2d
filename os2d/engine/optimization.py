import logging
from statistics import median

from torch import optim

from os2d.utils import ceildiv


def create_optimizer(parameters, cfg, optimizer_state=None):
    lr = cfg.lr
    optim_method = cfg.optim_method.casefold()
    if optim_method == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, weight_decay=cfg.weight_decay, momentum=cfg.sgd_momentum)
    elif optim_method == "adagrad":
        optimizer = optim.Adagrad(parameters, lr=lr, weight_decay=cfg.weight_decay)
    elif optim_method == "adadelta":
        optimizer = optim.Adadelta(parameters, lr=lr, weight_decay=cfg.weight_decay)
    elif optim_method == "adam":
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=cfg.weight_decay)
    elif optim_method == "adamax":
        optimizer = optim.Adamax(parameters, lr=lr, weight_decay=cfg.weight_decay)
    elif optim_method == "asgd":
        optimizer = optim.ASGD(parameters, lr=lr, t0=5000, weight_decay=cfg.weight_decay)
    elif optim_method == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=lr, weight_decay=cfg.weight_decay)
    elif optim_method == "rprop":
        optimizer = optim.Rprop(parameters, lr=lr)
    else:
        raise RuntimeError("Invalid optim method: " + cfg.optim_method)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        set_learning_rate(optimizer, cfg.lr)

    return optimizer


def set_learning_rate(optimizer, learning_rate):
    logger = logging.getLogger("OS2D")

    for p in optimizer.param_groups:
        if "lr" in p:
            if p["lr"] != learning_rate:
                logger.info("Changing learning rate from {} to {}".format(p["lr"], learning_rate))
                p["lr"] = learning_rate


def get_learning_rate(optimizer):
    for p in optimizer.param_groups:
        if "lr" in p:
            return p["lr"]

def setup_lr(optimizer, full_log, cfg, eval_iter):
    # annealing learning rate
    if cfg.type.lower() == "none":
        lr_scheduler = None
    elif cfg.type.lower() == "MultiStepLR".lower():
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[ceildiv(m, eval_iter) for m in cfg.milestones],
                                                      gamma=cfg.gamma)
    elif cfg.type.lower() == "ReduceLROnPlateau".lower():
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, factor=cfg.reduce_factor, min_lr=cfg.min_value,
            threshold=cfg.quantity_epsilon, threshold_mode="rel", mode=cfg.quantity_mode,
            patience=ceildiv(cfg.patience, eval_iter), cooldown=ceildiv(cfg.cooldown, eval_iter))

        # create a function and a closure
        averaging_buffer_max_length = ceildiv(cfg.quantity_smoothness, eval_iter)
        if averaging_buffer_max_length <= 1:
            averaging_buffer_max_length = 1
        averaging_buffer = []
    else:
        raise RuntimeError(f"Unknown annel_lr type: {cfg.type}")


    def anneal_lr_func(i_iter, anneal_now=True):
        if cfg.type.lower() == "none":
            pass
        elif cfg.type.lower() == "MultiStepLR".lower():
            lr_scheduler.step()
        elif cfg.type.lower() == "ReduceLROnPlateau".lower():
            value_to_monitor = full_log[cfg.quantity_to_monitor][-1]
            averaging_buffer.append(value_to_monitor)
            if len(averaging_buffer) > averaging_buffer_max_length:
                averaging_buffer.pop(0)
            averaged_value = median(averaging_buffer)
            counter = len(full_log[cfg.quantity_to_monitor])
            if anneal_now:
                lr_scheduler.step(averaged_value)
        else:
            raise RuntimeError(f"Unknown annel_lr type: {cfg.type}")
        return get_learning_rate(optimizer)

    return lr_scheduler, anneal_lr_func
