import datetime
import logging
import os
import time
from typing import Optional, Callable

import numpy as np
import torch
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    validation_period,
    checkpoint_period,
    arguments,
    run_validation):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    saved_models = {}
    best_metric = float("-inf")
    best_model_iter = None

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        batch_start = time.time()
        arguments["iteration"] = iteration

        if iteration % validation_period == 0:
            results = validate_and_log(model, run_validation, iteration)

            first_dataset_results = next(iter(results.items()))
            dataset_name = first_dataset_results[0]
            metric_name = "mAP@0.5"
            metric = first_dataset_results[1][metric_name]

            if metric > best_metric:
                logger.info( f"Found a new current best model: iter {iteration}, {metric_name} on {dataset_name} = {metric:0.4f}" )
                # checkpoint the best model
                best_metric = metric
                best_model_iter = iteration
                model_filename = 'model_best'
                checkpointer.save(model_filename, **arguments)

        if iteration % checkpoint_period == 0:
            model_filename = 'model_{:07d}'.format(iteration)
            checkpointer.save(model_filename, **arguments)
            saved_models[iteration] = model_filename + '.pth'

        model.train()

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - batch_start
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        logger.info(
            meters.delimiter.join(
                [
                    "eta: {eta}",
                    "iter: {iter}",
                    "{meters}",
                    "lr: {lr:.6f}",
                    "max mem: {memory:.0f}",
                ]
            ).format(
                eta=eta_string,
                iter=iteration,
                meters=str(meters),
                lr=optimizer.param_groups[0]["lr"],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            )
        )

        losses_str = meters.delimiter.join( ["Loss: {:.4f}".format(losses.item())] + ["{0}: {1:.4f}".format(k, v.item()) for k,v in loss_dict_reduced.items()])
        logger.info(losses_str)

    validate_and_log(model, run_validation, arguments["iteration"])
    checkpointer.save("model_final", **arguments)

    if max_iter > 0:
        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )


def validate_and_log(model, run_validation, iteration):
    results = run_validation(model, iteration)
    return results
