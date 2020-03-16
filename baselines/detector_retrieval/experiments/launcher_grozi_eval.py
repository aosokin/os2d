import os
import copy
from collections import OrderedDict

from os2d.utils import launcher as launcher


if __name__ == "__main__":
    # load default launcher parameters
    parser = launcher.create_args_parser()
    args = parser.parse_args()

    script_path = os.path.dirname(os.path.abspath(__file__))
    full_baseline_path = os.path.join(script_path, "..")

    retrieval_path = os.path.join(full_baseline_path, "retrieval")

    os2d_path = os.path.join(full_baseline_path, "..", "..")
    cirtorch_path = os.path.join(retrieval_path, "cnnimageretrieval-pytorch")
    detector_path = os.path.join(full_baseline_path, "detector")
    python_path = f"PYTHONPATH={os2d_path}:{cirtorch_path}"

    log_folder_path = f"{full_baseline_path}/output/grozi"
    config_job_name = "eval_grozi"

    full_eval_launcher = f"{full_baseline_path}/main_detector_retrieval.py"
    command_testing = python_path + " " + f"python {full_eval_launcher}"

    exp_commands = []
    exp_job_names = []
    exp_log_paths = []
    exp_log_file_prefix = []

    def add_job(training_dataset, # "grozi-train-retrieval" and "grozi-train-retrieval-rndCropPerImage10grozi_retrieval"
                arch, # "resnet50" or "resnet101"
                pool = "gem", # "mac", "spoc", "gem"
                test_whiten = False, # True or False
                local_whitening = False, # True or False
                regional = False, # True or False
                whitening = False, # True or False
                loss_margin = 0.85, # True or False
                image_size = 240, # 240, 360, 500, 750, 1024 
                learning_rate = 1e-6, # 1e-6, 5e-7, 5e-6
                pretrained = True,
                loss = "contrastive",
                optimizer = "adam",
                weight_decay = 1e-4,
                neg_num = 5, 
                query_size = 2000, 
                pool_size = 20000,
                batch_size = 5,
                eval_dataset = "",
                use_best_model = True,
                retrieval_multiscale = "ss",
                folder_suffix="",
                ):

        directory = "{}".format(training_dataset)
        directory += "_{}".format(arch)
        directory += "_{}".format(pool)
        if local_whitening:
            directory += "_lwhiten"
        if regional:
            directory += "_r"
        if whitening:
            directory += "_whiten"
        if not pretrained:
            directory += "_notpretrained"
        directory += "_{}_m{:.2f}".format(loss, loss_margin)
        directory += "_{}_lr{:.1e}_wd{:.1e}".format(optimizer, learning_rate, weight_decay)
        directory += "_nnum{}_qsize{}_psize{}".format(neg_num, query_size, pool_size)
        directory += "_bsize{}_imsize{}".format(batch_size, image_size)

        job_name = f"{config_job_name}.{eval_dataset}"
        # set output folder
        log_path = config_job_name
        if folder_suffix:
            log_path += "." + folder_suffix
        log_path = os.path.join(log_folder_path, log_path)

        commands = []

        # testing on top of the detector
        d_testing = OrderedDict()
        if retrieval_multiscale == "ms":
            d_testing["--retrieval_multiscale"] = ""
        
        if arch == "resnet50":
            d_testing["--maskrcnn_config_file"] = os.path.join(detector_path, "config", "e2e_faster_rcnn_R_50_FPN_1x_multiscale_noClasses.yaml")
        elif arch == "resnet101":
            d_testing["--maskrcnn_config_file"] = os.path.join(detector_path, "config", "e2e_faster_rcnn_R_101_FPN_1x_multiscale_noClasses.yaml")
        else:
            raise RuntimeError(f"Unknown arch {arch}")
        
        if training_dataset.startswith("grozi"):
            if arch == "resnet50":
                d_testing["--maskrcnn_weight_file"] = os.path.join(detector_path, "output", "exp0000-R-50-noCl-grozi", "model_best.pth")
            elif arch == "resnet101":
                d_testing["--maskrcnn_weight_file"] = os.path.join(detector_path, "output", "exp0003-R-101-noCl-grozi", "model_best.pth")
            else:
                raise RuntimeError(f"Unknown arch {arch}")
        else:
            raise f"Unknown training set {training_dataset}"

        if use_best_model:
            d_testing["--retrieval_network_path"] = os.path.join(retrieval_path, "output", "grozi", directory, "model_best.pth.tar")
        else:
            d_testing["--retrieval_network_path"] = os.path.join(retrieval_path, "output", "grozi", directory, "model_epoch0.pth.tar")

        d_testing["--retrieval_image_size"] = image_size
        d_testing["is_cuda"] = "True"

        assert training_dataset.startswith("grozi"), f"Unknown training set {training_dataset}"

        if eval_dataset == "grozi-val-new-cl":
            d_testing["eval.dataset_names"] = "\"[\\\"grozi-val-new-cl\\\"]\""
            d_testing["eval.dataset_scales"] = "\"[1280.0]\""
        elif eval_dataset == "grozi-val-old-cl":
            d_testing["eval.dataset_names"] = "\"[\\\"grozi-val-old-cl\\\"]\""
            d_testing["eval.dataset_scales"] = "\"[1280.0]\""
        elif eval_dataset == "dairy":
            d_testing["eval.dataset_names"] = "\"[\\\"dairy\\\"]\""
            d_testing["eval.dataset_scales"] = "\"[3500.0]\""
        elif eval_dataset == "paste-v":
            d_testing["eval.dataset_names"] = "\"[\\\"paste-v\\\"]\""
            d_testing["eval.dataset_scales"] = "\"[3500.0]\""
        elif eval_dataset == "paste-f":
            d_testing["eval.dataset_names"] = "\"[\\\"paste-f\\\"]\""
            d_testing["eval.dataset_scales"] = "\"[3500.0]\""
            # eval with rotations
            d_testing["eval.class_image_augmentation"] = "rotation90"
        else:
            raise f"Unknown eval set {eval_dataset}"
        
        d_testing["eval.mAP_iou_thresholds"] = "\"[0.5]\""
        commands.append(command_testing + " " + launcher.parameters_to_str(d_testing))

        exp_job_names.append(job_name)
        exp_log_paths.append(log_path)
        exp_commands.append(commands)
        exp_log_file_prefix.append(f"eval_{eval_dataset}_{'bestModel' if use_best_model else 'initModel'}_{retrieval_multiscale}_")


    training_dataset = "grozi-train-retrieval-rndCropPerImage10"
    arch = "resnet101"
    pool = "gem"
    retrieval_multiscale ="ms"

    for eval_dataset in ["grozi-val-new-cl", "grozi-val-old-cl", "dairy", "paste-v", "paste-f"]:
        add_job(training_dataset=training_dataset, arch=arch, pool=pool, whitening=True,
                retrieval_multiscale=retrieval_multiscale,
                eval_dataset=eval_dataset, use_best_model=True,
                folder_suffix="det-ret-baseline-train")

        add_job(training_dataset=training_dataset, arch=arch, pool=pool, whitening=False,
                retrieval_multiscale=retrieval_multiscale,
                eval_dataset=eval_dataset, use_best_model=False,
                folder_suffix="det-ret-baseline-init")



    for job_name, log_path, commands, log_file_prefix in zip(exp_job_names, exp_log_paths, exp_commands, exp_log_file_prefix):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands,
                         log_file_prefix=log_file_prefix)
    launcher.launch_all_jobs(args)
