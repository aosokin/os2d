import os
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

    log_folder_path = f"{full_baseline_path}/output/instre"
    config_job_name = "eval_instre"

    full_eval_launcher = f"{full_baseline_path}/main_detector_retrieval.py"
    command_testing = python_path + " " + f"python {full_eval_launcher}"

    exp_commands = []
    exp_job_names = []
    exp_log_paths = []
    exp_log_file_prefix = []

    def add_job(training_dataset,
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
                class_image_augmentation = "",
                folder_suffix=""):

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

        # add testing on top of the detector
        d_testing = OrderedDict()
        if retrieval_multiscale == "ms":
            d_testing["--retrieval_multiscale"] = ""

        if arch == "resnet50":
            d_testing["--maskrcnn_config_file"] = os.path.join(detector_path, "config", "e2e_faster_rcnn_R_50_FPN_1x_multiscale_noClasses.yaml")
        elif arch == "resnet101":
            d_testing["--maskrcnn_config_file"] = os.path.join(detector_path, "config", "e2e_faster_rcnn_R_101_FPN_1x_multiscale_noClasses.yaml")
        else:
            raise RuntimeError(f"Unknown arch {arch}")

        if training_dataset.startswith("instre-s1"):
            if arch == "resnet50":
                d_testing["--maskrcnn_weight_file"] = os.path.join(detector_path, "output", "exp0001-R-50-noCl-instre-s1/model_best.pth")
            elif arch == "resnet101":
                d_testing["--maskrcnn_weight_file"] = os.path.join(detector_path, "output", "exp0004-R-101-noCl-instre-s1/model_best.pth")
            else:
                raise RuntimeError(f"Unknown arch {arch}")
        elif training_dataset.startswith("instre-s2"):
            if arch == "resnet50":
                d_testing["--maskrcnn_weight_file"] = os.path.join(detector_path, "output", "exp0002-R-50-noCl-instre-s2/model_best.pth")
            elif arch == "resnet101":
                d_testing["--maskrcnn_weight_file"] = os.path.join(detector_path, "output", "exp0005-R-101-noCl-instre-s2/model_best.pth")
            else:
                raise RuntimeError(f"Unknown arch {arch}")
        else:
            raise f"Unknown training set {training_dataset}"

        if use_best_model:
            d_testing["--retrieval_network_path"] = os.path.join(retrieval_path, "output", "instre", directory, "model_best.pth.tar")
        else:
            d_testing["--retrieval_network_path"] = os.path.join(retrieval_path, "output", "instre", directory, "model_epoch0.pth.tar")


        d_testing["--retrieval_image_size"] = image_size
        d_testing["is_cuda"] = "True"
        if "instre-s1" in eval_dataset:
            d_testing["eval.dataset_names"] = f"\"[\\\"{eval_dataset}\\\"]\""
            d_testing["eval.dataset_scales"] = "\"[700]\""
        elif "instre-s2" in eval_dataset:
            d_testing["eval.dataset_names"] = f"\"[\\\"{eval_dataset}\\\"]\""
            d_testing["eval.dataset_scales"] = "\"[600]\""
        else:
            raise RuntimeError(f"Unknown eval dataset {eval_dataset}")

        d_testing["eval.mAP_iou_thresholds"] = "\"[0.5]\""
        if class_image_augmentation:
            d_testing["eval.class_image_augmentation"] = class_image_augmentation

        commands.append(command_testing + " " + launcher.parameters_to_str(d_testing))

        exp_job_names.append(job_name)
        exp_log_paths.append(log_path)
        exp_commands.append(commands)
        exp_log_file_prefix.append(f"eval_{eval_dataset}_{'bestModel' if use_best_model else 'initModel'}_{retrieval_multiscale}_")


    datasets = ["instre-s1", "instre-s2"]
    archs = ["resnet50", "resnet101"]
    pool = "gem"
    retrieval_multiscale = "ms"
    class_image_augmentation = "rotation90"

    for dataset in datasets:
        training_dataset = dataset + "-train-retrieval-rndCropPerImage10"
        eval_dataset = dataset + "-test"
        for arch in archs:
            add_job(training_dataset=training_dataset, arch=arch, pool=pool, whitening=True,
                    retrieval_multiscale=retrieval_multiscale,
                    eval_dataset=eval_dataset, use_best_model=True,
                    class_image_augmentation=class_image_augmentation,
                    folder_suffix=f"det-ret-baseline-train-{arch}-{dataset}")


    for job_name, log_path, commands, log_file_prefix in zip(exp_job_names, exp_log_paths, exp_commands, exp_log_file_prefix):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands,
                         log_file_prefix=log_file_prefix)
    launcher.launch_all_jobs(args)
