import os
import copy
from collections import OrderedDict

from os2d.utils import launcher as launcher


if __name__ == "__main__":
    # load default launcher parameters
    parser = launcher.create_args_parser()
    args = parser.parse_args()

    script_path = os.path.dirname(os.path.abspath(__file__))
    detector_root_path = os.path.join(script_path, "..")
    config_path = os.path.abspath(os.path.join(detector_root_path, "config"))
    log_path = os.path.abspath(os.path.join(detector_root_path, "output"))

    os2d_path = os.path.abspath(os.path.join(detector_root_path, "..", "..", ".."))

    main_command = f"PYTHONPATH={os2d_path}:$PYTHONPATH python {os.path.join(detector_root_path, 'train_detector.py')}"

    exp_job_names = []
    exp_log_paths = []
    exp_commands = []

    def add_job(job_id,
                model,
                dataset,
                init_weights,
                ):
        job_name = f"exp-{model}-{dataset}-{init_weights}"
        commands = []
        
        d = OrderedDict()
        if model == "R-50":
            if "pytorch" not in init_weights:
                config_file = os.path.join(config_path, "e2e_faster_rcnn_R_50_FPN_1x_multiscale_noClasses.yaml")
            else:
                config_file = os.path.join(config_path, "e2e_faster_rcnn_R_50_FPN_1x_multiscale_noClasses_fromPytorch.yaml")
        elif model == "R-101":
            if "pytorch" not in init_weights:
                config_file = os.path.join(config_path, "e2e_faster_rcnn_R_101_FPN_1x_multiscale_noClasses.yaml")
            else:
                config_file = os.path.join(config_path, "e2e_faster_rcnn_R_101_FPN_1x_multiscale_noClasses_fromPytorch.yaml")
        else:
            raise RuntimeError(f"Do not know config for model {model}")

        d["--validation_period"] = 5000
        d["--config-file"] = config_file

        if model == "R-50":
            if init_weights == "imagenet-repmet-pytorch":
                d["MODEL.WEIGHT"] = "../../../data/ImageNet-RepMet/pretrain/output/resnet50/model_best_maskrcnnbenchmark.pth.tar"
            elif init_weights == "imagenet-pytorch":
                d["MODEL.WEIGHT"] = "../../../models/resnet50-19c8e357.pth"
            elif init_weights == "imagenet-caffe":
                pass
            else:
                raise RuntimeError(f"Do not recognize weight initialization {init_weights}")
        elif model == "R-101":
            if init_weights == "imagenet-repmet-pytorch":
                d["MODEL.WEIGHT"] = "../../../data/ImageNet-RepMet/pretrain/output/resnet101/model_best_maskrcnnbenchmark.pth.tar"
            elif init_weights == "imagenet-pytorch":
                d["MODEL.WEIGHT"] = "../../../models/resnet101-5d3b4d8f.pth"
            elif init_weights == "imagenet-caffe":
                pass
            else:
                raise RuntimeError(f"Do not recognize weight initialization {init_weights}")
        else:
            raise RuntimeError(f"Do not know config for model {model}")

        if dataset == "imagenet-repmet":
            d["DATASETS.TRAIN"] = "[\\\"imagenet-repmet-train\\\"]"
            d["DATASETS.TEST"] = "[\\\"imagenet-repmet-val-5000\\\"]" # crop val set from 50k images to 5k GT boxes
            d["INPUT.MIN_SIZE_TRAIN"] = "[225,280,360,450,540,630,720]"
            d["INPUT.MAX_SIZE_TRAIN"] = 2048
            d["INPUT.MIN_SIZE_TEST"] = 450
            d["INPUT.MAX_SIZE_TEST"] = 1280
        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        log_folder = os.path.join(log_path, job_name)

        d["OUTPUT_DIR"] = log_folder

        commands.append(main_command + " " + launcher.parameters_to_str(d))

        # testing
        d_testing = OrderedDict()
        d_testing["--test_weights"] = os.path.join(log_folder, "model_best.pth")
        d_testing.update(d)

        datasets_test = ["[\\\"imagenet-repmet-val-5000\\\"]"]
        scales_test = ["[180,225,288,360,432,504,576]"]

        for dataset, scales in zip(datasets_test, scales_test):
            d_testing_local = copy.deepcopy(d_testing)
            d_testing_local["DATASETS.TEST"] = dataset
            d_testing_local["TEST.BBOX_AUG.ENABLED"] = True
            d_testing_local["TEST.BBOX_AUG.SCALES"] = scales

            commands.append(main_command + " " + launcher.parameters_to_str(d_testing_local))

        exp_job_names.append(job_name)
        exp_commands.append(commands)
        exp_log_paths.append(log_folder)


    job_id = 0
    dataset = "imagenet-repmet"
    # Train class-agnostic detectors for the detector-retrieval baseline
    for model in ["R-101"]:# ["R-50", "R-101"]:
        for init_weights in ["imagenet-repmet-pytorch"]: # ["imagenet-repmet-pytorch", "imagenet-pytorch", "imagenet-caffe"]:
            add_job(job_id, model, dataset, init_weights)
            job_id += 1

    for job_name, log_path, commands in zip(exp_job_names, exp_log_paths, exp_commands):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands)
    launcher.launch_all_jobs(args)
