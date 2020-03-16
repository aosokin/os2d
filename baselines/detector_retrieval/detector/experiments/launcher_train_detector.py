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
                use_classes,
                dataset,
                ):
        job_name = f"exp{job_id:04}-{model}-{'withCl' if use_classes else 'noCl'}-{dataset}"
        commands = []
        
        d = OrderedDict()
        if model == "R-50" and use_classes:
            config_file = os.path.join(config_path, "e2e_faster_rcnn_R_50_FPN_1x_multiscale.yaml")
        elif model == "R-101" and use_classes:
            config_file = os.path.join(config_path, "e2e_faster_rcnn_R_101_FPN_1x_multiscale.yaml")
        elif model == "R-50" and not use_classes:
            config_file = os.path.join(config_path, "e2e_faster_rcnn_R_50_FPN_1x_multiscale_noClasses.yaml")
        elif model == "R-101" and not use_classes:
            config_file = os.path.join(config_path, "e2e_faster_rcnn_R_101_FPN_1x_multiscale_noClasses.yaml")
        else:
            raise RuntimeError(f"Do not know config for model {model} and use_classes {use_classes}")

        d["--config-file"] = config_file

        if dataset == "grozi":
            d["DATASETS.TRAIN"] = "[\\\"grozi-train\\\"]"
            if use_classes:
                d["DATASETS.TEST"] = "[\\\"grozi-val-old-cl\\\"]"
            else:
                d["DATASETS.TEST"] = "[\\\"grozi-val-all\\\"]"
            d["INPUT.MIN_SIZE_TRAIN"] = "[480,600,768,960,1152,1344,1536]"
            d["INPUT.MAX_SIZE_TRAIN"] = 2048
            d["INPUT.MIN_SIZE_TEST"] = 960
            d["INPUT.MAX_SIZE_TEST"] = 1280
        elif dataset == "instre-s1":
            d["DATASETS.TRAIN"] = "[\\\"instre-s1-train\\\"]"
            d["DATASETS.TEST"] = "[\\\"instre-s1-val\\\"]"
            d["INPUT.MIN_SIZE_TRAIN"] = "[210,262,336,420,504,588,672]"
            d["INPUT.MAX_SIZE_TRAIN"] = 2048
            d["INPUT.MIN_SIZE_TEST"] = 420
            d["INPUT.MAX_SIZE_TEST"] = 1280
        elif dataset == "instre-s2":
            d["DATASETS.TRAIN"] = "[\\\"instre-s2-train\\\"]"
            d["DATASETS.TEST"] = "[\\\"instre-s2-val\\\"]"
            d["INPUT.MIN_SIZE_TRAIN"] = "[180,225,288,360,432,504,576]"
            d["INPUT.MAX_SIZE_TRAIN"] = 2048
            d["INPUT.MIN_SIZE_TEST"] = 360
            d["INPUT.MAX_SIZE_TEST"] = 1280
        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        log_folder = os.path.join(log_path, job_name)

        d["OUTPUT_DIR"] = log_folder

        commands.append(main_command + " " + launcher.parameters_to_str(d))

        # testing
        if not use_classes:
            d_testing = OrderedDict()
            d_testing["--test_weights"] = os.path.join(log_folder, "model_best.pth")
            d_testing.update(d)

            datasets_test = ["[\\\"grozi-val-all\\\"]",
                             "[\\\"instre-s1-val\\\",\\\"instre-s1-test\\\"]",
                             "[\\\"instre-s2-val\\\",\\\"instre-s2-test\\\"]"]
            scales_test = ["[480,600,768,960,1152,1344,1536]",
                           "[210,262,336,420,504,588,672]",
                           "[180,225,288,360,432,504,576]"]

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
    # Train class-agnostic detectors for the detector-retrieval baseline
    for model in ["R-50", "R-101"]:
        for dataset in ["grozi", "instre-s1", "instre-s2"]:
            add_job(job_id, model, False, dataset)
            job_id += 1

    # Train class-aware detectors as baselines
    for model in ["R-50", "R-101"]:
        add_job(job_id, model, True, "grozi")
        job_id += 1


    for job_name, log_path, commands in zip(exp_job_names, exp_log_paths, exp_commands):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands)
    launcher.launch_all_jobs(args)
