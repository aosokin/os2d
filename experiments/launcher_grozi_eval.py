import os
import yaml
from collections import OrderedDict

from os2d.utils import launcher as launcher


def load_yaml(config_file):
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    return config


if __name__ == "__main__":
    # load default launcher parameters
    parser = launcher.create_args_parser()
    args = parser.parse_args()

    main_command = "python main.py"

    config_path = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(config_path, "config_training.yml")
    config = load_yaml(config_file)
    config_job_name = "eval_grozi"

    log_path = os.path.abspath(os.path.join(config_path, "..", "output/eval_grozi"))

    config_dict_v1 = OrderedDict()
    config_dict_v1["model.use_inverse_geom_model"] = False
    config_dict_v1["model.use_simplified_affine_model"] = True
    config_dict_v1["train.objective.loc_weight"] = 0.2
    config_dict_v1["train.model.freeze_bn_transform"] = False

    config_dict_v2 = OrderedDict()
    config_dict_v2["model.use_inverse_geom_model"] = True
    config_dict_v2["model.use_simplified_affine_model"] = False
    config_dict_v2["train.objective.loc_weight"] = 0.0
    config_dict_v2["train.model.freeze_bn_transform"] = True
    config_dict_v2["init.transform"] = "models/weakalign_resnet101_affine_tps.pth.tar"

    exp_job_names = []
    exp_log_paths = []
    exp_commands = []
    exp_log_file_prefix = []


    def add_job(sub_index,
                job_type, # "v1" or "v2"
                backbone_arch,
                eval_dataset,
                model_path,
                model_checkpoint,
                folder_suffix="",
                extra_params=None):
        job_name = f"{config_job_name}.{sub_index}.{eval_dataset}"

        d = OrderedDict()

        d["--config-file"] = config_file

        if job_type == "v1":
            d.update(config_dict_v1)
        elif job_type == "v2":
            d.update(config_dict_v2)
        else:
            raise RuntimeError("Unknown job_type {0}".format(job_type))

        d["model.backbone_arch"] = backbone_arch

        if extra_params:
            d.update(extra_params)

        # set output folder
        log_folder = f"{config_job_name}.{sub_index}"
        if folder_suffix:
            log_folder += "." + folder_suffix
        log_folder = os.path.join(log_path, log_folder)

        # evaluation params
        d["train.do_training"] = False
        d["eval.mAP_iou_thresholds"] = "\"[0.5]\""
        d["eval.train_subset_for_eval_size"] = 0

        # choose init
        if "init.transform" in d:
            del d["init.transform"]
        if os.path.isfile(model_path):
            d["init.model"] = model_path
        else:
            d["init.model"] = os.path.join(model_path, model_checkpoint)

        # choose eval dataset
        if eval_dataset == "grozi-val-new-cl":
            d["eval.dataset_names"] = "\"[\\\"grozi-val-new-cl\\\"]\""
            d["eval.dataset_scales"] = "[1280.0]"
        elif eval_dataset == "grozi-val-old-cl":
            d["eval.dataset_names"] = "\"[\\\"grozi-val-old-cl\\\"]\""
            d["eval.dataset_scales"] = "[1280.0]"
        elif eval_dataset == "dairy":
            d["eval.dataset_names"] = "\"[\\\"dairy\\\"]\""
            d["eval.dataset_scales"] = "[3500.0]"
        elif eval_dataset == "paste-v":
            d["eval.dataset_names"] = "\"[\\\"paste-v\\\"]\""
            d["eval.dataset_scales"] = "[3500.0]"
        elif eval_dataset == "paste-f":
            d["eval.dataset_names"] = "\"[\\\"paste-f\\\"]\""
            d["eval.dataset_scales"] = "[2000.0]"
            # eval with rotations
            d["eval.class_image_augmentation"] = "rotation90"
        else:
            raise f"Unknown eval set {eval_dataset}"

        commands = [main_command + " " + launcher.parameters_to_str(d)]

        exp_job_names.append(job_name)
        exp_commands.append(commands)
        exp_log_paths.append(log_folder)
        exp_log_file_prefix.append(f"eval_{eval_dataset}_scale{d['eval.dataset_scales'][1:-1]}_")


    for eval_dataset in ["grozi-val-new-cl", "grozi-val-old-cl", "dairy", "paste-v", "paste-f"]:
        # Best v1 model
        add_job(0, "v1", "ResNet101", eval_dataset,
                "output/exp2/exp2.7.v1_seed0_ResNet101_init_buildingsCirtorch",
                model_checkpoint="checkpoint_best_model_grozi-val-new-cl_mAP@0.50.pth",
                folder_suffix="best_V1-train")
        # Best v2 trained model
        add_job(1, "v2", "ResNet50", eval_dataset,
                "output/exp1/exp1.8.lossRLL_remap_invFullAffine_initTranform_zeroLocLoss_mine_seed0_ResNet50_init_imageNetCaffe2",
                model_checkpoint="checkpoint_best_model_grozi-val-new-cl_mAP@0.50.pth",
                folder_suffix="best_V2-train")
        # Best v2 init model
        add_job(2, "v2", "ResNet50", eval_dataset,
                "output/exp1/exp1.8.lossRLL_remap_invFullAffine_initTranform_zeroLocLoss_mine_seed0_ResNet50_init_imageNetCaffe2",
                model_checkpoint="checkpoint_iter_0.pth",
                folder_suffix="best_V2-init")
        # Sliding window baseline (identity transformation model)
        add_job(3, "v1", "ResNet101", eval_dataset,
                "output/exp2/exp2.7.v1_seed0_ResNet101_init_buildingsCirtorch",
                model_checkpoint="checkpoint_iter_0.pth",
                folder_suffix="best_V1-init")


    for job_name, log_path, commands, log_file_prefix in zip(exp_job_names, exp_log_paths, exp_commands, exp_log_file_prefix):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands,
                         log_file_prefix=log_file_prefix)
    launcher.launch_all_jobs(args)
