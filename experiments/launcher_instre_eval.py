import os
import yaml
import copy
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
    config_job_name = "eval_instre"

    log_path = os.path.abspath(os.path.join(config_path, "..", "output/eval_instre"))

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
        commands = []

        d = OrderedDict()
        d["--config-file"] = config_file

        if job_type == "v1":
            d.update(config_dict_v1)
        elif job_type == "v2":
            d.update(config_dict_v2)
        else:
            raise RuntimeError("Unknown job_type {0}".format(job_type))

        train_data = eval_dataset + "-train"
        d["train.dataset_name"] = "\"" + train_data + "\""
        if train_data == "instre-s1-train":
            d["train.dataset_scale"] = 700.0
            main_val_dataset = "instre-s1-val"
            d["eval.dataset_scales"] = "[700.0]"
        elif train_data == "instre-s2-train":
            d["train.dataset_scale"] = 600.0
            main_val_dataset = "instre-s2-val"
            d["eval.dataset_scales"] = "[600.0]"
        else:
            raise RuntimeError(f"Unknown dataset {train_data}")

        d["output.best_model.dataset"] = main_val_dataset
        d["eval.dataset_names"] = f"[\\\"{main_val_dataset}\\\"]"

        d["eval.class_image_augmentation"] = "rotation90"
        d["eval.iter"] = 5000

        # extra augmentation for this run
        d["train.augment.mine_extra_class_images"] = True

        d["model.backbone_arch"] = backbone_arch

        if extra_params:
            d.update(extra_params)

        # set output folder
        log_folder = f"{config_job_name}"
        if folder_suffix:
            log_folder += "." + folder_suffix
        log_folder = os.path.join(log_path, log_folder)

        d["train.do_training"] = False
        if train_data == "instre-s1-train":
            d["eval.dataset_names"] = "[\\\"instre-s1-test\\\"]"
            d["eval.dataset_scales"] = "[700.0]"
        elif train_data == "instre-s2-train":
            d["eval.dataset_names"] = "[\\\"instre-s2-test\\\"]"
            d["eval.dataset_scales"] = "[600.0]"
        else:
            raise RuntimeError(f"Unknown dataset {train_data}")

        d["eval.class_image_augmentation"] = "rotation90"

        # choose init
        if "init.transform" in d:
            del d["init.transform"]
        if os.path.isfile(model_path):
            d["init.model"] = model_path
        else:
            d["init.model"] = os.path.join(model_path, model_checkpoint)

        commands.append(main_command + " " + launcher.parameters_to_str(d))

        exp_job_names.append(job_name)
        exp_commands.append(commands)
        exp_log_paths.append(log_folder)
        exp_log_file_prefix.append(f"eval_{eval_dataset}_scale{d['eval.dataset_scales'][1:-1]}_")


    index = 0
    for dataset in ["instre-s1", "instre-s2"]:
        for job_type in ["v1", "v2"]:
            for train_type in ["train", "init"]:

                if train_type == "train":
                    model_checkpoint = f"checkpoint_best_model_{dataset}-val_mAP@0.50.pth"
                else:
                    model_checkpoint = f"checkpoint_iter_0.pth"

                for num_layers in [50, 101]:
                    add_job(index, job_type, f"ResNet{num_layers}", dataset,
                            model_path=f"output/exp3/exp3.R{num_layers}.{job_type}_{dataset}-train_seed0_ResNet{num_layers}_init_imageNetCaffe2",
                            model_checkpoint=model_checkpoint,
                            folder_suffix=f"{dataset}_{job_type}-{train_type}_ResNet{num_layers}_imageNetCaffe2")
                    index += 1


    for job_name, log_path, commands, log_file_prefix in zip(exp_job_names, exp_log_paths, exp_commands, exp_log_file_prefix):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands,
                         log_file_prefix=log_file_prefix)
    launcher.launch_all_jobs(args)
