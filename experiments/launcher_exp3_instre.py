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
    config_job_name = "exp3"
    
    log_path = os.path.abspath(os.path.join(config_path, "..", "output/exp3"))

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

    def add_job(job_type, # "v1" or "v2"
                job_id,
                backbone_arch,
                init_model_nickname,
                init_model_path,
                extra_params=None,
                train_data="",
                train_data_scale=None):
        job_name = "{0}.{1}.{2}_{3}_seed{4}".format(config_job_name, job_id, job_type, train_data, config["random_seed"])

        d = OrderedDict()
        d["--config-file"] = config_file

        if job_type == "v1":
            d.update(config_dict_v1)
        elif job_type == "v2":
            d.update(config_dict_v2)
        else:
            raise RuntimeError("Unknown job_type {0}".format(job_type))

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
        d["init.model"] = init_model_path

        log_folder = job_name + "_" + backbone_arch + "_init_" + init_model_nickname
        log_folder = os.path.join(log_path, log_folder)

        d["output.path"] = log_folder

        if extra_params:
            d.update(extra_params)

        commands = []
        commands.append(main_command + " " + launcher.parameters_to_str(d))

        exp_job_names.append(job_name)
        exp_commands.append(commands)
        exp_log_paths.append(log_folder)


    for train_data in ["instre-s1-train", "instre-s2-train"]:
        for job_type in ["v1", "v2"]:
            add_job(job_type, "R50", "ResNet50", "imageNetCaffe2",
                    "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
                    train_data=train_data)
            add_job(job_type, "R101", "ResNet101", "imageNetCaffe2",
                    "models/imagenet-caffe-resnet101-features-10a101d-converted.pth",
                    train_data=train_data)


    for job_name, log_path, commands in zip(exp_job_names, exp_log_paths, exp_commands):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands)
    launcher.launch_all_jobs(args)
