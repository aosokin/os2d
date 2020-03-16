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
    config_job_name = "exp2"

    log_path = os.path.abspath(os.path.join(config_path, "..", "output/exp2"))

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
                sub_index,
                backbone_arch,
                init_model_nickname,
                init_model_path,
                extra_params=None,
                ):
        job_name = f"{config_job_name}.{sub_index}.{job_type}_seed{config['random_seed']}"

        d = OrderedDict()

        d["--config-file"] = config_file

        if job_type == "v1":
            d.update(config_dict_v1)
        elif job_type == "v2":
            d.update(config_dict_v2)
        else:
            raise RuntimeError("Unknown job_type {0}".format(job_type))

        d["model.backbone_arch"] = backbone_arch
        d["init.model"] = init_model_path

        log_folder = job_name + "_" + backbone_arch + "_init_" + init_model_nickname
        log_folder = os.path.join(log_path, log_folder)

        d["output.path"] = log_folder

        if extra_params:
            d.update(extra_params)

        commands = [main_command + " " + launcher.parameters_to_str(d)]

        exp_job_names.append(job_name)
        exp_commands.append(commands)
        exp_log_paths.append(log_folder)


    for job_type in ["v1", "v2"]:
        add_job(job_type, 0, "ResNet50", "fromScratch",
                "models/does_not_exist")
        add_job(job_type, 1, "ResNet50", "imageNetPth",
                "models/resnet50-19c8e357.pth")
        add_job(job_type, 2, "ResNet50", "imageNetCaffe2",
                "models/imagenet-caffe-resnet50-features-ac468af-converted.pth")
        add_job(job_type, 3, "ResNet50", "imageNetCaffe2GroupNorm",
                "models/resnet50_caffe2_groupnorm.pth", {"model.use_group_norm" : True})
        add_job(job_type, 4, "ResNet50", "cocoMaskrcnnFpn",
                "models/maskrcnn-benchmark/e2e_mask_rcnn_R_50_FPN_1x_converted.pth")
        add_job(job_type, 5, "ResNet101", "imageNetPth",
                "models/resnet101-5d3b4d8f.pth")
        add_job(job_type, 6, "ResNet101", "imageNetCaffe2",
                "models/imagenet-caffe-resnet101-features-10a101d-converted.pth")
        add_job(job_type, 7, "ResNet101", "buildingsCirtorch",
                "models/gl18-tl-resnet101-gem-w-a4d43db-converted.pth")
        add_job(job_type, 8, "ResNet101", "cocoMaskrcnnFpn",
                "models/maskrcnn-benchmark/e2e_mask_rcnn_R_101_FPN_1x_converted.pth")
        add_job(job_type, 9, "ResNet101", "pascalWeakalign",
                "models/weakalign_resnet101_affine_tps.pth.tar")

    for job_name, log_path, commands in zip(exp_job_names, exp_log_paths, exp_commands):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands)
    launcher.launch_all_jobs(args)
