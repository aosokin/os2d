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
    config_job_name = "exp1"

    log_path = os.path.abspath(os.path.join(config_path, "..", "output/exp1"))

    exp_job_names = []
    exp_log_paths = []
    exp_commands = []

    def add_job(job_name,
                sub_index,
                backbone_arch,
                init_model_nickname,
                init_model_path,
                extra_params=None,
                ):
        job_name = f"{config_job_name}.{sub_index}.{job_name}_seed{config['random_seed']}"

        d = OrderedDict()

        d["--config-file"] = config_file
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


    add_job("lossCL", 0, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": False,
                "model.use_simplified_affine_model": True,
                "train.objective.loc_weight": 0.2,
                "train.model.freeze_bn_transform": False,
                "train.objective.class_objective": "ContrastiveLoss",
                "train.objective.pos_margin": 1.0,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": False,
            }
            )
    add_job("lossRLL", 1, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": False,
                "model.use_simplified_affine_model": True,
                "train.objective.loc_weight": 0.2,
                "train.model.freeze_bn_transform": False,
                "train.objective.class_objective": "RLL",
                "train.objective.pos_margin": 0.6,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": False,
            }
            )
    add_job("lossRLL_remap", 2, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": False,
                "model.use_simplified_affine_model": True,
                "train.objective.loc_weight": 0.2,
                "train.model.freeze_bn_transform": False,
                "train.objective.class_objective": "RLL",
                "train.objective.pos_margin": 0.6,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": True,
            }
            )
    add_job("lossRLL_remap_mine", 3, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": False,
                "model.use_simplified_affine_model": True,
                "train.objective.loc_weight": 0.2,
                "train.model.freeze_bn_transform": False,
                "train.objective.class_objective": "RLL",
                "train.objective.pos_margin": 0.6,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": True,
                "train.mining.do_mining": True,
            }
            )
    add_job("lossRLL_remap_invFullAffine", 4, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": True,
                "model.use_simplified_affine_model": False,
                "train.objective.loc_weight": 0.2,
                "train.model.freeze_bn_transform": False,
                "train.objective.class_objective": "RLL",
                "train.objective.pos_margin": 0.6,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": True,
            }
            )
    add_job("lossRLL_remap_mine_fullAffine", 5, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": False,
                "model.use_simplified_affine_model": False,
                "train.objective.loc_weight": 0.2,
                "train.model.freeze_bn_transform": False,
                "train.objective.class_objective": "RLL",
                "train.objective.pos_margin": 0.6,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": True,
                "train.mining.do_mining": True,
            }
            )
    add_job("lossRLL_remap_invFullAffine_initTranform", 6, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": True,
                "model.use_simplified_affine_model": False,
                "train.objective.loc_weight": 0.2,
                "train.model.freeze_bn_transform": True,
                "train.objective.class_objective": "RLL",
                "train.objective.pos_margin": 0.6,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": True,
                "init.transform": "models/weakalign_resnet101_affine_tps.pth.tar",
            }
            )
    add_job("lossRLL_remap_invFullAffine_initTranform_zeroLocLoss", 7, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": True,
                "model.use_simplified_affine_model": False,
                "train.objective.loc_weight": 0.0,
                "train.model.freeze_bn_transform": True,
                "train.objective.class_objective": "RLL",
                "train.objective.pos_margin": 0.6,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": True,
                "init.transform": "models/weakalign_resnet101_affine_tps.pth.tar",
            }
            )
    add_job("lossRLL_remap_invFullAffine_initTranform_zeroLocLoss_mine", 8, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": True,
                "model.use_simplified_affine_model": False,
                "train.objective.loc_weight": 0.0,
                "train.model.freeze_bn_transform": True,
                "train.objective.class_objective": "RLL",
                "train.objective.pos_margin": 0.6,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": True,
                "train.mining.do_mining": True,
                "init.transform": "models/weakalign_resnet101_affine_tps.pth.tar",
            }
            )
    add_job("lossCL_invFullAffine_initTranform_zeroLocLoss", 9, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": True,
                "model.use_simplified_affine_model": False,
                "train.objective.loc_weight": 0.0,
                "train.model.freeze_bn_transform": True,
                "train.objective.class_objective": "ContrastiveLoss",
                "train.objective.pos_margin": 1.0,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": False,
                "init.transform": "models/weakalign_resnet101_affine_tps.pth.tar",
            }
            )
    add_job("lossCL_remap_invFullAffine_initTranform_zeroLocLoss", 10, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": True,
                "model.use_simplified_affine_model": False,
                "train.objective.loc_weight": 0.0,
                "train.model.freeze_bn_transform": True,
                "train.objective.class_objective": "ContrastiveLoss",
                "train.objective.pos_margin": 1.0,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": True,
                "init.transform": "models/weakalign_resnet101_affine_tps.pth.tar",
            }
            )

    add_job("lossRLL_invFullAffine_initTranform_zeroLocLoss", 11, "ResNet50", "imageNetCaffe2",
            "models/imagenet-caffe-resnet50-features-ac468af-converted.pth",
            {
                "model.use_inverse_geom_model": True,
                "model.use_simplified_affine_model": False,
                "train.objective.loc_weight": 0.0,
                "train.model.freeze_bn_transform": True,
                "train.objective.class_objective": "RLL",
                "train.objective.pos_margin": 0.6,
                "train.objective.neg_margin": 0.5,
                "train.objective.remap_classification_targets": False,
                "init.transform": "models/weakalign_resnet101_affine_tps.pth.tar",
            }
            )

    for job_name, log_path, commands in zip(exp_job_names, exp_log_paths, exp_commands):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands)
    launcher.launch_all_jobs(args)
