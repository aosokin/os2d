import os
from collections import OrderedDict

from os2d.utils import launcher as launcher


if __name__ == "__main__":
    # load default launcher parameters
    parser = launcher.create_args_parser()
    args = parser.parse_args()

    script_path = os.path.dirname(os.path.abspath(__file__))
    coae_path = os.path.join(script_path, "..")
    os2d_path = os.path.join(coae_path, "..", "..")

    train_launcher_path = os.path.join(coae_path, "test_net.py")
    main_command = f"PYTHONPATH={coae_path}:{os2d_path}:$PYTHONPATH python {train_launcher_path}"

    data_path = os.path.abspath(os.path.join(coae_path, "data"))

    config_job_name = "coae"
    log_path = os.path.abspath(os.path.join(coae_path, "output", "eval_grozi"))

    exp_job_names = []
    exp_log_paths = []
    exp_commands = []
    exp_log_file_prefix = []


    def add_job(arch,
                eval_dataset,
                model_path,
                model_checkpoint,
                folder_suffix="",
                extra_params=None,
                ):
        job_name = f"{config_job_name}.{eval_dataset}.{arch}"

        # set output folder
        log_folder = f"{config_job_name}"
        if folder_suffix:
            log_folder += "." + folder_suffix
        log_folder = os.path.join(log_path, log_folder)

        commands = []

        d = OrderedDict()
        d["--cuda"] = ""
        if os.path.isfile(model_path):
            d["--weights"] = os.path.join(log_folder, model_path)
        else:
            d["--weights"] = os.path.join(model_path, model_checkpoint)
        d["--dataset"] = eval_dataset
        if eval_dataset == "paste-f":
            d["--class_image_augmentation"] = "rotation90"
        d["--net"] = arch
        d["--set"] = ""
        d["TRAIN.USE_FLIPPED"] = "False"
        d["DATA_DIR"] = data_path
        # put smth here, but those are not used in CoAE
        d["TRAIN.MAX_SIZE"] = "5000"
        d["TEST.MAX_SIZE"] = "5000"

        if extra_params:
            d.update(extra_params)

        commands += [main_command + " " + launcher.parameters_to_str(d)]

        exp_job_names.append(job_name)
        exp_commands.append(commands)
        exp_log_paths.append(log_folder)
        exp_log_file_prefix.append(f"eval_{eval_dataset}_")


    sub_index = 0
    arch = "res101"
    init_net_name = "Pytorch"
    query_size = 192
    scale = 900
    multi_scale_training = "ms"
    best_job_name = f"{sub_index}.{arch}_init{init_net_name}_query{query_size}_scale{scale}_{multi_scale_training}"

    dataset_scale = {}
    # compute renormalizations by using the same ratio of scale between datasets as in OS2D
    dataset_scale["grozi-val-new-cl"] = 1
    dataset_scale["grozi-val-old-cl"] = 1
    dataset_scale["dairy"] = 3500.0 / 1280.0
    dataset_scale["paste-v"] = 3500.0 / 1280.0
    dataset_scale["paste-f"] = 3500.0 / 1280.0

    for eval_dataset in ["grozi-val-new-cl", "grozi-val-old-cl", "dairy", "paste-v", "paste-f"]:
        cur_scale = int(scale * dataset_scale[eval_dataset])
        scale_str = str(cur_scale) if multi_scale_training == "ss"\
                                   else ",".join(str(int(m * cur_scale)) for m in [0.5, 0.625, 0.8, 1, 1.2, 1.4, 1.6])
        add_job(arch, eval_dataset,
                "output/grozi/coae." + best_job_name, "best_model_1.pth",
                folder_suffix="best_train",
                extra_params={
                    "TRAIN.query_size": query_size,
                    "TRAIN.SCALES": f"[{scale_str}]",
                    "TEST.SCALES": f"[{cur_scale}]",
                }
                )


    for job_name, log_path, commands, log_file_prefix in zip(exp_job_names, exp_log_paths, exp_commands, exp_log_file_prefix):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands,
                         log_file_prefix=log_file_prefix)
    launcher.launch_all_jobs(args)
