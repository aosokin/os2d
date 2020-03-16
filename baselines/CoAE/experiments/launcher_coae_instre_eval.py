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
    log_path = os.path.abspath(os.path.join(coae_path, "output", "eval_instre"))

    exp_job_names = []
    exp_log_paths = []
    exp_commands = []
    exp_log_file_prefix = []


    def add_job(arch,
                eval_dataset,
                model_path,
                model_checkpoint,
                folder_suffix="",
                test_augment=None,
                extra_params=None,
                ):
        job_name = f"{config_job_name}.{eval_dataset}.{arch}"

        # set output folder
        log_folder = f"{config_job_name}"
        if folder_suffix:
            log_folder += "." + folder_suffix
        log_folder = os.path.join(log_path, log_folder)

        commands = []

        # stage 1
        d = OrderedDict()
        d["--cuda"] = ""
        if os.path.isfile(model_path):
            d["--weights"] = os.path.join(log_folder, model_path)
        else:
            d["--weights"] = os.path.join(model_path, model_checkpoint)
        d["--dataset"] = eval_dataset
        d["--net"] = arch
        if test_augment is not None:
            d["--class_image_augmentation"] = test_augment
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
        exp_log_file_prefix.append(f"eval_{eval_dataset}_{arch}_")

    query_size = 192
    scale = 900
    multi_scale_training = "ms"
    dataset_scale = {}
    # compute renormalizations by using the same ratio of scale between datasets as in OS2D
    dataset_scale["instre-s1"] = 700.0 / 1280.0
    dataset_scale["instre-s2"] = 600.0 / 1280.0

    job_id = 0
    for dataset in ["instre-s1", "instre-s2"]:
        for arch, init_net in zip(["res101", "res50"], ["Pytorch", "Caffe2"]):
            eval_dataset = f"{dataset}-test"
            dataset_train = f"{dataset}-train"
            cur_scale = int(scale * dataset_scale[dataset])
            scale_str = str(cur_scale) if multi_scale_training == "ss"\
                                       else ",".join(str(int(m * cur_scale)) for m in [0.5, 0.625, 0.8, 1, 1.2, 1.4, 1.6])

            best_job_name = f"{job_id}.{dataset_train}_{arch}_init{init_net}_query{query_size}_scale{cur_scale}_{multi_scale_training}"
            add_job(arch, eval_dataset,
                    "output/instre/coae." + best_job_name, "best_model_1.pth",
                    folder_suffix="best_train",
                    test_augment="rotation90",
                    extra_params={
                        "TRAIN.query_size": query_size,
                        "TRAIN.SCALES": f"[{scale_str}]",
                        "TEST.SCALES": f"[{cur_scale}]",
                    }
                    )
            job_id += 1


    for job_name, log_path, commands, log_file_prefix in zip(exp_job_names, exp_log_paths, exp_commands, exp_log_file_prefix):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands,
                         log_file_prefix=log_file_prefix)
    launcher.launch_all_jobs(args)
