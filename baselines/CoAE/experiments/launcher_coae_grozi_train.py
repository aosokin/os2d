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

    train_launcher_path = os.path.join(coae_path, "trainval_net.py")
    main_command = f"PYTHONPATH={coae_path}:{os2d_path}:$PYTHONPATH python {train_launcher_path}"

    data_path = os.path.abspath(os.path.join(coae_path, "data"))

    config_job_name = "coae"
    log_path = os.path.abspath(os.path.join(coae_path, "output", "grozi"))

    exp_job_names = []
    exp_log_paths = []
    exp_commands = []


    def add_job(job_name,
                sub_index,
                arch,
                init_net,
                extra_params=None,
                ):
        job_name = f"{config_job_name}.{sub_index}.{job_name}"

        log_folder = job_name
        log_folder = os.path.join(log_path, log_folder)

        commands = []

        # stage 1
        d = OrderedDict()
        d["--cuda"] = ""
        d["--dataset"] = "grozi-train"
        d["--dataset_val"] = "grozi-val-new-cl"
        d["--init_weights"] = init_net
        d["--disp_interval"] = "1"
        d["--val_interval"] = "10"
        d["--nw"] = "4"
        d["--bs"] = "4"
        d["--s"] = 1
        d["--epochs"] = "2000"
        d["--lr_decay_milestones"] = "1000 1500"
        d["--lr"] = 0.01 # default starting learning rate
        d["--lr_decay_gamma"] = 0.1
        d["--lr_reload_best_after_decay"] = "True"
        d["--save_dir"] = log_folder
        d["--net"] = arch
        d["--set"] = ""
        d["DATA_DIR"] = data_path
        # put smth here, but those are not used in CoAE
        d["TRAIN.MAX_SIZE"] = "3000"
        d["TEST.MAX_SIZE"] = "3000"

        if extra_params:
            d.update(extra_params)

        commands += [main_command + " " + launcher.parameters_to_str(d)]

        exp_job_names.append(job_name)
        exp_commands.append(commands)
        exp_log_paths.append(log_folder)


    arch = "res101"
    init_net = os.path.join(os2d_path, "models", "resnet101-5d3b4d8f.pth")
    init_net_name = "Pytorch"
    query_size = 192
    scale = 900
    ms = "ms"

    job_id = 0
    scale_str = str(scale) if ms == "ss" else ",".join(str(int(m * scale)) for m in [0.5, 0.625, 0.8, 1, 1.2, 1.4, 1.6])
    add_job(f"{arch}_init{init_net_name}_query{query_size}_scale{scale}_{ms}", job_id,
            arch, init_net,
            {
                "TRAIN.query_size": query_size,
                "TRAIN.SCALES": f"[{scale_str}]",
                "TEST.SCALES": f"[{scale}]", # CAUTION! single scale here
            }
            )

    for job_name, log_path, commands in zip(exp_job_names, exp_log_paths, exp_commands):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands)
    launcher.launch_all_jobs(args)
