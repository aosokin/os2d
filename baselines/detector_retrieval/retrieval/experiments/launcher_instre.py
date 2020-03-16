import os
import copy
from collections import OrderedDict

from os2d.utils import launcher as launcher


if __name__ == "__main__":
    # load default launcher parameters
    parser = launcher.create_args_parser()
    args = parser.parse_args()

    script_path = os.path.dirname(os.path.abspath(__file__))
    retrieval_path = os.path.join(script_path, "..")

    os2d_path = os.path.join(retrieval_path, "..", "..", "..")
    cirtorch_path = os.path.join(retrieval_path, "cnnimageretrieval-pytorch")
    detector_path = os.path.join(retrieval_path, "..", "detector")
    python_path = f"PYTHONPATH={os2d_path}:{cirtorch_path}"

    retrieval_train_launcher = f"{retrieval_path}/train.py"
    full_eval_launcher = f"{retrieval_path}/../main_detector_retrieval.py"

    main_command = python_path + " " + f"python {retrieval_train_launcher}"
    log_folder_path = f"{retrieval_path}/output/instre"
    main_command += " " + log_folder_path

    exp_commands = []
    exp_job_names = []
    exp_log_paths = []

    def add_job(sub_index,
                training_dataset,
                arch,
                pool = "gem",
                test_whiten = False,
                local_whitening = False,
                regional = False,
                whitening = False,
                loss_margin = 0.85,
                image_size = 240,
                learning_rate = 1e-6,
                pretrained = True,
                loss = "contrastive",
                optimizer = "adam",
                weight_decay = 1e-4,
                neg_num = 5,
                query_size = 2000,
                pool_size = 20000,
                batch_size = 5,
                ):

        directory = "{}".format(training_dataset)
        directory += "_{}".format(arch)
        directory += "_{}".format(pool)
        if local_whitening:
            directory += "_lwhiten"
        if regional:
            directory += "_r"
        if whitening:
            directory += "_whiten"
        if not pretrained:
            directory += "_notpretrained"
        directory += "_{}_m{:.2f}".format(loss, loss_margin)
        directory += "_{}_lr{:.1e}_wd{:.1e}".format(optimizer, learning_rate, weight_decay)
        directory += "_nnum{}_qsize{}_psize{}".format(neg_num, query_size, pool_size)
        directory += "_bsize{}_imsize{}".format(batch_size, image_size)

        log_path = os.path.join(log_folder_path, directory)

        job_name = "ret-instre-{0}.{1}".format(sub_index, directory)

        d = OrderedDict()
        d["--training-dataset"] = training_dataset
        if training_dataset == "instre-s1-train-retrieval":
            d["--test-datasets"] = "instre-s1-val-retrieval"
        elif training_dataset == "instre-s2-train-retrieval":
            d["--test-datasets"] = "instre-s2-val-retrieval"
        elif training_dataset == "instre-s1-train-retrieval-rndCropPerImage10":
            d["--test-datasets"] = "instre-s1-val-retrieval-rndCropPerImage10"
        elif training_dataset == "instre-s2-train-retrieval-rndCropPerImage10":
            d["--test-datasets"] = "instre-s2-val-retrieval-rndCropPerImage10"
        else:
            raise RuntimeError(f"Unknown training set {training_dataset}")

        if test_whiten:
            d["--test-whiten"] = training_dataset

        d["--arch"] = arch
        d["--pool"] = pool
        if local_whitening:
            d["--local-whitening"] = ""
        if regional:
            d["--regional"] = ""
        if whitening:
            d["--whitening"] = ""
        d["--loss-margin"] = loss_margin
        d["--image-size"] = image_size
        d["--learning-rate"] = learning_rate
        if not pretrained:
            d["--not-pretrained"] = ""
        d["--loss"] = loss
        d["--optimizer"] = optimizer
        d["--weight-decay"] = weight_decay
        d["--neg-num"] = neg_num
        d["--query-size"] = query_size
        d["--pool-size"] = pool_size
        d["--batch-size"] = batch_size

        commands = []
        commands.append(main_command + " " + launcher.parameters_to_str(d))

        exp_job_names.append(job_name)
        exp_log_paths.append(log_path)
        exp_commands.append(commands)


    job_index = 0

    training_datasets = ["instre-s1-train-retrieval-rndCropPerImage10",
                         "instre-s2-train-retrieval-rndCropPerImage10"]
    archs = ["resnet50", "resnet101"]
    pool = "gem"
    whitening = True

    for training_dataset in training_datasets:
        for arch in archs:
            add_job(job_index, training_dataset=training_dataset, arch=arch, pool=pool, whitening=whitening)
            job_index += 1

    for job_name, log_path, commands in zip(exp_job_names, exp_log_paths, exp_commands):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands)
    launcher.launch_all_jobs(args)
