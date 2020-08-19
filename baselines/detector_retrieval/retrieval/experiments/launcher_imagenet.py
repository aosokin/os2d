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
    log_folder_path = f"{retrieval_path}/output/imagenet-repmet"

    exp_commands = []
    exp_job_names = []
    exp_log_paths = []

    def add_job(sub_index,
                training_dataset,
                arch,
                init_weights="",
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

        log_path = os.path.join(log_folder_path, init_weights, directory)

        job_name = "ret-imagenet-{0}.{1}.{2}".format(sub_index, init_weights, directory)

        d = OrderedDict()
        d[os.path.join(log_folder_path, init_weights)] = ""
        d["--training-dataset"] = training_dataset
        if training_dataset == "instre-s1-train-retrieval":
            d["--test-datasets"] = "instre-s1-val-retrieval"
        elif training_dataset == "instre-s2-train-retrieval":
            d["--test-datasets"] = "instre-s2-val-retrieval"
        elif training_dataset == "instre-s1-train-retrieval-rndCropPerImage10":
            d["--test-datasets"] = "instre-s1-val-retrieval-rndCropPerImage10"
        elif training_dataset == "instre-s2-train-retrieval-rndCropPerImage10":
            d["--test-datasets"] = "instre-s2-val-retrieval-rndCropPerImage10"
        elif training_dataset == "imagenet-repmet-train-retrieval":
            d["--test-datasets"] = "imagenet-repmet-val-5000-retrieval"
        elif training_dataset == "imagenet-repmet-train-retrieval-rndCropPerImage10":
            d["--test-datasets"] = "imagenet-repmet-val-5000-retrieval-rndCropPerImage10"
        else:
            raise RuntimeError(f"Unknown training set {training_dataset}")

        if test_whiten:
            d["--test-whiten"] = training_dataset

        if arch == "resnet50":
            if init_weights == "imagenet-repmet-pytorch":
                d["--network-path"] = "../../../data/ImageNet-RepMet/pretrain/output/resnet50/model_best_cirtorch.pth.tar"
            elif init_weights == "imagenet-pytorch":
                d["--network-path"] = "../../../models/resnet50-19c8e357_cirtorch.pth"
            elif init_weights == "imagenet-caffe":
                # use the built-in caffe weights
                pass
            else:
                raise RuntimeError(f"Do not recognize weight initialization {init_weights}")
        elif arch == "resnet101":
            if init_weights == "imagenet-repmet-pytorch":
                d["--network-path"] = "../../../data/ImageNet-RepMet/pretrain/output/resnet101/model_best_cirtorch.pth.tar"
            elif init_weights == "imagenet-pytorch":
                d["--network-path"] = "../../../models/resnet101-5d3b4d8f_cirtorch.pth"
            elif init_weights == "imagenet-caffe":
                # use the built-in caffe weights
                pass
            else:
                raise RuntimeError(f"Do not recognize weight initialization {init_weights}")
        else:
            raise RuntimeError(f"Unknown arch: {arch}")

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

    training_datasets = ["imagenet-repmet-train-retrieval"]
    archs = ["resnet101"] # ["resnet50", "resnet101"]
    init_nets = ["imagenet-repmet-pytorch"] # ["imagenet-repmet-pytorch", "imagenet-pytorch", "imagenet-caffe"]
    pool = "gem"
    whitening = True

    for training_dataset in training_datasets:
        for arch in archs:
            for init_weights in init_nets:
                add_job(job_index, training_dataset=training_dataset, init_weights=init_weights,
                        arch=arch, pool=pool, whitening=whitening)
                job_index += 1

    for job_name, log_path, commands in zip(exp_job_names, exp_log_paths, exp_commands):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands)
    launcher.launch_all_jobs(args)
