import os
from collections import OrderedDict
import pandas as pd

from os2d.utils.logger import extract_map_value_from_os2d_log


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.abspath(__file__))
    config_job_name = "eval_instre"
    log_path = os.path.abspath(os.path.join(config_path, "..", "output/eval_instre"))

    def get_result(eval_dataset,
                   folder_suffix="",
                   result_suffix="out.txt"):
        d = OrderedDict()
        if "instre-s1" in eval_dataset:
            d["eval.dataset_scales"] = "[700.0]"
            folder_dataset_pattern = "instre-s1"
        elif "instre-s2" in eval_dataset:
            d["eval.dataset_scales"] = "[600.0]"
            folder_dataset_pattern = "instre-s2"
        else:
            raise RuntimeError(f"Unknown dataset {eval_dataset}")

        log_folder = f"{config_job_name}"
        if folder_suffix:
            log_folder += "." + folder_suffix
        log_folder = os.path.join(log_path, log_folder)

        result_file = f"eval_{folder_dataset_pattern}_scale{d['eval.dataset_scales'][1:-1]}_{result_suffix}"
        result_file = os.path.join(log_folder, result_file)

        return extract_map_value_from_os2d_log(result_file, eval_dataset)


    datasets = ["instre-s1", "instre-s2"]
    init = "imageNetCaffe2"
    table = pd.DataFrame(columns=["arch", "init", "type", "trained"] + datasets)

    for arch in ["ResNet50", "ResNet101"]:
        for job_type in ["v1", "v2"]:
            for train_type in ["train", "init"]:
                d = {}
                d["arch"] = arch
                d["init"] = init
                d["type"] = job_type
                d["trained"] = train_type

                for dataset in datasets:
                    folder_suffix = f"{dataset}_{job_type}-{train_type}_{arch}_{init}"

                    eval_dataset = dataset + "-test"
                    val = get_result(eval_dataset, folder_suffix=folder_suffix)
                    d[dataset] = val

                table = table.append(d, ignore_index=True)

    print(table, sep='\n')
