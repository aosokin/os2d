import os
from collections import OrderedDict
import pandas as pd

from os2d.utils.logger import extract_map_value_from_os2d_log


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.abspath(os.path.join(config_path, "..", "output/eval_grozi"))
    config_job_name = "eval_grozi"


    def get_result(sub_index,
                   eval_dataset,
                   folder_suffix="",
                   result_suffix="out.txt"):
        d = OrderedDict()

        log_folder = f"{config_job_name}.{sub_index}"
        if folder_suffix:
            log_folder += "." + folder_suffix
        log_folder = os.path.join(log_path, log_folder)

        # choose eval dataset
        if eval_dataset == "grozi-val-new-cl":
            d["eval.dataset_names"] = "\"[\\\"grozi-val-new-cl\\\"]\""
            d["eval.dataset_scales"] = "[1280.0]"
        elif eval_dataset == "grozi-val-old-cl":
            d["eval.dataset_names"] = "\"[\\\"grozi-val-old-cl\\\"]\""
            d["eval.dataset_scales"] = "[1280.0]"
        elif eval_dataset == "dairy":
            d["eval.dataset_names"] = "\"[\\\"dairy\\\"]\""
            d["eval.dataset_scales"] = "[3500.0]"
        elif eval_dataset == "paste-v":
            d["eval.dataset_names"] = "\"[\\\"paste-v\\\"]\""
            d["eval.dataset_scales"] = "[3500.0]"
        elif eval_dataset == "paste-f":
            d["eval.dataset_names"] = "\"[\\\"paste-f\\\"]\""
            d["eval.dataset_scales"] = "[2000.0]"
            # eval with rotations
            d["eval.class_image_augmentation"] = "rotation90"
        else:
            raise f"Unknown eval set {eval_dataset}"

        result_file = f"eval_{eval_dataset}_scale{d['eval.dataset_scales'][1:-1]}_{result_suffix}"
        result_file = os.path.join(log_folder, result_file)

        return extract_map_value_from_os2d_log(result_file, eval_dataset)


    datasets = ["grozi-val-old-cl", "grozi-val-new-cl", "dairy", "paste-v", "paste-f"]
    methods = ["V1-init", "V1-train", "V2-init", "V2-train"]
    ids= [3, 0, 2, 1]
    table = pd.DataFrame(columns=["method"] + datasets)

    for i, method in zip(ids, methods):
        row = {}
        row["method"] = method
        for eval_dataset in datasets:
            value = get_result(i, eval_dataset,
                               folder_suffix=f"best_{method}")
            row[eval_dataset] = value
        table = table.append(row, ignore_index=True)

    print(table, sep='\n')
