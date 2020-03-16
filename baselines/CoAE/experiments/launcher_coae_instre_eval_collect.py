import os
import pandas as pd

from os2d.utils.logger import extract_pattern_after_marked_line, numeric_const_pattern, mAP_percent_to_points


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    coae_path = os.path.join(script_path, "..")

    config_job_name = "coae"
    log_path = os.path.abspath(os.path.join(coae_path, "output", "eval_instre"))


    def get_result(arch,
                   eval_dataset,
                   folder_suffix="",
                   result_suffix="out.txt",
                  ):

        # set output folder
        log_folder = f"{config_job_name}"
        if folder_suffix:
            log_folder += "." + folder_suffix
        log_folder = os.path.join(log_path, log_folder)

        result_file = f"eval_{eval_dataset}_{arch}_{result_suffix}"
        result_file = os.path.join(log_folder, result_file)

        dataset_pattern = "Evaluating detections"
        eval_pattern = f"mAP@0.50:\s({numeric_const_pattern})"

        mAP_value = extract_pattern_after_marked_line(result_file, dataset_pattern, eval_pattern)
        return mAP_percent_to_points(mAP_value)


    datasets = ["instre-s1-test", "instre-s2-test"]
    archs = ["res50", "res101"]
    table = pd.DataFrame(columns=["arch"] + datasets)
    for arch in archs:
        d = {}
        d["arch"] = arch
        for eval_dataset in datasets:
            d[eval_dataset] = get_result(arch, eval_dataset, folder_suffix="best_train")
        table = table.append(d, ignore_index=True)

    print(table, sep='\n')
