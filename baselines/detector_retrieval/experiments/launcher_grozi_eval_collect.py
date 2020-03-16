import os
import pandas as pd

from os2d.utils.logger import extract_map_value_from_os2d_log


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.abspath(os.path.join(config_path, "..", "output/grozi"))
    config_job_name = "eval_grozi"

    datasets = ["grozi-val-old-cl", "grozi-val-new-cl", "dairy", "paste-v", "paste-f"]
    methods = ["det-ret-baseline-init", "det-ret-baseline-train"]
    table = pd.DataFrame(columns=["method"] + datasets)
    retrieval_multiscale = "ms"

    for method in methods:
        row = {}
        row["method"] = method
        for eval_dataset in datasets:
            suffix = "initModel" if "init" in method else "bestModel"
            suffix += "_" + retrieval_multiscale

            result_file = os.path.join(f"{config_job_name}.{method}",
                                       f"eval_{eval_dataset}_{suffix}_out.txt")
            value = extract_map_value_from_os2d_log(os.path.join(log_path, result_file),
                                                    eval_dataset)
            row[eval_dataset] = value
        table = table.append(row, ignore_index=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(table, sep='\n')
