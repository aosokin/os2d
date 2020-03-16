import os
import pandas as pd

from os2d.utils.logger import extract_map_value_from_os2d_log


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.abspath(os.path.join(config_path, "..", "output/instre"))
    config_job_name = "eval_instre"

    datasets = ["instre-s1", "instre-s2"]
    archs = ["resnet50", "resnet101"]
    method = "det-ret-baseline-train"
    retrieval_multiscale = "ms"
    table = pd.DataFrame(columns=["arch", "method"] + datasets)
    
    for arch in archs:
        row = {}
        row["arch"] = arch
        row["method"] = method
        for dataset in datasets:
            suffix = "initModel" if "init" in method else "bestModel"
            suffix += "_" + retrieval_multiscale

            eval_dataset = dataset + "-test"
            #eval_instre.det-ret-baseline-init-resnet101-instre-s1
            result_file = os.path.join(f"{config_job_name}.{method}-{arch}-{dataset}",
                                    #    f"eval_{suffix}_out.txt")
                                        f"eval_{eval_dataset}_{suffix}_out.txt")
            value = extract_map_value_from_os2d_log(os.path.join(log_path, result_file),
                                                    eval_dataset)
            row[dataset] = value
        table = table.append(row, ignore_index=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(table, sep='\n')
