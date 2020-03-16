import os
import pandas as pd

from os2d.utils.logger import extract_value_from_os2d_binary_log, mAP_percent_to_points


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.abspath(__file__))
    config_job_name = "exp2"
    log_path = os.path.abspath(os.path.join(config_path, "..", "output/exp2"))


    def get_result(job_type, # "v1" or "v2"
                   sub_index,
                   backbone_arch,
                   init_model_nickname,
                   random_seed,
                  ):
        job_name = f"{config_job_name}.{sub_index}.{job_type}_seed{random_seed}"

        log_folder = job_name + "_" + backbone_arch + "_init_" + init_model_nickname
        log_folder = os.path.join(log_path, log_folder)

        data_file = os.path.join(log_folder, "train_log.pkl")

        return mAP_percent_to_points(extract_value_from_os2d_binary_log(data_file, "mAP@0.50_grozi-val-new-cl", reduce="max")),\
               mAP_percent_to_points(extract_value_from_os2d_binary_log(data_file, "mAP@0.50_grozi-val-new-cl", reduce="first"))


    table = pd.DataFrame(columns=["arch", "init", "v1-train", "v2-init", "v2-train"])
    random_seed = 0

    for i, arch, init in zip(range(10),
                             ["ResNet50"] * 5 + ["ResNet101"] * 5,
                             ["fromScratch", "imageNetPth", "imageNetCaffe2", "imageNetCaffe2GroupNorm", "cocoMaskrcnnFpn",
                              "imageNetPth", "imageNetCaffe2", "buildingsCirtorch", "cocoMaskrcnnFpn", "pascalWeakalign"]
                            ):
        val_train_v1, val_init_v1 = get_result("v1", i, arch, init, random_seed)
        val_train_v2, val_init_v2 = get_result("v2", i, arch, init, random_seed)

        table = table.append({"arch":arch, "init":init,
                              "v1-train":val_train_v1, "v2-init":val_init_v2, "v2-train":val_train_v2},
                                ignore_index=True)

    print(table, sep='\n')
