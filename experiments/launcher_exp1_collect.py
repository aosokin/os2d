import os

from os2d.utils.logger import extract_value_from_os2d_binary_log, mAP_percent_to_points


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.abspath(__file__))
    config_job_name = "exp1"

    log_path = os.path.abspath(os.path.join(config_path, "..", "output/exp1"))


    def get_result(job_name,
                   sub_index,
                   backbone_arch,
                   init_model_nickname,
                   rand_seed,
                  ):
        job_name = f"{config_job_name}.{sub_index}.{job_name}_seed{rand_seed}"

        log_folder = job_name + "_" + backbone_arch + "_init_" + init_model_nickname
        log_folder = os.path.join(log_path, log_folder)

        data_file = os.path.join(log_folder, "train_log.pkl")

        return mAP_percent_to_points(extract_value_from_os2d_binary_log(data_file, "mAP@0.50_grozi-val-new-cl", reduce="max"))


    results = []
    random_seed = 0
    results.append(get_result(\
            "lossCL", 0, "ResNet50", "imageNetCaffe2", random_seed,
            ))

    results.append(get_result(\
            "lossRLL", 1, "ResNet50", "imageNetCaffe2", random_seed
            ))

    results.append(get_result(\
            "lossRLL_remap", 2, "ResNet50", "imageNetCaffe2", random_seed
            ))

    results.append(get_result(\
             "lossRLL_remap_mine", 3, "ResNet50", "imageNetCaffe2", random_seed
            ))

    results.append(get_result(\
            "lossRLL_remap_invFullAffine", 4, "ResNet50", "imageNetCaffe2", random_seed
            ))

    results.append(get_result(\
            "lossRLL_remap_mine_fullAffine", 5, "ResNet50", "imageNetCaffe2", random_seed
            ))

    results.append(get_result(\
            "lossRLL_remap_invFullAffine_initTranform", 6, "ResNet50", "imageNetCaffe2", random_seed,
            ))

    results.append(get_result(\
            "lossRLL_remap_invFullAffine_initTranform_zeroLocLoss", 7, "ResNet50", "imageNetCaffe2", random_seed
            ))

    results.append(get_result(\
            "lossRLL_remap_invFullAffine_initTranform_zeroLocLoss_mine", 8, "ResNet50", "imageNetCaffe2", random_seed
            ))

    results.append(get_result(\
            "lossCL_invFullAffine_initTranform_zeroLocLoss", 9, "ResNet50", "imageNetCaffe2", random_seed
            ))

    results.append(get_result(\
            "lossCL_remap_invFullAffine_initTranform_zeroLocLoss", 10, "ResNet50", "imageNetCaffe2", random_seed
            ))

    results.append(get_result(\
            "lossRLL_invFullAffine_initTranform_zeroLocLoss", 11, "ResNet50", "imageNetCaffe2", random_seed
            ))

    for r in results:
        print(r)
