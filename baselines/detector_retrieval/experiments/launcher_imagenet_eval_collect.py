import os
import statistics
import pandas as pd

from os2d.utils.logger import extract_pattern_after_marked_line, numeric_const_pattern


MISSING_VAL_CONSTANT = "None"


def mAP_percent_to_points(v):
    if v is not None:
        return float(v)*100
    else:
        return MISSING_VAL_CONSTANT


def extract_map_value_from_os2d_log(result_file, eval_dataset, metric_name="mAP@0.50"):
    dataset_search_pattern = "Evaluated on {0}"
    dataset_pattern = dataset_search_pattern.format(eval_dataset)
    eval_pattern = f"{metric_name}\s({numeric_const_pattern})"

    value = extract_pattern_after_marked_line(result_file, dataset_pattern, eval_pattern)
    return mAP_percent_to_points(value)


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.abspath(__file__))
    config_job_name = "eval_imagenet_repmet"

    log_path = os.path.abspath(os.path.join(config_path, "..", "output/imagenet-repmet"))


    def get_result(sub_index,
                   episodes,
                   metric_names=["mAP@0.50"],
                   folder_suffix="",
                   result_suffix="out.txt"):
        # set output folder
        log_folder = f"{config_job_name}"
        if folder_suffix:
            log_folder += "." + folder_suffix
        log_folder = os.path.join(log_path, log_folder)

        values = []
        for episode in episodes:
            eval_dataset = f"imagenet-repmet-test-episode-{episode}"

            result_file = f"eval_epi{min(episodes)}-{max(episodes)}_{result_suffix}"
            result_file = os.path.join(log_folder, result_file)
            values_one_run = {}
            for m in metric_names:
                values_one_run[m] = extract_map_value_from_os2d_log(result_file, eval_dataset, metric_name=m)
            values.append(values_one_run)

        return values


    def collect_run_results(folder_suffix, result_suffix):
        num_episodes = 500
        episode_per_job = 50
        index = 0
        i_episode = 0

        metric_names = ["mAP@0.50", "AP_joint_classes@0.50"]
        computed_episodes_metric = {m:[] for m in metric_names}

        while i_episode < num_episodes:
            list_of_episodes = list(range(i_episode, min(i_episode + episode_per_job, num_episodes)))
            results = get_result(sub_index=index,
                                 episodes=list_of_episodes,
                                 metric_names=metric_names,
                                 folder_suffix=folder_suffix,
                                 result_suffix=result_suffix,
                                 )

            for e, r in zip(list_of_episodes, results):
                for m in metric_names:
                    if r[m] == MISSING_VAL_CONSTANT:
                        print(f"Missing episode {e} from chunk {index}")
                    else:
                        computed_episodes_metric[m].append(r[m])

            index += 1
            i_episode += episode_per_job

        for metric_name in metric_names:
            collected_metric = computed_episodes_metric[metric_name]
            average_val = sum(collected_metric) / len(collected_metric)
            max_val = max(collected_metric)
            min_val = min(collected_metric)
            std_val = statistics.stdev(collected_metric)
            print(f"{folder_suffix}: {len(collected_metric)} episodes; average {metric_name} = {average_val:0.2f}; max {metric_name} = {max_val:0.2f}; min {metric_name} = {min_val:0.2f}; std {metric_name} = {std_val:0.2f};")


for init_weights in ["imagenet-repmet-pytorch"]: # ["imagenet-repmet-pytorch", "imagenet-pytorch", "imagenet-caffe"]:
    collect_run_results(folder_suffix=f"det-ret-baseline-train-resnet101-{init_weights}-imagenet-repmet", result_suffix="bestModel_ms_out.txt")
