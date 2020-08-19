import os
from collections import OrderedDict

from os2d.utils import launcher as launcher


if __name__ == "__main__":
    # load default launcher parameters
    parser = launcher.create_args_parser()
    args = parser.parse_args()

    main_command = "python main.py"

    config_path = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(config_path, "config_training.yml")
    config_job_name = "eval_imagenet"

    log_path = os.path.abspath(os.path.join(config_path, "..", "output/eval_imagenet"))

    exp_job_names = []
    exp_log_paths = []
    exp_commands = []
    exp_log_file_prefix = []


    def add_job(sub_index,
                backbone_arch,
                model_path,
                model_checkpoint,
                episodes,
                eval_scale,
                test_augmentation,
                folder_suffix="",
                extra_params=None):
        job_name = f"{config_job_name}.{sub_index}"
        commands = []

        d = OrderedDict()
        d["--config-file"] = config_file

        d["model.use_inverse_geom_model"] = True
        d["model.use_simplified_affine_model"] = False
        d["model.backbone_arch"] = backbone_arch

        d["eval.dataset_scales"] = f"[{eval_scale}]"

        if test_augmentation:
            d["eval.class_image_augmentation"] = test_augmentation

        if extra_params:
            d.update(extra_params)

        # set output folder
        log_folder = f"{config_job_name}"
        if folder_suffix:
            log_folder += "." + folder_suffix
        log_folder = os.path.join(log_path, log_folder)

        d["train.do_training"] = False

        # choose init
        if "init.transform" in d:
            del d["init.transform"]
        if os.path.isfile(model_path):
            d["init.model"] = model_path
        else:
            d["init.model"] = os.path.join(model_path, model_checkpoint)

        for episode in episodes:
            d["eval.dataset_names"] = f"[\\\"imagenet-repmet-test-episode-{episode}\\\"]"

            commands.append(main_command + " " + launcher.parameters_to_str(d))

        exp_job_names.append(job_name)
        exp_commands.append(commands)
        exp_log_paths.append(log_folder)
        exp_log_file_prefix.append(f"eval_scale{d['eval.dataset_scales'][1:-1]}_epi{min(episodes)}-{max(episodes)}_")


    test_augmentation = "horflip" #"horflip_rotation90"

    num_episodes = 500
    episode_per_job = 50

    scales_to_test = [250]

    for eval_scale in scales_to_test:
        index = 0
        i_episode = 0
        while i_episode < num_episodes:
            list_of_episodes = list(range(i_episode, min(i_episode + episode_per_job, num_episodes)))
            add_job(sub_index=index,
                    backbone_arch="ResNet50",
                    model_path="models",
                    model_checkpoint="os2d_v2-init.pth",
                    episodes=list_of_episodes,
                    eval_scale=eval_scale,
                    test_augmentation=test_augmentation,
                    folder_suffix=f"model_v2-init_scale_{int(eval_scale)}_aug_horFlip",
                    extra_params=None)
            index += 1
            i_episode += episode_per_job

    for eval_scale in scales_to_test:
        index = 0
        i_episode = 0
        while i_episode < num_episodes:
            list_of_episodes = list(range(i_episode, min(i_episode + episode_per_job, num_episodes)))
            add_job(sub_index=index,
                    backbone_arch="ResNet50",
                    model_path="output/exp2/exp2.2.v1_seed0_ResNet50_init_imageNetCaffe2",
                    model_checkpoint="checkpoint_iter_0.pth",
                    episodes=list_of_episodes,
                    eval_scale=eval_scale,
                    test_augmentation=test_augmentation,
                    folder_suffix=f"model_v1-init_scale_{int(eval_scale)}_aug_horFlip",
                    extra_params=None)
            index += 1
            i_episode += episode_per_job

    for job_name, log_path, commands, log_file_prefix in zip(exp_job_names, exp_log_paths, exp_commands, exp_log_file_prefix):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands,
                         log_file_prefix=log_file_prefix)
    launcher.launch_all_jobs(args)
