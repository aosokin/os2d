import os
import argparse
import errno
import subprocess


def create_args_parser():
    parser = argparse.ArgumentParser(description="Launching experiments locally or with SLURM")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--job-names",
        default=None,
        help="Launch a subset of jobs ~-- only jobs with these names, default - all",
        type=str,
        nargs="+",
    )
    group.add_argument(
        "--job-indices",
        default=None,
        help="Launch a subset of jobs ~-- only jobs with these indices, default - all",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
        help="Launch the job in this conda env",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Prepare jobs for SLURM and launch them with sbatch",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="No actual launch - only generate commands",
    )
    # cluster arguments
    parser.add_argument(
        "-p", "--partition",
        type=str,
        default=None,
        help="SLURM partition where to launch",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Request this number of GPUs",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=4,
        help="Request this number of CPUs",
    )
    parser.add_argument(
        "--exclusive-node",
        action="store_true",
        help="Request the node exclusively for yourself",
    )
    parser.add_argument(
        "--exclude-nodes",
        type=str,
        nargs="+",
        default=None,
        help="Exclude this nodes from the job",
    )
    parser.add_argument(
        "--nodelist",
        type=str,
        default=None,
        help="Request this list of nodes for the job",
    )
    parser.add_argument(
        "--stdout-file",
        type=str,
        default="out.txt",
        help="Save stdout to this file",
    )
    parser.add_argument(
        "--stderr-file",
        type=str,
        default="err.txt",
        help="Save stderr to this file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in hours after which the job is dead",
    )

    return parser


def parse_arguments():
    parser = create_args_parser()
    return parser.parse_args()


def get_bare_file_name(exp_config_file):
    # get the name of the file without path and extension
    return os.path.splitext(os.path.basename(exp_config_file))[0]


def mkdir(path):
    """
    From https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/miscellaneous.py
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def parameters_to_str(config_dict):
    job_command = " "
    if config_dict:
        for k, v in config_dict.items():
            job_command += str(k) + " " + str(v) + " "
    return job_command


JOB_QUEUE_NAMES = []
JOB_QUEUE_PATHS = []
JOB_QUEUE_COMMANDS = []
JOB_QUEUE_LOG_FILE_PREFIX = []


def add_job(job_name="",
            log_path="",
            commands="",
            log_file_prefix=""):
    JOB_QUEUE_NAMES.append(job_name)
    JOB_QUEUE_PATHS.append(log_path)
    JOB_QUEUE_COMMANDS.append(commands)
    JOB_QUEUE_LOG_FILE_PREFIX.append(log_file_prefix)


def launch_all_jobs(args):
    for i_job, (job_name, log_path, commands, log_file_prefix) in enumerate(zip(JOB_QUEUE_NAMES, JOB_QUEUE_PATHS, JOB_QUEUE_COMMANDS, JOB_QUEUE_LOG_FILE_PREFIX)):
        if ((args.job_names is None) and (args.job_indices is None)) or \
           ((args.job_names is not None) and (job_name in args.job_names)) or \
           ((args.job_indices is not None) and (i_job in args.job_indices)):
            
            if not args.no_launch:
                print("Launching job #{0}: {1}".format(i_job, job_name))
            else:
                print("Preparing job #{0}: {1}".format(i_job, job_name))
            if log_path:
                # create log folder
                # need to make the folder before calling sbatch, otherwise --output and --error parameters will not work
                mkdir(log_path)

            job_command = "\n\n".join(commands)
        
            if args.slurm:
                run_job_slurm_cluster(job_command, log_path, args, job_name=job_name, no_launch=args.no_launch, log_file_prefix=log_file_prefix)
            else:
                run_job_locally(job_command, log_path, args, no_launch=args.no_launch, log_file_prefix=log_file_prefix)
            print("success", flush=True)


def echo_and_execute_command(out_f, command):
    out_f.write("echo \"{}\"\n".format(command))
    out_f.write(command + "\n")
    out_f.write("echo")
    out_f.write("\n")


def echo_node_gpu_info(out_f):
    out_f.write("echo \"Working on node `hostname`\"\n")
    out_f.write("echo \"Assigned GPUs: $CUDA_VISIBLE_DEVICES\"\n")
    echo_and_execute_command(out_f, "gpustat -c -u -p --no-color")


def echo_conda_info(out_f):
    out_f.write("echo \"Checking current conda env\"\n")
    echo_and_execute_command(out_f, "conda info")


def echo_git_status(out_f):
    out_f.write("echo \"Checking the current git commit\"\n")
    git_command = "git show -s --pretty=format:'%H'"
    echo_and_execute_command(out_f, git_command)


def echo_system_info(out_f):
    echo_conda_info(out_f)
    echo_git_status(out_f)
    echo_node_gpu_info(out_f)


def set_num_cpu_threads(out_f, num_cpus):
    out_f.write("export EXP_NUM_CPU_THREADS={}\n".format(num_cpus))
    out_f.write("export OMP_NUM_THREADS=${EXP_NUM_CPU_THREADS}\n")
    out_f.write("export MKL_NUM_THREADS=${EXP_NUM_CPU_THREADS}\n")
    out_f.write("export NUMEXPR_NUM_THREADS=${EXP_NUM_CPU_THREADS}\n")
    out_f.write("\n")


def run_job_locally(job_command, log_path, args, no_launch=False, log_file_prefix=""):
    # save the launching command to file
    cmd_file = os.path.join(log_path, log_file_prefix + 'launch.sh')
    with open(cmd_file, "w") as out_f:
        if args.conda_env:
            print(f"source activate {args.conda_env}", file=out_f)

        echo_system_info(out_f)
        set_num_cpu_threads(out_f, args.num_cpus)
        print(job_command, file=out_f)
        echo_system_info(out_f)

    stdout_file_path = os.path.join(log_path, log_file_prefix + args.stdout_file)
    stderr_file_path = os.path.join(log_path, log_file_prefix + args.stderr_file)

    cmd = f"bash {cmd_file} 2>{stderr_file_path} | tee -a {stdout_file_path}"

    run_cmd_with_line_printing(cmd, no_launch=no_launch)


def run_job_slurm_cluster(job_command, log_path, args, job_name=None, no_launch=False, log_file_prefix=""):
    # save the sbatch script to file
    launcher_file = os.path.join(log_path, log_file_prefix + 'launch.sh')
    with open(launcher_file, "w") as out_f:
        out_f.write("#!/bin/bash\n")

        # params of sbatch
        if args.exclusive_node:
            out_f.write("#SBATCH --exclusive=user\n")
        if args.partition:
            out_f.write("#SBATCH --partition {}\n".format(args.partition))
        out_f.write("#SBATCH --gpus={}\n".format(args.num_gpus))
        out_f.write("#SBATCH --cpus-per-task={}\n".format(args.num_cpus))
        if job_name:
            out_f.write("#SBATCH --job-name={}\n".format(job_name))
        out_f.write("#SBATCH --output={}\n".format(os.path.join(log_path, log_file_prefix + args.stdout_file)))
        out_f.write("#SBATCH --error={}\n".format(os.path.join(log_path, log_file_prefix + args.stderr_file)))
        if args.exclude_nodes:
            out_f.write("#SBATCH --exclude={}\n".format(",".join(args.exclude_nodes)))
        if args.nodelist:
            out_f.write(f"#SBATCH --nodelist={args.nodelist}")
        if args.timeout:
            out_f.write(f"#SBATCH --time={int(args.timeout * 60)}") # slurm wants timeout in minutes
        out_f.write("\n")

        # activate conda environment
        if args.conda_env:
            conda_command = "source activate {}".format(args.conda_env)
            echo_and_execute_command(out_f, conda_command)

        # print system status
        echo_system_info(out_f)

        # env vars
        set_num_cpu_threads(out_f, args.num_cpus)
        out_f.write("export HDF5_USE_FILE_LOCKING='FALSE'\n")
        out_f.write("\n")

        # the command
        out_f.write("{}\n".format(job_command))
        out_f.write("\n")

        # print system status after the main job - to know what happened
        echo_system_info(out_f)

    cmd = "sbatch {}".format(launcher_file)
    run_cmd_with_line_printing(cmd, no_launch=no_launch)


def run_cmd_with_line_printing(cmd, no_launch=False):
    if not no_launch:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ## But do not wait till netstat finish, start displaying output immediately ##
        while True:
            out = p.stdout.readline().decode("utf-8") 
            if out == '' and p.poll() != None:
                break
            print(out, end="")
    else:
        print(cmd)


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    print("This file should be used as a lib to laucch implemented jobs")
