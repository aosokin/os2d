import os
import pickle
import argparse
import glob


parser = argparse.ArgumentParser()

parser.add_argument("--log_path", default=None, type=str, help="Folder to search for logs")
parser.add_argument("--log_names", default=[], nargs="+", type=str, help="Plot logs from these folder")
parser.add_argument("--log_file_names", default=["out.txt"], nargs="+", type=str, help="Plot logs from these folder")

args = parser.parse_args()

if args.log_path is not None and not os.path.isdir(args.log_path):
    raise RuntimeError("Log path %s does not exist" % args.log_path)

log_path = args.log_path if args.log_path else ""
if len(args.log_names) == 0:
    print("--log_names was not specified, scanning folder %s" % args.log_path)
    log_names = sorted(glob.glob( os.path.join(log_path, "*")))
else:
    log_names = [os.path.join(log_path, name) for name in args.log_names]

log_file_names = args.log_file_names
target_pickle_file = "train_log.pkl"

prefix = "mAP@0.50: "
plot_name = "mAP@0.50"

iter_log_step = 10
iter_plot_name = "iter"


def update_meter(log, name, num_log_steps, value):
    # create entry if needed
    if name not in log:
        log[name] = []
    meter = log[name]
    # add missing values if any
    while len(meter) < num_log_steps - 1:
        meter.append(float("nan"))
    # add the new value
    meter.append(value)

for log_name in log_names:
    log_pkl = {}
    num_log_steps = 0

    for log_file_name in log_file_names:
        log_file = os.path.join(log_name, log_file_name)
        if not os.path.isfile(log_file):
            print("Missing file", log_file)
            continue
        with open(log_file, 'r') as log:
            for l in log:
                if l.startswith(prefix):
                    num_log_steps += 1
                    value = float(l[len(prefix):])
                    update_meter(log_pkl, plot_name, num_log_steps, value)
                    update_meter(log_pkl, iter_plot_name, num_log_steps, num_log_steps * iter_log_step)
    if log_pkl:
        pickle.dump(log_pkl, open(os.path.join(log_name, "train_log.pkl"), "wb"))
