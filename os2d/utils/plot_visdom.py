import os
import argparse
import pickle
import glob

from visdom import Visdom
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--log_path", default=None, type=str, help="Folder to search for logs")
parser.add_argument("--log_names", default=[], nargs="+", type=str, help="Plot logs from these folder")

opt = parser.parse_args()

if opt.log_path is not None and not os.path.isdir(opt.log_path):
    raise RuntimeError("Log path %s does not exist" % opt.log_path)

viz = Visdom()
viz_plots = {}
x_name_all = ["iter", "time"]


def vizualize_log(log_path):
    # read the log_file
    log_file = os.path.join(log_path, "train_log.pkl")
    if not os.path.isfile(log_file):
        print("WARNING: Could not find file %s" % log_file)
        return
    logs = pickle.load(open(log_file, "rb"))

    for x_name in x_name_all:
        if x_name in logs:
            for y_name, y_data in logs.items():
                if not y_name in x_name_all:
                    x_data = logs[x_name]
                    plot_key = (y_name, x_name)

                    plot_opts = dict(
                        markers=False,
                        xlabel=x_name,
                        ylabel=y_name,
                        title="{0} vs. {1}".format(y_name, x_name),
                        showlegend=True
                    )
                    X = np.array(x_data).flatten()
                    Y = np.array(y_data).flatten()

                    # sync lengths
                    length = min(X.size, Y.size)
                    X = X[:length]
                    Y = Y[:length]

                    mask_non_nan = np.logical_not(np.isnan(Y))
                    X = X[mask_non_nan]
                    Y = Y[mask_non_nan]

                    viz_plots[plot_key] = "{0} vs. {1}".format(y_name, x_name)
                    line_name = os.path.basename(os.path.normpath(log_path))
                    viz.line(X=None, Y=None, win=viz_plots[plot_key], name=line_name, update="remove")
                    viz.line(
                        X=X,
                        Y=Y,
                        win=viz_plots[plot_key],
                        name=line_name,
                        opts=plot_opts,
                        update="append"
                        )

log_path = opt.log_path if opt.log_path else ""
if len(opt.log_names) == 0:
    print("--log_names was not specified, scanning folder %s" % opt.log_path)
    log_names = sorted(glob.glob( os.path.join(log_path, "*")))
else:
    log_names = [os.path.join(log_path, name) for name in opt.log_names]


n = len(log_names)
for i_log, path in enumerate(log_names):
    try:
        vizualize_log(path)
        print("Plot %d of %d: %s" % (i_log, n, path))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print("Failed to plot from %s. Error: %s" % (path, str(e)))
