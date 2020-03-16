# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name.
This file is a replacement of
https://github.com/timy90022/One-Shot-Object-Detection/blob/master/lib/datasets/factory.py
"""

from lib.datasets.os2d import build_os2d_dataset_by_name


# __sets = {}

# # Set up the grozi dataset
# for split in ["train", "val-old-cl", "val-new-cl", "val-all", "train-mini"]:
#     name = 'grozi-{}'.format(split)
#     __sets[name] = (lambda name=name: build_os2d_dataset_by_name(name, data_path=None))

# # Set up OS2D evaluation datasets
# for name in ["dairy", "paste-v", "paste-f"]:
#     __sets[name] = (lambda  name=name: build_os2d_dataset_by_name(name, data_path=None))


# # Set up the grozi dataset
# for split in ["train", "val-old-cl", "val-new-cl", "val-all", "train-mini"]:
#     name = 'grozi-{}'.format(split)
#     __sets[name] = (lambda name=name: build_os2d_dataset_by_name(name, data_path=None, eval_scale=1280.0))

# # Set up evaluation datasets
# dataset_eval_scale = {}
# dataset_eval_scale["dairy"] = 3500.0
# dataset_eval_scale["paste-v"] = 3500.0
# dataset_eval_scale["paste-f"] = 2000.0

# for name in dataset_eval_scale:
#     __sets[name] = (lambda  name=name: build_os2d_dataset_by_name(name, data_path=None, eval_scale=dataset_eval_scale[name]))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    return build_os2d_dataset_by_name(name)

#     if name not in __sets:
#         raise KeyError('Unknown dataset: {}'.format(name))
#     return __sets[name]()


# def list_imdbs():
#     """List all registered imdbs."""
#     return list(__sets.keys())
