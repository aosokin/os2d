import os
import math
import random
import errno
from PIL import Image

import numpy as np

import torch


def get_data_path():
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    data_path = os.path.expanduser(os.path.abspath(data_path))
    return data_path


def get_trainable_parameters(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def count_model_parameters(net):
    num_params = 0
    num_param_groups = 0
    for p in get_trainable_parameters(net):
        num_param_groups += 1
        num_params += p.numel()
    return num_params, num_param_groups


def get_image_size_after_resize_preserving_aspect_ratio(h, w, target_size):
    aspect_ratio_h_to_w = float(h) / w
    w = int(target_size / math.sqrt(aspect_ratio_h_to_w))
    h = int(target_size * math.sqrt(aspect_ratio_h_to_w))
    h, w = (1 if s <= 0 else s for s in (h, w))  # filter out crazy one pixel images
    return h, w


def masked_select_or_fill_constant(a, mask, constant=0):
    constant_tensor = torch.tensor([constant], dtype=a.dtype, device=a.device)
    return torch.where(mask, a, constant_tensor)


def set_random_seed(random_seed, cuda=False):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed_all(random_seed)


def mkdir(path):
    """From https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/miscellaneous.py
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_image(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.load()
    return img


def ceildiv(a, b):
    return -(-a // b)
