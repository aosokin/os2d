import os
import math
from collections import OrderedDict
from tqdm import tqdm

from os2d.utils import get_data_path, setup_logger
from os2d.data.dataset import build_dataset_by_name
from os2d.structures.feature_map import FeatureMapSize


DATASET_LIST = ["grozi-train", "grozi-val-new-cl", "dairy", "paste-v", "paste-f",
                "instre-s1-train", "instre-s1-val",
                "instre-s2-train", "instre-s2-val"]


def get_image_sizes(dataset):
    print("Reading images from {}".format(dataset.image_path))
    image_sizes_by_id = OrderedDict()
    images_in_dataset = dataset.gtboxframe.groupby(["imageid", "imagefilename"]).size().reset_index()
    for _, datum in tqdm(images_in_dataset.iterrows()):
        img = dataset._get_dataset_image_by_id(datum["imageid"])
        im_size = FeatureMapSize(img=img)
        image_sizes_by_id[datum["imageid"]] = im_size
    print("Found {} images".format(len(image_sizes_by_id)))
    return image_sizes_by_id


def compute_average_object_size(gtboxframe, image_sizes_by_id):
    object_sizes = []
    for _, datum in gtboxframe.iterrows():
        image_id = datum["imageid"]

        img_size = image_sizes_by_id[image_id]
        box_w = (datum['rx'] - datum['lx']) * img_size.w
        box_h = (datum['by'] - datum['ty']) * img_size.h
        box_size = math.sqrt(box_w * box_h)
        if not datum['difficult']:
            object_sizes.append(box_size)
    print("Found {} non-difficult objects".format(len(object_sizes)))
    object_sizes.sort()
    median = object_sizes[len(object_sizes) // 2]
    q90 = object_sizes[len(object_sizes) * 9 // 10]
    q10 = object_sizes[len(object_sizes) // 10]
    return sum(object_sizes) / len(object_sizes), median, q10, q90


def main():
    target_object_size = 240
    data_path = get_data_path()
    logger = setup_logger("get_dataset_scales", None)

    for name in DATASET_LIST:
        dataset = build_dataset_by_name(data_path, name, eval_scale=None, logger_prefix="get_dataset_scales")

        image_sizes_by_id = get_image_sizes(dataset)
        average_size, median, q10, q90 = compute_average_object_size(dataset.gtboxframe, image_sizes_by_id)
        print("Average size of object = {0:0.2f} for image size = {1}".format(average_size, dataset.image_size))
        print("Median = {0:0.2f}, q10 = {1:0.2f}, q90 = {2:0.2f}".format(median, q10, q90))
        print("To get objects to size {0}, images should be of size {1:d}".format(target_object_size, int(dataset.image_size * target_object_size / median)))


if __name__ == "__main__":
    main()
