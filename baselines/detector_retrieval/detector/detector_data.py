import os
import logging
from collections import OrderedDict, defaultdict

import torch
import torch.utils.data

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler
from maskrcnn_benchmark.data.collate_batch import BatchCollator, BBoxAugCollator
from maskrcnn_benchmark.data.transforms import build_transforms

from os2d.data.dataset import build_dataset_by_name
from os2d.structures.bounding_box import BoxList as BoxList_os2d
from os2d.structures.feature_map import FeatureMapSize


def make_data_loader(root_path, cfg, is_train=True, is_distributed=False, start_iter=0, class_ids=None, ignore_labels=False):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger("maskrcnn_benchmark.dataset_gtboxframe.make_data_loader")
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)

    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    if not is_train and not ignore_labels:
        assert class_ids is not None, "For validation datasets, class_ids has to be provided!"

    datasets = [build_detection_dataset_by_name(root_path, name, transforms, class_ids=class_ids, cache_images=False, ignore_labels=ignore_labels) for name in dataset_list]

    if is_train:
        assert len(datasets) == 1, "Can train on only one dataset, otherwise have to merge classes"
        class_ids = datasets[0].get_class_ids()

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0], class_ids
    return data_loaders


def build_detection_dataset_by_name(data_path, name, transforms, class_ids=None, cache_images=False, ignore_labels=False):
    dataset_os2d = build_dataset_by_name(data_path, name, eval_scale=None, cache_images=cache_images, no_image_reading=False)

    dataset = DatasetDetection(dataset_os2d, transforms, class_ids=class_ids, ignore_labels=ignore_labels)
    return dataset


class DatasetDetection(torch.utils.data.Dataset):
    def __init__(self, dataset_oneShot, transforms, class_ids=None, ignore_labels=False):
        self.dataset_os2d = dataset_oneShot
        self.transforms = transforms

        self.ignore_labels = ignore_labels # do foreground/background detection
        if self.ignore_labels:
            self.class_ids = ["bkg", "obj"]
            self.class_index_by_id = defaultdict(lambda: 1)
            self.class_index_by_id["bkg"] = 0
        else:
            # decide the class list
            if class_ids is None:
                # class list is not provided, build one from dataset
                self.class_ids = ["bkg"] + [idx for idx in self.dataset_os2d.gtboxframe['classid'].unique()]
            else:
                # use the predefined list of classes
                self.class_ids = class_ids

            self.class_index_by_id = OrderedDict()
            for index, class_id in enumerate(self.class_ids):
                self.class_index_by_id[class_id] = index

    def get_class_ids(self):
        return self.class_ids.copy()

    def get_image_annotation_for_imageid(self, image_id):
        # get data for this image
        boxes = self.dataset_os2d.get_image_annotation_for_imageid(image_id)
        return boxes

    def __getitem__(self, idx):
        image_id = self.dataset_os2d.image_ids[idx]

        # load the image as a PIL Image
        image = self.dataset_os2d._get_dataset_image_by_id(image_id)

        # get annotation
        boxlist = self.get_groundtruth(idx)
        assert boxlist.size == image.size, "Observing inconsistent image sizes: {0} and {1} at image {2}".format(boxlist.size, image.size, image_id)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def __len__(self):
        return self.dataset_os2d.num_images

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        image_id = self.dataset_os2d.image_ids[idx]
        img_size = self.dataset_os2d.get_image_size_for_image_id(image_id)
        return {"height":  img_size.h, "width":  img_size.w}

    def get_groundtruth(self, idx):
        image_id = self.dataset_os2d.image_ids[idx]
        boxlist_os2d = self.get_image_annotation_for_imageid(image_id)
        label_ids_global = boxlist_os2d.get_field("labels")

        if self.ignore_labels:
            # ignore all labels if needed
            label_ids_local = label_ids_global.clone().fill_(1)
        else:
            label_ids_local = self.convert_label_ids_global_to_local(label_ids_global, self.class_index_by_id)
        # update the labels to the ones comaptible with the trained detector
        boxlist_os2d.add_field("labels", label_ids_local)

        # convert BoxList of OS2D to BoxList of maskrcnn
        boxlist = convert_boxlist_os2d_to_maskrcnn(boxlist_os2d)
        return boxlist

    @staticmethod
    def convert_label_ids_global_to_local(label_ids_global, class_index_by_id):
        label_ids_local = [] # local indices w.r.t. batch_class_images
        if label_ids_global is not None:
            for label_id in label_ids_global:
                label_id = label_id.item()
                label_ids_local.append( class_index_by_id[label_id] )
        label_ids_local = torch.tensor(label_ids_local, dtype=torch.long)
        return label_ids_local


def convert_boxlist_os2d_to_maskrcnn(boxlist_os2d):
    image_size = (boxlist_os2d.image_size.w, boxlist_os2d.image_size.h)
    boxlist = BoxList(boxlist_os2d.bbox_xyxy, image_size, mode="xyxy")
    # add extra fields
    for f in boxlist_os2d.fields():
        boxlist.add_field(f, boxlist_os2d.get_field(f))
    return boxlist


def convert_boxlist_maskrcnn_to_os2d(boxlist_maskrcnn):
    image_size = FeatureMapSize(w=boxlist_maskrcnn.size[0], h=boxlist_maskrcnn.size[1])
    boxlist = BoxList_os2d(boxlist_maskrcnn.convert("xyxy").bbox, image_size, mode="xyxy")
    # add extra fields
    for f in boxlist_maskrcnn.fields():
        boxlist.add_field(f, boxlist_maskrcnn.get_field(f))
    return boxlist
