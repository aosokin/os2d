import os
import sys
import random
import math
import copy
import pickle
import errno
import argparse
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

from os2d.data.dataset import build_dataset_by_name
from os2d.utils import set_random_seed, get_data_path, setup_logger, mkdir
from os2d.structures.feature_map import FeatureMapSize
from os2d.structures.bounding_box import BoxList, cat_boxlist


def parse_opts():
    parser = argparse.ArgumentParser(description="Preparing the dataset to train the retrieval model")
    parser.add_argument(
        "--dataset-train",
        help="Name of the training set (from OS2D datasets)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset-train-scale",
        help="Scale of the training dataset - need this to make sure random boxes are not too small",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--dataset-val",
        help="Name of the validation set (from OS2D datasets)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--dataset-val-scale",
        help="Scale of the val dataset - need this to make sure random boxes are not too small",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--datasets-test",
        help="Names of the test datasets (from OS2D datasets)",
        default=[],
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--datasets-test-scale",
        help="Scales of the test datasets - need this to make sure random boxes are not too small",
        type=float,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--num-random-crops-per-image",
        type=int,
        default=0,
        help="Adding this number of random crops to form a set of negatives",
    )
    parser.add_argument(
        "--iou-pos-threshold",
        type=float,
        default=0.7,
        help="IoU threshold to use sampled crops as positives",
    )
    parser.add_argument(
        "--iou-neg-threshold",
        type=float,
        default=0.3,
        help="IoU threshold to use sampled crops as negatives",
    )
    parser.add_argument(
        "--num-queries-image-to-image",
        type=int,
        default=0,
        help="Add pairs from each object and this number of random positives"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    return args


def cid2filename(cid, prefix):
    """
    Creates a training image path out of its CID name

    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved

    Returns
    -------
    filename : full image filename
    Source: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/c4fca8958631c03cefff9e8ae6bfad4d9813b633/cirtorch/datasets/datahelpers.py
    """
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


def prepare_dataset(target_path,
                    retrieval_dataset_train_name, dataset_train,
                    retrieval_dataset_val_name, dataset_val,
                    iou_pos_threshold, iou_neg_threshold,
                    num_queries_image_to_image,
                    logger,
                    retrieval_dataset_test_names=None, datasets_test=None,
                    num_random_crops_per_image=0):
    # prepare data images for train and val
    tgt_image_path_trainval = os.path.join(target_path, "train", retrieval_dataset_train_name, "ims")
    mkdir(tgt_image_path_trainval)
    logger.info(f"Train set {retrieval_dataset_train_name}")
    db_images_train = save_cropped_boxes(dataset_train, tgt_image_path_trainval, extension="", num_random_crops_per_image=num_random_crops_per_image)

    # create val subset: add all boxes from images that have at least one validation box (can add some boxes from train as distractors)
    logger.info(f"Val set {retrieval_dataset_val_name}")
    db_images_val = save_cropped_boxes(dataset_val, tgt_image_path_trainval, extension="", num_random_crops_per_image=num_random_crops_per_image)

    # prepare data images for test
    dbs_images_test = {}
    if datasets_test:
        for dataset_test, dataset_name in zip(datasets_test, retrieval_dataset_test_names):
            tgt_image_path_test = os.path.join(target_path, "test", dataset_name, "jpg")  # the folder name should be always "test" - from cirtorch
            mkdir(tgt_image_path_test)
            logger.info(f"Eval dataset: {dataset_name}")
            dbs_images_test[dataset_name] = save_cropped_boxes(dataset_test, tgt_image_path_test, num_random_crops_per_image=num_random_crops_per_image)

    # save GT images from train
    db_classes_train = save_class_images(dataset_train,
                                         os.path.join(target_path, "train", retrieval_dataset_train_name, "ims"), extension="")

    # save GT images from val
    db_classes_val = save_class_images(dataset_val,
                                       os.path.join(target_path, "train", retrieval_dataset_train_name, "ims"), extension="")

    # save GT images for testing
    dbs_classes_test = {}
    if datasets_test:
        for dataset_test, dataset_name in zip(datasets_test, retrieval_dataset_test_names):
            dbs_classes_test[dataset_name] = save_class_images(dataset_test,
                                                    os.path.join(target_path, "test", dataset_name, "jpg"))

    # merge databases
    logger.info(f"Processing trainval set from {retrieval_dataset_train_name} and {retrieval_dataset_val_name}")
    db_train = create_train_database_queries(db_images_train, db_classes_train,
                                             iou_pos_threshold=iou_pos_threshold,
                                             iou_neg_threshold=iou_neg_threshold,
                                             logger=logger,
                                             num_queries_image_to_image=num_queries_image_to_image)
    db_val = create_train_database_queries(db_images_val, db_classes_val,
                                           iou_pos_threshold=iou_pos_threshold,
                                           iou_neg_threshold=iou_neg_threshold,
                                           logger=logger,
                                           num_queries_image_to_image=num_queries_image_to_image)

    dbs_test = {}
    if datasets_test:
        for dataset_name in retrieval_dataset_test_names:
            logger.info(f"Processing test set {dataset_name}")
            dbs_test[dataset_name] = create_test_database_queries(dbs_images_test[dataset_name], dbs_classes_test[dataset_name],
                                                                  iou_pos_threshold=iou_pos_threshold,
                                                                  iou_neg_threshold=iou_neg_threshold,
                                                                  logger=logger,
                                                                  num_queries_image_to_image=num_queries_image_to_image)

    # save trainval to disk
    db_trainval = {"train":db_train, "val":db_val}
    db_fn = os.path.join(os.path.join(target_path, "train", retrieval_dataset_train_name), f"{retrieval_dataset_train_name}.pkl")
    with open(db_fn, "wb") as f:
        pickle.dump(db_trainval, f)

    # save train separately for whitening
    db_fn = os.path.join(os.path.join(target_path, "train", retrieval_dataset_train_name), f"{retrieval_dataset_train_name}-whiten.pkl")
    with open(db_fn, "wb") as f:
        pickle.dump(db_train, f)

    # save test to disk
    if datasets_test:
        for dataset_name in retrieval_dataset_test_names:
            db_fn = os.path.join(os.path.join(target_path, "test", dataset_name ), f"gnd_{dataset_name}.pkl")
            with open(db_fn, "wb") as f:
                pickle.dump(dbs_test[dataset_name], f)


def main():
    args = parse_opts()
    set_random_seed(args.random_seed)

    logger_name = "retrieval_data"
    retrieval_dataset_name_suffix = "-retrieval"
    logger = setup_logger(logger_name, None)
    data_path = get_data_path()

    script_path = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(script_path, "cnnimageretrieval-pytorch", "data")
    mkdir(target_path)

    dataset_train = build_dataset_by_name(data_path, args.dataset_train,
                                        eval_scale=args.dataset_train_scale,
                                        logger_prefix=logger_name)
    retrieval_dataset_train_name = dataset_train.get_name() + retrieval_dataset_name_suffix

    dataset_val = build_dataset_by_name(data_path, args.dataset_val,
                                        eval_scale=args.dataset_val_scale,
                                        logger_prefix=logger_name)
    retrieval_dataset_val_name = dataset_val.get_name() + retrieval_dataset_name_suffix


    datasets_test = []
    retrieval_dataset_test_names = []
    if args.datasets_test:
        if len(args.datasets_test_scale) == 1:
            datasets_test_scale = args.datasets_test_scale * len(args.datasets_test)
        else:
            datasets_test_scale = args.datasets_test_scale
        assert len(args.datasets_test) == len(datasets_test_scale), "Arg datasets-test-scale should be of len 1 or of len equal to the len of datasets-test"

        for dataset_name, scale in zip(args.datasets_test, datasets_test_scale):
            dataset = build_dataset_by_name(data_path, dataset_name,
                                            eval_scale=scale,
                                            logger_prefix=logger_name)
            retrieval_dataset_test_names.append(dataset.get_name() + retrieval_dataset_name_suffix)
            datasets_test.append(dataset)

    # create dataset
    if args.num_random_crops_per_image > 0:
        crop_suffix = f"-rndCropPerImage{args.num_random_crops_per_image}"
        retrieval_dataset_train_name = retrieval_dataset_train_name + crop_suffix
        retrieval_dataset_val_name = retrieval_dataset_val_name + crop_suffix
        retrieval_dataset_test_names = [name + crop_suffix for name in retrieval_dataset_test_names]

    prepare_dataset(target_path,
                    retrieval_dataset_train_name, dataset_train,
                    retrieval_dataset_val_name, dataset_val,
                    args.iou_pos_threshold, args.iou_neg_threshold,
                    args.num_queries_image_to_image,
                    logger,
                    retrieval_dataset_test_names=retrieval_dataset_test_names, datasets_test=datasets_test,
                    num_random_crops_per_image=args.num_random_crops_per_image)


def create_train_database_queries(db_images, db_classes_train,
                                  iou_pos_threshold,
                                  iou_neg_threshold,
                                  logger,
                                  num_queries_image_to_image=0):
    db_train = merge_dicts_of_lists(db_images, db_classes_train)
   # create train queries
    db_train["qidxs"] = []
    db_train["pidxs"] = []
    query_offset = len(db_images["cids"])

    # create reverse indexing table
    query_hash = {}
    for i_label, query_label in enumerate(db_classes_train["cluster"]):
        query_hash[query_label] = i_label

    # hash GT boxes
    gtbox_hash = {}
    for i_crop in range(len(db_images["bbox"])):
        if db_images["type"][i_crop] == "gtproposal":
            imageid = db_images["imageid"][i_crop]
            if imageid in gtbox_hash:
                gtbox_hash[imageid].append(i_crop)
            else:
                gtbox_hash[imageid] = [i_crop]

    # get a lists of crops per class
    hash_crop_per_label = {}
    for i_crop in range(len(db_images["bbox"])):
        class_id = db_images["cluster"][i_crop]
        if class_id in hash_crop_per_label:
            hash_crop_per_label[class_id].append(i_crop)
        else:
            hash_crop_per_label[class_id] = [i_crop]
    for class_id in hash_crop_per_label:
        hash_crop_per_label[class_id] = set(hash_crop_per_label[class_id])

    # add pairs of queries and positives
    max_rand_crop_iou = 0
    for i_crop, prop_label in enumerate(db_images["cluster"]):
        if db_images["type"][i_crop] == "gtproposal":
            # exclude boxes with the difficult flag and missing classes
            if not db_images["difficult"][i_crop]:
                if prop_label in query_hash:
                    db_train["qidxs"].append(query_offset + query_hash[prop_label])
                    db_train["pidxs"].append(i_crop)
                # add pairs from positives of the same class
                if num_queries_image_to_image > 0:
                    candidate_positives = hash_crop_per_label[prop_label] - set([i_crop])
                    if len(candidate_positives) > num_queries_image_to_image:
                        new_positives = random.sample(list(candidate_positives), num_queries_image_to_image)
                    else:
                        new_positives = list(candidate_positives)
                    for j_crop in new_positives:
                        db_train["qidxs"].append(i_crop)
                        db_train["pidxs"].append(j_crop)
        elif db_images["type"][i_crop] == "randomcrop":
            # collect all GT boxes from that image
            imageid = db_images["imageid"][i_crop]
            gt_boxes = torch.stack( [torch.FloatTensor( db_images["bbox"][i_box_gt] ) for i_box_gt in gtbox_hash[imageid]],
                                  dim=0 )
            gt_labels = [db_images["cluster"][i_box_gt] for i_box_gt in gtbox_hash[imageid]]
            this_box = torch.FloatTensor(db_images["bbox"][i_crop]).view(1, 4)

            ious = box_iou(gt_boxes, this_box).view(-1)

            # do two passes: first exclude objects with iou > iou_neg_threshold
            # then add objects with iou > iou_pos_threshold as positives
            for i_gt, iou in enumerate(ious):
                max_rand_crop_iou = max(max_rand_crop_iou, iou)
                gt_label = gt_labels[i_gt]
                if gt_label in query_hash:
                    if iou > iou_neg_threshold:
                        db_train["cluster"][i_crop] = gt_label
            for i_gt, iou in enumerate(ious):
                gt_label = gt_labels[i_gt]
                if iou > iou_pos_threshold:
                    if gt_label in query_hash:
                        j_crop = query_offset + query_hash[gt_label]
                    elif len(hash_crop_per_label[gt_label]) > 0:
                        j_crop = random.sample(list(hash_crop_per_label[gt_label]), 1)[0]
                    else:
                        j_crop = None
                    if j_crop is not None:
                        db_train["cluster"][i_crop] = gt_label
                        db_train["qidxs"].append(j_crop)
                        db_train["pidxs"].append(i_crop)
        else:
            raise("Crop of unknown type {}".format(db_images["type"][i_crop]))

    logger.info("Created {} pairs out of {} labels and {} detections".format(len(db_train["qidxs"]), len(db_classes_train["cids"]), len(db_images["cids"])))
    if max_rand_crop_iou > 0:
        logger.info("Max IoU of randcrops and GT equals {}".format(max_rand_crop_iou))

    return db_train


def create_test_database_queries(db_images_test, db_classes_test,
                                 iou_pos_threshold,
                                 iou_neg_threshold,
                                 logger,
                                 num_queries_image_to_image=0):
    db_test = {"gnd":[], "imlist":[], "qimlist":[]}

    db_test["imlist"] = [cid2filename(cid, prefix="") for cid in db_images_test["cids"]]
    db_test["qimlist"] = [cid2filename(cid, prefix="") for cid in db_classes_test["cids"]]

    # hash queries
    query_hash = {}
    for i_query in range(len(db_test["qimlist"])):
        gnd = {"bbx": None, "ok":[], "junk":[]}

        gnd["bbx"] = db_classes_test["bbox"][i_query]  # format (x1,y1,x2,y2)

        db_test["gnd"].append(gnd)
        query_label = db_classes_test["cluster"][i_query]
        query_hash[query_label] = i_query

    # add extra queries from some gt boxes
    if num_queries_image_to_image > 0:
        # select num_queries_image_to_image queries in each class
        gtbox_hash_per_class = {}
        for i_crop in range(len(db_images_test["cluster"])):
            classid = db_images_test["cluster"][i_crop]
            if classid != -1 and classid in gtbox_hash_per_class: # -1 corresponds to background
                gtbox_hash_per_class[classid].append(i_crop)
            else:
                gtbox_hash_per_class[classid] = [i_crop]

        query_gt_per_label = {}
        query_original_id = {}
        for class_id in gtbox_hash_per_class:
            # remove all difficult boxes
            gtbox_hash_per_class[class_id] = [i_image for i_image in gtbox_hash_per_class[class_id] if not db_images_test["difficult"][i_image]]
            # leave at most num_queries_image_to_image queries in each class
            if len(gtbox_hash_per_class[class_id]) > num_queries_image_to_image:
                subsample = random.sample(gtbox_hash_per_class[class_id], num_queries_image_to_image)
                gtbox_hash_per_class[class_id] = subsample
            gtbox_hash_per_class[class_id] = sorted(gtbox_hash_per_class[class_id])
            # add selected GTs as queries
            query_gt_per_label[class_id] = []
            for i_image in gtbox_hash_per_class[class_id]:
                gnd = {"bbx": None, "ok":[], "junk":[]}
                crop_box = db_images_test["bbox"][i_image]  # format (x1,y1,x2,y2)
                full_image_box = [0, 0, crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]]
                gnd["bbx"] = full_image_box
                new_query_id = len(db_test["gnd"])
                query_gt_per_label[class_id].append(new_query_id)
                query_original_id[new_query_id] = i_image
                db_test["gnd"].append(gnd)
                db_test["qimlist"].append(cid2filename(db_images_test["cids"][i_image], prefix=""))

    # hash GT boxes
    gtbox_hash = {}
    for i_crop in range(len(db_images_test["bbox"])):
        if db_images_test["type"][i_crop] == "gtproposal":
            imageid = db_images_test["imageid"][i_crop]
            if imageid in gtbox_hash:
                gtbox_hash[imageid].append(i_crop)
            else:
                gtbox_hash[imageid] = [i_crop]

    # collect all positives
    # move the ones with difficult flags to junk
    num_ok_pairs = 0
    num_junk_pairs = 0
    max_rand_crop_iou = 0.0
    for i_crop, label in enumerate(db_images_test["cluster"]):
        if db_images_test["type"][i_crop] == "gtproposal":
            # add the gt box as positive to the query from the label image
            if label in query_hash:
                # check that this label exists in this set - used to distinguish test and test only
                if db_images_test["difficult"][i_crop] == 1:
                    db_test["gnd"][ query_hash[label] ]["junk"].append(i_crop)
                    num_junk_pairs += 1
                else:
                    db_test["gnd"][ query_hash[label] ]["ok"].append(i_crop)
                    num_ok_pairs += 1
            # add this box to the GT queries of the same label
            if num_queries_image_to_image > 0 and label in query_gt_per_label:
                for i_gt_query in query_gt_per_label[label]:
                    # add pair to itself as junk not to influence anything
                    if db_images_test["difficult"][i_crop] == 1 or query_original_id[i_gt_query] == i_crop:
                        db_test["gnd"][ i_gt_query ]["junk"].append(i_crop)
                        num_junk_pairs += 1
                    else:
                        db_test["gnd"][ i_gt_query ]["ok"].append(i_crop)
                        num_ok_pairs += 1

        elif db_images_test["type"][i_crop] == "randomcrop":
            # collect all GT boxes from that image
            imageid = db_images_test["imageid"][i_crop]
            if imageid in gtbox_hash:
                # filter out images that intersect with GT to much
                gt_boxes = torch.stack( [torch.FloatTensor( db_images_test["bbox"][i_box_gt] ) for i_box_gt in gtbox_hash[imageid]],
                                    dim=0 )
                gt_labels = [db_images_test["cluster"][i_box_gt] for i_box_gt in gtbox_hash[imageid]]
                this_box = torch.FloatTensor(db_images_test["bbox"][i_crop]).view(1, 4)

                ious = box_iou(gt_boxes, this_box).view(-1)
                for i_gt, iou in enumerate(ious):
                    max_rand_crop_iou = max(max_rand_crop_iou, iou)
                    gt_label = gt_labels[i_gt]
                    if gt_label in query_hash:
                        # add boxes only if the corresponding label is currently used
                        if iou > iou_pos_threshold:
                            # if iou is very big than make this crop positive to the corresponding label
                            db_test["gnd"][ query_hash[gt_label] ]["ok"].append(i_crop)
                            num_ok_pairs += 1
                        elif iou > iou_neg_threshold:
                            # if iou is somewhere in the middle than make this crop junk - unknown label
                            db_test["gnd"][ query_hash[gt_label] ]["junk"].append(i_crop)
                            num_junk_pairs += 1
                        else:
                            # if iou is small than make this crop negative - no mentions in db_test
                            pass
                     # add this box to the GT queries of the same label
                    if num_queries_image_to_image > 0 and gt_label in query_gt_per_label:
                        for i_gt_query in query_gt_per_label[gt_label]:
                            if iou > iou_pos_threshold:
                                # if iou is very big than make this crop positive to the corresponding label
                                db_test["gnd"][ i_gt_query ]["ok"].append(i_crop)
                                num_ok_pairs += 1
                            elif iou > iou_neg_threshold:
                                # if iou is somewhere in the middle than make this crop junk - unknown label
                                db_test["gnd"][ i_gt_query ]["junk"].append(i_crop)
                                num_junk_pairs += 1
                            else:
                                # if iou is small than make this crop negative - no mentions in db_test
                                pass

        else:
            raise("Crop of unknown type {}".format(db_images_test["type"][i_crop]))

    logger.info("Created {} ok and {} junk pairs out of {} labels and {} detections".format(num_ok_pairs, num_junk_pairs, len(db_classes_test["cids"]), len(db_images_test["cids"])))
    if max_rand_crop_iou > 0:
        logger.info("Max IoU of randcrops and GT equals {}".format(max_rand_crop_iou))

    return db_test


def box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4]; format: (xmin,ymin,xmax,ymax)
      box2: (tensor) bounding boxes, sized [M,4]; format: (xmin,ymin,xmax,ymax)

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


def merge_dicts_of_lists(A, B):
    assert set(B.keys()) == set(A.keys()), "Keys of the dicts to merge don't match"
    C = copy.deepcopy(A)
    list_len = None
    for key, val_list in C.items():
        val_list.extend( B[key] )
        if list_len is None:
            list_len = len(val_list)
        else:
            assert list_len == len(val_list), "List {} are not of same length with others".format(key)
    return C


def save_class_images(dataset, tgt_image_path, extension=".jpg"):
    db = {"cids":[], "cluster":[], "gtbboxid":[], "classid":[], "imageid":[], "difficult":[], "type":[], "size":[], "bbox":[]}

    for lbl, label_image in tqdm(dataset.gt_images_per_classid.items()):
        # create the file name to be used with cirtorch.datasets.datahelpers.cid2filename and their dataloader
        cid = "lbl{label:05d}{box_type}".format(label = lbl, box_type="CL")
        file_name = cid2filename(cid, prefix=tgt_image_path)

        # save the image
        image_path, _ = os.path.split(file_name)
        mkdir(image_path)
        if extension:
            label_image.save("{}{}".format(file_name, extension))
        else:
            # cirtorch uses files with empty extension for training for some reason, need to support that
            label_image.save("{}".format(file_name), format="jpeg")

        width, height = label_image.size
        box = [0, 0, width, height]  # format (x1,y1,x2,y2)

        # add to the db structure
        db["cids"].append(cid)
        db["cluster"].append(lbl)  # use labels as clusters not to sample negatives from the same object
        db["classid"].append(lbl)
        db["gtbboxid"].append(None)
        db["imageid"].append(None)
        db["difficult"].append(None)
        db["type"].append("classimage")
        db["size"].append(label_image.size)
        db["bbox"].append(box)  # format (x1,y1,x2,y2)

    return db


def save_cropped_boxes(dataset, tgt_image_path, extension=".jpg", num_random_crops_per_image=0):
    # crop all the boxes
    db = {"cids":[], "cluster":[], "gtbboxid":[], "classid":[], "imageid":[], "difficult":[], "type":[], "size":[], "bbox":[]}

    for image_id in tqdm(dataset.image_ids):
        img = dataset._get_dataset_image_by_id(image_id)
        boxes = dataset.get_image_annotation_for_imageid(image_id)

        assert boxes.has_field("labels"), "GT boxes need a field 'labels'"
        # remove all fields except "labels" and "difficult"
        for f in boxes.fields():
            if f not in ["labels", "difficult"]:
                boxes.remove_field(f)
        if not boxes.has_field("difficult"):
            boxes.add_field("difficult", torch.zeros(len(boxes), dtype=torch.bool))

        num_gt_boxes = len(boxes)
        im_size = FeatureMapSize(img=img)
        assert im_size == boxes.image_size

        eval_scale = dataset.get_eval_scale()

        # sample random boxes if needed
        if num_random_crops_per_image > 0:
            boxes_random = torch.rand(num_random_crops_per_image, 4)
            x1 = torch.min(boxes_random[:, 0], boxes_random[:, 2]) * im_size.w
            x2 = torch.max(boxes_random[:, 0], boxes_random[:, 2]) * im_size.w
            y1 = torch.min(boxes_random[:, 1], boxes_random[:, 3]) * im_size.h
            y2 = torch.max(boxes_random[:, 1], boxes_random[:, 3]) * im_size.h
            boxes_random = torch.stack([x1, y1, x2, y2], 1).floor()

            # crop boxes that are too small
            min_size = 10.0 / eval_scale * max(im_size.w, im_size.h)
            mask_bad_boxes = (boxes_random[:,0] + min_size > boxes_random[:,2]) | (boxes_random[:,1] + min_size > boxes_random[:,3])
            good_boxes = torch.nonzero(~mask_bad_boxes).view(-1)
            boxes_random = boxes_random[good_boxes]

            boxes_random = BoxList(boxes_random, im_size, mode="xyxy")

            boxes_random.add_field("labels", torch.full([len(boxes_random)], -1, dtype=torch.long))
            boxes_random.add_field("difficult", torch.zeros(len(boxes_random), dtype=torch.bool))
            boxes = cat_boxlist([boxes, boxes_random])

        if boxes is not None:
            for i_box in range(len(boxes)):
                # box format: left, top, right, bottom
                box = boxes[i_box].bbox_xyxy.view(-1)
                box = [b.item() for b in box]
                cropped_img = img.crop(box)

                if i_box < num_gt_boxes:
                    lbl = boxes[i_box].get_field("labels").item()
                    dif_flag = boxes[i_box].get_field("difficult").item()
                    box_id = i_box
                    box_type = "GT"
                else:
                    lbl = -1
                    dif_flag = 0
                    box_id = i_box
                    box_type = "RN"

                # create the file name to be used with cirtorch.datasets.datahelpers.cid2filename and their dataloader
                cid = "box{box_id:05d}_lbl{label:05d}_dif{dif:01d}_im{image_id:05d}{box_type}".format(box_id=box_id, image_id = image_id, label = lbl, dif = dif_flag, box_type=box_type)
                file_name = cid2filename(cid, prefix=tgt_image_path)

                # save the image
                image_path, _ = os.path.split(file_name)
                mkdir(image_path)
                if extension:
                    cropped_img.save("{}{}".format(file_name, extension))
                else:
                    # cirtorch uses files with empty extension for training for some reason, need to support that
                    cropped_img.save("{}".format(file_name), format="jpeg")

                # add to the db structure
                db["cids"].append(cid)
                db["cluster"].append(lbl)  # use labels as clusters not to sample negatives from the same object
                db["classid"].append(lbl)
                db["gtbboxid"].append(box_id)
                db["imageid"].append(image_id)
                db["difficult"].append(dif_flag)
                if i_box < num_gt_boxes:
                    db["type"].append("gtproposal")
                else:
                    db["type"].append("randomcrop")
                db["size"].append(cropped_img.size)
                db["bbox"].append(box)  # format (x1,y1,x2,y2)

    return db


if __name__ == "__main__":
    main()
