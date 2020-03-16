import random
from PIL import Image, ImageOps

import torch

import torchvision.transforms as transforms

from .feature_map import FeatureMapSize
from .bounding_box import BoxList, box_intersection_over_reference, FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM


class TransformList:
    """
    This class allows to store the sequence of transformations and allow to execute them in the reversed order.
    Implemented for storing the transformation of bounding boxes during data augmentation
    and for returning the boxes to the original coordinates.
    """
    def __init__(self):
        self._transforms = []

    def append(self, t):
        self._transforms.append(t)

    def __call__(self, x):
        for t in reversed(self._transforms):
            x = t(x)
        return x


def check_image_size(img, boxes):
    if boxes is not None:
        assert boxes.image_size == FeatureMapSize(img=img),\
            "Size of the image should match the size store in the accompanying BoxList"


def transpose(img, hflip=False, vflip=False, boxes=None, transform_list=None):
    check_image_size(img, boxes)
    if hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if boxes is not None:
            boxes = boxes.transpose(FLIP_LEFT_RIGHT)
            # add inverse box transform
            if transform_list is not None:
                transform_list.append(lambda boxes: boxes.transpose(FLIP_LEFT_RIGHT))
    if vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if boxes is not None:
            boxes = boxes.transpose(FLIP_TOP_BOTTOM)
            # add inverse box transform
            if transform_list is not None:
                transform_list.append(lambda boxes: boxes.transpose(FLIP_TOP_BOTTOM))
    return img, boxes


def resize(img, target_size, random_interpolation=False,
           boxes=None, transform_list=None):
    image_size = FeatureMapSize(img=img)

    if not isinstance(target_size, FeatureMapSize):
        size_max = max(image_size.w, image_size.h)
        scale = float(target_size) / size_max
        target_size = FeatureMapSize(w=int(image_size.w * scale + 0.5),
                                     h=int(image_size.h * scale + 0.5))

    method = random.choice([
        Image.BOX,
        Image.NEAREST,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
        Image.BILINEAR]) if random_interpolation else Image.BILINEAR
    img = img.resize((target_size.w, target_size.h), method)

    if boxes is not None:
        boxes = boxes.resize(target_size)
        if transform_list is not None:
            transform_list.append(lambda boxes: boxes.resize(image_size))
    else:
        assert transform_list is None
    return img, boxes


def crop(img,
         crop_size=None,
         crop_position=None,
         random_crop_size=None,  # (width, height)
         random_crop_scale=1.0, # image is virtually resized with this scale and then is cropped, default - do not rescale
         scale_jitter=1.0,
         jitter_aspect_ratio=1.0, 
         random_scale=1.0,
         coverage_keep_threshold = 0.7, coverage_remove_threshold = 0.3,
         max_trial=100, min_box_coverage=0.7,
         boxes=None, transform_list=None):
    use_boxes = boxes is not None

    image_size = FeatureMapSize(img=img)
    
    assert 0 < random_crop_scale, "Crop scale has to be > 0, we have random_crop_scale = {0}".format(random_crop_scale)
    assert 0 < scale_jitter <= 1.0, "Scale jitter has to be in (0, 1], we have scale_jitter = {0}".format(scale_jitter)
    assert 0 < jitter_aspect_ratio <= 1.0, "Sapect ratio jitter has to be in (0, 1], we have jitter_aspect_ratio = {0}".format(jitter_aspect_ratio)

    def good_crop(crop_xyxy, image_size=image_size):
        min_x = max(int(crop_xyxy[0]), 0)
        min_y = max(int(crop_xyxy[1]), 0)
        max_x = min(int(crop_xyxy[2]), image_size.w)
        max_y = min(int(crop_xyxy[3]), image_size.h)
        return min_x, min_y, max_x, max_y

    padding = [0,0,0,0]
    imh, imw = image_size.h, image_size.w
    if crop_position is not None:
        assert len(crop_position) == 1, "Precomputed crop position should have only one box, but have {0}".format(crop_position)
        crop_position_xyxy = crop_position.bbox_xyxy[0]

        if int(crop_position_xyxy[0]) < 0:
            # padding from the left side
            padding[0] = -int(crop_position_xyxy[0])
            crop_position_xyxy[0] += padding[0]
            crop_position_xyxy[2] += padding[0]
            imw += padding[0]
        
        if int(crop_position_xyxy[1]) < 0:
            # padding from the top side
            padding[1] = -int(crop_position_xyxy[1])
            crop_position_xyxy[1] += padding[1]
            crop_position_xyxy[3] += padding[1]
            imh += padding[1]

        if int(crop_position_xyxy[2]) > imw:
            # padding from the right side
            padding[2] = int(crop_position_xyxy[2])- imw
            imw += padding[2]

        if int(crop_position_xyxy[3]) > imh:
            # padding from the bottom side
            padding[3] = int(crop_position_xyxy[3])- imh
            imh += padding[3]

        img = ImageOps.expand(img, border=tuple(padding), fill=0) # border=(left, top, right, bottom), fill=label
        img_size = FeatureMapSize(img=img)
        assert img_size == FeatureMapSize(w=imw, h=imh), "computed and actual image sizes after padding should be equal"

        crop_xyxy = good_crop(crop_position_xyxy, image_size=img_size)
        # check that the crop was good should not have more than one pixel divergence
        for tuned, initial in zip(crop_xyxy, crop_position_xyxy):
            assert abs(tuned - initial) <= 1.01, "Mined crop is not fitting: mined {0}, tuned {1}".format(crop_position_xyxy, crop_xyxy)
    else:
        # do random crops
        crop_width, crop_height = random_crop_size.w, random_crop_size.h
        crop_ar = crop_width / crop_height
        crop_xyxy = good_crop((0, 0, crop_width / random_crop_scale, crop_height / random_crop_scale), image_size=image_size)
        for _ in range(max_trial):
            aspect_ratio = random.uniform(crop_ar * jitter_aspect_ratio, crop_ar / jitter_aspect_ratio)
            scale = random.uniform(random_crop_scale * scale_jitter, random_crop_scale / scale_jitter)
            w = min(crop_width / scale, imw)
            h = min(w / aspect_ratio, imh)
            w, h = int(w), int(h)

            assert imw - w >= 0, "Trying to sample a patch which is too wide: image width - {0}, patch width - {1}".format(imw, w)
            x = random.randrange(imw - w) if imw - w > 0 else 0
            
            assert imh - h >= 0, "Trying to sample a patch which is too high: image height - {0}, patch height - {1}".format(imh, h)
            y = random.randrange(imh - h) if imh - h > 0 else 0

            cur_crop_xyxy = good_crop((x, y, x+w, y+h), image_size=image_size)
            if not use_boxes:
                crop_xyxy = cur_crop_xyxy
                break

            cur_crop_boxlist = BoxList(torch.FloatTensor([cur_crop_xyxy]), image_size, mode="xyxy")
            coverage_ratio = box_intersection_over_reference(boxes, cur_crop_boxlist)
            if len(boxes) == 0 or coverage_ratio.max() >= min_box_coverage:
                crop_xyxy = cur_crop_xyxy
                break

    # crop the image
    img = img.crop(crop_xyxy)

    if use_boxes:
        # box coverage: ratio of the bbox are covered by crop - this will be used to decide the box fate: keep, remove, mark difficult
        crop_boxlist = BoxList(torch.FloatTensor([crop_xyxy]), image_size, mode="xyxy")
        coverage_ratio = box_intersection_over_reference(boxes, crop_boxlist)

        boxes = boxes.crop(crop_xyxy)

        coverage_ratio = coverage_ratio.squeeze()
        mask_cutoff_boxes = coverage_ratio < coverage_remove_threshold
        mask_difficult_boxes = coverage_ratio < coverage_keep_threshold

        if transform_list is not None:
            # implement "uncrop" operation using the saem crop function
            uncrop_xyxy = (-crop_xyxy[0], -crop_xyxy[1], -crop_xyxy[0] + image_size.w, -crop_xyxy[1] + image_size.h)
            transform_list.append(lambda boxes: boxes.crop(uncrop_xyxy))

        return img, boxes, mask_cutoff_boxes, mask_difficult_boxes
    else:
        return img, None, None, None


def random_distort(img,
                   brightness_delta=32/255.,
                   contrast_delta=0.5,
                   saturation_delta=0.5,
                   hue_delta=0.1):
    """A color related data augmentation used in SSD.

    Args:
      img: (PIL.Image) image to be color augmented.
      brightness_delta: (float) shift of brightness, range from [1-delta,1+delta].
      contrast_delta: (float) shift of contrast, range from [1-delta,1+delta].
      saturation_delta: (float) shift of saturation, range from [1-delta,1+delta].
      hue_delta: (float) shift of hue, range from [-delta,delta].

    Returns:
      img: (PIL.Image) color augmented image.

    This function was taken from https://github.com/kuangliu/torchcv/blob/master/torchcv/transforms/random_distort.py
    """
    def brightness(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(brightness=delta)(img)
        return img

    def contrast(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(contrast=delta)(img)
        return img

    def saturation(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(saturation=delta)(img)
        return img

    def hue(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(hue=delta)(img)
        return img

    img = brightness(img, brightness_delta)
    if random.random() < 0.5:
        img = contrast(img, contrast_delta)
        img = saturation(img, saturation_delta)
        img = hue(img, hue_delta)
    else:
        img = saturation(img, saturation_delta)
        img = hue(img, hue_delta)
        img = contrast(img, contrast_delta)
    return img
