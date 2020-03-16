import random

from os2d.structures.transforms import random_distort, crop


class DataAugmentation():
    """
    Class stores the parameters of all the data augmentations
    """
    def __init__(self, random_flip_batches,
                       random_crop_size,
                       random_crop_scale,
                       jitter_aspect_ratio,
                       scale_jitter,
                       random_color_distortion,
                       random_crop_label_images,
                       min_box_coverage):
        # random crop size is (width, height)
        self.batch_random_hflip = random_flip_batches
        self.batch_random_vflip = random_flip_batches

        # color distortions
        self.do_random_color = random_color_distortion
        self.brightness_delta = 32/255.
        self.contrast_delta = 0.5
        self.saturation_delta = 0.5
        self.hue_delta = 0.1

        self.scale_jitter = scale_jitter
        self.jitter_aspect_ratio = jitter_aspect_ratio

        # random crop parameters
        self.do_random_crop = True if random_crop_size is not None else False
        if self.do_random_crop:
            self.random_crop_size = random_crop_size
            self.random_crop_scale = random_crop_scale
            self.random_interpolation = True
            self.coverage_keep_threshold = 0.7
            self.coverage_remove_threshold = 0.3
            self.max_trial = 100
            self.min_box_coverage = min_box_coverage  # need this to help random crops contain at least one object

        # random crops of label images
        self.do_random_crop_label_images = random_crop_label_images

    def random_distort(self, img):
        if self.do_random_color:
            img = random_distort(img,
                                 brightness_delta=self.brightness_delta,
                                 contrast_delta=self.contrast_delta,
                                 saturation_delta=self.saturation_delta,
                                 hue_delta=self.hue_delta)
        return img

    def random_crop(self, img, boxes=None, transform_list=None):
        if not self.do_random_crop:
            raise(RuntimeError("Random crop data augmentation is not initialized"))
        return self.crop_image(img, crop_position=None,
                               boxes=boxes, transform_list=transform_list,
                               random_crop_size=self.random_crop_size)

    def crop_image(self, img, crop_position, boxes=None, transform_list=None, random_crop_size=None):
        img, boxes, mask_cutoff_boxes, mask_difficult_boxes = \
                crop(img,
                     crop_position=crop_position,
                     random_crop_size=random_crop_size,
                     random_crop_scale=self.random_crop_scale,
                     crop_size=self.random_crop_size,
                     scale_jitter=self.scale_jitter,
                     jitter_aspect_ratio=self.jitter_aspect_ratio,
                     coverage_keep_threshold=self.coverage_keep_threshold,
                     coverage_remove_threshold=self.coverage_remove_threshold,
                     max_trial=self.max_trial,
                     min_box_coverage=self.min_box_coverage,
                     boxes=boxes, transform_list=transform_list)
        return img, boxes, mask_cutoff_boxes, mask_difficult_boxes

    def random_crop_label_image(self, img):
        if self.do_random_crop_label_images:
            ar = img.size[0] / img.size[1]
            new_ar = random.uniform(ar * self.jitter_aspect_ratio, ar / self.jitter_aspect_ratio)
            w = int( min(img.size[0], img.size[1] * new_ar) )
            h = int( min(img.size[0] / new_ar, img.size[1]) )
            random_crop_size = (w, h)
            img = self.crop_image(img, None, random_crop_size=random_crop_size)[0]
        return img
