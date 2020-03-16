import math
import itertools
from functools import lru_cache

import torch

from torchvision.models.detection._utils import Matcher, BoxCoder, encode_boxes

from os2d.structures.bounding_box import BoxList, cat_boxlist, boxlist_iou, nms
from os2d.utils import masked_select_or_fill_constant


BOX_ENCODING_WEIGHTS = torch.tensor([10, 10, 5, 5])


@lru_cache()
def create_strided_boxes_columnfirst(grid_size, box_size, box_stride):
    """Create a list of boxes, shifted horizontally and vertically with some stride. The boxes are appearinf in the column-first (vertical shift first) order starting from the top left. The boxes are in the x1y1x2y2 format.
  
    Args:
      grid_size: (tuple of len 2) height and width of the grid, the number of boxes equals grid_size.w * grid_size.h
      box_size: (tuple of len 2) height and width of all the boxes
      box_stride: (tuple of len 2) vertical and horizontal strides, respectively

    Returns:
      (Tensor) tensor of boxes, size [grid_size.w * grid_size.h, 4]

    Comment: even vectorized this functions can be quite slow, thus I put it into functools.lru_cache decorator to cache the calls
    """
    # # slow code
    # boxes_cXcYWH = []
    # for h in range(grid_size.h):
    #     for w in range(grid_size.v):
    #         cx = (w + 0.5) * box_stride.w
    #         cy = (h + 0.5) * box_stride.h
    #         boxes_cXcYWH.append((cx, cy, box_size.w, box_size.h))
    # boxes_cXcYWH = torch.FloatTensor(boxes)  # 'cx cy w h'

    # vectorized code

    # get center positions
    h = torch.arange(0, grid_size.h, dtype=torch.float)
    cy = (h + 0.5) * box_stride.h
    w = torch.arange(0, grid_size.w, dtype=torch.float)
    cx = (w + 0.5) * box_stride.w

    # make tuples of coordinates
    cx = cx.unsqueeze(0).expand(cy.size(0), -1).contiguous()
    cy = cy.unsqueeze(1).expand(-1, cx.size(1)).contiguous()
    cx = cx.view(-1)
    cy = cy.view(-1)

    # create sizes of appropriate length
    sx = torch.FloatTensor( [box_size.w] ).expand_as(cx)
    sy = torch.FloatTensor( [box_size.h] ).expand_as(cy)

    boxes_cXcYWH = torch.stack( [cx, cy, sx, sy], dim=1 )

    boxes_xyxy = BoxList.convert_bbox_format(boxes_cXcYWH, "cx_cy_w_h", "xyxy")
    return boxes_xyxy


class BoxGridGenerator:
    """This class allows to specialize the call of the create_strided_boxes_columnfirst to the specific network stride and box sizes/
    """
    def __init__(self, box_size, box_stride):
        """
        Args:
            box_size : FeatureMapSize
            box_stride : FeatureMapSize
        """
        self.box_size = box_size
        self.box_stride = box_stride

    def create_strided_boxes_columnfirst(self, fm_size):
        return create_strided_boxes_columnfirst(fm_size, self.box_size, self.box_stride)

    def  get_box_to_cut_anchor(self, img_size, crop_size, fm_size, default_box_transform=None):
        """For each anchor box, obtain the box of the size crop_size such that
            2) the anchor box is roughly in the middle of the crop
            2) it is aligned with the stride of the anchor boxes
        Need this function so make sure that after cropping the original image we get the same cropped feature map
        (problems are caused by the network stride).
        Used in train.mine_hard_patches.
        Args:
            img_size (FeatureMapSize) - size of the original image
            crop_size (FeatureMapSize) - size of the crop needed for training
            fm_size (FeatureMapSize) - size of the feature map from this image
            default_box_transform (TransformList) - transformation to convert the boxes to the img_size scale
        Returns:
            crop_boxes_xyxy, anchor_box (BoxList)
            anchor_index (tensor of indices)
        """

        # anchors are encoded in the column-first row-last order
        # converting position (anchor_index) to (anchor_y_index, anchor_x_index)
        anchor_index = torch.arange( fm_size.h * fm_size.w )
        anchor_y_index = anchor_index // fm_size.w
        anchor_x_index = anchor_index % fm_size.w

        # get the center of the anchor
        cx = (anchor_x_index.float() + 0.5) * self.box_stride.w
        cy = (anchor_y_index.float() + 0.5) * self.box_stride.h

        # get the top-left corner of the box to crop
        box_left = cx - crop_size.w / 2
        box_top = cy - crop_size.h / 2

        anchor_box = torch.stack([cx, cy, torch.full_like(cx, self.box_size.w), torch.full_like(cx, self.box_size.h)], 1)
        anchor_box = BoxList.convert_bbox_format(anchor_box, "cx_cy_w_h", "xyxy")

        # round down to strided positions in the image
        def floor_to_stride(pos, stride):
            return (torch.floor(pos) // stride) * stride

        def ceil_to_stride(pos, stride):
            return torch.floor(torch.ceil(torch.floor(pos) / stride)) * stride

        box_left = masked_select_or_fill_constant(floor_to_stride(box_left, self.box_stride.w), box_left > 0, 0)
        box_top = masked_select_or_fill_constant(floor_to_stride(box_top, self.box_stride.h), box_top > 0, 0)

        # get another corner
        box_right = box_left + crop_size.w
        box_bottom = box_top + crop_size.h

        # make sure the crop is in the image: this stratery should be compatible with the one used in augmentation.crop_image
        mask_have_to_move_right = box_left < 0
        box_right[mask_have_to_move_right] -= box_left[mask_have_to_move_right]
        box_left[mask_have_to_move_right] = 0

        mask = box_right > img_size.w
        shift_left = ceil_to_stride(box_right - img_size.w, self.box_stride.w)
        mask_good_fit = (box_left - shift_left >= 0)
        # can safely shift left
        box_left[mask & mask_good_fit] -= shift_left[mask & mask_good_fit]
        box_right[mask & mask_good_fit] -= shift_left[mask & mask_good_fit]
        # just output full width
        box_left[ mask & ~mask_good_fit ] = 0
        box_right[ mask & ~mask_good_fit ] = crop_size.w

        
        mask_have_to_move_down = box_top < 0
        box_bottom[mask_have_to_move_down] -= box_top[mask_have_to_move_down]
        box_top[mask_have_to_move_down] = 0

        mask = box_bottom > img_size.h
        shift_up = ceil_to_stride(box_bottom - img_size.h, self.box_stride.h)
        mask_good_fit = (box_top - shift_up >= 0)
        # can safely shift up
        box_top[mask & mask_good_fit] -= shift_up[mask & mask_good_fit]
        box_bottom[mask & mask_good_fit] -= shift_up[mask & mask_good_fit]
        # just output full height
        box_top[ mask & ~mask_good_fit ] = 0
        box_bottom[ mask & ~mask_good_fit ] = crop_size.h

        # assemble the box
        crop_boxes_xyxy = torch.stack([box_left, box_top, box_right, box_bottom], 1) # lx ty rx by

        # convert boxes to the original image coordinates
        crop_boxes_xyxy = BoxList(crop_boxes_xyxy, img_size, mode="xyxy")
        anchor_box = BoxList(anchor_box, img_size, mode="xyxy")
        if default_box_transform is not None:
            crop_boxes_xyxy = default_box_transform(crop_boxes_xyxy)
            anchor_box = default_box_transform(anchor_box)

        return crop_boxes_xyxy, anchor_box, anchor_index


class Os2dBoxCoder:
    """This class implements the analogue of the BoxCoder from torchvision, but supports image pyramids and has NMS inside.
    """
    def __init__(self, positive_iou_threshold, negative_iou_threshold,
                       remap_classification_targets_iou_pos, remap_classification_targets_iou_neg,
                       output_box_grid_generator, function_get_feature_map_size,
                       do_nms_across_classes=False):
        self.get_feature_map_size = function_get_feature_map_size
        self.output_box_grid_generator = output_box_grid_generator
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.remap_classification_targets_iou_pos = remap_classification_targets_iou_pos
        self.remap_classification_targets_iou_neg = remap_classification_targets_iou_neg
        self.do_nms_across_classes = do_nms_across_classes

        self.weights = BOX_ENCODING_WEIGHTS
        self.box_coder = BoxCoder(self.weights)
        self.matcher = Matcher(self.positive_iou_threshold,
                               self.negative_iou_threshold)
        self.matcher_remap = Matcher(self.remap_classification_targets_iou_pos,
                                     self.remap_classification_targets_iou_neg)

    def _get_default_boxes(self, img_size):
        """Compute the default (anchor) bounding boxes given the image size.
        Not caching this because both self._get_feature_map_size_per_image_size and self.output_box_grid_generator.create_strided_boxes_columnfirst are cached.

        Args:
            img_size (FeatureMapSize)
        Return:
            boxes_xyxy (BoxList)
        """
        feature_map_size = self._get_feature_map_size_per_image_size(img_size)
        boxes_xyxy = self.output_box_grid_generator.create_strided_boxes_columnfirst(feature_map_size)
        boxes_xyxy = BoxList(boxes_xyxy, image_size=img_size, mode="xyxy")
        return boxes_xyxy

    @lru_cache()
    def _get_feature_map_size_per_image_size(self, img_size):
        """Compute feature_map_size for this particular image size.
        The calls are cached with @lru_cache() for speed.
        """
        return self.get_feature_map_size(img_size)

    @staticmethod
    def assign_anchors_to_boxes_threshold(detection_boxes, annotation_boxes, matcher):
        """assign_anchors_to_boxes_threshold is a wrapper to call the Matcher class of torchvision.
        Assigns proposal boxes to the annotation boxes.
        detection_boxes, annotation_boxes are BoxList.
        matcher is a torchvision Matcher instance.
        """
        ious = boxlist_iou(annotation_boxes, detection_boxes)

        index = matcher(ious)

        # assign difficult flags
        class_difficult_flags = annotation_boxes.get_field("difficult")
        good_index_mask = index >= 0
        if good_index_mask.any():
            good_index = good_index_mask.nonzero()
            difficult_mask = class_difficult_flags[index[good_index]]
            difficult_index = good_index[difficult_mask]
            index[difficult_index] = -2

        return index, ious

    def remap_anchor_targets(self, loc_scores, batch_img_size, class_image_sizes, batch_boxes,
                             box_reverse_transform=None):
        """Target remapping: changing detection targets (positive/negatives) after computing the localization from the model
        Used in train.train_one_batch and train.mine_hard_patches
        """
        cls_targets_remapped = []
        ious_anchor_corrected = []
        ious_anchor = []
        for i_image in range(loc_scores.size(0)):
            default_boxes_xyxy = self._get_default_boxes(batch_img_size[i_image]) # num_anchors x 4
            image_cls_targets_remapped = []
            image_ious_anchor_corrected = []
            image_ious_anchor = []
            for i_label in range(loc_scores.size(1)):
                cur_loc_scores = loc_scores[i_image, i_label].transpose(0,1)  # num_anchors x 4
                cur_default_boxes_xyxy = default_boxes_xyxy.to(cur_loc_scores) # num_anchors x 4
                box_predictions = self.build_boxes_from_loc_scores(cur_loc_scores, cur_default_boxes_xyxy) # num_anchors x 4

                if box_reverse_transform is not None:
                    box_predictions = box_reverse_transform[i_image](box_predictions)
                    cur_default_boxes_xyxy = box_reverse_transform[i_image](cur_default_boxes_xyxy)
                
                # match boxes to the GT objects
                cur_labels = batch_boxes[i_image].get_field("labels")
                label_mask = cur_labels == i_label
                ids = torch.nonzero(label_mask).view(-1)
                device = box_predictions.bbox_xyxy.device
                if ids.numel() > 0:
                    class_boxes = batch_boxes[i_image][ids].to(device=device)

                    # compute IoUs with anchors
                    _, ious = self.assign_anchors_to_boxes_threshold(cur_default_boxes_xyxy,
                                                                     class_boxes,
                                                                     self.matcher_remap)
                    ious_anchors_max_gt = ious.max(0)[0] # IoU with the best fitting GT box

                    # compute IoUs with corrected anchors
                    index, ious = self.assign_anchors_to_boxes_threshold(box_predictions,
                                                                         class_boxes,
                                                                         self.matcher_remap)
                    ious_corrected_max_gt = ious.max(0)[0] # IoU with the best fitting GT box

                    # assign labels
                    #   for index == -2 assign -1 (ignore at training) 
                    #   for index == -1 assign 0 (negatives)
                    #   for others assign 1 (positives)
                    image_class_cls_targets_remapped = 1 + index.clamp(min=-2, max=0)
                else:
                    # no GT boxes of class i_label in image i_image
                    image_class_cls_targets_remapped = torch.LongTensor(len(cur_default_boxes_xyxy)).zero_().to(device=device)
                    ious_anchors_max_gt = torch.FloatTensor(len(cur_default_boxes_xyxy)).zero_().to(device=device)
                    ious_corrected_max_gt = torch.FloatTensor(len(cur_default_boxes_xyxy)).zero_().to(device=device)
                image_cls_targets_remapped.append(image_class_cls_targets_remapped)
                image_ious_anchor_corrected.append(ious_corrected_max_gt)
                image_ious_anchor.append(ious_anchors_max_gt)

            image_cls_targets_remapped = torch.stack(image_cls_targets_remapped, 0)  # num_labels x num_anchors
            cls_targets_remapped.append(image_cls_targets_remapped)

            image_ious_anchor_corrected = torch.stack(image_ious_anchor_corrected, 0)  # num_labels x num_anchors
            ious_anchor_corrected.append(image_ious_anchor_corrected)
            image_ious_anchor = torch.stack(image_ious_anchor, 0)  # num_labels x num_anchors
            ious_anchor.append(image_ious_anchor)
        
        cls_targets_remapped = torch.stack(cls_targets_remapped, 0) # num_images x num_labels x num_anchors
        
        ious_anchor_corrected = torch.stack(ious_anchor_corrected, 0) # num_images x num_labels x num_anchors
        ious_anchor = torch.stack(ious_anchor, 0) # num_images x num_labels x num_anchors

        return cls_targets_remapped, ious_anchor, ious_anchor_corrected
        
    @staticmethod
    def build_loc_targets(class_boxes, default_boxes):
        """build_loc_targets is a wrapper for the torchvision implemetation of box encoding
        Mush be a static method as it is used in Os2dHead.forward, when there is no access to the boxcoder object

        Ref: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/_utils.py
        """
        # make sure that no spatial size of any box is too small
        # too small boxes cause NaN in the gradient of torch.log inside encode_boxes
        class_boxes.clip_to_min_size(min_size=1)
        default_boxes.clip_to_min_size(min_size=1)
        class_loc_targets = encode_boxes(class_boxes.bbox_xyxy, default_boxes.bbox_xyxy, BOX_ENCODING_WEIGHTS)
        return class_loc_targets

    def build_boxes_from_loc_scores(self, loc_scores, default_boxes):
        """build_boxes_from_loc_scores is a wrapper for the torchvision implemetation of box decoding
        Cannot be static because the torchvision method for decode is not static (this can be easily fixed id needed).

        build_boxes_from_loc_scores and build_loc_targets implement inverse functionality:
        self.build_loc_targets(self.build_boxes_from_loc_scores(loc_scores, default_boxes), default_boxes)
        should be very close to loc_scores

        Ref: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/_utils.py
        """
        box_preds = self.box_coder.decode_single(loc_scores, default_boxes.bbox_xyxy)
        return BoxList(box_preds, image_size=default_boxes.image_size, mode="xyxy")

    def encode(self, boxes, img_size, num_labels, default_box_transform=None):
        """Encode target bounding boxes and class labels.
        Classification target assignment is done with self.assign_anchors_to_boxes_threshold.
        Localization target assignment is doen with self.build_loc_targets.
        The set of anchor boxes is defined by self._get_default_boxes(img_size).

        Args:
          boxes (BoxList) - bounding boxes that need to be encoded
          img_size (FeatureMapSize) - size of the image at which to do decoding
            Note that img_size can be not equal to boxes.image_size
          num_labels (int) - labels from 0,...,num_labels-1 can appear in boxes.get_field("labels")
          default_box_transform (TransformList) - transformation to convert the boxes to the img_size scale

        Outputs:
          loc_targets (tensor) - encoded bounding boxes, sized num_labels x 4 x num_anchors
          cls_targets (tensor) - encoded class labels, sized num_labels x num_anchors
        """
        difficult_flags = boxes.get_field("difficult")
        labels = boxes.get_field("labels")

        default_boxes = self._get_default_boxes(img_size)
        if default_box_transform is not None:
            default_boxes = default_box_transform(default_boxes)

        loc_targets = []
        cls_targets = []
        for i_label in range(num_labels):

            # select boxes of this class
            label_mask = labels == i_label
            ids = torch.nonzero(label_mask).view(-1)

            if ids.numel() > 0:
                # there are boxes of this class on this image
                class_boxes = boxes[ids]

                index, ious = self.assign_anchors_to_boxes_threshold(default_boxes, class_boxes, self.matcher)
                ious_max_gt = ious.max(0)[0] # IoU with the best fitting GT box

                # copy the GT boxes to match anchors
                # anchors that are not assigned to anything get a dummy box (with index 0)
                # it's done this way to avoid complicated tensor operations
                class_boxes = class_boxes[index.clamp(min=0)]  # negative index not supported

                class_loc_targets = self.build_loc_targets(class_boxes, default_boxes)

                # assign labels
                #   for index == -2 assign -1 (ignore at training) 
                #   for index == -1 assign 0 (negatives)
                #   for others assign 1 (positives)
                class_cls_targets = 1 + index.clamp(min=-2, max=0)
            else:
                class_loc_targets = torch.zeros(len(default_boxes), 4, dtype=torch.float)
                class_cls_targets = torch.zeros(len(default_boxes), dtype=torch.long)

            loc_targets.append(class_loc_targets.transpose(0, 1).contiguous()) 
            # for the network implementation, we want this order of dimensions
            cls_targets.append(class_cls_targets)

        loc_targets = torch.stack(loc_targets, 0)
        cls_targets = torch.stack(cls_targets, 0)

        return loc_targets, cls_targets

    def encode_pyramid(self, boxes, img_size_pyramid, num_labels,
                       default_box_transform_pyramid=None):
        """encode_pyramid is a wrapper that applies encode to each pyramid level.
        See encode for mode details.

        Args:
          boxes (BoxList) - bounding boxes that need to be encoded
          img_size_pyramid (list of FeatureMapSize) - list of sizes for all pyramid levels
          num_labels (int) - labels from 0,...,num_labels-1 can appear in boxes.get_field("labels")
          default_box_transform_pyramid (list TransformList) - for each pyramid level, a transformation to convert the boxes to the img_size scale of that level

        Outputs:
          loc_targets_pyramid (tensor) - encoded bounding boxes for each pyramid level
          cls_targets_pyramid (tensor) - encoded class labels for each pyramid level
        """
        num_pyramid_levels = len(img_size_pyramid)
        
        loc_targets_pyramid = []
        cls_targets_pyramid = []
        for i_p in range(num_pyramid_levels):
            loc_targets_this_level, cls_targets_this_level = \
                    self.encode(boxes, img_size_pyramid[i_p], num_labels,
                                default_box_transform=default_box_transform_pyramid[i_p])
            loc_targets_pyramid.append(loc_targets_this_level)
            cls_targets_pyramid.append(cls_targets_this_level)

        return loc_targets_pyramid, cls_targets_pyramid

    @staticmethod
    def _nms_box_lists(boxlists, nms_iou_threshold):
        boxes = cat_boxlist(boxlists)
        scores = boxes.get_field("scores")
    
        ids_boxes_keep = nms(boxes, nms_iou_threshold)
        
       # sort by the scores in the decreasing order: some NMS codes do not do this
        scores = scores[ids_boxes_keep]
        _, score_sorting_index = torch.sort(scores, dim=0, descending=True)
        # sort indices and boxes in the same order
        ids_boxes_keep = ids_boxes_keep[score_sorting_index]

        return boxes[ids_boxes_keep]

    @staticmethod
    def apply_transform_to_corners(masked_transform_corners, transform, img_size):
        # need to have 8 numbers of corners in the format x,y,x,y,x,y,x,y for the following:
        masked_transform_corners = masked_transform_corners.contiguous().view(-1, 4)
        corners_as_boxes = BoxList(masked_transform_corners, img_size, mode="xyxy")
        corners_as_boxes = transform(corners_as_boxes)
        masked_transform_corners = corners_as_boxes.bbox_xyxy.contiguous().view(-1, 8)
        return masked_transform_corners

    def decode_pyramid(self, loc_scores_pyramid, cls_scores_pyramid, img_size_pyramid, class_ids,
               nms_score_threshold=0.0, nms_iou_threshold=0.3,
               inverse_box_transforms=None, transform_corners_pyramid=None):
        """Decode pyramids of classification and localization scores to actual detections.

        Args:
            loc_scores_pyramid (list of tensors) - localization scores for all pyramid levels,
                each level is of size num_labels x 4 x num_anchors
            cls_scores_pyramid (list of tensors) - classification scores for all pyramid levels,
                each level is of size num_labels x num_anchors
            img_size_pyramid (list of FeatureMapSize) - sizes of images for all the pyramid levels
            class_ids (list of int) - global ids of classes, loc_scores_pyramid/cls_scores_pyramid correspond to local class
                need to output the global ones
            nms_score_threshold (float) - remove detection with too small scores
            nms_iou_threshold (float) - IoU threshold for NMS
            inverse_box_transforms (list of TransformList) - for each level, the transformation to convert boxes to the original image size
            transform_corners_pyramid (list of tensors)- for each level, give the end points of the parallelogram defining the transformation,
                each level is of size num_labels x 8 x num_anchors

        Returns:
          boxes_stacked (BoxList) - the detections
        """
        num_classes = len(class_ids)
        num_pyramid_levels = len(img_size_pyramid)
        default_boxes_per_level = [self._get_default_boxes(img_size) for img_size in img_size_pyramid]

        device = cls_scores_pyramid[0].device
        for cl, loc in zip (cls_scores_pyramid, loc_scores_pyramid):
            assert cl.device == device, "scores and boxes should be on the same device"
            assert loc.device == device, "scores and boxes should be on the same device"

        boxes_per_label = []
        transform_corners_per_label = []

        # can have identical entries in class ids - need to merge those together
        for real_label in set(class_ids):
            masked_boxes_pyramid, masked_score_pyramid, masked_default_boxes_pyramid, masked_labels_pyramid = [], [], [], []
            masked_transform_corners_pyramid = []
            for i_label in range(num_classes):
                if class_ids[i_label] != real_label:
                    continue
                # decode boxes at each pyramid level and joint NMS afterwards
                for i_p, (loc_scores, cls_scores) in enumerate(zip(loc_scores_pyramid, cls_scores_pyramid)):
                    default_boxes = default_boxes_per_level[i_p]
                    default_boxes = default_boxes.to(device=device)

                    box_preds = self.build_boxes_from_loc_scores(loc_scores[i_label].transpose(0,1), default_boxes)
                    box_preds.add_field("scores", cls_scores[i_label, :].float())
                    box_preds.add_field("default_boxes", default_boxes)
                    box_preds.add_field("labels", torch.zeros(len(box_preds), dtype=torch.long, device=device).fill_(int(real_label)))

                    if transform_corners_pyramid is not None:
                        box_preds.add_field("transform_corners", transform_corners_pyramid[i_p][i_label].transpose(0,1))

                    # clamp box to the image
                    assert img_size_pyramid[i_p] == box_preds.image_size
                    box_preds.clip_to_image(remove_empty=False)
                    bad_boxes = box_preds.get_mask_empty_boxes()
                    # threshold boxes by score
                    mask = (box_preds.get_field("scores").float() > nms_score_threshold) & ~bad_boxes
                    if mask.any():
                        masked_boxes = box_preds[mask]

                        # convert boxes to global coordinates
                        if inverse_box_transforms is not None:
                            img_size = masked_boxes.image_size
                            masked_boxes = inverse_box_transforms[i_p](masked_boxes)
                            masked_boxes.add_field("default_boxes",
                                                   inverse_box_transforms[i_p](masked_boxes.get_field("default_boxes")))
                            if masked_boxes.has_field("transform_corners"):
                                masked_transform_corners = masked_boxes.get_field("transform_corners")
                                masked_transform_corners = self.apply_transform_to_corners(masked_transform_corners, inverse_box_transforms[i_p], img_size)
                                masked_boxes.add_field("transform_corners", masked_transform_corners)

                        # merge boxes from pyramid levels
                        masked_boxes_pyramid.append(masked_boxes)

            # NMS has to be done across pyramid levels
            if len(masked_boxes_pyramid) > 0:
                boxes_after_nms = self._nms_box_lists(masked_boxes_pyramid, nms_iou_threshold)
                boxes_per_label.append(boxes_after_nms)
            
        if self.do_nms_across_classes:
            boxes_stacked = \
                self._nms_box_lists(boxes_per_label, nms_iou_threshold)
        else:
            boxes_stacked = cat_boxlist(boxes_per_label)

        return boxes_stacked
