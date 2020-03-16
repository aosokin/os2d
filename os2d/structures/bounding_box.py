import math
import torch

from torchvision.ops.boxes import nms as _box_nms
from torchvision.ops.boxes import box_iou, box_area, clip_boxes_to_image

from .feature_map import FeatureMapSize


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.

    This code is based on the corresponding class of the maskrcnn-benchmark:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/structures/bounding_box.py
    """
    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        
        BoxList.assert_bbox_tensor_dims(bbox)
        BoxList.assert_bbox_mode(mode)

        self.bbox_xyxy = self.convert_bbox_format(bbox, mode, "xyxy")
        self.image_size = image_size  # FeatureMapSize
        self.extra_fields = {}

    @staticmethod
    def create_empty(image_size):
        bbox = torch.zeros(0, 4)
        return BoxList(bbox, image_size)

    def __len__(self):
        return self.bbox_xyxy.shape[0]

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def remove_field(self, field):
        if field in self.extra_fields:
            del self.extra_fields[field]
        else:
            raise ValueError("bbox has not field {}".format(field))

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    @staticmethod
    def assert_bbox_tensor_dims(bbox):
        if not isinstance(bbox, torch.Tensor):
            raise ValueError(
                "bbox should be of type torch.Tensor"
            ) 
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )

    @staticmethod
    def assert_bbox_mode(mode):
        allowed_modes = ("xyxy", "xywh", "cx_cy_w_h")
        if mode not in allowed_modes:
            raise ValueError("mode should be in {0}".format(allowed_modes))

    @staticmethod
    def convert_bbox_format(bbox, source_format, target_format):
        BoxList.assert_bbox_mode(source_format)
        BoxList.assert_bbox_mode(target_format)
        BoxList.assert_bbox_tensor_dims(bbox)
        if source_format == target_format:
            return bbox
        # convert to "xyxy" first
        bbox_xyxy = BoxList._convert_to_xyxy(bbox, source_format)
        # convert to the target format
        bbox = BoxList._convert_from_xyxy(bbox_xyxy, target_format)
        return bbox

    @staticmethod
    def _convert_to_xyxy(bbox, source_format):
        if source_format == "xyxy":
            return bbox
        if source_format == "xywh":
            xmin, ymin, w, h = bbox.split(1, dim=-1)
            xmax, ymax = xmin + w, ymin + h
        elif source_format == "cx_cy_w_h":
            cx, cy, w, h = bbox.split(1, dim=-1)
            xmin, ymin = cx - w/2, cy - h/2
            xmax, ymax = cx + w/2, cy + h/2
        else:
            raise RuntimeError("Should not be here")
        return torch.cat([xmin, ymin, xmax, ymax], dim=-1)

    @staticmethod
    def _convert_from_xyxy(bbox, target_format):
        if target_format == "xyxy":
            return bbox
        xmin, ymin, xmax, ymax = bbox.split(1, dim=-1)
        if target_format == "xywh":
            w, h = xmax - xmin, ymax - ymin
            return torch.cat([xmin, ymin, w, h], dim=-1)
        elif target_format == "cx_cy_w_h":
            cx, cy = (xmin + xmax)/2, (ymin + ymax)/2
            w, h = xmax - xmin, ymax - ymin
            return torch.cat([cx, cy, w, h], dim=-1)
        else:
            raise RuntimeError("Should not be here")

    @staticmethod
    def _split_bbox_into_params(bbox):
        BoxList.assert_bbox_tensor_dims(bbox)
        return bbox.split(1, dim=-1)

    def resize(self, target_size):
        """
        Returns a resized copy of this bounding box list

        Args:
            size: The requested size in pixels, as FeatureMapSize
        """

        ratio_w = float(target_size.w) / self.image_size.w
        ratio_h = float(target_size.h) / self.image_size.h

        if ratio_w == ratio_h:
            ratio = ratio_w
            scaled_box = self.bbox_xyxy * ratio
            bbox = BoxList(scaled_box, target_size, mode="xyxy")
        else:
            xmin, ymin, xmax, ymax = self._split_bbox_into_params(self.bbox_xyxy)
            scaled_xmin = xmin * ratio_w
            scaled_xmax = xmax * ratio_w
            scaled_ymin = ymin * ratio_h
            scaled_ymax = ymax * ratio_h
            scaled_box = torch.cat((scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1)
            bbox = BoxList(scaled_box, target_size, mode="xyxy")

        bbox._copy_extra_fields(self)
        return bbox

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.image_size.w, self.image_size.h
        xmin, ymin, xmax, ymax = self._split_bbox_into_params(self.bbox_xyxy)
        if method == FLIP_LEFT_RIGHT:
            transposed_xmin = image_width - xmax
            transposed_xmax = image_width - xmin
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.image_size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox

    def crop(self, box):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.

        Note: this function does not clip boxes to image size, use clip_to_image for this
        """
        xmin, ymin, xmax, ymax = self._split_bbox_into_params(self.bbox_xyxy)
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = xmin - box[0]
        cropped_ymin = ymin - box[1]
        cropped_xmax = xmax - box[0]
        cropped_ymax = ymax - box[1]

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, FeatureMapSize(w=w, h=h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox

    def to(self, device):
        bbox = BoxList(self.bbox_xyxy.to(device), self.image_size, mode="xyxy")
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def cuda(self):
        bbox = BoxList(self.bbox_xyxy.cuda(), self.image_size, mode="xyxy")
        for k, v in self.extra_fields.items():
            if hasattr(v, "cuda"):
                v = v.cuda()
            bbox.add_field(k, v)
        return bbox

    def cpu(self):
        bbox = BoxList(self.bbox_xyxy.cpu(), self.image_size, mode="xyxy")
        for k, v in self.extra_fields.items():
            if hasattr(v, "cpu"):
                v = v.cpu()
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        selected_boxes = self.bbox_xyxy[item]
        if selected_boxes.ndimension() == 1 and selected_boxes.numel() == 4:
            selected_boxes = selected_boxes.view(1, 4)
        bbox = BoxList(selected_boxes, self.image_size)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def clip_to_image(self, remove_empty=True):
        self.bbox_xyxy = clip_boxes_to_image(self.bbox_xyxy, (self.image_size.h, self.image_size.w))
        if remove_empty:
            return self[~self.get_mask_empty_boxes()]
        return self

    def clip_to_min_size(self, min_size=1):
        box = self.bbox_xyxy
        # clip width
        mask = (box[:, 0] + min_size) > box[:, 2]
        box[mask, 0] = box[mask, 0].detach()
        box[mask, 2] = box[mask, 0] + min_size
        # clip height
        mask = (box[:, 1] + min_size) > box[:, 3]
        box[mask, 1] = box[mask, 1].detach()
        box[mask, 3] = box[mask, 1] + min_size
        return self

    def get_mask_empty_boxes(self):
        box = self.bbox_xyxy
        return (box[:, 3] <= box[:, 1]) | (box[:, 2] <= box[:, 0])

    def area(self):
        return box_area(self.bbox_xyxy)

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.image_size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox


    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.image_size.w)
        s += "image_height={}, ".format(self.image_size.h)
        s += ")"
        return s


def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    Reference:
    """
    if boxlist1.image_size != boxlist2.image_size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1.image_size, boxlist2.image_size))
    return box_iou(boxlist1.bbox_xyxy, boxlist2.bbox_xyxy)

# implementation from https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
# with slight modifications
def box_intersection_over_reference(boxes_reference, boxes):
    if boxes_reference.image_size != boxes.image_size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxes_reference.image_size, boxes.image_size))
    # N = len(boxes_reference)
    # M = len(boxes)

    area_ref = boxes_reference.area()

    box1, box2 = boxes_reference.bbox_xyxy, boxes.bbox_xyxy

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    ratio = inter / area_ref[:, None]
    return ratio

def nms(boxes, nms_iou_threshold, nms_max_batch=10000, nms_score_threshold=float("-inf"), do_separate_per_label=False):
    """Need this function when there are too many boxes - the standard approach blows up GPU memory
    """
    if not do_separate_per_label:
        finished = False
        scores = boxes.get_field("scores")
        surviving_box_ids = torch.nonzero(scores > nms_score_threshold).squeeze(1)
        # surviving_box_ids = torch.arange(boxes.size(0))
        is_cuda = boxes.bbox_xyxy.is_cuda
        device = boxes.bbox_xyxy.device
        if is_cuda:
            surviving_box_ids = surviving_box_ids.cuda()
        while not finished:
            num_boxes = surviving_box_ids.size(0)
            num_batches = int(math.ceil(float(num_boxes) / nms_max_batch))
            # print('Processing {0} boxes in {1} batches'.format(num_boxes, num_batches))
            survived = []
            if len(surviving_box_ids) > 0:
                for i_start in range(0, len(surviving_box_ids), nms_max_batch):
                    batch_ids = torch.arange(i_start, min(len(surviving_box_ids), i_start + nms_max_batch))
                    if is_cuda:
                        batch_ids = batch_ids.cuda()
                    batch_ids_global = surviving_box_ids[batch_ids]
                    batch_ids_boxes_keep = _box_nms(boxes[batch_ids_global].bbox_xyxy, scores[batch_ids_global], nms_iou_threshold)
                    survived.append(batch_ids_global[batch_ids_boxes_keep])
            if len(survived) > 0:
                surviving_box_ids = torch.cat(survived, 0)
            else:
                surviving_box_ids = torch.tensor([], device=device, dtype=torch.long)
            if num_batches <= 1 or surviving_box_ids.size(0) == num_boxes:
                finished = True
    else:
        labels = boxes.get_field("labels")
        unique_labels = labels.unique()
        surviving_box_ids = []
        for l in unique_labels:
            cur_ids = torch.nonzero(labels == l).view(-1)
            cur_boxes = boxes[cur_ids]
            cur_good_indices = nms(cur_boxes, nms_iou_threshold, nms_max_batch, nms_score_threshold, do_separate_per_label=False)
            cur_good_indices = cur_ids[cur_good_indices]
            surviving_box_ids.append(cur_good_indices)
        surviving_box_ids = torch.cat(surviving_box_ids, dim=0)
    
    return surviving_box_ids


def cat_boxlist(bboxes):
    """    Concatenates a list of BoxList (having the same image size) into a
    single BoxList
    Arguments:
        bboxes (list[BoxList])

    Inspired by cat_boxlist from maskrnn-benchmark:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/structures/boxlist_ops.py
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    image_size = bboxes[0].image_size
    assert all(bbox.image_size == image_size for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(torch.cat([bbox.bbox_xyxy for bbox in bboxes], dim=0), image_size)

    for field in fields:
        list_fields = [bbox.get_field(field) for bbox in bboxes]
        if list_fields and type(list_fields[0]) == BoxList:
            data = cat_boxlist(list_fields)
        else:
            data = torch.cat(list_fields, dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
