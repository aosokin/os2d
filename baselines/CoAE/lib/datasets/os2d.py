import os
import pickle
import uuid
import scipy.sparse
import numpy as np

import torch

from datasets.imdb import imdb
from model.utils.config import cfg

from os2d.utils import read_image
from os2d.structures.feature_map import FeatureMapSize
from os2d.structures.bounding_box import BoxList, cat_boxlist
from os2d.data.voc_eval import do_voc_evaluation
from os2d.utils.visualization import vis_image
from os2d.data.dataset import build_dataset_by_name


class Os2dDataset(imdb):
    def __init__(self, dataset_src):
        imdb.__init__(self, dataset_src.name)

        self._gtboxframe = dataset_src.gtboxframe
        self._image_size = dataset_src.image_size
        self._gt_path = dataset_src.gt_path
        self._image_path = dataset_src.image_path
        self._image_ids = dataset_src.image_ids
        self._image_file_names = dataset_src.image_file_names

        self._num_images = len(self._image_ids)
        self._num_boxes = len(self._gtboxframe)
        self._image_index = list(range(self._num_images))

        # add background class
        # '__background__' - always index 0
        bkg_tag = '__background__'
        self._classes = (bkg_tag,) + tuple(self._gtboxframe["classid"].unique())
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._gt_path), 'GT path does not exist: {}'.format(self._gt_path)
        assert os.path.exists(self._image_path), 'Image path does not exist: {}'.format(self._image_path)

        self.cat_data = {}
        for i in self._class_to_ind.values():
            class_id = self._classes[i]
            if class_id != bkg_tag:
                class_entries = self._gtboxframe[self._gtboxframe["classid"] == class_id]
                gt_file = class_entries['classfilename'].unique()
                assert len(gt_file) == 1
                gt_file = gt_file[0]
                gt_file = os.path.join(self._gt_path, gt_file)

                curimage = read_image(gt_file)
                height, width = curimage.height, curimage.width

                self.cat_data[i] = [{
                                        'image_path': gt_file,
                                        'boxes': [0, 0, width, height]
                                   }]

    def gt_roidb(self):
        """Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                [roidb, self.cat_data] = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump([gt_roidb, self.cat_data], fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def image_path_at(self, i):
        """Return the absolute path to image i in the image sequence.
        """
        return os.path.join(self._image_path, self._image_file_names[i])


    def image_id_at(self, i):
        return self._image_ids[i]

    def filter(self):
        self.inverse_list = list(range(1,len(self._classes)))

    def _load_annotation(self, index):
        imageframe = self._gtboxframe[self._gtboxframe['imageid'] == self._image_ids[index]]

        # get the image
        curimagefilename = self._image_file_names[index]
        curimagepath = os.path.join(self._image_path, curimagefilename)
        curimage = read_image(curimagepath)
        height, width = curimage.height, curimage.width

        # create roidb entry
        roi_rec = {}
        roi_rec['gt_classes'] = []
        boxes = []
        difficult_flag = []
        seg_areas = []
        overlaps = np.zeros((len(imageframe), self.num_classes), dtype=np.float32)
        for ix, gt_index in enumerate(imageframe.index):
            lx = np.int32(imageframe.loc[gt_index, 'lx'] * width)
            rx = np.int32(imageframe.loc[gt_index, 'rx'] * width)
            ty = np.int32(imageframe.loc[gt_index, 'ty'] * height)
            by = np.int32(imageframe.loc[gt_index, 'by'] * height)
            gt_class = self._class_to_ind[imageframe.loc[gt_index, 'classid']]
            seg_areas.append( (rx - lx) * (by - ty) )

            boxes.append([lx, ty, rx, by])
            roi_rec['gt_classes'].append(np.int32(gt_class))
            overlaps[ix, gt_class] = 1.0
            difficult_flag.append(imageframe.loc[gt_index, 'difficult'])

        roi_rec['boxes'] = np.array(boxes, dtype=np.int32)
        roi_rec['height'] = height
        roi_rec['width'] = width
        roi_rec['flipped'] = False

        roi_rec['gt_classes'] = np.asarray(roi_rec['gt_classes'], dtype=np.int32)
        roi_rec['gt_ishard'] = np.asarray(difficult_flag, dtype=np.int32)
        roi_rec['seg_areas'] = np.asarray(seg_areas, dtype=np.float32)
        roi_rec['gt_overlaps'] = scipy.sparse.csr_matrix(overlaps)

        return roi_rec

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def evaluate_detections(self, all_boxes, output_dir, mAP_iou_threshold=0.5):
        predictions = []
        gt_boxes = []
        roidb = self.roidb
        for i_image, roi in enumerate(roidb):
            image_size = FeatureMapSize(w=roi["width"], h=roi["height"])
            if roi["boxes"].size > 0:
                roi_gt_boxes = BoxList(roi["boxes"], image_size, mode="xyxy")
            else:
                roi_gt_boxes = BoxList.create_empty(image_size)
            roi_gt_boxes.add_field("labels", torch.as_tensor(roi["gt_classes"], dtype=torch.int32))
            roi_gt_boxes.add_field("difficult", torch.as_tensor(roi["gt_ishard"], dtype=torch.int32))

            gt_boxes.append(roi_gt_boxes)

            roi_detections = []
            for i_class, class_boxes in enumerate(all_boxes):
                assert len(class_boxes) == len(roidb), \
                    "Number of detection for class {0} image{1} ({2}) inconsistent with the length of roidb ({3})".format(i_class, i_image, len(class_boxes), len(roidb))
                boxes = class_boxes[i_image]
                if len(boxes) > 0:
                    assert boxes.shape[1] == 5, "Detections should be of shape (:,5), but are {0} for class {1}, image {2}".format(boxes.shape, i_class, i_image)
                    bbox = BoxList(boxes[:,:4], image_size, mode="xyxy")
                    scores = boxes[:,-1]
                    bbox.add_field("scores", torch.as_tensor(scores, dtype=torch.float32))
                    bbox.add_field("labels", torch.full(scores.shape, i_class, dtype=torch.int32))
                    roi_detections.append(bbox)

            if roi_detections:
                roi_detections = cat_boxlist(roi_detections)
            else:
                roi_detections = BoxList.create_empty(image_size)
                roi_detections.add_field("scores", torch.zeros((0,), dtype=torch.float32))
                roi_detections.add_field("labels", torch.zeros((0,), dtype=torch.int32))
            predictions.append(roi_detections)

            if False:
                self.visualize_detections(i_image, gt=roi_gt_boxes, dets=roi_detections)

        ap_data = do_voc_evaluation(predictions, gt_boxes, iou_thresh=mAP_iou_threshold, use_07_metric=False)
        print("mAP@{:0.2f}: {:0.4f}".format(mAP_iou_threshold, ap_data["map"]))
        print("mAPw@{:0.2f}: {:0.4f}".format(mAP_iou_threshold, ap_data["map_weighted"]))
        print("recall@{:0.2f}: {:0.4f}".format(mAP_iou_threshold, ap_data["recall"]))

        return ap_data['map']

    def visualize_detections(self, i_image, gt=None, dets=None, num_dets_to_show=30):
        # get the image
        imageframe = self._gtboxframe[self._gtboxframe['imageid'] == self._image_ids[i_image]]
        curimagefilename = imageframe['imagefilename'].unique()
        assert len(curimagefilename) == 1
        curimagefilename = curimagefilename[0]
        curimagepath = os.path.join(self._image_path, curimagefilename)
        curimage = read_image(curimagepath)

        # add GT boxes
        if gt is not None:
            boxes_gt = gt.bbox_xyxy
            colors_gt = ['y'] * boxes_gt.shape[0]
            labels_gt = [str(self._classes[lb]) for lb in gt.get_field("labels")]
            scores_gt = [np.nan] * boxes_gt.shape[0]

        # add detections
        if dets is not None:
            box_ids = dets.get_field("scores").topk(num_dets_to_show)[1]
            dets = dets[box_ids]
            boxes_dets = dets.bbox_xyxy
            colors_dets = ['m'] * boxes_dets.shape[0]
            labels_dets = [str(self._classes[lb]) for lb in dets.get_field("labels")]
            scores_dets = [float(s) for s in dets.get_field("scores")]

        # merge data
        if gt is not None and dets is not None:
            vis_boxes = torch.cat([boxes_gt, boxes_dets], 0)
            vis_labels = labels_gt + labels_dets
            vis_scores = scores_gt + scores_dets
            vis_colors = colors_gt + colors_dets
        elif gt is not None:
            vis_boxes = boxes_gt
            vis_labels = labels_gt
            vis_scores = scores_gt
            vis_colors = colors_gt
        elif dets is not None:
            vis_boxes = boxes_dets
            vis_labels = labels_dets
            vis_scores = scores_dets
            vis_colors = colors_dets
        else:
            vis_boxes = None
            vis_labels = None
            vis_scores = None
            vis_colors = None

        # show image
        vis_image(curimage, boxes=vis_boxes, label_names=vis_labels, scores=vis_scores, colors=vis_colors, showfig=True)


def build_os2d_dataset_by_name(name, data_path=None):
    if data_path is None:
        data_path = cfg.DATA_DIR

    dataset = build_dataset_by_name(data_path, name, eval_scale=None, cache_images=False)
    return Os2dDataset(dataset)