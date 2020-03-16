import os
import math
import numbers
import time
import logging
from collections import OrderedDict
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import build_feature_extractor

from .box_coder import Os2dBoxCoder, BoxGridGenerator
from os2d.engine.objective import Os2dObjective
from os2d.utils import count_model_parameters, masked_select_or_fill_constant
from os2d.structures.feature_map import FeatureMapSize
from .head import build_os2d_head_creator


def build_os2d_from_config(cfg):
    logger = logging.getLogger("OS2D")

    logger.info("Building the OS2D model")
    img_normalization = {"mean":cfg.model.normalization_mean, "std": cfg.model.normalization_std}
    net = Os2dModel(logger=logger,
                    is_cuda=cfg.is_cuda,
                    backbone_arch=cfg.model.backbone_arch,
                    merge_branch_parameters=cfg.model.merge_branch_parameters,
                    use_group_norm=cfg.model.use_group_norm,
                    use_inverse_geom_model=cfg.model.use_inverse_geom_model,
                    simplify_affine=cfg.model.use_simplified_affine_model,
                    img_normalization=img_normalization)
    box_coder = Os2dBoxCoder(positive_iou_threshold=cfg.train.objective.positive_iou_threshold,
                             negative_iou_threshold=cfg.train.objective.negative_iou_threshold,
                             remap_classification_targets_iou_pos=cfg.train.objective.remap_classification_targets_iou_pos,
                             remap_classification_targets_iou_neg=cfg.train.objective.remap_classification_targets_iou_neg,
                             output_box_grid_generator=net.os2d_head_creator.box_grid_generator_image_level,
                             function_get_feature_map_size=net.get_feature_map_size,
                             do_nms_across_classes=cfg.eval.nms_across_classes)
    criterion = Os2dObjective(class_loss=cfg.train.objective.class_objective,
                              margin=cfg.train.objective.neg_margin,
                              margin_pos=cfg.train.objective.pos_margin,
                              class_loss_neg_weight=cfg.train.objective.class_neg_weight,
                              remap_classification_targets=cfg.train.objective.remap_classification_targets,
                              localization_weight=cfg.train.objective.loc_weight,
                              neg_to_pos_ratio=cfg.train.objective.neg_to_pos_ratio,
                              rll_neg_weight_ratio=cfg.train.objective.rll_neg_weight_ratio)
    optimizer_state = net.init_model_from_file(cfg.init.model, init_affine_transform_path=cfg.init.transform)

    num_params, num_param_groups = count_model_parameters(net)
    logger.info("OS2D has {0} blocks of {1} parameters (before freezing)".format(num_param_groups, num_params))

    # freeze transform parameters if requested
    if cfg.train.model.freeze_transform:
        logger.info("Freezing the transform parameters")
        net.freeze_transform_params()

    num_frozen_extractor_blocks = cfg.train.model.num_frozen_extractor_blocks
    if num_frozen_extractor_blocks > 0:
        logger.info("Freezing {0} of {1} blocks of the feature extractor network".format(num_frozen_extractor_blocks, net.get_num_blocks_in_feature_extractor()))
        net.freeze_extractor_blocks(num_blocks=num_frozen_extractor_blocks)

    num_params, num_param_groups = count_model_parameters(net)
    logger.info("OS2D has {0} blocks of {1} trainable parameters".format(num_param_groups, num_params))

    return net, box_coder, criterion, img_normalization, optimizer_state


class LabelFeatureExtractor(nn.Module):
    """LabelFeatureExtractor implements the feature extractor from query images.
    The main purpose pf this class is to run on a list of images of different sizes.
    """
    def __init__(self, feature_extractor):
        super(LabelFeatureExtractor, self).__init__()
        # backbone extractor
        self.net_class_features = feature_extractor
        
    def forward(self, class_image_list):
        list_of_feature_maps = []

        for class_image in class_image_list:
            # extract features from images
            class_feature_maps = self.net_class_features(class_image.unsqueeze(0))
            list_of_feature_maps.append(class_feature_maps)
        
        return list_of_feature_maps

    def freeze_bn(self):
        # Freeze BatchNorm layers
        self.net_class_features.freeze_bn()

    def freeze_blocks(self, num_blocks=0):
        self.net_class_features.freeze_blocks(num_blocks)


def get_feature_map_size_for_network(img_size, net, is_cuda=False):
    """get_feature_map_size_for_network computes the size of the feature map when the network is applied to an image of specific size.
    The function creates a dummy image of required size, and just runs a network on it.
    This approach is very robust, but can be quite slow, so these calls shoulb be cached.
    Args:
        img_size (FeatureMapSize) - size of the input image
        net - the net to run
        is_cuda (bool) -flag showing where to put the dummy image on a GPU.
    Output:
        feature_map_size (FeatureMapSize) - the size of the feature map
    """
    dummy_image = torch.zeros(1, 3, img_size.h, img_size.w)  # batch_size, num_channels, height, width
    if is_cuda:
        dummy_image = dummy_image.cuda()

    with torch.no_grad():
        dummy_feature_maps = net(dummy_image)
        feature_map_size = FeatureMapSize(img=dummy_feature_maps)

    if is_cuda:
        torch.cuda.empty_cache()

    return feature_map_size


class Os2dModel(nn.Module):
    """The main class implementing the OS2D model.
    """
    default_normalization = {}
    default_normalization["mean"] = (0.485, 0.456, 0.406)
    default_normalization["std"] = (0.229, 0.224, 0.225)

    def __init__(self, logger,
                       is_cuda=False,
                       merge_branch_parameters=False, use_group_norm=False,
                       backbone_arch="resnet50",
                       use_inverse_geom_model=True,
                       simplify_affine=False,
                       img_normalization=None):
        super(Os2dModel, self).__init__()
        self.logger = logger
        self.use_group_norm = use_group_norm
        if img_normalization:
            self.img_normalization = img_normalization
        else:
            self.img_normalization = self.default_normalization
        self.net_feature_maps = build_feature_extractor(backbone_arch, use_group_norm)
        self.merge_branch_parameters = merge_branch_parameters
        extractor = self.net_feature_maps if self.merge_branch_parameters else build_feature_extractor(backbone_arch, use_group_norm)

        # net to regress parameters of the transform
        self.simplify_affine = simplify_affine
        self.use_inverse_geom_model = use_inverse_geom_model

        # new code fot the network heads
        self.os2d_head_creator = build_os2d_head_creator(self.simplify_affine, is_cuda, self.use_inverse_geom_model,
                                                         self.net_feature_maps.feature_map_stride,
                                                         self.net_feature_maps.feature_map_receptive_field)

        self.net_label_features = LabelFeatureExtractor(feature_extractor=extractor)

        # set the net to the eval mode by default, other wise batchnorm statistics get screwed when using dummy images to find feature map sizes
        self.eval()

        # decide GPU usage
        self.is_cuda = is_cuda

        if self.is_cuda:
            self.logger.info("Creating model on one GPU")
            self.cuda()
        else:            
            self.logger.info("Creating model on CPU")

    def train(self, mode=True, freeze_bn_in_extractor=False, freeze_transform_params=False, freeze_bn_transform=False):
        super(Os2dModel, self).train(mode)
        if freeze_bn_in_extractor:
            self.freeze_bn()
        if freeze_transform_params:
            self.freeze_transform_params()
        if freeze_bn_transform:
            self.os2d_head_creator.aligner.parameter_regressor.freeze_bn()

    def freeze_bn(self):
        # Freeze BatchNorm layers
        self.net_feature_maps.freeze_bn()
        self.net_label_features.freeze_bn()

    def freeze_transform_params(self):
        self.os2d_head_creator.aligner.parameter_regressor.eval()
        for param in self.os2d_head_creator.aligner.parameter_regressor.parameters():
            param.requires_grad = False
               
    def freeze_extractor_blocks(self, num_blocks=0):
        self.net_feature_maps.freeze_blocks(num_blocks)
        self.net_label_features.freeze_blocks(num_blocks)

    def get_num_blocks_in_feature_extractor(self):
        return self.net_feature_maps.get_num_blocks_in_feature_extractor()

    def apply_class_heads_to_feature_maps(self, feature_maps, class_head):
        """Applies class heads to the feature maps

        Args:
            feature_maps (Tensor) - feature maps of size batch_size x num_labels x height x width
            class_head (Os2dHead) - heads detecting some classes, created by an instance of Os2dHeadCreator

        Outputs:
            loc_scores (Tensor) - localization scores, size batch_size x num_labels x 4 x num_anchors
            class_scores (Tensor) - classification scores, size batch_size x num_labels x num_anchors
            class_scores_transform_detached (Tensor) - same as class_scores, but with transformations detached from the computational graph
            transform_corners (Tensor) - points representings transformations, size batch_size x num_labels x 8 x num_anchors
        """
        num_images = feature_maps.size(0)

        outputs = class_head(feature_maps)
        loc_scores = outputs[0]
        class_scores = outputs[1]
        num_labels = class_scores.size(1)
        class_scores_transform_detached = outputs[2]
        transform_corners = outputs[3]

        assert loc_scores.size(-2) == class_scores.size(-2), "Class and loc score should have same spatial sizes, but have {0} and {1}".format(class_scores.size(), loc_scores.size())
        assert loc_scores.size(-1) == class_scores.size(-1), "Class and loc score should have same spatial sizes, but have {0} and {1}".format(class_scores.size(), loc_scores.size())

        fmH = class_scores.size(-2)
        fmW = class_scores.size(-1)

        class_scores = class_scores.view(num_images, -1, fmH, fmW)
        class_scores_transform_detached = class_scores_transform_detached.view(num_images, -1, fmH, fmW)

        class_scores = class_scores.contiguous().view(num_images, num_labels, -1)
        class_scores_transform_detached = class_scores_transform_detached.contiguous().view(num_images, num_labels, -1)
        loc_scores = loc_scores.contiguous().view(num_images, num_labels, 4, -1) if loc_scores is not None else None
        transform_corners = transform_corners.contiguous().view(num_images, num_labels, 8, -1) if transform_corners is not None else None

        return loc_scores, class_scores, class_scores_transform_detached, transform_corners

    def forward(self, images=None, class_images=None,
                      feature_maps=None, class_head=None,
                      train_mode=False, fine_tune_features=True):
        """ Forward pass of the OS2D model. Cant function in several different regimes:
            [training mode] Extract features from input and class images, and applies the model to get 
                clasificaton/localization scores of all classes on all images
                Args:
                    images (tensor) - batch of input images
                    class_images (list of tensors) - list of class images (possibly of different sizes)
                    train_mode (bool) - should be True
                    fine_tune_features (bool) - flag showing whether to enable gradients over features
            [evaluation mode]
                    feature_maps (tensor) - pre-extracted feature maps, sized batch_size x feature_dim x height x width
                    class_head (Os2dHead) - head created to detect some classes,
                        inside has class_feature_maps, sized class_batch_size x feature_dim x class_height x class_width
                    train_mode (bool) - should be False
        Outputs:
            loc_scores (tensor) - localization prediction, sized batch_size x num_classes x 4 x num_anchors (bbox parameterization)
            class_scores (tensor) - classification prediction, sized batch_size x num_classes x num_anchors
            class_scores_transform_detached (tensor) - same, but with transofrms detached from the computational graph
                used not to tune transofrmation on the negative examples
            fm_sizes (FeatureMapSize) - size of the output score map, num_anchors == fm_sizes.w * fm_sizes.h
            transform_corners (tensor) - points defining parallelograms showing transformations, sized batch_size x num_classes x 8 x num_anchors
        """
        with torch.set_grad_enabled(train_mode and fine_tune_features):
            # extract features
            if feature_maps is None:
                assert images is not None, "If feature_maps is None than images cannot be None"
                feature_maps = self.net_feature_maps(images)

            # get features for labels
            if class_head is None:
                assert class_images is not None, "If class_conv_layer is None than class_images cannot be None"
                class_feature_maps = self.net_label_features(class_images)
                class_head = self.os2d_head_creator.create_os2d_head(class_feature_maps)

        # process features maps of different pyramid levels
        loc_scores, class_scores, class_scores_transform_detached, transform_corners = \
            self.apply_class_heads_to_feature_maps(feature_maps, class_head)

        fm_size = FeatureMapSize(img=feature_maps)
        return loc_scores, class_scores, class_scores_transform_detached, fm_size, transform_corners

    @lru_cache()
    def get_feature_map_size(self, img_size):
        """Computes the size of the feature map when the feature extractor is applied to the image of specific size.
        The calls are cached with @lru_cache() for speed.
        Args:
            img_size (FeatureMapSize)
        Output: FeatureMapSize
        """
        return get_feature_map_size_for_network(img_size=img_size,
                                                net=self.net_feature_maps,
                                                is_cuda=self.is_cuda)

    def init_model_from_file(self, path, init_affine_transform_path=""):
        """init_model_from_file loads weights from a binary file.
        It will try several ways to load the weights (in the order below) by doing the follwoing steps:
        1) Load full checkpoint (created by os2d.utils.logger.checkpoint_model)
            - reads file with torch.load(path)
            - expects a dict with "net" as a key which is expected to load with self.load_state_dict
            - if finds key "optimizer" as well outputs it to try to init from it later
        2) in (1) is not a success will try to init the backbone separately see _load_network
        3) if init_affine_transform_path is provided will try to additionally load the transformation model
            (CAREFUL! it will override weights from both (1) and (2))
        """
        # read the checkpoint file
        optimizer = None
        try:
            if path:
                self.logger.info("Reading model file {}".format(path))
                checkpoint = torch.load(path)
            else:
                checkpoint = None

            if checkpoint and "net" in checkpoint:
                self.load_state_dict(checkpoint["net"])
                self.logger.info("Loaded complete model from checkpoint")
            else:
                self.logger.info("Cannot find 'net' in the checkpoint file")
                raise RuntimeError()

            if checkpoint and "optimizer" in checkpoint:
                optimizer = checkpoint["optimizer"]
                self.logger.info("Loaded optimizer from checkpoint")
            else:
                self.logger.info("Cannot find 'optimizer' in the checkpoint file. Initializing optimizer from scratch.")

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.logger.info("Failed to load the full model, trying to init feature extractors")
            self._load_network(self.net_label_features.net_class_features, path=path)
            if not self.merge_branch_parameters:
                self._load_network(self.net_feature_maps, model_data=self.net_label_features.net_class_features.state_dict())

        if init_affine_transform_path:
            try:
                self.logger.info("Trying to init affine transform from {}".format(init_affine_transform_path))
                try:
                    model_data = torch.load(init_affine_transform_path)
                except:
                    self.logger.info("Could not read the model file {0}.".format(path))
                assert hasattr(self, "os2d_head_creator") and hasattr(self.os2d_head_creator, "aligner") and hasattr(self.os2d_head_creator.aligner, "parameter_regressor"), "Need to have the affine regressor part to inialize it"
                init_from_weakalign_model(model_data["state_dict"], None,
                                          affine_regressor=self.os2d_head_creator.aligner.parameter_regressor)
                self.logger.info("Successfully initialized the affine transform from the provided weakalign model.")
            except:
                self.logger.info("Could not init affine transform from {0}.".format(init_affine_transform_path))

        return optimizer

    def _load_network(self, net, path=None, model_data=None):
        """_load_network loads weights from the provided path to a network net
        It will try several ways to load the weights (in the order below) by doing the follwoing steps:
        0) in model_data one can provide the already loaded weights (useful not to load many times) otherwise reads from path with torch.load
        1) tries to load complete network as net.load_state_dict(model_data)
        2) if fails, tries to load as follows: net.load_state_dict(model_data["net"], strict=False)
        3) if fails, tries to load from the weak align format with init_from_weakalign_model(model_data["state_dict"], self.net_feature_maps)
        4) if fails tries to partially init backbone with net.load_state_dict(model_data, strict=False) - works for the standard pytorch models
        """
        if model_data is None:
            try:
                self.logger.info("Trying to init from {}".format(path))
                model_data = torch.load(path)
            except:
                self.logger.info("Could not read the model file {0}. Starting from scratch.".format(path))
        else:
            self.logger.info("Initializing from provided weights")
        if model_data is not None:
            try:
                net.load_state_dict(model_data)
            except:
                self.logger.info("FAILED to load as network")
                try:
                    self.logger.info("Trying to init from {} as checkpoint".format(path))
                    net.load_state_dict(model_data["net"], strict=False)
                    self.logger.info("Loaded epoch {0} with loss {1}".format(model_data["epoch"], model_data["loss"]))
                except:
                    self.logger.info("FAILED to load as checkpoint")
                    self.logger.info("Could not init the full feature extractor. Trying to init form a weakalign model")
                    try:
                        init_from_weakalign_model(model_data["state_dict"],
                                                  self.net_feature_maps)
                        self.logger.info("Successfully initialized form the provided weakalign model.")
                    except:
                        self.logger.info("Could not init from the weakalign network. Trying to init backbone from {}.".format(path))
                        try:
                            net.load_state_dict(model_data, strict=False)
                            self.logger.info("Successfully initialized backbone.")
                        except:
                            self.logger.info("Could not init anything. Starting from scratch.")


def init_from_weakalign_model(src_state_dict, feature_extractor=None, affine_regressor=None, tps_regressor=None):
    # init feature extractor - hve three blocks of ResNet101
    layer_prefix_map = {}  # layer_prefix_map[target prefix] = source prefix
    layer_prefix_map["conv1."] = "FeatureExtraction.model.0."
    layer_prefix_map["bn1."] = "FeatureExtraction.model.1."
    for idx in range(3):
        layer_prefix_map["layer1." + str(idx)] = "FeatureExtraction.model.4." + str(idx)
    for idx in range(4):
        layer_prefix_map["layer2." + str(idx)] = "FeatureExtraction.model.5." + str(idx)
    for idx in range(23):
        layer_prefix_map["layer3." + str(idx)] = "FeatureExtraction.model.6." + str(idx)

    if feature_extractor is not None:
        for k, v in feature_extractor.state_dict().items():
            found_init = False
            for k_map in layer_prefix_map:
                if k.startswith(k_map):
                    found_init = True
                    break
            if found_init:
                k_target = k.replace(k_map, layer_prefix_map[k_map])
                if k.endswith("num_batches_tracked"):
                    continue
                # print("Copying from {0} to {1}, size {2}".format(k_target, k, v.size()))
                v.copy_(src_state_dict[k_target])
    
    for regressor, prefix in zip([affine_regressor, tps_regressor], ["FeatureRegression.", "FeatureRegression2."]):
        if regressor is not None:
            for k, v in regressor.state_dict().items():
                k_target = prefix + k
                if k.endswith("num_batches_tracked"):
                    continue
                # print("Copying from {0} to {1}, size {2}".format(k_target, k, v.size()))
                if k != "linear.weight":
                    v.copy_(src_state_dict[k_target])
                else:
                    # HACK to substitute linear layer with convolution
                    v.copy_(src_state_dict[k_target].view(-1, 64, 5, 5))
