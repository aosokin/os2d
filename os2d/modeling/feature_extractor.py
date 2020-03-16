from itertools import chain

import torch

from torchvision.models.resnet import ResNet, resnet50, resnet101

from os2d.structures.feature_map import FeatureMapSize


GROUPNORM_NUMGROUPS = 32


def build_feature_extractor(backbone_arch, use_group_norm=False):
    if backbone_arch.lower() == "resnet50":
        net = resnet50_c4(use_group_norm=use_group_norm)
    elif backbone_arch.lower() == "resnet101":
        net = resnet101_c4(use_group_norm=use_group_norm)
    else:
        raise(RuntimeError("Unknown backbone arch: {0}".format(backbone_arch)))
    return net


class ResNetFeatureExtractor(ResNet):
    """
    This class implements the feature extractor based on the ResNet backbone
    """
    def __init__(self, resnet_full, level,
                       feature_map_stride, feature_map_receptive_field):
        """
        Args:
            resnet_full - a resnet model: an instance of the ResNet class from torchvision
                https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
            level (int) - level at which to create the feature extractor, can be 1, 2, 3, 4, 5
            feature_map_stride (FeatureMapSize) - the stride of the feature map, should be set manually
            feature_map_receptive_field (FeatureMapSize) - the effective receptive field of the feature map, should be set manually
        """
        self.__dict__ = resnet_full.__dict__.copy()
        self.feature_map_receptive_field = feature_map_receptive_field
        self.feature_map_stride = feature_map_stride
        self._feature_level = level

        # remove unused layers to free memory
        delattr(self, "fc")
        delattr(self, "avgpool")

        assert level in [1, 2, 3, 4, 5], "Feature level should be one of 1, 2, 3, 4, 5"
        # level == 5 - use all blocks
        # note that for level == 4, self.layer4 is chopped off
        # inconsistency in numbers comes from the inconsistency between layer names in the ResNet paper and torchvision ResNet code
        self.resnet_blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        layer_names = ["layer1", "layer2", "layer3", "layer4"]

        self.resnet_blocks = self.resnet_blocks[:level-1]
        for name in layer_names[level-1:]:
            delattr(self, name)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.resnet_blocks:
            x = layer(x)
        return x

    def freeze_bn(self):
        # Freeze BatchNorm layers
        for layer in self.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()

    def freeze_blocks(self, num_blocks=0):
        # join conv1 and bn1 into one block
        layer0 = [torch.nn.ModuleList([self.conv1, self.bn1])]

        num_remaining_blocks = num_blocks
        blocks = chain(layer0, chain.from_iterable(self.resnet_blocks))
        for b in blocks:
            if num_remaining_blocks > 0:
                self.freeze_layer_parameters(b)
                num_remaining_blocks -= 1

    @staticmethod
    def freeze_layer_parameters(layer):
        for p in layer.parameters():
            p.requires_grad = False

    def get_num_blocks_in_feature_extractor(self):
        # one block - self.conv1 + self.bn1
        # the following blocks: self.layer1, self.layer2, self.layer3, until cut off
        num_blocks = 1 + sum(len(b) for b in self.resnet_blocks)
        return num_blocks


def get_norm_layer(use_group_norm):
    if use_group_norm:
        return lambda width: torch.nn.GroupNorm(GROUPNORM_NUMGROUPS, width)
    else:
        return torch.nn.BatchNorm2d


def _resnet_fe(resnet, level, use_group_norm, feature_map_stride, feature_map_receptive_field):
    return ResNetFeatureExtractor(resnet(norm_layer=get_norm_layer(use_group_norm)), level,
                                  feature_map_stride, feature_map_receptive_field)


def resnet50_c4(use_group_norm=False):
    """
    Constructs the ResNet50 C4 feature extractor (R-50-C4 in maskrcnn-benchmark)
    Args:
        use_group_norm (bool) - if True use torch.nn.GroupNorm with GROUPNORM_NUMGROUPS groups as normalization layers,
            otherwise use torch.nn.BatchNorm2d
    """
    return _resnet_fe(resnet50, 4, use_group_norm,
                      feature_map_stride=FeatureMapSize(h=16, w=16),
                      feature_map_receptive_field=FeatureMapSize(h=16, w=16))


def resnet101_c4(use_group_norm=False):
    """
    Constructs the ResNet101 C4 feature extractor
    Args:
        use_group_norm (bool) - if True use torch.nn.GroupNorm with GROUPNORM_NUMGROUPS groups as normalization layers,
            otherwise use torch.nn.BatchNorm2d
    """
    return _resnet_fe(resnet101, 4, use_group_norm,
                      feature_map_stride=FeatureMapSize(h=16, w=16),
                      feature_map_receptive_field=FeatureMapSize(h=16, w=16))

