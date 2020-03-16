import os
import argparse
from collections import OrderedDict

import torch
from torchvision.models.resnet import resnet50, resnet101

from maskrcnn_benchmark.config import cfg as maskrcnn_cfg


def convert_model(maskrcnn_weight_path, maskrcnn_config_path):
    model_name, ext = os.path.splitext(maskrcnn_weight_path)
    target_path = model_name + "_converted" + ext

    print("Converting", maskrcnn_weight_path,
          "with config", maskrcnn_config_path,
          "to", target_path)

    if "R_50" in maskrcnn_weight_path:
        target_net = "resnet50"
        model = resnet50().state_dict()
    elif "R_101" in maskrcnn_weight_path:
        target_net = "resnet101"
        model = resnet101().state_dict()
    else:
        raise RuntimeError("Could not recognize architecture from file name {0}".format(maskrcnn_weight_path))

    maskrcnn_model = torch.load(maskrcnn_weight_path)
    maskrcnn_model = maskrcnn_model["model"]
    maskrcnn_cfg.merge_from_file(maskrcnn_config_path)

    # create a map of prefix renamings
    prefix_map = OrderedDict()
    prefix_map["conv1."] = "module.backbone.body.stem.conv1."
    prefix_map["bn1."] = "module.backbone.body.stem.bn1."
    prefix_map["layer1."] = "module.backbone.body.layer1."
    prefix_map["layer2."] = "module.backbone.body.layer2."
    prefix_map["layer3."] = "module.backbone.body.layer3."
    prefix_map["layer4."] = None
    prefix_map["fc."] = None

    new_model = OrderedDict()
    num_added_tensors = 0
    for k, v in model.items():
        found = False
        for prefix in prefix_map.keys():
            if k.startswith(prefix):
                found = prefix
        if not found:
            print("Layer {0} was not found in the prefix map".format(k))
            continue

        if prefix_map[found] is None:
            # chop off these
            continue

        if k.endswith("num_batches_tracked"):
            # skip these parameters
            continue

        layer_to_init_from = prefix_map[found] + k[len(found):]
        if layer_to_init_from not in maskrcnn_model:
            print("Layer {0} to init {1} was not found in the maskrcnn model".format(layer_to_init_from, k))

        assert maskrcnn_model[layer_to_init_from].size() == v.size(), "Size {0} of the source {1} does not match size {2} of target {3}".format(maskrcnn_model[layer_to_init_from].size(), layer_to_init_from, v.size(), k )

        new_model[k] = maskrcnn_model[layer_to_init_from].cpu()
        num_added_tensors += 1
    print("Converted {0} tensors".format(num_added_tensors))

    # adjust the first layer convolution
    assert new_model['conv1.weight'].size(1) == 3, "the first layer is of the wrong size: {}".format(new_model['conv1.weight'].size())
    w = new_model['conv1.weight']

    # deal with different normalization and BGR
    # their normalization
    # maskrcnn_cfg.INPUT.PIXEL_MEAN - can't deal mean mean easily
    # maskrcnn_cfg.INPUT.PIXEL_STD / 255

    for c in range(3):
        w[:,c] = w[:,c] * 255.0 / maskrcnn_cfg.INPUT.PIXEL_STD[c]

    # deal with BGR
    if maskrcnn_cfg.INPUT.TO_BGR255:
        # remap the first layer from BGR to RGB
        w = torch.stack([w[:,2], w[:,1], w[:,0]], 1)

    # pytorch normalization:
    normalization = {}
    normalization['mean'] = (0.485, 0.456, 0.406)
    normalization['std'] = (0.229, 0.224, 0.225)

    for c in range(3):
        w[:,c] = w[:,c] * normalization['std'][c]

    new_model['conv1.weight'] = w

    print("saving model to {0}".format(target_path))
    torch.save(new_model, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converting maskrcnn-benchmark ResNets to pytorch")
    parser.add_argument("model", help="Path to the model to convert, the result will be save to the same folder")
    parser.add_argument("config", help="Path to the config file corresponding to the model")
    args = parser.parse_args()

    convert_model(args.model, args.config)
