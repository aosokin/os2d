import os
import argparse
from collections import OrderedDict

import torch


def convert_model(model_file):
    model_cirtorch = torch.load(model_file)
    model_name, ext = os.path.splitext(model_file)
    target_path = model_name + "-converted" + ext

    print("Converting", model_file,
          "to", target_path)

    state_dict_cirtorch = model_cirtorch["state_dict"]

    # Create the pytorch state_dict
    model_pth = OrderedDict()

    # create a map of prefix renamings
    prefix_map = OrderedDict()
    prefix_map["features.0."] = "conv1."
    prefix_map["features.1."] = "bn1."
    prefix_map["features.4."] = "layer1."
    prefix_map["features.5."] = "layer2."
    prefix_map["features.6."] = "layer3."
    prefix_map["features.7."] = "layer4."

    # rename layers and add to the pytorch model
    num_added_tensors = 0
    for k, v in state_dict_cirtorch.items():
        # find good prefix
        prefix = None
        for p in prefix_map.keys():
            if k.startswith(p):
                if prefix is None:
                    prefix = p
                else:
                    print("For layer {0} found two prefixes: {1} or {2}".format(k, prefix, p))
        if prefix is None:
            print("For layer {0} did not find any matching prefix!".format(k))
        else:
            new_name = prefix_map[prefix] + k[len(prefix):]
            # print("Renaming {0} to {1}".format(k, new_name))
            model_pth[new_name] = v
            num_added_tensors += 1
    print("Converted {0} tensors".format(num_added_tensors))

    # saving the model
    torch.save(model_pth, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converting Caffe2-cirtorch ResNets to pytorch")
    parser.add_argument("model", help="Path to the model to convert, the result will be save to the same folder")
    args = parser.parse_args()

    convert_model(args.model)
