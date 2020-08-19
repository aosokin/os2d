import argparse
from collections import OrderedDict

import torch


def convert_model(model_file):
    model_pth = torch.load(model_file, map_location=torch.device("cpu"))
    exts = [".pth.tar", ".pth"]
    ext = None
    for e in exts:
        if model_file.endswith(e):
            ext = e
            break
    assert ext is not None, "Can only parse models saved to one of {} formats".format(exts)
    model_name = model_file[:-len(ext)]
    target_path = model_name + "_cirtorch" + ext

    print("Converting", model_file,
          "to", target_path)

    if "state_dict" in model_pth:
        state_dict_pth = model_pth["state_dict"]
    else:
        state_dict_pth = model_pth

    # Create the pytorch state_dict
    state_dict_cirtorch = OrderedDict()

    # create a map of prefix renamings
    prefix_map = OrderedDict()
    prefix_map["conv1."] = "features.0."
    prefix_map["bn1."] = "features.1."
    prefix_map["layer1."] = "features.4."
    prefix_map["layer2."] = "features.5."
    prefix_map["layer3."] = "features.6."
    prefix_map["layer4."] = "features.7."

    # rename layers and add to the pytorch model
    num_added_tensors = 0
    for k, v in state_dict_pth.items():
        # find good prefix
        if k.startswith("module."):
            k = k[len("module."):]
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
            state_dict_cirtorch[new_name] = v
            num_added_tensors += 1
    print("Converted {0} tensors".format(num_added_tensors))

    # saving the model
    state_dict_cirtorch = {"state_dict": state_dict_cirtorch}
    torch.save(state_dict_cirtorch, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converting pytorch ResNets to Caffe2-cirtorch")
    parser.add_argument("model", help="Path to the model to convert, the result will be save to the same folder")
    args = parser.parse_args()

    convert_model(args.model)
