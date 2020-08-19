import argparse

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
    target_path = model_name + "_maskrcnnbenchmark" + ext

    print("Converting", model_file,
          "to", target_path)

    if "state_dict" in model_pth:
        state_dict_pth = model_pth["state_dict"]
    else:
        state_dict_pth = model_pth 

    # saving the model
    torch.save(state_dict_pth, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converting pytorch ResNets to the ones maskrcnn-benchmark can init from")
    parser.add_argument("model", help="Path to the model to convert, the result will be save to the same folder")
    args = parser.parse_args()

    convert_model(args.model)
