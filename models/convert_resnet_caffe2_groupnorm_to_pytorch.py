"""Models that can be converted:
ResNet-50-GN: https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl
ResNet-101-GN: https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47592356/R-101-GN.pkl

Code is base on this script:
https://github.com/ruotianluo/pytorch-resnet/blob/master/convert_gn.py
"""

import os
import argparse
import pickle
import numpy as np

import torch
import torchvision.models.resnet as resnet


def load_caffe2_model(path):
    with open(path, 'rb') as fp:
        src_blobs = pickle.load(fp, encoding='latin1')
        if 'blobs' in src_blobs:
            src_blobs = src_blobs['blobs']
        pretrained_state_dict = src_blobs
    return pretrained_state_dict


def detectron_weight_mapping(self):
    mapping_to_detectron = {
        'conv1.weight': 'conv1_w',
        'bn1.weight': 'conv1_gn_s',
        'bn1.bias': 'conv1_gn_b'
    }

    for res_id in range(1, 5):
        stage_name = 'layer%d' % res_id
        mapping = residual_stage_detectron_mapping(
            getattr(self, stage_name), res_id)
        mapping_to_detectron.update(mapping)

    return mapping_to_detectron


def residual_stage_detectron_mapping(module_ref, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    pth_norm_suffix = '_bn'
    norm_suffix = '_gn'
    mapping_to_detectron = {}
    for blk_id in range(len(module_ref)):
        detectron_prefix = 'res%d_%d' % (res_id+1, blk_id)
        my_prefix = 'layer%s.%d' % (res_id, blk_id)

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix
                                 + '.downsample.0.weight'] = dtt_bp + '_w'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'

        # conv branch
        for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
            dtt_bp = detectron_prefix + '_branch2' + c
            mapping_to_detectron[my_prefix
                                 + '.conv%d.weight' % i] = dtt_bp + '_w'
            mapping_to_detectron[my_prefix
                                 + '.' + pth_norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.' + pth_norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'

    return mapping_to_detectron


def convert_model(path, num_layers=50, num_groups=32):
    target_path = "resnet{}_caffe2_groupnorm.pth".format(num_layers)

    print("Converting ResNet-{0}-GN from {1} to {2}".format(num_layers, path, target_path))

    # load Caffe2 model
    model_caffe2 = load_caffe2_model(path)

    # create pytorch model
    norm_layer = lambda width: torch.nn.GroupNorm(num_groups, width)
    model_pth = getattr(resnet, 'resnet{}'.format(num_layers))(norm_layer=norm_layer)
    model_pth.eval()
    model_pth_state_dict = model_pth.state_dict()

    name_mapping = detectron_weight_mapping(model_pth)
    name_mapping.update({
        'fc.weight': 'pred_w',
        'fc.bias': 'pred_b'
    })

    assert set(model_pth_state_dict.keys()) == set(name_mapping.keys())
    assert set(model_caffe2.keys()) == set(name_mapping.values())

    num_added_tensors = 0
    for k, v in name_mapping.items():
        if isinstance(v, str):  # maybe a str, None or True
            assert(model_pth_state_dict[k].shape == torch.Tensor(model_caffe2[v]).shape)
            model_pth_state_dict[k].copy_(torch.Tensor(model_caffe2[v]))
            if k == 'conv1.weight':
                tmp = model_pth_state_dict[k]
                # BGR to RGB
                tmp = tmp[:, [2, 1, 0]].numpy()
                # renormalize
                tmp *= 255.0
                tmp *= np.array([0.229, 0.224, 0.225])[np.newaxis, :, np.newaxis, np.newaxis]

                model_pth_state_dict[k].copy_(torch.from_numpy(tmp))
            num_added_tensors += 1

    torch.save(model_pth_state_dict, target_path)
    print("Converted {0} tensors".format(num_added_tensors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converting Caffe2-GroupNorm ResNets to pytorch")
    parser.add_argument("model", help="Path to the model to convert, the result will be save to the same folder")
    parser.add_argument("--num_layers", default=50, type=int, help="Number of residual blocks in ResNet: X from ResNet-X")
    parser.add_argument("--num_groups", default=32, type=int, help="Number of groups in gorup norm")
    args = parser.parse_args()

    convert_model(args.model, args.num_layers, args.num_groups)
