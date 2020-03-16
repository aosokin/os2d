"""Script to train a network on a dataset.
This script is a refactored version of
https://github.com/timy90022/One-Shot-Object-Detection/blob/master/trainval_net.py
"""
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import math
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

import _init_paths

from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.resnet import resnet as resnet_coae

from test_net import test


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='grozi-train', type=str)
  parser.add_argument('--net', dest='net',
                    help='res50, res101',
                    default='res50', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--init_weights',
                      help='model weights to init from',
                      default=None, type=str)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=10, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of epochs to save models',
                      default=None, type=int)
  parser.add_argument('--val_interval', dest='val_interval',
                      help='number of epochs after which to validate the model',
                      default=10, type=int)
  parser.add_argument('--dataset_val', dest='dataset_val',
                      help='name of th evaludation dataset',
                      default=None, type=str)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=8, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=128, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      default=True, type=bool)
  parser.add_argument('--class_image_augmentation', default=None, type=str,
                      help='augment images at test time: None or "rotation90"')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.01, type=float)
  parser.add_argument('--lr_decay_milestones', dest='lr_decay_milestones',
                      help='steps when to do learning rate decay, unit is epoch',
                      default=None, type=int, nargs='+')
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)
  parser.add_argument('--lr_reload_best_after_decay',
                      help='after decaying learning rate one can reload the best sees model',
                      default=True, type=bool)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      default=False)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


collate_tensors_err_msg_format = (
    "collate_tensors: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def tensor_list_to_blob(ims):
    """torch.stack(ims, dim=0) but when tensors in ims can have different dimensions
    Modification of im_list_to_blob of
    https://github.com/timy90022/One-Shot-Object-Detection/blob/master/lib/model/utils/blob.py
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = torch.zeros([num_images] + list(max_shape),
                       dtype=ims[0].dtype, device=ims[0].device)
    for i in range(num_images):
        im = ims[i]
        blob[i][tuple(slice(0,s) for s in im.shape)] = im

    return blob


def collate_tensors(batch):
    r"""Puts each data field into a tensor with outer dimension batch size
    Modification of default_collate from:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    This modification uses tensor_list_to_blob to stack tensors of different sizes.
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return tensor_list_to_blob(batch) # torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_tensors_err_msg_format.format(elem.dtype))

            return collate_tensors([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_tensors([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_tensors(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_tensors(samples) for samples in transposed]

    raise TypeError(collate_tensors_err_msg_format.format(elem_type))


class resnet(resnet_coae):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    super(resnet, self).__init__(classes, num_layers, pretrained, class_agnostic)
    if num_layers==50:
      self.model_path = './data/pretrain_imagenet_resnet50/model_best.pth.tar'
    elif num_layers==101:
      self.model_path = './data/pretrain_imagenet_resnet101/model_best.pth.tar'

  def init_from_pytorch_backbone(self, net):

    def init_subnet(subnet, prefix_map):
      state_dict = subnet.state_dict()
      new_state_dict = OrderedDict()
      for k in state_dict:
        prefix_found = None
        for prefix in prefix_map:
          if k.startswith(prefix):
            prefix_found = prefix
            break
        if prefix_found:
          src_layer = k.replace(prefix, prefix_map[prefix], 1)
          if "num_batches_tracked" in src_layer:
            # do not have this parameter in older pytorch nets
            new_state_dict[k] = state_dict[k]
          else:
            assert src_layer in net, "Src layer {0} (to init {1}) was not found in net".format(src_layer, k)
            src_tensor = net[src_layer]
            assert state_dict[k].shape == src_tensor.shape, "Layer {0} of backbone and found layer {1} have different sizes: {2}, {3}".format(k, src_layer, state_dict[k].shape, src_tensor.shape)
            # print("Replacing {0} (norm {1}) with {2} (norm {3})".format(k, state_dict[k].norm().item(), src_layer, src_tensor.norm().item()))
            new_state_dict[k] = src_tensor
      subnet.load_state_dict(new_state_dict)

    prefix_map_RCNN_base = OrderedDict()
    prefix_map_RCNN_base["0."] = "conv1."
    prefix_map_RCNN_base["1."] = "bn1."
    prefix_map_RCNN_base["4."] = "layer1."
    prefix_map_RCNN_base["5."] = "layer2."
    prefix_map_RCNN_base["6."] = "layer3."
    init_subnet(self.RCNN_base, prefix_map_RCNN_base)

    prefix_map_RCNN_top = OrderedDict()
    prefix_map_RCNN_top["0."] = "layer4."
    init_subnet(self.RCNN_top, prefix_map_RCNN_top)


if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  args.imdb_name = args.dataset
  args.cfg_file = "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  random_seed = cfg.RNG_SEED
  np.random.seed(random_seed)
  torch.manual_seed(random_seed)
  if args.cuda:
    torch.cuda.manual_seed_all(random_seed)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  cfg.USE_GPU_NMS = args.cuda

  # create dataloader
  imdb, roidb, ratio_list, ratio_index, query = combined_roidb(args.imdb_name, True)
  train_size = len(roidb)
  print('{:d} roidb entries'.format(len(roidb)))
  sampler_batch = sampler(train_size, args.batch_size)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, query, args.batch_size, imdb._classes, training=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers,
                            collate_fn=collate_tensors)

  # create output directory
  output_dir = args.save_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  query = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    query   = query.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    cfg.CUDA = True

  # initilize the network here.
  have_explicit_init = args.init_weights is not None and args.init_weights != "None"
  if args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=not have_explicit_init, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=not have_explicit_init, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")

  fasterRCNN.create_architecture()
  lr = args.lr

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN.cuda()

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)
  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))
  elif args.init_weights and args.init_weights != "None":
    load_name = args.init_weights
    print("loading weight file %s" % (load_name))
    checkpoint = torch.load(load_name)
    try:
      # try initializing as from checkpoint
      fasterRCNN.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr_old = optimizer.param_groups[0]['lr']
      if lr != lr_old:
        adjust_learning_rate(optimizer, lr / lr_old)
      print("Successfully initialized as from checkpoint")
    except:
      print("Could not initialized as from checkpoint. Trying to initialize from Pytorch backbone")
      fasterRCNN.init_from_pytorch_backbone(checkpoint)


  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(math.ceil(train_size / args.batch_size))

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  best_model_mAP = float("-inf")

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if args.lr_decay_milestones and (epoch in args.lr_decay_milestones):
      if args.lr_reload_best_after_decay:
        best_model_save_name = os.path.join(output_dir, 'best_model_{0}.pth'.format(args.session))
        print("Reloading the best model:", best_model_save_name)
        checkpoint = torch.load(best_model_save_name)
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_old = optimizer.param_groups[0]['lr']
        if lr != lr_old:
          adjust_learning_rate(optimizer, lr / lr_old)
      print("Decaying LR from {0} by factor {1}".format(lr, args.lr_decay_gamma))
      adjust_learning_rate(optimizer, args.lr_decay_gamma)
      lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)

    for step in range(iters_per_epoch):
      data = next(data_iter)
      im_data.resize_(data[0].size()).copy_(data[0])
      query.resize_(data[1].size()).copy_(data[1])
      im_info.resize_(data[2].size()).copy_(data[2])
      gt_boxes.resize_(data[3].size()).copy_(data[3])
      num_boxes.resize_(data[4].size()).copy_(data[4])

      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, margin_loss, RCNN_loss_bbox, \
      rois_label, _ = fasterRCNN(im_data, query, im_info, gt_boxes, num_boxes)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + margin_loss.mean() + RCNN_loss_bbox.mean()
      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls  = rpn_loss_cls.mean().item()
          loss_rpn_box  = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_margin   = margin_loss.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_margin = margin_loss.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.3f, rpn_box: %.3f, rcnn_cls: %.3f, margin: %.3f, rcnn_box %.3f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_margin, loss_rcnn_box))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_margin': loss_margin,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    if (args.val_interval and (epoch + 1) % args.val_interval == 0)\
      and (args.dataset_val is not None):
      # validate the current model
      args_val = copy.deepcopy(args)
      args_val.imdbval_name = args.dataset_val

      mAP = test(args_val, model=fasterRCNN)

      if mAP > best_model_mAP:
          # save the model as the current best
          print("New best model: mAP={0:.4f}, {1} epochs".format(mAP, epoch + 1))
          best_model_mAP = mAP
          best_model_save_name = os.path.join(output_dir, 'best_model_{0}.pth'.format(args.session))
          save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
          }, best_model_save_name)
          print('save model: {}'.format(best_model_save_name))

    if args.checkpoint_interval and (epoch + 1) % args.checkpoint_interval == 0:
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch + 1 , iters_per_epoch))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch + 1,
        'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
