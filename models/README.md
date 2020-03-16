## Pretrained models

### The semantic alignment model trained on [PF-PASCAL dataset](https://www.di.ens.fr/willow/research/proposalflow/)
The semantic alignment model of [Rocco et al.](https://github.com/ignacio-rocco/weakalign) contains the weights of both feature extractor and the transformtion network. The model can be downloaded as follows:
```bash
cd $OS2D_ROOT/models
wget http://www.di.ens.fr/willow/research/weakalign/trained_models/weakalign_resnet101_affine_tps.pth.tar
```

### Models trained on [ImageNet](http://www.image-net.org/) for classification in PyTorch
The standard PyTorch models can be downloaded as follows (links from [torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.pyhttps://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)):
```bash
cd $OS2D_ROOT/models
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
```

### Models trained on [ImageNet](http://www.image-net.org/) for classification in Caffe2
Some projects have reported that specific weights os ResNets originally trained in Caffe2 work better, e.g., in image retrieval. We use these weights ported to PyTorch by [Radenović F. et al.](https://github.com/filipradenovic/cnnimageretrieval-pytorch) the models can be downloaded as follows (links from [here](https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/networks/imageretrievalnet.py)):
```bash
cd $OS2D_ROOT/models
wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth
wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth
```
These models are in a slightly different format from standard PyTorch, so we need to do some weight surgery to convert them.
```bash
conda activate os2d
python convert_resnet_caffe2_cirtorch_to_pytorch.py imagenet-caffe-resnet50-features-ac468af.pth
python convert_resnet_caffe2_cirtorch_to_pytorch.py imagenet-caffe-resnet101-features-10a101d.pth
```
This should produce files `imagenet-caffe-resnet50-features-ac468af-converted.pth` and `imagenet-caffe-resnet101-features-10a101d-converted.pth`.


### Model with GroupNorm instead of BatchNorm trained on [ImageNet](http://www.image-net.org/) for classification in Caffe2
[Group normalization](https://arxiv.org/abs/1803.08494) has been reported to work better than BatchNorm when the batch size is small. We have tried using ResNet-50 with GroupNorm. Download the model with 32 groups:
```bash
cd $OS2D_ROOT/models
wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl
```
Converting the model:
```bash
conda activate os2d
python convert_resnet_caffe2_groupnorm_to_pytorch.py R-50-GN.pkl --num_layers 50
```
This should produce file resnet50_caffe2_groupnorm.pth

### Models trained on [COCO](http://cocodataset.org/) for object detection
We have tried to initialize our models from the weights of detection models trained in the [maskrcnn-benchmark framework](https://github.com/facebookresearch/maskrcnn-benchmark). Download the models and their configs:
```bash
cd $OS2D_ROOT/models
wget -P maskrcnn-benchmark https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth
wget -P maskrcnn-benchmark https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
wget -P maskrcnn-benchmark https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_101_FPN_1x.pth
wget -P maskrcnn-benchmark https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_mask_rcnn_R_101_FPN_1x.yaml
```
To convert the models one needs to have [maskrcnn-benchmark installed](../baselines/detector_retrieval/INSTALL.md) (we used v0.1) or just have their [default config](https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/config). Scripts for convertion:
```bash
conda activate os2d
python convert_resnet_maskrcnnbenchmark_to_pytorch.py maskrcnn-benchmark/e2e_mask_rcnn_R_50_FPN_1x.pth maskrcnn-benchmark/e2e_mask_rcnn_R_50_FPN_1x.yaml
python convert_resnet_maskrcnnbenchmark_to_pytorch.py maskrcnn-benchmark/e2e_mask_rcnn_R_101_FPN_1x.pth maskrcnn-benchmark/e2e_mask_rcnn_R_101_FPN_1x.yaml
```
This should produce files `maskrcnn-benchmark/e2e_mask_rcnn_R_50_FPN_1x_converted.pth` and `maskrcnn-benchmark/e2e_mask_rcnn_R_101_FPN_1x_converted.pth`

### Model trained on [google-landmarks-2018](https://www.kaggle.com/google/google-landmarks-dataset) for image retrieval
We have tried to initialize from a model trained for large scale image retrieval in the [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) project. Download the model:
```bash
cd $OS2D_ROOT/models
wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth
```
Scripts for convertion:
```bash
conda activate os2d
python convert_resnet_cirtorch_to_pytorch.py gl18-tl-resnet101-gem-w-a4d43db.pth
```