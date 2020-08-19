## Experiments on the ImageNet dataset
This file describes our experiments in the 1-shot setting of [Karlinsky et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Karlinsky_RepMet_Representative-Based_Metric_Learning_for_Classification_and_Few-Shot_Object_Detection_CVPR_2019_paper.html)
based on the ImageNet dataset.

### Installing the dataset
The first step is to install the ImageNet dataset.
Please, follow [the official webcite](http://image-net.org/) for instructions.
We used the training and validation subsets of ILSVRC2012.

Create the simlink `$OS2D_ROOT/data/ImageNet-RepMet/ILSVRC` pointing to your ImageNet installation.
Make sure you have paths `$OS2D_ROOT/data/ImageNet-RepMet/ILSVRC/Data/CLS-LOC` and `$OS2D_ROOT/data/ImageNet-RepMet/ILSVRC/Annotations/CLS-LOC` available.

Download the episodic data of [Karlinsky et al. (RepMet)](https://github.com/jshtok/RepMet) released on [Google Drive](https://drive.google.com/drive/folders/1MZ6HWQpR_Oseo5_v5gmrlAyubrPL-ciO?usp=sharing) and put it to `$OS2D_ROOT/data/ImageNet-RepMet/RepMet_CVPR19_data`.
We need only two files `RepMet_CVPR19_data/data/Imagenet_LOC/voc_inloc_roidb.pkl` and `RepMet_CVPR19_data/data/Imagenet_LOC/episodes/epi_inloc_in_domain_1_5_10_500.pkl` which can be downloded with these commands:
```bash
mkdir -p $OS2D_ROOT/data/ImageNet-RepMet/RepMet_CVPR2019_data/data/Imagenet_LOC/episodes
$OS2D_ROOT/os2d/utils/wget_gdrive.sh $OS2D_ROOT/data/ImageNet-RepMet/RepMet_CVPR2019_data/data/Imagenet_LOC/voc_inloc_roidb.pkl 1VFQkO4WToV7OMggzu6F_sOuuHno_qEFE
$OS2D_ROOT/os2d/utils/wget_gdrive.sh $OS2D_ROOT/data/ImageNet-RepMet/RepMet_CVPR2019_data/data/Imagenet_LOC/episodes/epi_inloc_in_domain_1_5_10_500.pkl 1yjBvPoVO-PAnTEXnpHAfTv5XQ1Xg1pJS
```

### Train ResNet101 on data with RepMet test classes excluded
Preparations:
```bash
conda activate os2d
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
cd $OS2D_ROOT/data/ImageNet-RepMet/pretrain
```
Prepare dataset (scripts will create subfolders of `$OS2D_ROOT/data/ImageNet-RepMet/pretrain/imagenet-repmet` with simlinks to the original ImageNet files):
```bash
python prepare_data_exclude_test_classes.py
```
Train the model with [the script from the PyTorch examples](https://github.com/pytorch/examples/tree/master/imagenet):
```bash
ARCH=resnet101
mkdir -p output/${ARCH}
cd output/${ARCH}
python ../../train_imagenet.py -a ${ARCH} --dist-url 'tcp://127.0.0.1:23455' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ../../imagenet-repmet
```
We trained on 4 V100 GPUs, the script ran for 90 epochs and obtained Acc@1 of 75.819 and Acc@5 of 92.735 (note that these numbers are not comparable with the standard ImageNet results because of different set of classes).

Convert the trained model for further usage:
```bash
python convert_resnet_pytorch_to_maskrcnnbenchmark.py output/${ARCH}/model_best.pth.tar
python convert_resnet_pytorch_to_cirtorch.py output/${ARCH}/model_best.pth.tar
```

### Train the detector of all classes
```bash
conda activate os2d
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
cd $OS2D_ROOT/baselines/detector_retrieval/detector
python experiments/launcher_train_detector_imagenet.py
```

### Train the retrieval system
```bash
conda activate os2d
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
cd $OS2D_ROOT/baselines/detector_retrieval/retrieval
bash ./prepare_datasets_imagenet.sh
python experiments/launcher_imagenet.py
```

### Evaluate the detector-retrieval baseline
```bash
conda activate os2d
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
cd $OS2D_ROOT/baselines/detector_retrieval
python experiments/launcher_imagenet_eval.py
python experiments/launcher_imagenet_eval_collect.py
```

### Evaluate the OS2D models
```bash
conda activate os2d
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
cd $OS2D_ROOT
python experiments/launcher_imagenet_eval.py
python experiments/launcher_imagenet_eval_collect.py
```
