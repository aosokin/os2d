# OS2D: One-Stage One-Shot Object Detection by Matching Anchor Features

This repo is the implementation of the following paper:

OS2D: One-Stage One-Shot Object Detection by Matching Anchor Features<br>
Anton Osokin, Denis Sumin, Vasily Lomakin<br>
In proceedings of the European Conference on Computer Vision (ECCV), 2020

If you use our ideas, code or data, please, cite our paper ([available on arXiv](https://arxiv.org/abs/2003.06800)).

<details>
<summary>Citation in bibtex</summary>

```
@inproceedings{osokin20os2d,
    title = {{OS2D}: One-Stage One-Shot Object Detection by Matching Anchor Features},
    author = {Anton Osokin and Denis Sumin and Vasily Lomakin},
    booktitle = {proceedings of the European Conference on Computer Vision (ECCV)},
    year = {2020} }
```
</details>

## License
This software is released under the [MIT license](./LICENSE), which means that you can use the code in any way you want.

## Requirements
1. python >= 3.7
2. pytorch >= 1.4, torchvision >=0.5
3. NVIDIA GPU, tested with V100 and GTX 1080 Ti
4. Installed CUDA, tested with v10.0

See [INSTALL.md](INSTALL.md) for the package installation.

## Demo
See our [demo-notebook](./demo.ipynb) for an illustration of our method.

## Dataset installation
1. Grozi-3.2k dataset with our annotation (0.5GB): download from [Google Drive](https://drive.google.com/open?id=1Fx9lvmjthe3aOqjvKc6MJpMuLF22I1Hp) or with the [magic command](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99) and unpack to $OS2D_ROOT/data
```bash
cd $OS2D_ROOT
./os2d/utils/wget_gdrive.sh data/grozi.zip 1Fx9lvmjthe3aOqjvKc6MJpMuLF22I1Hp
unzip data/grozi.zip -d data
```
2. Extra test sets of retail products (0.1GB):  download from [Google Drive](https://drive.google.com/open?id=1Vp8sm9zBOdshYvND9EPuYIu0O9Yo346J) or with the [magic command](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99) and unpack to $OS2D_ROOT/data
```bash
cd $OS2D_ROOT
./os2d/utils/wget_gdrive.sh data/retail_test_sets.zip 1Vp8sm9zBOdshYvND9EPuYIu0O9Yo346J
unzip data/retail_test_sets.zip -d data
```
3. INSTRE datasets (2.3GB) are re-hosted in Center for Machine Perception in Prague (thanks to [Ahmet Iscen](http://cmp.felk.cvut.cz/~iscenahm/code.html)!): 
```bash
cd $OS2D_ROOT
wget ftp://ftp.irisa.fr/local/texmex/corpus/instre/gnd_instre.mat -P data/instre  # 200KB
wget ftp://ftp.irisa.fr/local/texmex/corpus/instre/instre.tar.gz -P data/instre  # 2.3GB
tar -xzf data/instre/instre.tar.gz -C data/instre
```
4. If you want to add your own dataset you should create an instance of the `DatasetOneShotDetection` class and then pass it into the functions creating dataloaders `build_train_dataloader_from_config` or `build_eval_dataloaders_from_cfg` from [os2d/data/dataloader.py](os2d/data/dataloader.py). See [os2d/data/dataset.py](os2d/data/dataset.py) for docs and examples.

## Trained models
We release three pretrained models:
| Name | mAP on "grozi-val-new-cl" | link |
| -- | -- | -- |
| OS2D V2-train | 90.65 | [Google Drive](https://drive.google.com/open?id=1l_aanrxHj14d_QkCpein8wFmainNAzo8) |
| OS2D V1-train | 88.71 | [Google Drive](https://drive.google.com/open?id=1ByDRHMt1x5Ghvy7YTYmQjmus9bQkvJ8g) |
| OS2D V2-init  | 86.07 | [Google Drive](https://drive.google.com/open?id=1sr9UX45kiEcmBeKHdlX7rZTSA4Mgt0A7) |

The results (mAP on "grozi-val-new-cl") can be computed with the commands given [below](#evaluation).

You can download the released datasets  with [the magic commands](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99):
```bash
cd $OS2D_ROOT
./os2d/utils/wget_gdrive.sh models/os2d_v2-train.pth 1l_aanrxHj14d_QkCpein8wFmainNAzo8
./os2d/utils/wget_gdrive.sh models/os2d_v1-train.pth 1ByDRHMt1x5Ghvy7YTYmQjmus9bQkvJ8g
./os2d/utils/wget_gdrive.sh models/os2d_v2-init.pth 1sr9UX45kiEcmBeKHdlX7rZTSA4Mgt0A7
```

## Evaluation
1. OS2D V2-train (best model)

For a fast eval on a validation set, one can do use a single scale of images with this script (will give 85.58 mAP on the validation set "grozi-val-new-cl"):
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False model.backbone_arch ResNet50 train.do_training False eval.dataset_names "[\"grozi-val-new-cl\"]" eval.dataset_scales "[1280.0]" init.model models/os2d_v2-train.pth eval.scales_of_image_pyramid "[1.0]"
```

Multiscale evaluation gives better results - scripts below use the default setting with 7 scales: 0.5, 0.625, 0.8, 1, 1.2, 1.4, 1.6. Note that this evaluation can be slower because of the multiple scale and *a lot of* classes in the dataset.

To evaluate on the validation set with multiple scales, run:
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False model.backbone_arch ResNet50 train.do_training False eval.dataset_names "[\"grozi-val-new-cl\"]" eval.dataset_scales "[1280.0]" init.model models/os2d_v2-train.pth
```

2. OS2D V1-train

To evaluate on the validation set run:
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model False model.use_simplified_affine_model True model.backbone_arch ResNet101 train.do_training False eval.dataset_names "[\"grozi-val-new-cl\"]" eval.dataset_scales "[1280.0]" init.model models/os2d_v1-train.pth
```


3. OS2D V2-init

To evaluate on the validation set run:
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False model.backbone_arch ResNet50 train.do_training False eval.dataset_names "[\"grozi-val-new-cl\"]" eval.dataset_scales "[1280.0]" init.model models/os2d_v2-init.pth
```


## Training

### Pretrained models
In this project, we do not train models from scratch but start from some pretrained models. For instructions how to get them, see [models/README.md](models/README.md).

### Best models
Our V2-train model on the Grozi-3.2k dataset was trained using this command:
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False train.objective.loc_weight 0.0 train.model.freeze_bn_transform True model.backbone_arch ResNet50 init.model models/imagenet-caffe-resnet50-features-ac468af-renamed.pth init.transform models/weakalign_resnet101_affine_tps.pth.tar train.mining.do_mining True output.path output/os2d_v2-train
```
Dut to hard patch mining, this process is quite slow. Without it, training is faster, but produces slightly worse results:
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False train.objective.loc_weight 0.0 train.model.freeze_bn_transform True model.backbone_arch ResNet50 init.model models/imagenet-caffe-resnet50-features-ac468af-renamed.pth init.transform models/weakalign_resnet101_affine_tps.pth.tar train.mining.do_mining False output.path output/os2d_v2-train-nomining
```
For the V1-train model, we used this command:
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model False model.use_simplified_affine_model True train.objective.loc_weight 0.2 train.model.freeze_bn_transform False model.backbone_arch ResNet101 init.model models/gl18-tl-resnet101-gem-w-a4d43db-converted.pth train.mining.do_mining False output.path output/os2d_v1-train
```
Note that these runs need a lot of RAM due to caching of the whole training set. If this does not work for you you can use parameters `train.cache_images False`, which will load images on the fly, but can be slow. Also note that several first iterations of training can be slow bacause of "warming up", i.e., computing the grids of anchors in Os2dBoxCoder. Those computations are cached, so everyhitng will eventually run faster.

For the rest of the training scripts see [below](#rerunning-experiments-on-retail-and-instre-datasets).

### Rerunning experiments
All the experiments ob this project were run with [our job helper](./os2d/utils/launcher.py).
For each experiment, one program an experiment structure (in python) and calls several technical function provided by the launcher.
See, e.g., [this file](./experiments/launcher_exp1.py) for an example.

The launch happens as follows:
```bash
# add OS2D_ROOT to the python path - can be done, e.g., as follows
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
# call the experiment script
python ./experiments/launcher_exp1.py LIST_OF_LAUNCHER_FLAGS
```
Extra parameters in `LIST_OF_LAUNCHER_FLAGS` are parsed by [the launcher](./os2d/utils/launcher.py) and contain some useful options about the launch:
1. `--no-launch` allows to prepare all the scripts of the experiment without the actual launch.
2. `--slurm` allows to prepare SLURM jobs and launches (if the is no `--no-launch`) with sbatch.
3. `--stdout-file` and `--stderr-file` - files where to save stdout and stderr, respectively (relative to the log_path defined in the experiment description).
4. For many SLURM related parameters, see [the launcher](./os2d/utils/launcher.py).

Our experiments can be found here:
1. [Experiments with OS2D](experiments/README.md)
2. [Experiments with the detector-retrieval baseline](baselines/detector_retrieval/README.md)
3. [Experiments with the CoAE baseline](baselines/CoAE/README.md)
4. [Experiments on the ImageNet dataset](experiments/README_ImageNet.md)


### Baselines
We have added two baselines in this repo:
1. Class-agnostic detector + image retrieval system: see [README](baselines/detector_retrieval/README.md) for details.
2. Co-Attention and Co-Excitation, CoAE ([original code](https://github.com/timy90022/One-Shot-Object-Detection), [paper](https://arxiv.org/abs/1911.12529)): see [README](baselines/CoAE/README.md) for details.


### Acknowledgements
We would like to personally thank [Ignacio Rocco](https://www.irocco.info/), [Relja Arandjelović](http://www.relja.info/), [Andrei Bursuc](https://abursuc.github.io/), [Irina Saparina](https://github.com/saparina) and [Ekaterina Glazkova](https://github.com/EkaterinaGlazkova) for amazing discussions and insightful comments without which this project would not be possible.

This research was partly supported by Samsung Research, Samsung Electronics, by the Russian Science Foundation grant 19-71-00082 and through computational resources of [HPC facilities](https://it.hse.ru/hpc) at NRU HSE.

This software was largely inspired by a number of great repos: [weakalign](https://github.com/ignacio-rocco/weakalign), [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch), [torchcv](https://github.com/kuangliu/torchcv), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
Special thanks goes to the amazing [PyTorch](https://pytorch.org/).
