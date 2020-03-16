## Installation of the CoAE code
Needed for the CoAE baseline. These instructions assume OS2D is [installed](../../INSTALL.md), and maskrcnn_benchmark from the detector-retrieval baseline is [installed](../detector_retrieval/INSTALL.md).
The instructions are based on [the CoAE repo](https://github.com/timy90022/One-Shot-Object-Detection).
```bash
# activate the os2d env
conda activate os2d
cd ${OS2D_ROOT}/baselines/CoAE

# extra dependencies (from https://github.com/timy90022/One-Shot-Object-Detection/blob/master/requirements.txt)
pip install easydict==1.9

# clone the repo
git clone https://github.com/timy90022/One-Shot-Object-Detection.git coae
cd coae
# I was on this commit:
git checkout 2098ad3e90cb4aa9f1dd188a40efa29927ac3ab1

# build binaries (see https://github.com/timy90022/One-Shot-Object-Detection/blob/master/README.md)
cd ${OS2D_ROOT}/baselines/CoAE/coae/lib
python setup.py build develop
```
