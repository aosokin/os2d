# !/bin/bash

# dependencies
pip install ninja>=1.9.0 cython>=0.29.15
pip install opencv-python>=4.2.0.32

# the rest will be compiling from sources - set the path for that
INSTALL_DIR=$HOME/local/software/pytorch/os2d
mkdir -p $INSTALL_DIR

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# difference from the standard instruction: get v0.1
git checkout v0.1

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR