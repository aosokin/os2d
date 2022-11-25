# !/bin/bash

cd $OS2D_ROOT/baselines/detector_retrieval/retrieval

git clone https://github.com/filipradenovic/cnnimageretrieval-pytorch.git cnnimageretrieval-pytorch
cd cnnimageretrieval-pytorch
git checkout v1.1
cd ..