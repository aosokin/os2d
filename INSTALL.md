## Installation of OS2D
```bash
# create a separate conda env
conda create -n os2d python=3.7

# activate it 
conda activate os2d

# this installs the right pip and dependencies for the fresh python
conda install ipython pip

# get pytorch and torchvision
conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.0 -c pytorch

# more dependencies
conda install tqdm=4.42.1 pandas=1.0.1 matplotlib=3.1.3 pyyaml=5.3 scipy=1.4.1
conda install -c conda-forge yacs=0.1.6

# to monitor GPU usage on a cluster
pip install gpustat==0.6.0

# to view train logs in visdom
pip install visdom==0.1.8.9
```

## Installation of the baselines
1. To install the detector-retrieval baselines, see [instructions](baselines/detector_retrieval/INSTALL.md).
2. To install the CoAE baselines, see [instructions](baselines/CoAE/INSTALL.md).
