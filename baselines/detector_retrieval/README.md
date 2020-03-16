## Experiments with the detector-retrieval baseline
The detector-retrieval baseline consists of the two models that need to be trained independently.

### Installation
See [INSTALL.md](./INSTALL.md)

### Preparations
```bash
conda activate os2d
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
```

### Train the detector
The main training script is `detector/train_detector.py`. The training runs can be launched with `detector/launcher_train_detector.py`.
```bash
cd $OS2D_ROOT/baselines/detector_retrieval/detector

#6 jobs training class agnostic detectors, 2 jobs for training class-aware detectors
python experiments/launcher_train_detector.py
```
Trained model will be stored in `$OS2D_ROOT/baselines/detector_retrieval/detector/output`.


### Train the retrieval system

#### Prepare dataset
One needs to convert datasets to the format of [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch). This csript should do this:
```bash
cd $OS2D_ROOT/baselines/detector_retrieval/retrieval

bash prepare_all_datasets.sh
```

#### Run training
```bash
cd $OS2D_ROOT/baselines/detector_retrieval/retrieval

# Grozi
python experiments/launcher_grozi.py
# INSTRE
python experiments/launcher_instre.py
```

### Run evaluation
```bash
cd $OS2D_ROOT/baselines/detector_retrieval

# Grozi
python experiments/launcher_grozi_eval.py
# INSTRE
python experiments/launcher_instre_eval.py
```

### View results
```bash
cd $OS2D_ROOT/baselines/detector_retrieval

# Create tables
# Table 3
python experiments/launcher_grozi_eval_collect.py
# Table 4
python experiments/launcher_instre_eval_collect.py
```
