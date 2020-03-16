## Experiments with the CoAE baseline

### Installation
See [INSTALL.md](./INSTALL.md)

### Setup datasets
```bash
cd ${OS2D_ROOT}/baselines/CoAE/data
ln -s ${OS2D_ROOT}/data/grozi grozi
ln -s ${OS2D_ROOT}/data/dairy dairy
ln -s ${OS2D_ROOT}/data/paste paste
ln -s ${OS2D_ROOT}/data/instre instre
```

### Preparations
```bash
cd ${OS2D_ROOT}/baselines/CoAE
conda activate os2d
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
```

### Train models
```bash
# training runs
python experiments/launcher_coae_grozi_train.py
python experiments/launcher_coae_instre_train.py
```

### View training logs
```bash
# Convert text logs to the OS2D binary format
python experiments/parse_logs_to_pkl.py --log_path output/grozi
python experiments/parse_logs_to_pkl.py --log_path output/instre

# View in Visdom
python ../../os2d/utils/plot_visdom.py --log_path output/grozi
python ../../os2d/utils/plot_visdom.py --log_path output/instre
```

### Run evaluation
```bash
# evaluation of the best models (selection of the best model on the validation set was done manually)
python experiments/launcher_coae_grozi_eval.py
python experiments/launcher_coae_instre_eval.py
```

### View results
```bash
# Create tables
# Table 3
python experiments/launcher_coae_grozi_eval_collect.py
# Table 4
python experiments/launcher_coae_instre_eval_collect.py
```
