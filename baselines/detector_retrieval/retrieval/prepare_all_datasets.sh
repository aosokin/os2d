SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CIRTORCH_PATH=${SCRIPT_DIR}/cnnimageretrieval-pytorch
OS2D_PATH=${SCRIPT_DIR}/../../..
export PYTHONPATH=${OS2D_PATH}:${CIRTORCH_PATH}:${PYTHONPATH}

python prepare_dataset_retrieval.py --dataset-train grozi-train --dataset-train-scale 1280 --dataset-val grozi-val-new-cl --dataset-val-scale 1280 --datasets-test grozi-val-old-cl grozi-val-new-cl --datasets-test-scale 1280
python prepare_dataset_retrieval.py --dataset-train grozi-train --dataset-train-scale 1280 --dataset-val grozi-val-new-cl --dataset-val-scale 1280 --datasets-test grozi-val-old-cl grozi-val-new-cl --datasets-test-scale 1280 --num-random-crops-per-image 10

python prepare_dataset_retrieval.py --dataset-train instre-s1-train --dataset-train-scale 700 --dataset-val instre-s1-val --dataset-val-scale 700 --datasets-test instre-s1-val instre-s1-test --datasets-test-scale 700
python prepare_dataset_retrieval.py --dataset-train instre-s1-train --dataset-train-scale 700 --dataset-val instre-s1-val --dataset-val-scale 700 --datasets-test instre-s1-val instre-s1-test --datasets-test-scale 700 --num-random-crops-per-image 10

python prepare_dataset_retrieval.py --dataset-train instre-s1-train --dataset-train-scale 700 --dataset-val instre-s1-val --dataset-val-scale 700 --datasets-test instre-s1-val instre-s1-test --datasets-test-scale 700
python prepare_dataset_retrieval.py --dataset-train instre-s1-train --dataset-train-scale 700 --dataset-val instre-s1-val --dataset-val-scale 700 --datasets-test instre-s1-val instre-s1-test --datasets-test-scale 700 --num-random-crops-per-image 10
