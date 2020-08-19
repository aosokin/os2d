SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CIRTORCH_PATH=${SCRIPT_DIR}/cnnimageretrieval-pytorch
OS2D_PATH=${SCRIPT_DIR}/../../..
export PYTHONPATH=${OS2D_PATH}:${CIRTORCH_PATH}:${PYTHONPATH}

python prepare_dataset_retrieval.py --dataset-train imagenet-repmet-train --dataset-train-scale 600 --dataset-val imagenet-repmet-val-5000 --dataset-val-scale 600 --datasets-test imagenet-repmet-val-5000 --datasets-test-scale 600 --num-queries-image-to-image 10

