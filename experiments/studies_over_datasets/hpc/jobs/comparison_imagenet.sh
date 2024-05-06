#!/bin/bash -l

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/experiments/studies_over_datasets

TASK_NAME=comparison_imagenet

python compression_comparison_over_imagenet.py --data_dir=${VSC_SCRATCH}/imagenet --experiment_name=${TASK_NAME} --selected_methods "JPEG" "SVD" "IMF"
echo "comparison over Imagenet done."
