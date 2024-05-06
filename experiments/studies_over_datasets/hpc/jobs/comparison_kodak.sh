#!/bin/bash -l

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/experiments/studies_over_datasets

TASK_NAME=comparison_kodak

python compression_comparison_over_dataset.py --experiment_name=${TASK_NAME} --data_dir "data/kodak" --selected_methods "JPEG" "SVD" "IMF"
echo "comparison over Kodak done."
