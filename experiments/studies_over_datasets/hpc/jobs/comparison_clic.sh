#!/bin/bash -l

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/experiments/studies_over_datasets

TASK_NAME=comparison_clic

python compression_comparison_over_dataset.py --experiment_name=${TASK_NAME} --data_dir "data/clic/clic2024_test_image" --selected_methods "JPEG" "SVD" "IMF"
echo "comparison over Clic done."
