#!/bin/bash -l

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/experiments/studies_over_datasets

TASK_NAME=ablation_patchsize

patch_sizes=(4 8 16)
# Train
for size in ${patch_sizes[@]}; do
    python compression_comparison_over_dataset.py --experiment_name=${TASK_NAME} --patch_size=${size}
    echo "patch_size=$size done."
done