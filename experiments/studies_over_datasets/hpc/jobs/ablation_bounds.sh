#!/bin/bash -l

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/experiments/studies_over_datasets

TASK_NAME=ablation_bounds

bounds=(16 32 64 128)
# Train
for size in ${bounds[@]}; do
    python compression_comparison_over_dataset.py --data_dir=${VSC_SCRATCH}/kodak --experiment_name=${TASK_NAME} --bounds=${size}
    echo "bounds=$size done."
done