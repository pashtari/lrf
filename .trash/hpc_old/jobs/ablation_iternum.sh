#!/bin/bash -l

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/experiments/studies_over_datasets

TASK_NAME=ablation_iternum

iter_nums=(0 1 2 5 10)
# Train
for num in ${iter_nums[@]}; do
    python compression_comparison_over_dataset.py --data_dir=${VSC_SCRATCH}/kodak --experiment_name=${TASK_NAME} --num_iters=${num}
    echo "num_iters=$num done."
done