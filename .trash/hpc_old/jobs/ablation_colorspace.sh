#!/bin/bash -l

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/experiments/studies_over_datasets

TASK_NAME=ablation_colorspace

python compression_comparison_over_dataset.py --data_dir=${VSC_SCRATCH}/kodak --experiment_name=${TASK_NAME} --color_space="RGB"
echo "colorspace RGB done."

python compression_comparison_over_dataset.py --data_dir=${VSC_SCRATCH}/kodak --experiment_name=${TASK_NAME} --color_space="YCbCr"
echo "colorspace YCbCr done."
