#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"

cd ${ROOT_DIR}/experiments

python compression_comparison_over_dataset_imagenet.py