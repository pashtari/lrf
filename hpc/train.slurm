#!/bin/bash

#SBATCH --account=lp_inspiremed
#SBATCH --mail-type=NONE

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
conda activate deepenv

cd $VSC_DATA/scripts/lsvd/src
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py run --backend=nccl data=${data} model=${model}