#!/bin/bash

#SBATCH --account=lp_inspiremed
#SBATCH --mail-type=NONE

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
conda activate deepenv

cd $VSC_DATA/scripts/lsvd/src
torchrun --backend=nccl --nproc_per_node=2 --nnodes=1 train.py data=${data} model=${model} metric=${metric}


torchrun train.py data=cifar10 model=resnet metric=cifar10

python train.py data=cifar10 model=resnet metric=cifar10