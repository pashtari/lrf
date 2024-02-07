#!/bin/bash

#SBATCH --account=lp_inspiremed
#SBATCH --mail-type=NONE

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd $VSC_DATA/scripts/lsvd/src
# torchrun --backend=nccl --nproc_per_node=2 --nnodes=1 train.py data=${data} model=${model} metric=${metric}


# torchrun train.py data=cifar10 model=resnet metric=cifar10

python train.py distributed.backend=nccl distributed.nproc_per_node=2 distributed.nnodes=1 data=cifar10 model=resnet metric=cifar10