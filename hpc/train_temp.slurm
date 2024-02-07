#!/bin/bash
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
init conda
conda activate deepenv

cd $VSC_DATA/scripts/lsvd/src
python train.py distributed.backend=nccl distributed.nproc_per_node=2 distributed.nnodes=1 data=cifar10 model=resnet metric=cifar10