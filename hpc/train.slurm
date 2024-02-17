#!/bin/bash
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
conda activate deepenv

cd $VSC_DATA/projects/lsvd/src
# python train.py dist.backend=null dist.nproc_per_node=null dist.nnodes=null data=cifar10 model=interpolate_model_cifar10 metric=cifar10
torchrun train.py dist.backend=null dist.nproc_per_node=null dist.nnodes=null data=cifar10 model=interpolate_model_cifar10 metric=cifar10
