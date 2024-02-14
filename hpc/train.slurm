#!/bin/bash
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
init conda
conda activate deepenv

cd $VSC_DATA/scripts/lsvd/src
python train.py distributed.backend=null distributed.nproc_per_node=null distributed.nnodes=nulle data=cifar10 model=interpolate_model model.net._target_=src.models.resnet_cifar.ResNet50 metric=cifar10

python train.py data=cifar10 model=interpolate_model model.net._target_=src.models.resnet_cifar.ResNet50 metric=cifar10

# torchrun --backend=nccl --nproc_per_node=2 --nnodes=1 train.py data=${data} model=${model} metric=${metric}