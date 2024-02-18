#!/bin/bash
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
conda activate deepenv

cd $VSC_DATA/projects/lsvd/src
# python train.py dist.backend=null dist.nproc_per_node=null dist.nnodes=null data=cifar10 model=interpolate_model_cifar10 metric=cifar10
torchrun train.py dist.backend=null dist.nproc_per_node=null dist.nnodes=null data=cifar100 model=interpolate_model_cifar10 metric=cifar10

python train.py data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.rescale=true
python eval.py data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.rescale=true model.new_size=12, 16, 20, 24, 28, 32 checkpoint_path="/logs/"
python eval.py data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.rescale=false model.new_size=12, 16, 20, 24, 28, 32 checkpoint_path="/logs/"

python train.py data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.rescale=false
python eval.py data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.rescale=true model.new_size=12, 16, 20, 24, 28, 32 checkpoint_path="/logs/"
python eval.py data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.rescale=false model.new_size=12, 16, 20, 24, 28, 32 checkpoint_path="/logs/"

python train.py data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.new_size=32
python eval.py data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.rescale=true model.new_size=12, 16, 20, 24, 28, 32 checkpoint_path="/logs/"
python eval.py data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.rescale=false model.new_size=12, 16, 20, 24, 28, 32 checkpoint_path="/logs/"