#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd ${VSC_DATA}/projects/lsvd/src

TASK_NAME=dct_cifar100_com

# Train
torchrun train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar10 metric=cifar10 model=dct_model_cifar model.num_classes=100 model.domain=compressed

CKPT_PATH=$(cat ../.temp/${TASK_NAME}.txt)

# Evals -- comressed domain
for size in 12 16 20 24 28 32; do
    torchrun eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=cifar10 metric=cifar10 model=dct_model_cifar model.num_classes=100 model.domain=compressed model.new_size=$size
    echo "new_size=$size done."
done

rm ../.temp/${TASK_NAME}.txt

