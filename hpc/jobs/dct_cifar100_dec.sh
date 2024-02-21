#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv
cd ${VSC_DATA}/projects/lsvd/src

############
TASK_NAME=dct_cifar100_dec

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar100 metric=cifar10 model=dct_model_cifar model.num_classes=100 model.domain=decompressed model.pad=false

CKPT_PATH=$(cat ../.temp/${TASK_NAME}.txt)

# Evals
for size in 12 16 20 24 28 32; do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=cifar100 metric=cifar10 model=dct_model_cifar model.num_classes=100 model.domain=decompressed model.new_size=$size model.pad=false
    echo "new_size=$size done."
done

rm ../.temp/${TASK_NAME}.txt


############
TASK_NAME=dct_cifar100_decpad

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar100 metric=cifar10 model=dct_model_cifar model.num_classes=100 model.domain=decompressed model.pad=true

CKPT_PATH=$(cat ../.temp/${TASK_NAME}.txt)

# Evals
for size in 12 16 20 24 28 32; do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=cifar100 metric=cifar10 model=dct_model_cifar model.num_classes=100 model.domain=decompressed model.new_size=$size model.pad=true
    echo "new_size=$size done."
done

rm ../.temp/${TASK_NAME}.txt






