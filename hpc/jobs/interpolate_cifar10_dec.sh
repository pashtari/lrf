#!/bin/bash


export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd ${VSC_DATA}/projects/lsvd/src

TASK_NAME=interpolate_cifar10_dec

# Train
torchrun train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar10 metric=cifar10 model=interpolate_model_cifar model.num_classes=10 model.rescale=true

CKPT_PATH=$(cat ../.temp/${TASK_NAME}.txt)


# Evals 
for size in 12 16 20 24 28 32; do
    torchrun eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=cifar10 metric=cifar10 handler.checkpoint.load_from=$CKPT_PATH model=interpolate_model_cifar model.num_classes=10 model.rescale=true model.new_size=$size
    echo "new_size=$size done."
done

rm ../.temp/${TASK_NAME}.txt




