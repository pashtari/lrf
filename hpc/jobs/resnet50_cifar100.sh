#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/src

TASK_NAME=resnet50_cifar100

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar100 metric=cifar model=resnet50_model_cifar model.num_classes=100
