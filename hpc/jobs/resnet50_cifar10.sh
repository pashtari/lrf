#!/bin/bash

#SBATCH --account=lp_inspiremed
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=2
#SBATCH --time=12:00:00
#SBATCH --job-name=resnet50_cifar10

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/src

TASK_NAME=resnet50_cifar10

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar10 metric=cifar model=resnet50_model_cifar model.num_classes=10
