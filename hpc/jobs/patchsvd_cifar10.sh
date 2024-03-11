#!/bin/bash

#SBATCH --account=lp_inspiremed
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=2
#SBATCH --time=9:00:00
#SBATCH --job-name=patchsvd_cifar10


export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/src

TASK_NAME=patchsvd_cifar10

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar10 metric=cifar model=patchsvd_model_cifar model.num_classes=10 model.patch_size=2 model.domain=compressed model.rank=-1

CKPT_PATH=$(cat ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt)

# Evals
for rank in $(seq 1 1 12); do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=cifar10 metric=cifar handler.checkpoint.load_from=${CKPT_PATH} model=patchsvd_model_cifar model.num_classes=10 model.patch_size=2 model.domain=compressed model.rank=$rank 
    echo "rank=$size done."
done

rm ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt
