#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"

cd ${ROOT_DIR}/src

# ############
# TASK_NAME=dct_cifar10_dec

# # Train
# python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar10 metric=cifar model=dct_model_cifar model.num_classes=10 model.domain=decompressed model.pad=false

# CKPT_PATH=$(cat ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt)

# # Evals
# for size in $(seq 2 1 36); do
#     python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=cifar10 metric=cifar handler.checkpoint.load_from=${CKPT_PATH} model=dct_model_cifar model.num_classes=10 model.domain=decompressed model.new_size=$size model.pad=false
#     echo "new_size=$size done."
# done

# rm ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt


############
TASK_NAME=dct_cifar10_decpad

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar10 metric=cifar handler.checkpoint.load_from="${ROOT_DIR}/logs/train_dct_cifar10_decpad/runs/2024-02-27_14-20-53/checkpoints/checkpoint_356.pt" model=dct_model_cifar model.num_classes=10 model.domain=decompressed model.pad=true

CKPT_PATH=$(cat ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt)

# Evals
for size in $(seq 2 1 36); do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=cifar10 metric=cifar handler.checkpoint.load_from=${CKPT_PATH} model=dct_model_cifar model.num_classes=10 model.domain=decompressed model.new_size=$size model.pad=true
    echo "new_size=$size done."
done

rm ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt






