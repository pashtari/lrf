#!/bin/bash


export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv


ROOT_DIR="${VSC_DATA}/projects/lsvd"

cd ${ROOT_DIR}/src

TASK_NAME=interpolate_cifar100_dec

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar100 metric=cifar model=interpolate_model_cifar model.num_classes=100 model.rescale=true

CKPT_PATH=$(cat ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt)


# Evals 
for size in $(seq 2 1 36); do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=cifar100 metric=cifar handler.checkpoint.load_from=${CKPT_PATH} model=interpolate_model_cifar model.num_classes=100 model.rescale=true model.new_size=$size
    echo "new_size=$size done."
done

rm ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt




