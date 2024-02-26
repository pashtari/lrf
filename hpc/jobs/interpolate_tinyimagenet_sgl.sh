#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"

cd ${ROOT_DIR}/src

TASK_NAME=interpolate_tinyimagenet_sgl

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=tinyimagenet metric=imagenet model=interpolate_model_imagenet model.num_classes=200 model.new_size=64

CKPT_PATH=$(cat ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt)

# Evals -- comressed domain
for size in $(seq 2 2 72); do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=tinyimagenet metric=imagenet handler.checkpoint.load_from=${CKPT_PATH} model=interpolate_model_imagenet model.num_classes=200 model.rescale=false model.new_size=$size
    echo "new_size=$size done."
done

# Evals -- decomressed domain
for size in $(seq 2 2 72); do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=tinyimagenet metric=imagenet handler.checkpoint.load_from=${CKPT_PATH} model=interpolate_model_imagenet model.num_classes=200 model.rescale=true model.new_size=$size
    echo "new_size=$size done."
done

rm ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt




