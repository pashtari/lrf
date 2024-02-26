#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv


ROOT_DIR="${VSC_DATA}/projects/lsvd"

cd ${ROOT_DIR}/src

TASK_NAME=dct_tinyimagenet_sgl

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=tinyimagenet metric=imagenet model=dct_model_tinyimagenet model.domain=compressed model.new_size=64 model.pad=false

CKPT_PATH=$(cat ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt)

# Evals
for size in $(seq 2 2 72); do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=tinyimagenet metric=imagenet handler.checkpoint.load_from=${CKPT_PATH} model=dct_model_tinyimagenet model.domain=compressed model.new_size=$size model.pad=false
    echo "new_size=$size done."
done

rm ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt




