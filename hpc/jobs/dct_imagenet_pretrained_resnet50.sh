#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/src

TASK_NAME=dct_imagenet_pretrained_resnet50

# Evals -- decomressed domain
for size in $(seq 2 1 36); do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=imagenet metric=imagenet model=dct_model_imagenet model.net.__target__=torchvision.models.resnet50 +model.net.weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 model.num_classes=1000 model.rescale=true model.new_size=$size
    echo "new_size=$size done."
done