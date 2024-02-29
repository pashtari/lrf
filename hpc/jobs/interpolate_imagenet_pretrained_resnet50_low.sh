#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/src

TASK_NAME=interpolate_imagenet_pretrained_resnet50_low

# Evals -- decomressed domain
for size in $(seq 32 8 224); do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=imagenet data.val_set.root=${VSC_SCRATCH}/ImageNet/ILSVRC/Data/CLS-LOC/val metric=imagenet model=interpolate_model_imagenet model.net._target_=torchvision.models.resnet50 +model.weights._target_=src.utils.get_pretrained_resnet_weights +model.weights.name="ResNet50_Weights.IMAGENET1K_V2" model.num_classes=1000 model.rescale=false model.new_size=$size
    echo "new_size=$size done."
done