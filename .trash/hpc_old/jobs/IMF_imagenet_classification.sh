#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd/experiments"
cd ${ROOT_DIR}/pytorch-ignite-hydra-template/src

TASK_NAME=IMF_imagenet_classification

# Evals -- decomressed domain
for quality in $(seq 0 1 18); do
    python eval.py dist.backend=null dist.nproc_per_node=null dist.nnodes=null task_name=${TASK_NAME} data=imagenet data.val_set.root=${VSC_SCRATCH}/imagenet data.val_transform.transforms.1._target_=src.data.imf_transformer +data.val_transform.transforms.1.quality=$quality +data.val_transform.transforms.1.num_iters=5 metric=[imagenet,bpp] model=trans_resnet50 +model.weights._target_=src.utils.get_pretrained_resnet_weights +model.weights.name="ResNet50_Weights.IMAGENET1K_V2"

    echo "quality=$quality done."
done