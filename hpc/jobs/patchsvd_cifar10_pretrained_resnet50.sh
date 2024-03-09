#!/bin/bash 

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/src

TASK_NAME=patchsvd_cifar10_pretrained_resnet50

CKPT_PATH="${ROOT_DIR}/logs/pretrained_checkpoints/train_resnet50_cifar10/checkpoints/CIFAR10_V1.pt"

# Evals -- decomressed domain
for rank in $(seq 1 1 12); do
    python eval.py dist.backend=null dist.nproc_per_node=null dist.nnodes=null task_name=eval_${TASK_NAME} data=cifar10 metric=cifar model=patchsvd_model_cifar +model.net.pretrained_weights_path=${CKPT_PATH} +model.spatial_dims=2 +model.conv1_kernel_size=3 +model.conv1_stride=1 +model.no_maxpool=true model.num_classes=10 model.patch_size=2 model.domain=decompressed model.rank=$rank

    echo "rank=$rank done."
done

