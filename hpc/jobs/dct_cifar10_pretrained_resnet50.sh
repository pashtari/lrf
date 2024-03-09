#!/bin/bash 

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/src

TASK_NAME=dct_cifar10_pretrained_resnet50

CKPT_PATH="${ROOT_DIR}/logs/pretrained_checkpoints/train_resnet50_cifar10/checkpoints/CIFAR10_V1.pt"

# Evals -- decomressed domain
for size in $(seq 2 2 36); do
    python eval.py dist.backend=null dist.nproc_per_node=null dist.nnodes=null task_name=eval_${TASK_NAME} data=cifar10 metric=cifar model=dct_model_cifar +model.net.pretrained_weights_path=${CKPT_PATH} +model.spatial_dims=2 +model.conv1_kernel_size=3 +model.conv1_stride=1 +model.no_maxpool=true model.num_classes=10 model.domain=decompressed model.pad=true model.new_size=$size

    echo "new_size=$size done."
done

