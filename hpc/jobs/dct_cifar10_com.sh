#!/bin/bash -l

# SBATCH --account=lp_inspiremed
# SBATCH --clusters=genius
# SBATCH --partition=gpu_v100
# SBATCH --nodes=1
# SBATCH --ntasks=8
# SBATCH --cpus-per-task=1
# SBATCH --gpus-per-node=2
# SBATCH --time=1:00:00
# SBATCH --job-name=dct_cifar10_com

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

ROOT_DIR="${VSC_DATA}/projects/lsvd"
cd ${ROOT_DIR}/src

TASK_NAME=dct_cifar10_com

# Train
python train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=train_${TASK_NAME} data=cifar10 metric=cifar model=dct_model_cifar model.num_classes=10 model.domain=compressed

CKPT_PATH=$(cat ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt)

# Evals
for size in $(seq 2 1 36); do
    python eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_${TASK_NAME} data=cifar10 metric=cifar handler.checkpoint.load_from=${CKPT_PATH} model=dct_model_cifar model.num_classes=10 model.domain=compressed model.new_size=$size
    echo "new_size=$size done."
done

rm ${ROOT_DIR}/.temp/train_${TASK_NAME}.txt

