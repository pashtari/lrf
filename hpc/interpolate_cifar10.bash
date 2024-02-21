# #!/bin/bash
# export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
# source activate deepenv

# cd $VSC_DATA/projects/lsvd/src

cd ../src
TASK_NAME=train_interpolate_cifar10
# Train
torchrun train.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=$TASK_NAME data=cifar10 metric=cifar10 model=interpolate_model_cifar10 model.num_classes=10 model.rescale=false

CKPT_PATH=$(cat ../.temp/${TASK_NAME}.txt)

torchrun eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_interpolate_cifar10 data=cifar10 metric=cifar10 handler.checkpoint.load_from=$CKPT_PATH model=interpolate_model_cifar10 model.num_classes=10 model.rescale=false model.new_size=12

# Evals
for size in 12 16 20 24 28 32; do
    torchrun eval.py dist.backend=nccl dist.nproc_per_node=2 dist.nnodes=1 task_name=eval_interpolate_cifar10 data=cifar10 metric=cifar10 handler.checkpoint.load_from=$CKPT_PATH model=interpolate_model_cifar10 model.num_classes=10 model.rescale=false model.new_size=$size
    echo "new_size=$size done."
done

rm ../.temp/${TASK_NAME}.txt




