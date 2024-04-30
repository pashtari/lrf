#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd $VSC_DATA/projects/experiments/pytorch-ignite-hydra-template/hpc/jobs


##### Train vanilla models
sbatch resnet50_cifar10.sh
sbatch resnet50_cifar100.sh

#### Tensorboard
cd $VSC_DATA/projects/experiments/pytorch-ignite-hydra-template/logs
tensorboard --logdir=.  --port=6006



