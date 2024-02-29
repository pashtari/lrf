#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd $VSC_DATA/projects/lsvd/hpc/jobs


##### Pooya
sbatch --account=lp_inspiremed --job-name=interpolate_cifar10_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 interpolate_cifar10_com.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=interpolate_cifar10_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 interpolate_cifar10_dec.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=interpolate_cifar10_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 interpolate_cifar10_sgl.sh
sleep 1

sbatch --account=lp_inspiremed --job-name=dct_cifar10_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 dct_cifar10_com.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=dct_cifar10_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 dct_cifar10_dec.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=dct_cifar10_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 dct_cifar10_sgl.sh
sleep 1


##### Pourya
sbatch --account=lp_inspiremed --job-name=interpolate_cifar100_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 interpolate_cifar100_com.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=interpolate_cifar100_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 interpolate_cifar100_dec.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=interpolate_cifar100_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 interpolate_cifar100_sgl.sh
sleep 1

sbatch --account=lp_inspiremed --job-name=dct_cifar100_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 dct_cifar100_com.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=dct_cifar100_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 dct_cifar100_dec.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=dct_cifar100_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=12:00:00 --partition=gpu_p100 dct_cifar100_sgl.sh
sleep 1


##### Amir
sbatch --account=lp_inspiremed --job-name=interpolate_tinyimagenet_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:4 --time=72:00:00 --partition=gpu_v100 interpolate_tinyimagenet_com.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=interpolate_tinyimagenet_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:4 --time=72:00:00 --partition=gpu_v100 interpolate_tinyimagenet_dec.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=interpolate_tinyimagenet_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:4 --time=72:00:00 --partition=gpu_v100 interpolate_tinyimagenet_sgl.sh
sleep 1

sbatch --account=lp_inspiremed --job-name=dct_tinyimagenet_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:4 --time=72:00:00 --partition=gpu_v100 dct_tinyimagenet_com.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=dct_tinyimagenet_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:4 --time=72:00:00 --partition=gpu_v100 dct_tinyimagenet_dec.sh
sleep 1
sbatch --account=lp_inspiremed --job-name=dct_tinyimagenet_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:4 --time=72:00:00 --partition=gpu_v100 dct_tinyimagenet_sgl.sh
sleep 1


##### pretrained resnet50
sbatch --account=lp_inspiremed --job-name=interpolate_imagenet_pretrained_resnet50 --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=06:00:00 --partition=gpu_p100 interpolate_imagenet_pretrained_resnet50.sh
sbatch --account=lp_inspiremed --job-name=interpolate_imagenet_pretrained_resnet50_low --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=06:00:00 --partition=gpu_p100 interpolate_imagenet_pretrained_resnet50_low.sh
sbatch --account=lp_inspiremed --job-name=dct_imagenet_pretrained_resnet50 --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=06:00:00 --partition=gpu_p100 dct_imagenet_pretrained_resnet50.sh
sbatch --account=lp_inspiremed --job-name=dct_imagenet_pretrained_resnet50_low --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=06:00:00 --partition=gpu_p100 dct_imagenet_pretrained_resnet50_low.sh
sbatch --account=lp_inspiremed --job-name=patchsvd_imagenet_pretrained_resnet50 --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=06:00:00 --partition=gpu_p100 patchsvd_imagenet_pretrained_resnet50.sh

##### training vanilla models
sbatch --account=lp_inspiremed --job-name=resnet50_cifar10 --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=11:00:00 --partition=gpu_p100 resnet50_cifar10.sh
sbatch --account=lp_inspiremed --job-name=resnet50_cifar100 --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --gres=gpu:2 --time=13:00:00 --partition=gpu_p100 resnet50_cifar100.sh


#### for tensorboard
cd $VSC_DATA/projects/lsvd/logs
tensorboard --logdir=.  --port=6006



