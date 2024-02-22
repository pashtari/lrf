#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd $VSC_DATA/projects/lsvd/hpc/jobs

sbatch --account=lp_inspiremed --job-name=interpolate_cifar100_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 interpolate_cifar100_sgl.sh
sbatch --account=lp_inspiremed --job-name=dct_cifar10_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 dct_cifar10_com.sh
sbatch --account=lp_inspiremed --job-name=interpolate_cifar10_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 interpolate_cifar10_sgl.sh
sbatch --account=lp_inspiremed --job-name=interpolate_cifar100_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 interpolate_cifar100_com.sh
sbatch --account=lp_inspiremed --job-name=interpolate_cifar10_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 interpolate_cifar10_com.sh
sbatch --account=lp_inspiremed --job-name=dct_cifar100_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 dct_cifar100_dec.sh
sbatch --account=lp_inspiremed --job-name=dct_cifar10_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 dct_cifar10_sgl.sh
sbatch --account=lp_inspiremed --job-name=interpolate_cifar10_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 interpolate_cifar10_dec.sh
sbatch --account=lp_inspiremed --job-name=dct_cifar100_com --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 dct_cifar100_com.sh
sbatch --account=lp_inspiremed --job-name=interpolate_cifar100_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 interpolate_cifar100_dec.sh
sbatch --account=lp_inspiremed --job-name=dct_cifar10_dec --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 dct_cifar10_dec.sh
sbatch --account=lp_inspiremed --job-name=dct_cifar100_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 dct_cifar100_sgl.sh

# cd $VSC_DATA/projects/lsvd/logs

# tensorboard --logdir=.  --port=6006