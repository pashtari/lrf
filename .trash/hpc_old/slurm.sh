#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd $VSC_DATA/projects/lsvd/experiments/studies_over_datasets/hpc/jobs

# Ablation study on Kodak
sbatch --account=lp_inspiremed --job-name=abl_patchsize --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=6:00:00 ablation_patchsize.sh
sbatch --account=lp_inspiremed --job-name=abl_iternum --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=6:00:00 ablation_iternum.sh
sbatch --account=lp_inspiremed --job-name=abl_bounds --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=6:00:00 ablation_bounds.sh
sbatch --account=lp_inspiremed --job-name=abl_colorspace --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=4:00:00 ablation_colorspace.sh

# Comparison over datasets
sbatch --account=lp_inspiremed --job-name=cmp_kodak --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=12:00:00 comparison_kodak.sh
sbatch --account=lp_inspiremed --job-name=cmp_clic --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=12:00:00 comparison_clic.sh

# Comparison of classification performance over Imagenet
sbatch --account=lp_inspiremed --job-name=clsi_IMF --clusters=genius --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --gres=gpu:1 --time=60:00:00 --partition=gpu_p100 IMF_imagenet_classification.sh
sbatch --account=lp_inspiremed --job-name=clsi_SVD --clusters=genius --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --gres=gpu:1 --time=36:00:00 --partition=gpu_p100 SVD_imagenet_classification.sh
sbatch --account=lp_inspiremed --job-name=clsi_JPG --clusters=genius --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --gres=gpu:1 --time=36:00:00 --partition=gpu_p100 JPEG_imagenet_classification.sh