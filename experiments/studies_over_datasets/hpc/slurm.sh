#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd $VSC_DATA/projects/lsvd/experiments/studies_over_datasets/hpc/jobs

# Ablation study on Kodak
sbatch --account=lp_inspiremed --job-name=abl_patchsize --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=12:00:00 ablation_patchsize.sh
sbatch --account=lp_inspiremed --job-name=abl_iternum --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=12:00:00 ablation_iternum.sh
sbatch --account=lp_inspiremed --job-name=abl_bounds --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=12:00:00 ablation_bounds.sh
sbatch --account=lp_inspiremed --job-name=abl_colorspace --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=12:00:00 ablation_colorspace.sh
sbatch --account=lp_inspiremed --job-name=abl_patchification --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=12:00:00 ablation_patchification_svd.sh

# Comparison over datasets
sbatch --account=lp_inspiremed --job-name=cmp_kodak --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=24:00:00 comparison_kodak.sh
sbatch --account=lp_inspiremed --job-name=cmp_clic --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=24:00:00 comparison_clic.sh
sbatch --account=lp_inspiremed --job-name=cmp_imgnet --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=72:00:00 comparison_imagenet.sh

# Comparison of classification performance over a subset of Imagenet
sbatch --account=lp_inspiremed --job-name=clsi_IMF --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:1 --time=24:00:00 --partition=gpu_v100 IMF_imagenet_classification.sh
sbatch --account=lp_inspiremed --job-name=clsi_IMF --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:1 --time=24:00:00 --partition=gpu_v100 SVD_imagenet_classification.sh
sbatch --account=lp_inspiremed --job-name=clsi_IMF --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:1 --time=24:00:00 --partition=gpu_v100 JPEG_imagenet_classification.sh