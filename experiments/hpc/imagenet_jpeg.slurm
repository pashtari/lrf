#!/bin/bash -l

#SBATCH --account=lp_inspiremed
#SBATCH --job-name=imagenet_jpeg
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --partition=gpu_p100

# Activate the conda environment
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

# Set the project directory
cd ${VSC_DATA}/projects/lrf/experiments/imagenet_classification/src

# Evaluate
for quality in $(seq 0 1 75); do
    python eval.py dist.backend=null dist.nproc_per_node=null dist.nnodes=null task_name=imagenet_jpeg data=imagenet data.val_set.root=${VSC_SCRATCH}/ImageNet data.val_transform.transforms.1._target_=src.data.JPEGTransform +data.val_transform.transforms.1.quality=$quality metric=[imagenet,bpp] model=resnet +model.weights._target_=src.utils.get_pretrained_resnet_weights +model.weights.name="ResNet50_Weights.IMAGENET1K_V2"
    echo "quality=$quality done."
done
