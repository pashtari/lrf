#!/bin/bash

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd $VSC_DATA/projects/lsvd/experiments/hpc

sbatch --account=lp_inspiremed --job-name=imagenet_compression_comparison --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=9 --time=12:00:00 imagenet.sh
