#! /bin/bash

#SBATCH --account=lp_inspiremed
#SBATCH --job-name=clic2024
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --time=4:00:00

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd ${VSC_DATA}/projects/lrf/experiments

python comparison/eval.py --data=clic2024
