#! /bin/bash

######## Log in to HPC ########
ssh -L 6006:127.0.0.1:6006 vsc36284@login.hpc.kuleuven.be

######## Install miniconda ########
cd $VSC_DATA
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $VSC_DATA/miniconda3

######## Create a conda environment ########
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
conda create -n deepenv python
source activate deepenv

######## Install PyTorch ########
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

######### Download ImageNet ########
# Method 1
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

pip install kaggle

cd $VSC_SCRATCH

export KAGGLE_USERNAME="pooya1989"
export KAGGLE_KEY="3f84f2b0dd27886e92aa19882bdb1678"

kaggle competitions download -c imagenet-object-localization-challenge
python -c "import zipfile; zipfile.ZipFile('imagenet-object-localization-challenge.zip', 'r').extractall('ImageNet')"


# Method 2 (easier)
cd $VSC_SCRATCH
wget -P ImageNet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
wget -P ImageNet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
wget -P ImageNet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate


######## Install requirements ########
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv

cd $VSC_DATA/projects/lrf/experiments
pip install -r requirements.txt

cd $VSC_DATA/projects/lrf/experiments/imagenet_classification
pip install -r requirements.txt

######## Submit jobs ########
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate deepenv
cd $VSC_DATA/projects/lrf/experiments

sbatch hpc/kodak.slurm
sbatch hpc/clic2024.slurm

sbatch hpc/ablation_bounds.slurm
sbatch hpc/ablation_colorspace.slurm
sbatch hpc/ablation_numiters.slurm
sbatch hpc/ablation_patchsize.slurm

sbatch hpc/imagenet_jpeg.slurm
sbatch hpc/imagenet_svd.slurm
sbatch hpc/imagenet_imf.slurm

