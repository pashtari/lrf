#!/bin/bash
ssh -L 6006:127.0.0.1:6006 vsc36284@login.hpc.kuleuven.be
cd $VSC_DATA
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $VSC_DATA/miniconda3
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
conda create -n deepenv python
source activate deepenv
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
cd $VSC_DATA/projects


