cd $VSC_DATA/projects/lsvd/hpc/jobs

sbatch --account=lp_inspiremed --job-name=interpolate_cifar10_sgl --clusters=genius --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --time=15:00 --partition=gpu_p100 interpolate_cifar10_sgl.sh

cd $VSC_DATA/projects/lsvd/logs

tensorboard --logdir=.  --port=6006