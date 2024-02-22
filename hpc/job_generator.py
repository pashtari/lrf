import glob
from pathlib import Path

for job_path in glob.glob("./hpc/jobs/*.sh"):
    job_name = Path(job_path).stem
    print(
        f"sbatch --account=lp_inspiremed --job-name={job_name} --clusters=genius --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --gres=gpu:2 --time=30:00 --partition=gpu_v100 {job_name}.sh"
    )
