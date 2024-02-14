import subprocess

# Define the name of your Conda environment and the script you want to run
conda_env_name = "deepenv"
script_to_run = "train.py distributed.backend=nccl distributed.nproc_per_node=2 distributed.nnodes=1 data=cifar10 model=interpolate_model model.net._target_=models.resnet_cifar.ResNet50 metric=cifar10"

# Command to activate Conda environment and run your script
command = f"conda run -n {conda_env_name} python {script_to_run}"

# Execute the command
process = subprocess.run(
    command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# Print the stdout and stderr
print("STDOUT:", process.stdout.decode())
print("STDERR:", process.stderr.decode())
