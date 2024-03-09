import os
import sys
import matplotlib.pyplot as plt
from src.utils import get_eval_results

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

os.chdir(parent_dir)
os.environ["PROJECT_ROOT"] = parent_dir


task_name_list = ["eval_interpolate_cifar10_pretrained_resnet50", "eval_interpolate_cifar10_pretrained_resnet50_low", "eval_dct_cifar10_pretrained_resnet50", "eval_dct_cifar10_pretrained_resnet50_low", "eval_patchsvd_cifar10_pretrained_resnet50"]
task_name_list_abv = ["Interpolate", "Interpolate-low", "DCT", "DCT-low", "patchSVD"]
x, y = {}, {}

original_size = 32
for (task_name, task_name_abv) in zip(task_name_list, task_name_list_abv):
    directory = f"{parent_dir}/logs/{task_name}"
    
    if task_name == "eval_patchsvd_cifar10_pretrained_resnet50":
        x, y = get_eval_results(root_dir=directory, x_par_name="model.rank", y_par_name="val_accuracy", config_rel_path=".hydra/overrides.yaml")
        patch_size = 2
        M = (original_size/patch_size) ** 2
        N = 3 * patch_size ** 2
        x = (M*N)/(x * (M+N))
    else:
        x, y = get_eval_results(root_dir=directory, x_par_name="model.new_size", y_par_name="val_accuracy", config_rel_path=".hydra/overrides.yaml")
        x = (original_size/x) ** 2

    plt.plot(x[x<50], y[x<50], label=task_name_abv)


plt.title(f"model comparison on CIFAR10")
plt.ylabel("evaluation accuracy")
plt.xlabel("compression ratio")
plt.legend()
plt.grid()
plt.savefig(
    "experiments/compression_methods_comparison_CIFAR10.pdf",
    format="pdf",
    dpi=600,
)
plt.show()



