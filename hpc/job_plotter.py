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


task_name = "eval_dct_cifar100_sgl"
directory = f"{parent_dir}/logs/{task_name}"
x, y = get_eval_results(root_dir=directory, x_par_name="model.new_size", y_par_name="val_accuracy", yaml_rel_path=".hydra/overrides.yaml")

plt.plot(x, y)
plt.title(f"{task_name}")
plt.ylabel("evaluation accuracy")
plt.xlabel("compressed image sizes")
plt.grid()
plt.show()



