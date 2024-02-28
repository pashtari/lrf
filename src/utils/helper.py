import os
import glob
import yaml 
import numpy as np

from pathlib import Path
from typing import Union, Sequence, Callable
from contextlib import contextmanager
from functools import partial

import torch


@contextmanager
def null_context():
    yield


def wrap_class(obj: Union[Sequence, Callable]) -> Callable:
    assert isinstance(
        obj, (Sequence, Callable)
    ), f"{obj} should be a sequence or callable."

    if isinstance(obj, Sequence) and isinstance(obj[0], Callable):
        args = []
        kwargs = {}
        for i, a in enumerate(obj):
            if i == 0:
                callable_obj = a
            elif isinstance(a, Sequence):
                args.extend(a)
            elif isinstance(a, dict):
                kwargs.update(a)

        return partial(callable_obj, *args, **kwargs)
    else:
        return obj


def get_dtype(name: str):
    return getattr(torch, name)



def get_eval_results(root_dir, filename=None, extension=".log", x_par_name="model.new_size", y_par_name="val_accuracy", yaml_rel_path=".hydra/overrides.yaml"):
    filename = Path(root_dir).stem if filename==None else filename

    pattern = os.path.join(root_dir, '**', f'{filename}*{extension}')
    files = glob.glob(pattern, recursive=True)
    
    file_num = len(files)
    plot_data = np.zeros((2, file_num))

    for i, file_path in enumerate(files):
        file_dir = os.path.dirname(file_path)
        yaml_path = os.path.join(file_dir, yaml_rel_path)

        with open(yaml_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            for line in yaml_data:
                if x_par_name in line:
                    new_size = int(line.split('=')[1])
                    plot_data[0, i] = new_size

        with open(file_path, 'r') as log_file:
            for line in log_file:
                if y_par_name in line:
                    val_accuracy = float(line.split(':')[-1].strip())
                    plot_data[1, i] = val_accuracy

    return plot_data[0], plot_data[1]
