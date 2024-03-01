import os
import glob
from pathlib import Path
from typing import Union, Sequence, Callable
from functools import partial
from contextlib import contextmanager
import re

import numpy as np
import torch
import torchvision


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


def get_pretrained_resnet_weights(name):
    class_name, attr_name = name.split(".")
    return getattr(getattr(torchvision.models.resnet, class_name), attr_name)


def get_eval_results(
    root_dir,
    filename=None,
    extension=".log",
    x_par_name="model.new_size",
    y_par_name="val_accuracy",
    config_rel_path=".hydra/overrides.config",
):
    filename = Path(root_dir).stem if filename == None else filename

    pattern = os.path.join(root_dir, "**", f"{filename}*{extension}")
    files = glob.glob(pattern, recursive=True)

    x = [None for f in files]
    y = [None for f in files]

    for i, file_path in enumerate(files):
        file_dir = os.path.dirname(file_path)
        config_path = os.path.join(file_dir, config_rel_path)

        with open(config_path, "r") as config_file:
            config_string = config_file.read()
            match = re.search(
                rf"{x_par_name}\s*=\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?",
                config_string,
            )
            if match:
                x[i] = float(match.group(1))

        with open(file_path, "r") as log_file:
            log_string = log_file.read()
            match = re.search(
                rf"{y_par_name}\s*:\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", log_string
            )
            if match:
                y[i] = float(match.group(1))

    return np.array(x, dtype=float), np.array(y, dtype=float)
