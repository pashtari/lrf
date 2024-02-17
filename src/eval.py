import sys
import os
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig
from ignite.utils import manual_seed
import ignite.distributed as idist
from ignite.engine import (
    create_supervised_evaluator,
)
from ignite.handlers import ModelCheckpoint


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

os.chdir(parent_dir)
os.environ["PROJECT_ROOT"] = parent_dir


def evaluating(local_rank, cfg) -> None:
    device = idist.device()
    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)

    model = hydra.utils.instantiate(cfg.model)
    
    ckpt_path = Path(cfg.checkpoint_path)
    assert ckpt_path.exists(), f"Checkpoint '{ckpt_path.as_posix()}' is not found"
    ckpt = torch.load(ckpt_path.as_posix(), map_location="cpu")
    ModelCheckpoint.load_objects(to_load={"model": model}, checkpoint=ckpt)

    val_loader = hydra.utils.instantiate(cfg.data.val_loader)
    metrics = {k: hydra.utils.instantiate(v) for k, v in cfg.metric.items()}

    evaluator = create_supervised_evaluator(
        model,
        metrics,
        device,
    )

    objects = {"evaluator": evaluator}

    ###### loggers ######
    for value in cfg.logger.values():
        hydra.utils.instantiate(value)(objects=objects)

    evaluator.run(val_loader, max_epochs=1)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:

    for k, v in cfg.path.items():
        cfg.path[k] = v

    with idist.Parallel(**cfg.dist) as parallel:
        parallel.run(evaluating, cfg)


if __name__ == "__main__":
    main()
