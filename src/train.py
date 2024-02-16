import sys
import os
from typing import cast

import hydra
from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LRScheduler as PyTorchLRScheduler
from ignite.utils import manual_seed
import ignite.distributed as idist
from ignite.engine import (
    Events,
    Engine,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Loss


def training(local_rank, cfg) -> None:
    device = idist.device()
    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)

    model = idist.auto_model(hydra.utils.instantiate(cfg.model))

    train_loader = hydra.utils.instantiate(cfg.data.train_loader)
    val_loader = hydra.utils.instantiate(cfg.data.val_loader)

    optimizer = idist.auto_optim(
        hydra.utils.instantiate(cfg.trainer.optimizer)(params=model.parameters())
    )
    lr_scheduler = hydra.utils.instantiate(cfg.trainer.lr_scheduler)(optimizer=optimizer)
    loss = hydra.utils.instantiate(cfg.trainer.loss).to(device)

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        device=device,
        non_blocking=cfg.trainer.non_blocking,
        amp_mode="amp" if cfg.trainer.amp else None,
        scaler=cfg.trainer.amp,
    )
    metrics = {k: hydra.utils.instantiate(v) for k, v in cfg.metric.items()}
    metrics["loss"] = Loss(loss)

    train_evaluator = create_supervised_evaluator(
        model,
        metrics,
        device,
        non_blocking=cfg.trainer.non_blocking,
        amp_mode="amp" if cfg.trainer.amp else None,
    )
    val_evaluator = create_supervised_evaluator(
        model,
        metrics,
        device,
        non_blocking=cfg.trainer.non_blocking,
        amp_mode="amp" if cfg.trainer.amp else None,
    )

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, PyTorchLRScheduler):
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED,
                lambda engine: cast(PyTorchLRScheduler, lr_scheduler).step(),
            )
        else:
            trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)

    @trainer.on(
        Events.EPOCH_COMPLETED(every=cfg.trainer.val_every_epochs) or Events.COMPLETED
    )
    def run_validation(engine):
        train_evaluator.run(train_loader)
        val_evaluator.run(val_loader)

    if isinstance(train_loader.sampler, DistributedSampler):

        @trainer.on(Events.EPOCH_STARTED)
        def distrib_set_epoch(engine: Engine) -> None:
            train_loader.sampler.set_epoch(engine.state.epoch - 1)

    objects = {
        "model": model,
        "trainer": trainer,
        "train_evaluator": train_evaluator,
        "val_evaluator": val_evaluator,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }

    ###### handlers ######
    for value in cfg.handler.values():
        hydra.utils.instantiate(value)(objects=objects)

    ###### loggers ######
    for value in cfg.logger.values():
        hydra.utils.instantiate(value)(objects=objects)

    trainer.run(train_loader, max_epochs=cfg.trainer.max_epochs)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:

    for k, v in cfg.path.items():
        cfg.path[k] = v

    with idist.Parallel(**cfg.distributed) as parallel:
        parallel.run(training, cfg)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    os.chdir(parent_dir)
    os.environ["PROJECT_ROOT"] = parent_dir
    main()
