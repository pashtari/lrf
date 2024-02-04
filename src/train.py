from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
from ignite.utils import manual_seed
import ignite.distributed as idist
from ignite.engine import (
    Events,
    Engine,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Loss

from ignite.contrib.engines import common
from torch.utils.data.distributed import DistributedSampler
from ignite.handlers import Checkpoint, global_step_from_engine


def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    return checkpoint


def create_trainer(model, optimizer, loss, lr_scheduler, cfg):
    device = idist.device()
    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        device=device,
        non_blocking=cfg.trainer.non_blocking,
        amp_mode="amp" if cfg.trainer.amp else None,
        scaler=cfg.trainer.amp,
    )

    if cfg.trainer.resume_from is not None:
        to_load = {
            "trainer": trainer,
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
        checkpoint = load_checkpoint(cfg.trainer.resume_from)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

    return trainer


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    device = idist.device()
    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)

    # # PyLogger
    # logger = setup_logger(cfg.task_name)
    # log_basic_info(logger, cfg)

    # if rank == 0:
    #     setup_basic_logger(logger, cfg)

    model = idist.auto_model(hydra.utils.instantiate(cfg.model))

    train_loader = hydra.utils.instantiate(cfg.data.train_loader)
    val_loader = hydra.utils.instantiate(cfg.data.val_loader)

    optimizer = idist.auto_optim(
        hydra.utils.instantiate(cfg.trainer.optimizer)(params=model.parameters())
    )
    loss = hydra.utils.instantiate(cfg.trainer.loss).to(device)
    lr_scheduler = hydra.utils.instantiate(cfg.trainer.lr_scheduler)(optimizer=optimizer)

    trainer = create_trainer(model, optimizer, loss, lr_scheduler, cfg)

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


    object_dict = {"model": model,
                   "trainer": trainer,
                   "train_evaluator": train_evaluator,
                   "val_evaluator": val_evaluator,
                   "optimizer": optimizer,
                   "lr_scheduler": lr_scheduler,
                   }
    
    for key, value in cfg.handler.items():
        handler = hydra.utils.instantiate(value)(object_dict=object_dict)
        object_dict[f"{key}"] = handler



    ###### handlers ######
    # if rank == 0:
    #     evaluators = {"train": train_evaluator, "val": val_evaluator}
    #     tb_logger = common.setup_tb_logging(
    #         cfg.logger.output_path,
    #         trainer,
    #         optimizer,
    #         evaluators=evaluators,
    #         log_every_iters=cfg.logger.log_every_iter,
    #     )  # ??

    trainer.run(train_loader, max_epochs=cfg.trainer.num_epochs)

    # if rank == 0:
    #     tb_logger.close()
    print("All is fine!")


if __name__ == "__main__":
    main()
