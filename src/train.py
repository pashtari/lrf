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
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import common
from torch.utils.data.distributed import DistributedSampler
from ignite.handlers import Checkpoint


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
        Events.EPOCH_COMPLETED(every=cfg.trainer.val_every_epochs) | Events.COMPLETED
    )
    def run_validation(engine):
        train_evaluator.run(train_loader)
        val_evaluator.run(val_loader)

    # progress bar for iters
    iter_pbar = ProgressBar(persist=True)
    iter_pbar.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=cfg.trainer.log_every_iters),
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.trainer.log_every_iters))
    def log_train_pbar(engine):
        train_metrics = train_evaluator.state.metrics
        val_metrics = val_evaluator.state.metrics
        metrics = [
            f"train_{k}: {train_metrics[k]:.2f} - val_{k}: {val_metrics[k]:.2f}"
            for k in train_metrics
        ]
        metrics = " - ".join(metrics)
        iter_pbar.log_message(f"{metrics} - loss: {trainer.state.output:.2f}")

    # progress bar for epochs
    epoch_pbar = ProgressBar(persist=True)
    epoch_pbar.attach(
        trainer,
        metric_names="all",
        event_name=Events.EPOCH_STARTED,
        closing_event_name=Events.COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_eval_pbar(engine):
        train_metrics = train_evaluator.state.metrics
        val_metrics = val_evaluator.state.metrics
        metrics = [
            f"train_{k}: {train_metrics[k]:.2f} - val_{k}: {val_metrics[k]:.2f}"
            for k in train_metrics
        ]
        metrics = " - ".join(metrics)
        epoch_pbar.log_message(metrics)

    if isinstance(train_loader.sampler, DistributedSampler):

        @trainer.on(Events.EPOCH_STARTED)
        def distrib_set_epoch(engine: Engine) -> None:
            train_loader.sampler.set_epoch(engine.state.epoch - 1)

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

    # best_model_handler = Checkpoint(
    #     {"model": model},  # ??
    #     global_step_transform=global_step_from_engine(trainer),
    #     score_function=Checkpoint.get_default_score_fn("Accuracy"),  # ??
    #     **cfg.logger.model_checkpoint,
    # )
    # val_evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    trainer.run(train_loader, max_epochs=cfg.trainer.num_epochs)

    # if rank == 0:
    #     tb_logger.close()
    print("All is fine!")


if __name__ == "__main__":
    main()
