import torch
import ignite
import hydra
import ignite.distributed as idist

from pathlib import Path
from omegaconf import DictConfig, omegaconf
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.utils import manual_seed, setup_logger
from ignite.contrib.engines import common
from ignite.handlers import Checkpoint
from ignite.metrics import Loss
from datetime import datetime

from utils.pylogger import log_basic_info, setup_basic_logger


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    device = idist.device()
    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)

    # PyLogger
    logger = setup_logger(cfg.task_name)
    log_basic_info(logger, cfg)

    if rank == 0:
        setup_basic_logger(logger, cfg)

    model = idist.auto_model(hydra.utils.instantiate(cfg.model))

    train_loader = hydra.utils.instantiate(cfg.data.train_loader)
    val_loader = hydra.utils.instantiate(cfg.data.val_loader)

    optimizer = idist.auto_optim(
        hydra.utils.instantiate(cfg.trainer.optimizer)(params=model.parameters())
    )
    loss = hydra.utils.instantiate(cfg.trainer.loss).to(device)
    lr_scheduler = hydra.utils.instantiate(cfg.trainer.lr_scheduler)(optimizer=optimizer)

    val_metrics = {k: hydra.utils.instantiate(v) for k, v in cfg.metric.items()}
    val_metrics["loss"] = Loss(loss)

    trainer = create_trainer(
        model, optimizer, loss, lr_scheduler, train_loader.sampler, cfg, logger
    )

    train_evaluator = create_supervised_evaluator(
        model,
        val_metrics,
        device,
        non_blocking=cfg.trainer.non_blocking,
        amp_mode=cfg.trainer.amp,
    )
    val_evaluator = create_supervised_evaluator(
        model,
        val_metrics,
        device,
        non_blocking=cfg.trainer.non_blocking,
        amp_mode=cfg.trainer.amp,
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.handler.log_every_epochs))  # ??
    def run_validation(engine):
        epoch = trainer.state.epoch  # ?? engine.state?
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "train", state.metrics)
        state = val_evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "val", state.metrics)

    if rank == 0:
        evaluators = {"train": train_evaluator, "val": val_evaluator}
        tb_logger = common.setup_tb_logging(
            cfg.logger.output_path,
            trainer,
            optimizer,
            evaluators=evaluators,
            log_every_iters=cfg.logger.log_every_iter,
        )  # ??

    best_model_handler = Checkpoint(
        {"model": model},  # ??
        global_step_transform=global_step_from_engine(trainer),
        score_function=Checkpoint.get_default_score_fn("Accuracy"),  # ??
        **cfg.logger.model_checkpoint,
    )
    val_evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    trainer.run(train_loader, max_epochs=cfg.num_epochs)

    if rank == 0:
        tb_logger.close()


def create_trainer(model, optimizer, loss, lr_scheduler, train_sampler, cfg, logger):
    device = idist.device()
    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        device=device,
        non_blocking=True,
        output_transform=lambda x, y, y_pred, loss: {"batch loss": loss.item()},
        amp_mode=cfg.trainer.amp,
        scaler=cfg.trainer.amp,
    )
    trainer.logger = logger

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    metric_names = [
        "batch loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=cfg.handler.save_every_iters,
        save_handler=cfg.handler.save_handler,  # ??
        lr_scheduler=lr_scheduler,
        output_names=metric_names if cfg.handler.save_every_iters > 0 else None,  # ??
        with_pbars=cfg.handler.with_pbars,
        clear_cuda_cache=cfg.handler.clear_cuda_cache,
    )

    if cfg.trainer.resume_from is not None:
        checkpoint = load_checkpoint(cfg.trainer.resume_from)
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    return checkpoint


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


if __name__ == "__main__":
    main()
