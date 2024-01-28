from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
import hydra

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, ImageNet
from torchvision.transforms import Compose, Normalize, ToTensor, RandomResizedCrop, RandomHorizontalFlip

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    device = cfg.device
    log_interval = cfg.log_interval
    model = hydra.utils.instantiate(cfg.model).to(device)
    max_epochs = cfg.max_epochs
    n_saved_model = cfg.n_saved_model

    # train_loader = hydra.utils.instantiate(cfg.train_loader)
    # val_loader = hydra.utils.instantiate(cfg.val_loader)

    optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model.parameters())
    loss = hydra.utils.instantiate(cfg.loss)
    val_metrics = {
        "accuracy": hydra.utils.instantiate(cfg.val_metrics_accuracy), 
        "loss": hydra.utils.instantiate(cfg.val_metrics_loss)
        }

    trainer = create_supervised_trainer(model, optimizer, loss, device)
    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    def score_function(engine):
        return engine.state.metrics["accuracy"]

    model_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=n_saved_model,
        filename_prefix="best",
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer),
    )
    
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    tb_logger = TensorboardLogger(log_dir="tb-logger")
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    
    trainer.run(train_loader, max_epochs=max_epochs)
    tb_logger.close()

if __name__ == "__main__":
    main()
