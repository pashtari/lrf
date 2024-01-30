import rootutils
import hydra 

from omegaconf import DictConfig
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

# rootutils.setup_root(__file__, indicator="../", pythonpath=True)

@hydra.main(
    version_base=None, config_path="./models/svdresnet/configs", config_name="train"
)
def main(cfg: DictConfig) -> None:
    device = cfg.device
    model = hydra.utils.instantiate(cfg.model).to(device)

    train_loader = hydra.utils.instantiate(cfg.train_loader)
    val_loader = hydra.utils.instantiate(cfg.val_loader)

    device = cfg.device
    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model.parameters())
    loss = hydra.utils.instantiate(cfg.loss)

    trainer = create_supervised_trainer(model, optimizer, loss, device)

    val_metrics = {k: hydra.utils.instantiate(v) for k, v in cfg.val_metrics.items()}

    train_evaluator = create_supervised_evaluator(model, val_metrics, device)
    val_evaluator = create_supervised_evaluator(model, val_metrics, device)

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.log_interval))
    def log_training_loss(engine):
        print(
            f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
        )

    def score_function(engine):
        return engine.state.metrics["accuracy"]

    model_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=cfg.num_saved_model,
        filename_prefix="best",
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer),
    )
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    tb_logger = TensorboardLogger(log_dir="tb-logger")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=cfg.log_interval),
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

    trainer.run(train_loader, max_epochs=cfg.num_epochs)
    tb_logger.close()


if __name__ == "__main__":
    main()
