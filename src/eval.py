import hydra
import torch
import rootutils  # I prefer to avoid such an uncommon depondency!!

from omegaconf import DictConfig
from ignite.engine import (
    Events,
    create_supervised_evaluator,
)
from ignite.handlers import Checkpoint
from ignite.contrib.handlers import TensorboardLogger

rootutils.set_root(
    path=rootutils.find_root(search_from=__file__), pythonpath=True, cwd=True
)


@hydra.main(version_base=None, config_path="../models/resnet/configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    device = cfg.device
    model = hydra.utils.instantiate(cfg.model).to(device)
    checkpoint_fp = "../checkpoints" + cfg.checkpoint
    checkpoint = torch.load(checkpoint_fp, map_location=device)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

    val_loader = hydra.utils.instantiate(cfg.val_loader)
    val_metrics = {k: hydra.utils.instantiate(v) for k, v in cfg.val_metrics.items()}
    evaluator = create_supervised_evaluator(model, val_metrics, device)

    @evaluator.on(Events.COMPLETED)
    def log_test_results(engine):
        metrics = engine.state.metrics
        printed_metrics = " - ".join(f"{k}: {v:.2f}" for k, v in metrics.items())
        print(f"Test Results - {printed_metrics}")

    tb_logger = TensorboardLogger(log_dir="tb-test-logger")
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.COMPLETED,
        tag="test",
        metric_names="all",
    )

    evaluator.run(val_loader)
    tb_logger.close()


if __name__ == "__main__":
    main()
