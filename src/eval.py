import hydra
import torch
import rootutils

from omegaconf import DictConfig
from ignite.engine import (
    Events,
    create_supervised_evaluator,
)
from ignite.handlers import Checkpoint
from ignite.contrib.handlers import TensorboardLogger

rootutils.set_root(path=rootutils.find_root(search_from=__file__), pythonpath=True, cwd=True)

@hydra.main(
    version_base=None, config_path="../models/svdresnet/configs", config_name="eval"
)
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
        print(
            f"Test Results - Avg top-1 accuracy: {metrics['top_1_accuracy']:.2f} Avg top-5 accuracy: {metrics['top_5_accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
        )

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
