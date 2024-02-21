import logging

import torch
import ignite
import ignite.distributed as idist
from ignite.utils import setup_logger
from ignite.engine import Events


def pylogger(objects, log_every_iters=10, **kwargs):
    logger = setup_logger(
        name="PyLogger",
        format="[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
        distributed_rank=0,
        **kwargs,
    )
    if idist.get_rank() == 0:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Ignite version: {ignite.__version__}")

        if torch.cuda.is_available():
            # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
            from torch.backends import cudnn

            logger.info(
                f"GPU device: {torch.cuda.get_device_name(idist.get_local_rank())}"
            )
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDNN version: {cudnn.version()}")

        if idist.get_world_size() > 1:
            logger.info(f"Distributed Backend: {idist.backend()}")
            logger.info(f"World size: {idist.get_world_size()}")

    if "trainer" in objects:
        trainer = objects["trainer"]
        train_evaluator = objects["train_evaluator"]
        val_evaluator = objects["val_evaluator"]

        @trainer.on(Events.ITERATION_COMPLETED(every=log_every_iters) or Events.COMPLETED)
        def log_train(engine):
            train_metrics = train_evaluator.state.metrics
            val_metrics = val_evaluator.state.metrics
            metrics = [
                f"train_{k}: {train_metrics[k]:.2f} - val_{k}: {val_metrics[k]:.2f}"
                for k in train_metrics
            ]
            metrics = " - ".join(metrics)
            logger.info(
                f"Rank[{idist.get_local_rank()}]: Epoch[{trainer.state.epoch}/{trainer.state.max_epochs}], iter[{trainer.state.iteration}] - {metrics} - loss: {trainer.state.output:.2f}"
            )

    elif "evaluator" in objects:
        ignite_logger = logging.getLogger("ignite.engine.engine.Engine")
        ignite_logger.setLevel(logging.WARN)
        evaluator = objects["evaluator"]

        @evaluator.on(Events.COMPLETED)
        def log_val(engine):
            val_metrics = engine.state.metrics
            metrics = [f"val_{k}: {val_metrics[k]:.2f}" for k in val_metrics]
            metrics = " - ".join(metrics)
            logger.info(f"Rank[{idist.get_local_rank()}]: {metrics}")
