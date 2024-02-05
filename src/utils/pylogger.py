import torch
import ignite
import ignite.distributed as idist
from datetime import datetime
from pathlib import Path
from ignite.utils import setup_logger


def my_pylogger(**kwargs):

    if kwargs["cfg"] is not None:
        cfg = kwargs["cfg"]
        logger = setup_logger(cfg.task_name)
        log_basic_info(logger, cfg)

        rank = idist.get_rank()
        if rank == 0:
            setup_basic_logger(logger, cfg)

def log_basic_info(logger, cfg):
    logger.info(cfg.task_name)
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in cfg.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def setup_basic_logger(logger, cfg):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = cfg.model._target_.split(".")[-1]
    output_path = cfg.logger_output_path
    folder_name = f"{model_name}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
    output_path = Path(output_path) / folder_name
    if not output_path.exists():
        output_path.mkdir(parents=True)
    cfg.logger_output_path = output_path.as_posix()
    logger.info(f"Output path: {cfg.logger_output_path}")


def log_metrics(logger, trainer, elapsed, tag, metrics):
    metrics_output = " - ".join(f"{k}: {v:.2f}" for k, v in metrics.items())
    e = trainer.state.epoch
    n = trainer.state.max_epochs
    logger.info(
        f"\n{tag} results --- epoch[{e}/{n}] - time (sec): {elapsed:.2f} -   {metrics_output}"
    )
