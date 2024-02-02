import torch
import ignite
import ignite.distributed as idist
from datetime import datetime
from pathlib import Path


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
    output_path = cfg.logger.basic_logger_output_path
    folder_name = f"{cfg.model}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
    output_path = Path(output_path) / folder_name
    if not output_path.exists():
        output_path.mkdir(parents=True)
    cfg.logger.basic_logger_output_path = output_path.as_posix()
    logger.info(f"Output path: {cfg.logger.basic_logger_output_path}")
