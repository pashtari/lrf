from pathlib import Path
import torch
import ignite
import ignite.distributed as idist
from ignite.utils import setup_logger


def pylogger(objects, output_path="./", **kwargs):
    logger = setup_logger(**kwargs)

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDNN version: {cudnn.version()}")

    if idist.get_world_size() > 1:
        logger.info("Distributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\ttworld size: {idist.get_world_size()}")
