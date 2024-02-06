from pathlib import Path
import torch
import ignite
import ignite.distributed as idist
from ignite.utils import setup_logger


def pylogger(objects, output_path="./", **kwargs):
    logger = setup_logger(**kwargs)

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"\nIgnite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"\nGPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"\nCUDA version: {torch.version.cuda}")
        logger.info(f"\nCUDNN version: {cudnn.version()}")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\n\tbackend: {idist.backend()}")
        logger.info(f"\n\ttworld size: {idist.get_world_size()}")

    rank = idist.get_rank()
    if rank == 0:
        output_path = Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True)
