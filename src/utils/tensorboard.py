import ignite.distributed as idist
from ignite.contrib.engines import common


def tensorboard(objects, output_path="./", **kwargs):
    rank = idist.get_rank()

    if rank == 0:
        trainer = objects["trainer"]
        optimizer = objects["optimizer"]
        evaluators = {
            "train": objects["train_evaluator"],
            "val": objects["val_evaluator"],
        }

        tb_logger = common.setup_tb_logging(
            output_path,
            trainer=trainer,
            optimizers=optimizer,
            evaluators=evaluators,
            **kwargs
        )

    return tb_logger
