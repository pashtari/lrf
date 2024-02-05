import ignite.distributed as idist
from ignite.contrib.engines import common

def my_tensorboard(output_path="./tb", log_every_iter=100, **kwargs):
    rank = idist.get_rank()
    if rank == 0:
        evaluators = {"train": kwargs["train_evaluator"], "val": kwargs["val_evaluator"]}
        tb_logger = common.setup_tb_logging(
            output_path,
            kwargs["trainer"],
            kwargs["optimizer"],
            evaluators=evaluators,
            log_every_iters=log_every_iter,
        )