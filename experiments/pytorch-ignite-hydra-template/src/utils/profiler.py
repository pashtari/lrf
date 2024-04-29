from ignite.engine import Events
from ignite.handlers import HandlersTimeProfiler
import ignite.distributed as idist


def profiler(objects):
    rank = idist.get_rank()
    if rank == 0:
        trainer = objects["trainer"]
        time_profiler = HandlersTimeProfiler()
        time_profiler.attach(trainer)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_intermediate_results():
            time_profiler.print_results(time_profiler.get_results())
