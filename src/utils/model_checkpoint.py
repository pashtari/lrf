from ignite.engine import Events
from ignite.handlers import Checkpoint, global_step_from_engine

class myCheckpoint:
    def __init__(self, object_dict, save_handler = "./outputs/checkpoints", n_saved=2, filename_prefix="best", score_name="top1_accuracy"):
        self.object_dict = object_dict
        self.save_handler = save_handler
        self.n_saved = n_saved
        self.filename_prefix = filename_prefix
        self.score_name = score_name

        self.instantiate()

    def instantiate(self):
        trainer = self.object_dict["trainer"]
        val_evaluator = self.object_dict["val_evaluator"]
        model = self.object_dict["model"]
        optimizer = self.object_dict["optimizer"]
        lr_scheduler = self.object_dict["lr_scheduler"]

        to_save = {
            "trainer": trainer,
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

        best_model_handler = Checkpoint(
            to_save,  
            global_step_transform=global_step_from_engine(trainer),
            score_function=Checkpoint.get_default_score_fn(self.score_name),
            save_handler=self.save_handler,
            n_saved=self.n_saved,
            filename_prefix=self.filename_prefix, 
            )
        val_evaluator.add_event_handler(Events.COMPLETED, best_model_handler)