from task4feedback.ml.algorithms import *
import hydra 

def create_optimizer(cfg: DictConfig):
    if cfg.optimizer is not None:
        return hydra.utils.instantiate(cfg.optimizer)
    else:
        return None

def create_lr_scheduler(cfg: DictConfig):
    if cfg.lr_scheduler is None or cfg.lr_scheduler._target_ is None:
        return None
    lr_sched = hydra.utils.instantiate(cfg.lr_scheduler)
    return lr_sched