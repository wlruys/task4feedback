from .definitions import *
from omegaconf import DictConfig
from task4feedback.ml.algorithms import *
import hydra 

def create_optimizer(cfg: DictConfig):
    if cfg.optimizer.algorithm is None:
        optim = None
    else:
        optim =  hydra.utils.instantiate(cfg.optimizer.algorithm)

    if cfg.optimizer.schedule is None or cfg.optimizer.schedule._target_ is None:
        lr_sched = None
    else:
        lr_sched = hydra.utils.instantiate(cfg.optimizer.schedule)
    return optim, lr_sched