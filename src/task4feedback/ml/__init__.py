from .models import *
from .util import *
from dataclasses import dataclass
from typing import Callable


@dataclass
class PPOConfig:
    states_per_collection: int = 1000
    minibatch_size: int = 250
    num_epochs_per_collection: int = 4
    num_collections: int = 1000


def run_ppo_cleanrl(
    actor_critic_model: nn.Module, make_env: Callable[[], EnvBase], config: PPOConfig
):
    pass
