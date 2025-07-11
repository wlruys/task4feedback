from task4feedback.interface import SimulatorFactory, SimulatorInput, create_graph_spec 
import gmsh 
from typing import Callable
from .graph import GraphBuilder
import hydra
from omegaconf import DictConfig, OmegaConf

from torchrl.envs import (
    TransformedEnv,
    Compose,
    InitTracker,
    StepCounter,
    TrajCounter,
    ObservationNorm,
)
from torchrl.modules import LSTMModule
from typing import Optional
from dataclasses import dataclass

def create_system(cfg: DictConfig):
    system = hydra.utils.instantiate(cfg.system)
    return system


def create_conditions(cfg: DictConfig):
    transition_conditions = hydra.utils.instantiate(cfg.runtime)
    return transition_conditions


def create_runtime_reward(cfg: DictConfig):
    runtime_env_t = hydra.utils.instantiate(cfg.reward)
    return runtime_env_t


def create_observer_factory(cfg: DictConfig):
    graph_spec = hydra.utils.instantiate(cfg.feature.observer.spec)
    observer_factory = hydra.utils.instantiate(cfg.feature.observer)
    return observer_factory, graph_spec


@dataclass
class NormalizationDetails:
    task_norm: dict


def make_env(
    graph_builder: GraphBuilder,
    cfg: DictConfig,
    lstm: Optional[LSTMModule] = None,
    normalization: Optional[NormalizationDetails] = None,
):
    gmsh.initialize()

    s = create_system(cfg)
    graph = graph_builder.function()

    d = graph.get_blocks()
    m = graph

    transition_conditions = create_conditions(cfg)
    runtime_env_t = create_runtime_reward(cfg)
    observer_factory, graph_spec = create_observer_factory(cfg)
    input = SimulatorInput(m, d, s, transition_conditions=transition_conditions)

    env = runtime_env_t(
        SimulatorFactory(input, graph_spec, observer_factory),
        device="cpu",
        change_priority=cfg.graph.env.change_priority,
        change_locations=cfg.graph.env.change_locations,
        seed=cfg.graph.env.seed,
        max_samples_per_iter=len(graph)+1
        if cfg.algorithm.rollout_steps == 0
        else cfg.algorithm.rollout_steps+1,
    )
    env = TransformedEnv(env, StepCounter())
    env.append_transform(TrajCounter())
    env.append_transform(InitTracker())
        
    if lstm is not None:
        print("Adding LSTM module to environment", flush=True)
        env.append_transform(lstm.make_tensordict_primer())

    if normalization is None:
        task_norm_transform = ObservationNorm(
            in_keys=[("observation", "nodes", "tasks", "attr")],
            eps=1e-4,
            standard_normal=True,
        )
        env.append_transform(task_norm_transform)
        if isinstance(env.transform, Compose):
            for transform in env.transform:
                if isinstance(transform, ObservationNorm) and not transform.initialized:
                    transform.init_stats(
                        num_iter=1000, key=("observation", "nodes", "tasks", "attr")
                    )
        new_norm = NormalizationDetails(task_norm=task_norm_transform.state_dict())
    else:
        task_norm_transform = ObservationNorm(
            in_keys=[("observation", "nodes", "tasks", "attr")],
            eps=1e-4,
            standard_normal=True,
            loc=normalization.task_norm["loc"],
            scale=normalization.task_norm["scale"],
        )
        task_norm_transform.load_state_dict(normalization.task_norm)

        env.append_transform(task_norm_transform)
        new_norm = None

    if new_norm is not None:
        return env, new_norm

    else:
        return env
