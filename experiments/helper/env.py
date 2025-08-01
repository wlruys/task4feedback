from task4feedback.interface import SimulatorFactory, SimulatorInput, create_graph_spec 
from task4feedback.interface import TaskNoise 
from typing import Callable
from .graph import GraphBuilder
import hydra
from omegaconf import DictConfig, OmegaConf
from task4feedback.ml.env import RuntimeEnv
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
    if (
        hasattr(cfg.feature.observer, "width")
        and hasattr(cfg.feature.observer, "prev_frames")
        and hasattr(cfg.feature.observer, "batched")
    ):
        if cfg.feature.observer.batched:
            graph_spec.max_candidates = cfg.graph.config.n**2
        observer_factory = hydra.utils.instantiate(
            cfg.feature.observer,
            spec=graph_spec,
            width=cfg.graph.config.n,
            prev_frames=cfg.feature.observer.prev_frames,
            batched=cfg.feature.observer.batched,
        )
    else:
        observer_factory = hydra.utils.instantiate(cfg.feature.observer)
    return observer_factory, graph_spec


def create_task_noise(cfg: DictConfig, static_graph):
    task_noise = hydra.utils.instantiate(cfg.noise)

    print("Task noise configuration:", OmegaConf.to_yaml(cfg.noise))

    if task_noise is None:
        task_noise = TaskNoise(tasks=static_graph)
    else:
        task_noise = task_noise(tasks=static_graph)

    return task_noise

@dataclass
class NormalizationDetails:
    task_norm: dict


def make_env(
    graph_builder: GraphBuilder,
    cfg: DictConfig,
    lstm: Optional[LSTMModule] = None,
    normalization: Optional[NormalizationDetails] = None,
    eval=False,
):
    from task4feedback.graphs.mesh import gmsh, initialize_gmsh, finalize_gmsh
    gmsh.initialize()

    s = create_system(cfg)
    graph = graph_builder.function()

    d = graph.get_blocks()
    m = graph

    transition_conditions = create_conditions(cfg)
    if not eval:
        runtime_env_t = create_runtime_reward(cfg)
    else:
        runtime_env_t = RuntimeEnv
    observer_factory, graph_spec = create_observer_factory(cfg)

    print("CONFIG", cfg)
    task_noise = create_task_noise(cfg, graph.static_graph)

    print("NOISE")
    print(task_noise)
    
    input = SimulatorInput(m, d, s, transition_conditions=transition_conditions, task_noise=task_noise)



    env = runtime_env_t(
        SimulatorFactory(input, graph_spec, observer_factory),
        device="cpu",
        change_priority=cfg.graph.env.change_priority,
        change_locations=cfg.graph.env.change_locations,
        seed=cfg.graph.env.seed,
        max_samples_per_iter=(
            len(graph) + 1
            if cfg.algorithm.rollout_steps == 0
            else cfg.algorithm.rollout_steps + 1
        ),
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
                        num_iter=env.size() * 10,
                        key=("observation", "nodes", "tasks", "attr"),
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
