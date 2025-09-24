from task4feedback.interface import SimulatorFactory, SimulatorInput, create_graph_spec 
from task4feedback.interface import TaskNoise 
from task4feedback.graphs.jacobi import get_length_from_config
from typing import Callable, Dict, Any, Optional, Tuple, List, Sequence
from .graph import GraphBuilder
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig 
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
import torch
from pathlib import Path
import numpy as np 

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
    graph_config = hydra.utils.instantiate(cfg.graph.config)

    if (
        hasattr(cfg.feature.observer, "width")
        and hasattr(cfg.feature.observer, "prev_frames")
        and hasattr(cfg.feature.observer, "batched")
    ):
        if cfg.feature.observer.batched:
            width = graph_config.n
            length = get_length_from_config(graph_config)
            graph_spec.max_candidates = width * length
        observer_factory = hydra.utils.instantiate(
            cfg.feature.observer,
            spec=graph_spec,
            width=width,
            length=get_length_from_config(graph_config),
            prev_frames=cfg.feature.observer.prev_frames,
            batched=cfg.feature.observer.batched,
        )
    else:
        observer_factory = hydra.utils.instantiate(cfg.feature.observer)
    return observer_factory, graph_spec


def create_task_noise(cfg: DictConfig, static_graph):
    task_noise = hydra.utils.instantiate(cfg.noise)
    if task_noise is None:
        task_noise = TaskNoise(tasks=static_graph)
    else:
        task_noise = task_noise(tasks=static_graph)

    return task_noise


@dataclass
class NormalizationDetails:
    states: Dict[str, Dict[str, Any]]

def _oc_to_py(x: Any) -> Any:
    if isinstance(x, (DictConfig, ListConfig)):
        return OmegaConf.to_container(x, resolve=True)
    return x

def _parse_norm_specs(cfg: DictConfig) -> Tuple[bool, int, List[dict]]:
    norm_cfg = getattr(cfg.feature, "normalization", None)
    if norm_cfg is None:
        return False, 0, []
    enabled = bool(_oc_to_py(getattr(norm_cfg, "enabled", True)))
    warmup = int(_oc_to_py(getattr(norm_cfg, "warmup", 1)))
    specs = _oc_to_py(getattr(norm_cfg, "specs", []))
    return enabled and bool(specs), warmup, specs

def _setup_observation_norms(
    env: TransformedEnv,
    cfg: DictConfig,
    normalization: Optional[NormalizationDetails],
) -> Optional[NormalizationDetails]:
    enabled, warmup, specs = _parse_norm_specs(cfg)
    if not enabled:
        return None

    created: Dict[str, ObservationNorm] = {}
    # Build & attach; seed shapes from saved state if available
    for spec in specs:
        name = spec["name"]
        in_keys = [tuple(k) for k in _oc_to_py(spec["in_keys"])]
        eps = float(spec.get("eps", 1e-4))
        standard_normal = bool(spec.get("standard_normal", True))
        state = normalization.states.get(name) if normalization else None

        norm = ObservationNorm(
            in_keys=in_keys,
            eps=eps,
            standard_normal=standard_normal,
            loc=(state.get("loc") if state else None),
            scale=(state.get("scale") if state else None),
        )
        env.append_transform(norm)
        created[name] = norm

    # Load or init
    to_init: List[Tuple[str, ObservationNorm, dict]] = []
    for spec in specs:
        name = spec["name"]
        state = normalization.states.get(name) if normalization else None
        if state is not None:
            try:
                print(f"Loading saved observation norm state {name}")
                created[name].load_state_dict(state)
                continue
            except Exception:
                pass  # shape mismatch -> init
        to_init.append((name, created[name], spec))

    if to_init:
        print(f"Initializing observation norms: {[n for n, _, _ in to_init]}")
        num_iter = max(1, getattr(env, "size", lambda: 1)()) * max(1, int(getattr(cfg.feature.normalization, "warmup", 1)))
        env.disable_reward()
        try:
            for name, norm, spec in to_init:
                in_keys = [tuple(k) for k in _oc_to_py(spec["in_keys"])]
                reduce_dim = tuple(spec.get("reduce_dim", (0, 1)))
                cat_dim = int(spec.get("cat_dim", 0))
                try:
                    norm.init_stats(num_iter=num_iter, key=in_keys[0], reduce_dim=reduce_dim, cat_dim=cat_dim)
                except TypeError:
                    norm.init_stats(num_iter=num_iter, key=in_keys[0])
        finally:
            env.enable_reward()
        return NormalizationDetails(states={n: t.state_dict() for n, t in created.items()})

    return None


def make_env(
    graph_builder: GraphBuilder,
    cfg: DictConfig,
    lstm: Optional[LSTMModule] = None,
    normalization: Optional[NormalizationDetails] = None,
    eval=False,
)->RuntimeEnv | tuple[RuntimeEnv, NormalizationDetails]:
    from task4feedback.graphs.mesh import gmsh, initialize_gmsh, finalize_gmsh
    gmsh.initialize()

    s = create_system(cfg)
    graph = graph_builder.function(s)

    d = graph.get_blocks()
    m = graph

    transition_conditions = create_conditions(cfg)
    if not eval:
        runtime_env_t = create_runtime_reward(cfg)
    else:
        runtime_env_t = RuntimeEnv
    observer_factory, graph_spec = create_observer_factory(cfg)

    task_noise = create_task_noise(cfg, graph.static_graph)

    if cfg.feature.observer.batched:
        assert(hasattr(graph, "nx") and hasattr(graph, "ny"))
        top_k_candidates = graph.nx * graph.ny
    else:
        top_k_candidates = 1

    input = SimulatorInput(m, d, s, transition_conditions=transition_conditions, task_noise=task_noise, top_k_candidates=top_k_candidates)

    env = runtime_env_t(
        SimulatorFactory(input, graph_spec, observer_factory),
        device="cpu",
        change_priority=cfg.graph.env.change_priority if hasattr(cfg.graph.env, "change_priority") else False,
        change_location=cfg.graph.env.change_location if hasattr(cfg.graph.env, "change_location") else False,
        change_duration=cfg.graph.env.change_duration if hasattr(cfg.graph.env, "change_duration") else False,
        change_workload=cfg.graph.env.change_workload if hasattr(cfg.graph.env, "change_workload") else False,
        seed=cfg.graph.env.seed,
        max_samples_per_iter=(
            (len(graph)//(graph.nx * graph.ny) + 1
            if cfg.algorithm.rollout_steps == 0 
            else cfg.algorithm.rollout_steps + 1) if cfg.feature.observer.batched else (len(graph) + 1
            if cfg.algorithm.rollout_steps == 0 
            else cfg.algorithm.rollout_steps + 1)
        ),
    )
    env = TransformedEnv(env, StepCounter())
    env.append_transform(TrajCounter())
    env.append_transform(InitTracker())

    if lstm is not None:
        print("Adding LSTM module to environment", flush=True)
        env.append_transform(lstm.make_tensordict_primer())

    if normalization != False:
        new_norm = _setup_observation_norms(env, cfg, normalization)
    else:
        new_norm = None  

    return (env, new_norm) if new_norm is not None else env 