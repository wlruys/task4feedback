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

    grid_override = bool(cfg.feature.observer.get("grid_override", False))
    graph_override = bool(cfg.feature.observer.get("graph_override", False))
    use_grid_observer = grid_override or graph_override

    if use_grid_observer:
        width = graph_config.n
        length = graph_config.n
        graph_spec.max_candidates = width * length

        observer_factory = hydra.utils.instantiate(
            cfg.feature.observer,
            spec=graph_spec,
            width=width,
            length=length,
            prev_frames=cfg.feature.observer.prev_frames,
        )
    else:
        graph_spec.max_candidates = cfg.feature.observer.get("n_candidates", 1)
        print(f"Setting max candidates to {graph_spec.max_candidates}")
        observer_factory = hydra.utils.instantiate(cfg.feature.observer, spec=graph_spec)
        observer_factory.set_graph_spec(graph_spec)
        print(observer_factory.graph_spec)
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


def _norm_cache_enabled(cfg: DictConfig) -> bool:
    norm_cfg = getattr(cfg.feature, "normalization", None)
    if norm_cfg is None:
        return False
    # Default is disabled: always recompute normalization stats per env instance.
    return bool(_oc_to_py(getattr(norm_cfg, "cache", False)))


def _setup_observation_norms(
    env: TransformedEnv,
    cfg: DictConfig,
    normalization: Optional[NormalizationDetails],
) -> Optional[NormalizationDetails]:
    enabled, warmup, specs = _parse_norm_specs(cfg)
    if not enabled:
        return None

    use_cache = _norm_cache_enabled(cfg)
    created: Dict[str, ObservationNorm] = {}
    # Build & attach; seed shapes from saved state if available
    for spec in specs:
        name = spec["name"]
        in_keys = [tuple(k) for k in _oc_to_py(spec["in_keys"])]
        eps = float(spec.get("eps", 1e-4))
        standard_normal = bool(spec.get("standard_normal", True))
        state = normalization.states.get(name) if (use_cache and normalization) else None

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
        state = normalization.states.get(name) if (use_cache and normalization) else None
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
        step_lb_fn = getattr(env, "min_num_steps", None)
        if callable(step_lb_fn):
            base_steps = int(step_lb_fn())
        else:
            base_env = getattr(env, "base_env", None)
            if base_env is not None and callable(getattr(base_env, "min_num_steps", None)):
                base_steps = int(base_env.min_num_steps())
            else:
                base_steps = int(getattr(env, "size", lambda: 1)())
        num_iter = max(1, base_steps) * max(1, int(getattr(cfg.feature.normalization, "warmup", 1)))
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
                # if cfg.feature.observer.version in "DFGH":
                #     norm.loc[-4:] = 0.0
                #     norm.scale[-4:] = 1.0
                print(norm.loc)
                print(norm.scale)
                # import sys 
                # sys.exit(0)
        finally:
            env.enable_reward()
        return NormalizationDetails(states={n: t.state_dict() for n, t in created.items()})

    return None

def _plot_graph_with_networkx(g):
    import matplotlib.pyplot as plt

    dag = nx.DiGraph()
    dag.add_nodes_from((task.id, {"label": task.name}) for task in g.tasks.values())
    dag.add_edges_from((dep_id, task.id) for task in g for dep_id in g.get_task_dependencies(task.id))
    try:

        coords = graphviz_layout(dag, prog="dot")
    except (ImportError, nx.NetworkXException):
        coords = nx.spring_layout(dag, seed=getattr(cfg, "seed", None))
    labels = {node_id: data["label"] for node_id, data in dag.nodes(data=True)}
    nx.draw_networkx(
        dag,
        coords,
        labels=labels,
        node_size=700,
        node_color="#74add1",
        edge_color="#4c72b0",
        arrows=True,
        font_size=8,
    )
    plt.title("Task dependency graph")
    plt.tight_layout()
    plt.show()

def make_env(
    graph_builder: GraphBuilder,
    cfg: DictConfig,
    lstm: Optional[LSTMModule] = None,
    normalization: Optional[NormalizationDetails] = None,
    eval=False,
) -> RuntimeEnv | tuple[RuntimeEnv, NormalizationDetails]:
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout

    s = create_system(cfg)
    graph = graph_builder.function(s)
    if hasattr(graph, "is_finalized") and not graph.is_finalized:
        graph.finalize()

    d = graph.get_blocks()
    m = graph

    transition_conditions = create_conditions(cfg)
    runtime_env_t = create_runtime_reward(cfg)
    observer_factory, graph_spec = create_observer_factory(cfg)

    task_noise = create_task_noise(cfg, graph.static_graph)
    top_k_candidates = graph_spec.max_candidates

    print(f"Using top_k_candidates = {top_k_candidates}")

    input = SimulatorInput(m, d, s, transition_conditions=transition_conditions, task_noise=task_noise, top_k_candidates=top_k_candidates)


    if cfg.algorithm.rollout_steps <= 0:
        # Default to task count as a transition budget. Keep independent of candidate batch size.
        rollout_steps = len(graph)
    else:
        # Treat rollout_steps as an explicit number of environment transitions.
        rollout_steps = int(cfg.algorithm.rollout_steps)

    env = runtime_env_t(
        SimulatorFactory(input, graph_spec, observer_factory),
        device="cpu",
        random_start=False,
        random_offset=False,
        offset=0,
        change_priority=cfg.graph.env.change_priority if hasattr(cfg.graph.env, "change_priority") else False,
        change_location=cfg.graph.env.change_location if hasattr(cfg.graph.env, "change_location") else False,
        change_duration=cfg.graph.env.change_duration if hasattr(cfg.graph.env, "change_duration") else False,
        change_workload=cfg.graph.env.change_workload if hasattr(cfg.graph.env, "change_workload") else False,
        seed=cfg.seed,
        max_samples_per_iter=rollout_steps,
    )
    env = TransformedEnv(env, StepCounter())
    env.append_transform(TrajCounter())
    env.append_transform(InitTracker())

    if eval:
        env.disable_reward()

    if lstm is not None:
        print("Adding LSTM module to environment", flush=True)
        env.append_transform(lstm.make_tensordict_primer())

    if normalization != False:
        new_norm = _setup_observation_norms(env, cfg, normalization)
    else:
        new_norm = None

    # Keep the historical API shape expected by callers:
    # - first call (normalization=None): return (env, norm_state)
    # - subsequent calls (normalization provided): return env
    if normalization is None and new_norm is not None:
        return env, new_norm
    return env
