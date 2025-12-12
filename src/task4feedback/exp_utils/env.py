from .definitions import *
from task4feedback.interface import SimulatorFactory, SimulatorInput, create_graph_spec
from task4feedback.interface import TaskNoise
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
from .logging_helpers import get_helper_logger
logger = get_helper_logger(__name__)


def create_system(cfg: DictConfig):
    system = hydra.utils.instantiate(cfg.system)
    return system


def create_conditions(cfg: DictConfig):
    transition_conditions = hydra.utils.instantiate(cfg.env.runtime)
    return transition_conditions


def create_runtime_reward(cfg: DictConfig):
    runtime_env_t = hydra.utils.instantiate(cfg.env.reward)
    return runtime_env_t


def create_observer_factory(cfg: DictConfig):
    observer_cfg = cfg.feature.observer
    spec_cfg = OmegaConf.select(observer_cfg, "spec", default=None)
    if spec_cfg is None:
        spec_cfg = OmegaConf.create({"_target_": "task4feedback.interface.create_graph_spec"})
    elif not isinstance(spec_cfg, (DictConfig, ListConfig)):
        spec_cfg = OmegaConf.create(spec_cfg)

    spec_candidates = OmegaConf.select(spec_cfg, "max_candidates", default=0)
    graph_spec = hydra.utils.instantiate(spec_cfg)

    params = {}
    width = OmegaConf.select(observer_cfg, "width", default=None)
    length = OmegaConf.select(observer_cfg, "length", default=None)
    params["width"] = width
    params["length"] = length

    if max_candidates := OmegaConf.select(cfg.env.action, "max_candidates", default=None):
        if max_candidates > spec_candidates:
            logger.warning(
                f"Overriding graph_spec.max_candidates from {spec_candidates} to {max_candidates}"
            )
            graph_spec.max_candidates = max_candidates
            
    params["spec"] = graph_spec
    observer_factory = hydra.utils.instantiate(observer_cfg, **params)
    return observer_factory, graph_spec


def create_task_noise(cfg: DictConfig, static_graph):
    enabled = OmegaConf.select(cfg.env.randomize.task_duration, "enabled", default=False)
    if not enabled:
        task_noise = TaskNoise(tasks=static_graph)
    else:
        task_noise = hydra.utils.instantiate(cfg.env.randomize.task_duration.config)
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

    to_init: List[Tuple[str, ObservationNorm, dict]] = []
    for spec in specs:
        name = spec["name"]
        state = normalization.states.get(name) if normalization else None
        if state is not None:
            try:
                logger.debug("Loading saved observation norm state %s", name)
                created[name].load_state_dict(state)
                continue
            except Exception as e:
                logger.warning("Failed to load observation norm state %s: %s", name, e)
                pass
        to_init.append((name, created[name], spec))

    if to_init:
        logger.info("Initializing observation norms: %s", [n for n, _, _ in to_init])
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


def rollout_length(total_tasks: int, rollout_steps: int, max_candidates: int) -> int:
    if rollout_steps <= 0:
        steps = total_tasks // max_candidates
    else:
        steps = rollout_steps

    #Og én til javanissen
    steps = steps + 1

    return steps 

def make_env(
    graph_builder: GraphBuilder,
    cfg: DictConfig,
    lstm: Optional[LSTMModule] = None,
    normalization: Optional[NormalizationDetails] = None,
    eval=False,
) -> RuntimeEnv | tuple[RuntimeEnv, NormalizationDetails]:
    from task4feedback.graphs.mesh import gmsh
    gmsh.initialize()

    s = create_system(cfg)
    graph = graph_builder.function(s)
    d = graph.get_blocks()
    m = graph

    transition_conditions = create_conditions(cfg)
    runtime_env_t = create_runtime_reward(cfg)
    observer_factory, graph_spec = create_observer_factory(cfg)
    task_noise = create_task_noise(cfg, graph.static_graph)

    max_candidates = OmegaConf.select(cfg.env.action, "max_candidates",default=1)

    input = SimulatorInput(m, d, s, transition_conditions=transition_conditions, task_noise=task_noise, top_k_candidates=max_candidates)

    change_priority = OmegaConf.select(cfg.env.randomize.task_priority, "enabled", default=False)
    change_location = OmegaConf.select(cfg.env.randomize.data_location, "enabled", default=False)
    change_duration = OmegaConf.select(cfg.env.randomize.task_duration, "enabled", default=False)
    change_workload = OmegaConf.select(cfg.env.randomize.workload, "enabled", default=False)

    steps = rollout_length(len(graph), cfg.algorithm.rollout_steps, max_candidates)

    env = runtime_env_t(
        SimulatorFactory(input, graph_spec, observer_factory),
        device="cpu",
        change_priority=change_priority,
        change_location=change_location,
        change_duration=change_duration,
        change_workload=change_workload,
        seed=cfg.seed,
        max_samples_per_iter=steps,
    )
    env = TransformedEnv(env, StepCounter())
    env.append_transform(TrajCounter())
    env.append_transform(InitTracker())

    if eval:
        env.disable_reward()

    if lstm is not None:
        logger.info("Adding LSTM module to environment")
        env.append_transform(lstm.make_tensordict_primer())

    if normalization:
        new_norm = _setup_observation_norms(env, cfg, normalization)
    else:
        new_norm = None

    return (env, new_norm) if new_norm is not None else env
