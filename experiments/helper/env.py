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
import torch
from pathlib import Path

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
        top_k_candidates = cfg.graph.config.n**2
    else:
        top_k_candidates = 1

    input = SimulatorInput(m, d, s, transition_conditions=transition_conditions, task_noise=task_noise, top_k_candidates=top_k_candidates)



    env = runtime_env_t(
        SimulatorFactory(input, graph_spec, observer_factory),
        device="cpu",
        change_priority=cfg.graph.env.change_priority if hasattr(cfg.graph.env, "change_priority") else False,
        change_location=cfg.graph.env.change_location if hasattr(cfg.graph.env, "change_location") else False,
        change_duration=cfg.graph.env.change_duration if hasattr(cfg.graph.env, "change_duration") else False,
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
    new_norm = None
    if normalization is None and normalization is not False:
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
    elif isinstance(normalization, NormalizationDetails):
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


def load_policy_from_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> bool:
    """Load a policy module state_dict from `ckpt_path` into `model`.

    The checkpoint may be either a full training checkpoint with a `policy_module`
    key (as saved by `save_checkpoint`) or a raw state_dict for the policy itself.
    Returns True if parameters were loaded; False otherwise.
    """
    # We trust local checkpoints produced by our code; allow full unpickling.
    # If you are loading an untrusted checkpoint, you may wish to set weights_only=True for safety.
    try:
        # We trust local checkpoints produced by our code; allow full unpickling.
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        return False

    # Determine which state_dict to use
    state_dict = None
    if isinstance(obj, dict) and "policy_module" in obj and isinstance(obj["policy_module"], dict):
        state_dict = obj["policy_module"]
    elif isinstance(obj, dict):
        # Heuristic: treat as a raw state_dict if values are tensors
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj

    if state_dict is None:
        print(f"Checkpoint at {ckpt_path} does not contain a recognizable policy state_dict.")
        return False

    # Common attribute names for the policy head/module
    candidate_attrs = ["policy_module", "policy", "actor", "pi", "actor_net"]
    target_module = None
    for attr in candidate_attrs:
        if hasattr(model, attr):
            m = getattr(model, attr)
            if isinstance(m, torch.nn.Module):
                target_module = m
                break

    # Fallback: try loading into the model itself
    if target_module is None:
        target_module = model

    try:
        missing, unexpected = target_module.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded policy weights from {ckpt_path} into "
            f"{target_module.__class__.__name__} (missing={len(missing)}, unexpected={len(unexpected)})."
        )
        return True
    except Exception as e:
        print(f"Failed to load policy weights into target module: {e}")
        return False