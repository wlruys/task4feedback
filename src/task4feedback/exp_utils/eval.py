from .definitions import *
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from .graph import make_graph_builder
from .env import make_env
from .model import create_td_models
from .artifacts import (
    atomic_pickle_dump,
    eval_cache_context,
    load_normalization_state,
)
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *
from task4feedback.graphs.jacobi import (
    ColCyclicMapper,
    LevelPartitionMapper,
    JacobiRoundRobinMapper,
    JacobiQuadrantMapper,
    BlockCyclicMapper,
    GraphMETISMapper,
    RowCyclicMapper,
)
import torch
import numpy
import random
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.trip import ParMETIS_wrapper
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional
from .logging_helpers import get_helper_logger

comm = None
rank = 0
size = 1

logger = get_helper_logger(__name__)

@dataclass(slots=True)
class EvalLocation:
    folder: Path
    file_path: Path
    name: str

@dataclass(slots=True)
class EvalState:
    cfg: str = ""
    init_locs: defaultdict = field(default_factory=lambda: defaultdict(list))
    workloads: defaultdict = field(default_factory=lambda: defaultdict(list))
    policy_times: defaultdict = field(default_factory=lambda: defaultdict(list))
    policies: list = field(default_factory=list)
    best_policy: Optional[Dict] = None
    best_time: Optional[float] = None
    normalization: Optional[Dict[str, Any]] = None  # Normalization state for reproducibility


@dataclass(slots=True, frozen=True)
class PolicyType:
    name: str
    params: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __hash__(self):
        return hash((self.name, tuple(sorted(self.params.items()))))


def to_pickable_policy(policy: PolicyType):
    return {"name": policy.name, "params": dict(policy.params)}


def _graph_type(cfg: DictConfig) -> str:
    graph_config = cfg.graph
    return graph_config.type if "type" in graph_config else "Unknown"


def eval_location(cfg):
    ctx = eval_cache_context(cfg)
    target_file = ctx.dir / "eval.pkl"
    if target_file.exists():
        logger.warning("Eval file %s already exists and will be overwritten.", target_file)
    return EvalLocation(folder=ctx.dir, file_path=target_file, name=target_file.name)

def lookup_eval_location(cfg, raise_if_missing=False):
    loc = eval_location(cfg)
    if not loc.file_path.exists():
        if raise_if_missing:
            raise FileNotFoundError(f"Eval file {loc.file_path} does not exist.")
        logger.warning("Eval file %s does not exist.", loc.file_path)
        return None
    return loc


def ensure_eval_location(cfg: DictConfig) -> EvalLocation:
    loc = lookup_eval_location(cfg)
    if loc is None:
        create_evals(cfg)
        loc = lookup_eval_location(cfg, raise_if_missing=True)
        logger.info("Created evaluations at %s", loc.file_path)
        return loc

    # Validate that the pickle loads; otherwise regenerate.
    try:
        with open(loc.file_path, "rb") as f:
            pickle.load(f)
    except Exception as exc:
        logger.warning("Eval file %s invalid (%s); regenerating.", loc.file_path, exc)
        try:
            loc.file_path.unlink()
        except Exception:
            pass
        create_evals(cfg)
        loc = lookup_eval_location(cfg, raise_if_missing=True)

    logger.info("Loading evaluations from %s", loc.file_path)
    return loc

def get_time(eval_state: EvalState, policy: PolicyType, idx: int):
    policy_str = policy_to_str(policy)
    if policy_str in eval_state.policy_times:
        times = eval_state.policy_times[policy_str]
        if idx < len(times):
            return times[idx]
    return None

def find_best_policy(eval_state: EvalState):
    best_policy = None
    best_time = float('inf')
    best_idx = -1
    idx = 0
    for policy_str, times in eval_state.policy_times.items():
        avg_time = sum(times) / len(times) if times else float('inf')
        if avg_time < best_time:
            best_time = avg_time
            best_policy = policy_str
            best_idx = idx 
        idx += 1
    best_policy = eval_state.policies[best_idx] if best_idx >= 0 else None
    return best_policy, best_time

def evaluate_and_save(policy: PolicyType, cfg: DictConfig, eval_state: EvalState, n_samples=20):
        torch.manual_seed(cfg.seed)
        numpy.random.seed(cfg.seed)
        random.seed(cfg.seed)
        graph_builder = make_graph_builder(cfg)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
        env.set_reset_counter(0)
        env._reset()

        policy_str = policy_to_str(policy)

        logger.info(
            "Evaluating policy %s with %s samples", policy_str, n_samples
        )
        for i in range(n_samples):
            env.reset()
            eval_state.init_locs[policy_str].append(env.get_graph().get_cell_locations(as_dict=False))
            graph = env.get_graph()
            geom = graph.data.geometry
            if isinstance(graph, DynamicJacobiGraph):
                eval_state.workloads[policy_str].append(dict(graph.get_workload().level_workload))
            else:
                eval_state.workloads[policy_str].append(None)

            if policy.name == "EFT":
                env.simulator.disable_external_mapper()
                #env._get_baseline("EFT")
                #NOTE(wlr): Assumes default env policy is EFT
            elif policy.name == "Oracle":
                graph.mincut_per_levels(
                    bandwidth=cfg.system.d2d_bw,
                    mode="metis",
                    offset=1,
                    level_chunks=1,
                )
                graph.align_partitions()
                env.simulator.enable_external_mapper()
                env.simulator.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
            elif policy.name == "BlockCyclic":
                block_size = policy.params.get("block_size", 2)
                env.simulator.enable_external_mapper()
                env.simulator.external_mapper = BlockCyclicMapper(geometry=geom, n_devices=cfg.system.n_devices - 1, block_size=2, offset=1)
            elif policy.name == "GraphMETISMapper":
                env.simulator.enable_external_mapper()
                env.simulator.external_mapper = GraphMETISMapper(graph=graph, n_devices=cfg.system.n_devices - 1, offset=1)
            elif policy.name == "Quad":
                env.simulator.enable_external_mapper()
                env.simulator.external_mapper = JacobiQuadrantMapper(n_devices=cfg.system.n_devices - 1, graph=graph, offset=1)
            elif policy.name == "RowCyclic":
                env.simulator.enable_external_mapper()
                env.simulator.external_mapper = RowCyclicMapper(geometry=geom, n_devices=cfg.system.n_devices - 1)
            elif policy.name == "ColCyclic":
                env.simulator.enable_external_mapper()
                env.simulator.external_mapper = ColCyclicMapper(geometry=geom, n_devices=cfg.system.n_devices - 1)
            elif policy.name == "Cyclic":
                env.simulator.enable_external_mapper()
                env.simulator.external_mapper = JacobiRoundRobinMapper(n_devices=cfg.system.n_devices - 1, offset=1, setting=0)
            elif policy.name == "ParMETIS":
                pass
                # run_parmetis(sim=env.simulator if rank == 0 else None, cfg=cfg)
            else:
                raise ValueError(f"Unknown option: {policy.name}")
            
            env.simulator.run()
            eval_state.policy_times[policy_to_str(policy)].append(env.simulator.time)
            logger.debug(
                "Sample %s: %s time %.4f",
                i,
                policy_to_str(policy),
                env.simulator.time,
            )

def check_eval(location: EvalLocation, cfg: DictConfig):
    eval_state: EvalState = pickle.load(open(location.file_path, "rb"))
    normalization = load_normalization_state(cfg)
    env = make_env(graph_builder=make_graph_builder(cfg), cfg=cfg, normalization=normalization)
    if isinstance(env, tuple):
        env = env[0]
    #env.set_reset_counter(0)
    #env._reset()


    eft_policy = PolicyType(name="EFT", params={})
    policy_str = policy_to_str(eft_policy)
    n_samples = len(eval_state.init_locs[policy_str])

    for i in range(n_samples):
        saved_loc = eval_state.init_locs[policy_str][i]
        workload = eval_state.workloads[policy_str][i]
        env.reset_to_state(saved_loc, workload)
        eft_policy = PolicyType(name="EFT", params={})
        sim_time = env._get_baseline("EFT")
        saved_time = get_time(eval_state, eft_policy, i)
        if not numpy.isclose(saved_time, sim_time, atol=1e-3):
            raise Warning(f"Eval check failed for sample {i}: saved time {saved_time}, recomputed time {sim_time}")

def policy_to_str(policy: PolicyType):
    param_str = "_".join([f"{k}={v}" for k, v in policy.params.items()])
    if param_str:
        return f"{policy.name}_{param_str}"
    else:
        return policy.name
    

def create_evals(cfg: DictConfig):
    # start_logger()
    graph_type = cfg.graph.type if "type" in cfg.graph else "Unknown"
    if graph_type == "jacobi":
        policies = [
            PolicyType(name="EFT", params={}),
            PolicyType(name="Quad", params={}),
            PolicyType(name="BlockCyclic", params={"block_size": 1}),
            PolicyType(name="BlockCyclic", params={"block_size": 2}),
        ]
        n_steps = int(cfg.graph.config.steps) 
        oracle_chunks_to_test = [2**i for i in range(0, int(n_steps).bit_length()) if 2**i <= n_steps]
        policies += [PolicyType(name="Oracle", params={"chunk_size": chunk_size}) for chunk_size in oracle_chunks_to_test]
    elif graph_type == "cholesky":
        policies = [
            PolicyType(name="RowCyclic", params={}),
            PolicyType(name="ColCyclic", params={}),
            PolicyType(name="BlockCyclic", params={"block_size": 1}),
            PolicyType(name="BlockCyclic", params={"block_size": 2}),
            PolicyType(name="EFT", params={}),
        ]

    eval_state = EvalState(
        cfg=OmegaConf.to_yaml(cfg),
        policies=policies,
    )

    location = eval_location(cfg)


    logger.info(
        "Creating evaluations for policies: %s",
        [policy.name for policy in policies],
    )
    for policy in policies:
        logger.info(
            "Running evaluation for policy: %s with params: %s",
            policy.name,
            policy.params,
        )
        evaluate_and_save(policy, cfg, eval_state, n_samples=20)

    logger.info("Saving evaluation to %s", location.file_path)
    best_policy, best_time = find_best_policy(eval_state)
    eval_state.best_policy = to_pickable_policy(best_policy)
    eval_state.best_time = best_time
    eval_state.policies = [to_pickable_policy(p) for p in eval_state.policies]
    atomic_pickle_dump(eval_state, location.file_path)

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    check_eval(location, cfg)
