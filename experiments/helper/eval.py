from email import policy
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from helper.graph import make_graph_builder
from helper.env import make_env
from helper.model import create_td_actor_critic_models
from helper.algorithm import create_optimizer, create_lr_scheduler

from task4feedback.ml.algorithms.ppo import run_ppo, run_ppo_lstm
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *
from task4feedback.graphs.jacobi import (
    JacobiGraph,
    LevelPartitionMapper,
    JacobiRoundRobinMapper,
    JacobiQuadrantMapper,
    BlockCyclicMapper,
    GraphMETISMapper,
)

# from task4feedback.graphs.mesh.plot_fast import *
# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")

from hydra.experimental.callbacks import Callback
from hydra.core.utils import JobReturn
from omegaconf import DictConfig, open_dict
from pathlib import Path
import git
import os
from hydra.core.hydra_config import HydraConfig
from helper.run_name import make_run_name, cfg_hash
import torch
import numpy
import random
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.mesh.plot import animate_mesh_graph
from task4feedback.ml.util import EvaluationConfig
from helper.parmetis import run_parmetis
from helper.run_name import cfg_hash
from dataclasses import dataclass, field
from task4feedback.graphs.jacobi import *


import pickle
import os 
import pathlib 
from collections import defaultdict

from types import MappingProxyType
from typing import Any, Mapping

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

@dataclass(slots=True)
class EvalLocation:
    folder: pathlib.Path
    file_path: pathlib.Path
    name: str

@dataclass(slots=True)
class EvalState:
    cfg: str
    init_locs: defaultdict = field(default_factory=lambda: defaultdict(list))
    workloads: defaultdict = field(default_factory=lambda: defaultdict(list))
    policy_times: defaultdict = field(default_factory=lambda: defaultdict(list))
    policies: list = field(default_factory=list)
    best_policy: Optional[Dict] = None
    best_time: Optional[float] = None


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
    

def eval_location(cfg):
    graph_config = cfg.graph
    system_config = cfg.system

    seed = cfg.get("seed", 0)

    graph_type = graph_config.type if "type" in graph_config else "Unknown"
    graph_config_hash = cfg_hash(graph_config)
    system_config_hash = cfg_hash(system_config)


    target_dir = Path("saved_evals") / graph_type / f"{system_config_hash}" 
    target_file_name = f"{graph_config_hash}_{seed}.pkl"
    target_file = target_dir / target_file_name
    if target_file.exists():
        print(f"Warning: Eval file {target_file} already exists and will be overwritten.")

    target_dir.mkdir(parents=True, exist_ok=True)

    #save system config and graph config
    with open(target_dir / f"system_config_{system_config_hash}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(system_config))

    with open(target_dir / f"graph_config_{graph_config_hash}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(graph_config))

    return EvalLocation(folder=target_dir, file_path=target_file, name=f"evaluation_{graph_config_hash}_{seed}.pkl")

def lookup_eval_location(cfg, raise_if_missing=False):
    graph_config = cfg.graph
    system_config = cfg.system

    seed = cfg.get("seed", 0)

    graph_type = graph_config.type if "type" in graph_config else "Unknown"
    graph_config_hash = cfg_hash(graph_config)
    system_config_hash = cfg_hash(system_config)


    target_dir = Path("saved_evals") / graph_type / f"{system_config_hash}" 
    target_file_name = f"{graph_config_hash}_{seed}.pkl"
    target_file = target_dir / target_file_name
    if not target_file.exists() and raise_if_missing:
        raise FileNotFoundError(f"Eval file {target_file} does not exist.")
    if not target_file.exists():
        print(f"Warning: Eval file {target_file} does not exist.")
        return None 

    return EvalLocation(folder=target_dir, file_path=target_file, name=f"evaluation_{graph_config_hash}_{seed}.pkl")

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

        print(f"Evaluating policy {policy_str} with {n_samples} samples")
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
                run_parmetis(sim=env.simulator if rank == 0 else None, cfg=cfg)
            else:
                raise ValueError(f"Unknown option: {policy.name}")
            
            env.simulator.run()
            eval_state.policy_times[policy_to_str(policy)].append(env.simulator.time)
            print(f"  Sample {i}: {policy_to_str(policy)} time {env.simulator.time:.4f}")

def check_eval(location: EvalLocation, cfg: DictConfig):
    eval_state: EvalState = pickle.load(open(location.file_path, "rb"))
    env = make_env(graph_builder=make_graph_builder(cfg), cfg=cfg, normalization=False)
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

    eval_state = EvalState(cfg=OmegaConf.to_yaml(cfg), policies=policies)

    location = eval_location(cfg)


    for policy in policies:
        print(policies)
        print(f"Policy: {policy}")

        print(f"Running evaluation for policy: {policy.name} with params: {policy.params}")
        evaluate_and_save(policy, cfg, eval_state, n_samples=20)

    print(f"Saving evaluation to {location.file_path}")
    best_policy, best_time = find_best_policy(eval_state)
    eval_state.best_policy = to_pickable_policy(best_policy)
    eval_state.best_time = best_time
    eval_state.policies = [to_pickable_policy(p) for p in eval_state.policies]
    pickle.dump(eval_state, open(location.file_path, "wb"))

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    check_eval(location, cfg)