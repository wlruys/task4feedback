from email import policy
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate, get_original_cwd
from types import SimpleNamespace

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
from mpi4py import MPI
from tqdm import tqdm
import gc


def generate_ordered_sequences(blocks, target):
    """
    All ordered sequences of 'blocks' summing to 'target'.
    NOTE: This can blow up combinatorially; consider constraining 'blocks' or 'target'
    if needed for practicality.
    """
    result = []

    def backtrack(remaining, path):
        if remaining == 0:
            result.append(path[:])
            return
        for b in blocks:
            if b <= remaining:
                path.append(b)
                backtrack(remaining - b, path)
                path.pop()

    backtrack(target, [])
    return result


def _run_with_partitions(env, graph):
    """
    Runs a copy of the simulator using current graph.partitions as external mapping.
    Returns the resulting runtime (float).
    """
    sim_copy = env.simulator.copy()
    sim_copy.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
    sim_copy.run()
    return sim_copy.time


def configure_training(cfg: DictConfig):
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # deterministic seeds per user config (keep identical across ranks)
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    # Build env/graph on every rank (kept identical so comparisons are meaningful)
    # Expose a simple 'args' shim for requested CLI-like access
    graph_builder = make_graph_builder(cfg)
    env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
    env._reset()
    sim = env.simulator
    graph = env.get_graph()
    if not isinstance(graph, DynamicJacobiGraph):
        raise ValueError("Graph is not a DynamicJacobiGraph")

    # Enable external mapping
    env.simulator.enable_external_mapper()

    # --- 1) Synchronized baseline run to verify identical envs across ranks ---
    # Everyone computes the same 'level_chunks=12' METIS mincut, then runs a copy.
    baseline_time = env._get_baseline("EFT")

    # Bring all baselines to root and check consistency (within small tolerance)
    all_baselines = comm.gather(baseline_time, root=0)
    if rank == 0:
        tol = 1e-9
        ref = all_baselines[0]
        identical = all(abs(bt - ref) <= tol for bt in all_baselines)
        if identical:
            print(f"[MPI] Baseline check passed across {size} ranks. Time = {ref:.9f}")
        else:
            print("[MPI][WARNING] Baseline times differ across ranks:")
            for r, bt in enumerate(all_baselines):
                print(f"  rank {r}: baseline_time = {bt:.12f}")
            print("Proceeding anyway, but results may not be directly comparable.")
    # Make sure everyone knows we did the check
    comm.Barrier()

    # Keep the best-known baseline as the starting 'global best' reference
    global_best_time = baseline_time
    global_best_seq = None  # represents the baseline mapping (level_chunks=12)

    # --- 2) Build the full search space (locally, identically on all ranks) ---
    # blocks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    blocks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    target = cfg.graph.config.steps
    sequences = generate_ordered_sequences(blocks, target)

    # Partition sequences among ranks in a straightforward strided manner
    my_sequences = sequences[rank::size]

    # --- 3) Parallel search: each rank evaluates its shard, tracks local best ---
    local_best_time = global_best_time
    local_best_seq = global_best_seq

    # Progress bar per-rank (safe if run with a single shared stdout; otherwise falls back to prints)
    total = len(my_sequences)
    pbar = tqdm(
        total=total, desc=f"rank {rank}", position=rank, leave=True, dynamic_ncols=True
    )

    for ll in my_sequences:
        # Important: we must re-run mincut_per_levels to refresh graph.partitions for ll
        graph.mincut_per_levels(
            bandwidth=cfg.system.d2d_bw,
            mode="predict",
            offset=1,
            levels_list=ll,
            levels_per_chunk=cfg.sweep.future,
        )
        t = _run_with_partitions(env, graph)

        if t < local_best_time:
            local_best_time = t
            local_best_seq = tuple(ll)  # make it hashable / serializable
            # Optional per-rank progress print (can be chatty for huge spaces)
            pbar.set_postfix(
                best_time=f"{local_best_time:.6f}", list=f"{local_best_seq}"
            )
        gc.collect()
        pbar.update(1)

    pbar.close()

    # --- 4) Reduction to find the global best over all ranks ---
    # Gather all local bests at root and choose the globally best one.
    gathered = comm.gather((local_best_time, local_best_seq), root=0)

    if rank == 0:
        # Include baseline from every rank's starting point (already in gathered)
        # Find min by time; tie-break by shorter sequence then lexicographically.
        def key_fn(item):
            t, seq = item
            # Treat None as baseline; prefer real sequences over None only if strictly better time
            length = len(seq) if seq is not None else float("inf")
            return (t, length, seq if seq is not None else ())

        global_best_time, global_best_seq = min(gathered, key=key_fn)

        if global_best_seq is None:
            print(
                f"[MPI] Global best is the baseline mapping (level_chunks=12) with time {global_best_time}"
            )
        else:
            print(
                f"[MPI] Global best sequence: {list(global_best_seq)} with time {global_best_time}"
            )

        # Append final result to a single result.txt at project root and print as requested
        final_msg = f"k={cfg.sweep.future} {global_best_seq} -> {global_best_time} ({baseline_time/global_best_time}x)"
        print(final_msg)
        try:
            root_dir = Path(get_original_cwd())
        except Exception:
            root_dir = Path.cwd()
        out_path = root_dir / "result.txt"
        with open(out_path, "a") as f:
            f.write(final_msg + "\n")

    # Optionally broadcast the global winner to all ranks (if others need it)
    # so downstream code could use it. Here we just finish after printing on root.
    return


@hydra.main(config_path="conf", config_name="size_sweep.yaml", version_base=None)
def main(cfg: DictConfig):
    configure_training(cfg)


if __name__ == "__main__":
    main()
