from ..ml.env import RuntimeEnv
from ..graphs.jacobi import JacobiGraph
from ..graphs.base import weighted_cell_partition
from ..interface.wrappers import DeviceType, SimulatorDriver
import task4feedback.fastsim2 as fastsim
from task4feedback.fastsim2 import ParMETIS_wrapper
from mpi4py import MPI
import torch
import numpy as np
from ..graphs.jacobi import get_length_from_config
import hydra
import hashlib
import pickle
from pathlib import Path
from omegaconf import OmegaConf
import optuna
from enum import Enum


class ParMETISState(Enum):
    FAILED = 0
    SUCCESS = 1
    TIMEOUT = 2


def run_parmetis(
    sim: SimulatorDriver,
    cfg,
    verbose=False,
    offset=1,
    future_levels=0,
    itr: float = 1000,
    unbalance: float = 1.225,
    n_compute_devices: int = 4,
    ParMETIS=None,
    best_time=float("inf"),
    skip_error=False,
) -> ParMETISState:
    if itr == 0.0001:
        itr = 0.0001001  # ParMETIS limitation
    if unbalance == 1.0:
        unbalance = 1.001  # ParMETIS limitation
    if unbalance == 4.0:
        unbalance = 3.999  # ParMETIS limitation

    d2d_bandwidth = cfg.system.d2d_bw
    graph_config = hydra.utils.instantiate(cfg.graph.config)
    width = graph_config.n
    length = get_length_from_config(graph_config)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    target_loads = [1.0 / n_compute_devices for _ in range(n_compute_devices)]
    if size != n_compute_devices:
        raise ValueError(f"Expected {n_compute_devices} ranks, but got {size}. Please run with {n_compute_devices} ranks.")
    partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    if ParMETIS is None:
        ParMETIS = ParMETIS_wrapper()
    done = False

    if rank == 0:
        graph = sim.input.graph
        assert isinstance(graph, JacobiGraph), "Graph must be a JacobiGraph"
        if cfg.graph.init.partitioner == "metis":
            cell_graph = graph.get_weighted_cell_graph(
                DeviceType.GPU,
                bandwidth=d2d_bandwidth,
                levels=[0, 1],
            )
            edge_cut, partition = weighted_cell_partition(cell_graph, nparts=(cfg.system.n_devices - 1))
        elif cfg.graph.init.partitioner == "quad":
            partition = graph.quadrant_partition(
                arch=DeviceType.GPU,
                bandwidth=cfg.system.d2d_bw,
                n_parts=cfg.system.n_devices - 1,
                offset=0,
            )
        partition = graph.maximize_matches(partition)
        cell_to_device = [x + offset for x in partition]
        partition = [-1 for _ in range(sim.observer.graph_spec.max_candidates)]
        sim.enable_external_mapper()
        done = sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE
        candidates = torch.zeros((sim.observer.graph_spec.max_candidates), dtype=torch.int64)
        sim.get_mappable_candidates(candidates)
        actions = []
        for i, id in enumerate(candidates):
            mapping_priority = sim.get_mapping_priority(id)
            actions.append(
                fastsim.Action(
                    i,
                    cell_to_device[graph.task_to_cell[id.item()]],
                    mapping_priority,
                    mapping_priority,
                )
            )
        sim.simulator.map_tasks(actions)
        done = sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE
        if verbose:
            for i, loc in enumerate(cell_to_device):
                print(loc, end=" ")
                if (i + 1) % width == 0:
                    print()
    while True:
        done = comm.bcast(done, root=0)
        if done:
            break
        if rank == 0:
            sim.get_mappable_candidates(candidates)
            for i, id in enumerate(candidates):
                partition[i] = cell_to_device[graph.task_to_cell[id.item()]] - offset
            partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = graph.get_distributed_weighted_graph(
                bandwidth=d2d_bandwidth,
                task_ids=candidates.tolist(),
                partition=partition,
                future_levels=future_levels,
                width=width,
                n_compute_devices=n_compute_devices,
            )
        vtxdist = comm.bcast(vtxdist, root=0)
        xadj = comm.bcast(xadj, root=0)
        adjncy = comm.bcast(adjncy, root=0)
        vwgt = comm.bcast(vwgt, root=0)
        adjwgt = comm.bcast(adjwgt, root=0)
        vsize = comm.bcast(vsize, root=0)

        xadj = xadj[rank]
        adjncy = adjncy[rank]
        vwgt = vwgt[rank]
        adjwgt = adjwgt[rank]
        vsize = vsize[rank]
        wgtflag = 3
        numflag = 0
        ncon = 1
        tpwgts = np.array(target_loads, dtype=np.float32)
        ubvec = np.array([unbalance], dtype=np.float32)
        part = np.array([-1 for _ in range(width**2)], dtype=np.int32)
        comm.Barrier()
        status = ParMETIS.callParMETIS(
            vtxdist,
            xadj,
            adjncy,
            vwgt,
            vsize,
            adjwgt,
            wgtflag,
            numflag,
            ncon,
            tpwgts,
            ubvec,
            itr,
            part,
        )
        parts = comm.gather(part, root=0)

        if not status and not skip_error:
            if rank == 0:
                print("ParMETIS failed!", flush=True)
            return ParMETISState.FAILED

        if rank == 0:
            for i, p in enumerate(parts):
                for j, dev in enumerate(p):
                    if dev == -1:
                        break
                    task_id = partitioned_tasks[i][j]
                    cell_to_device[graph.task_to_cell[task_id]] = int(dev) + offset  # Offset by 1 to ignore CPU
            actions = []
            for i, id in enumerate(candidates):
                mapping_priority = sim.get_mapping_priority(id)
                actions.append(
                    fastsim.Action(
                        i,
                        cell_to_device[graph.task_to_cell[id.item()]],
                        mapping_priority,
                        mapping_priority,
                    )
                )
            sim.simulator.map_tasks(actions)
            done = sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE
            if verbose:
                print()
                for i, loc in enumerate(cell_to_device):
                    print(loc, end=" ")
                    if (i + 1) % 8 == 0:
                        print()
        time = comm.bcast(sim.time if rank == 0 else None, root=0)
        if time > best_time:
            if rank == 0:
                print(f"Terminating early: current time {time} >= best time {best_time}", flush=True)
            return ParMETISState.TIMEOUT
    return ParMETISState.SUCCESS


# def query_parmetis(
#     ParMETIS,
#     env: RuntimeEnv,
#     cfg,
#     prev_mapping=None,
#     verbose=False,
#     first_call=False,
#     offset=1,
#     future_levels=0,
#     itr: float = 1000,
#     unbalance: float = 1.225,
#     target_loads: list[float] = [0.25, 0.25, 0.25, 0.25],
#     n_compute_devices: int = 4,
# ) -> bool:
#     d2d_bandwidth = cfg.system.d2d_bw
#     graph_config = hydra.utils.instantiate(cfg.graph.config)
#     width = graph_config.n
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = (
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#     )

#     if rank == 0 and first_call:
#         graph = env.simulator.input.graph
#         assert isinstance(graph, JacobiGraph), "Graph must be a JacobiGraph"
#         cell_graph = graph.get_weighted_cell_graph(
#             DeviceType.GPU,
#             bandwidth=d2d_bandwidth,
#             levels=[0, 1],
#         )
#         edge_cut, partition = weighted_cell_partition(cell_graph, nparts=(cfg.system.n_devices - 1))
#         cell_to_device = [x + offset for x in partition]
#         return cell_to_device, True
#     elif rank != 0 and first_call:
#         return None, True
#     else:
#         if rank == 0:
#             assert prev_mapping is not None, "prev_mapping must be provided after the first step"
#             graph = env.simulator.input.graph
#             candidates = torch.zeros((env.simulator.observer.graph_spec.max_candidates), dtype=torch.int64)
#             partition = [-1 for _ in range(env.simulator.observer.graph_spec.max_candidates)]
#             env.simulator.get_mappable_candidates(candidates)
#             for i, id in enumerate(candidates):
#                 partition[i] = prev_mapping[graph.task_to_cell[id.item()]] - offset
#             partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = graph.get_distributed_weighted_graph(
#                 bandwidth=d2d_bandwidth,
#                 task_ids=candidates.tolist(),
#                 partition=partition,
#                 future_levels=future_levels,
#                 width=width,
#                 n_compute_devices=n_compute_devices,
#             )
#         vtxdist = comm.bcast(vtxdist, root=0)
#         xadj = comm.bcast(xadj, root=0)
#         adjncy = comm.bcast(adjncy, root=0)
#         vwgt = comm.bcast(vwgt, root=0)
#         adjwgt = comm.bcast(adjwgt, root=0)
#         vsize = comm.bcast(vsize, root=0)

#         xadj = xadj[rank]
#         adjncy = adjncy[rank]
#         vwgt = vwgt[rank]
#         adjwgt = adjwgt[rank]
#         vsize = vsize[rank]
#         wgtflag = 3
#         numflag = 0
#         ncon = 1
#         tpwgts = np.array(target_loads, dtype=np.float32)
#         ubvec = np.array([unbalance], dtype=np.float32)
#         part = np.array([-1 for _ in range(width**2)], dtype=np.int32)
#         comm.Barrier()
#         status = ParMETIS.callParMETIS(
#             vtxdist,
#             xadj,
#             adjncy,
#             vwgt,
#             vsize,
#             adjwgt,
#             wgtflag,
#             numflag,
#             ncon,
#             tpwgts,
#             ubvec,
#             itr,
#             part,
#         )
#         parts = comm.gather(part, root=0)

#         if not status:
#             if rank == 0:
#                 print("ParMETIS failed!", flush=True)
#             return prev_mapping, False

#         if rank == 0:
#             for i, p in enumerate(parts):
#                 for j, dev in enumerate(p):
#                     if dev == -1:
#                         break
#                     task_id = partitioned_tasks[i][j]
#                     prev_mapping[graph.task_to_cell[task_id]] = int(dev) + offset  # Offset by 1 to ignore CPU
#             return prev_mapping, status
#         else:
#             return None, status


def hash_graph_cfg(graph_cfg) -> str:
    data = OmegaConf.to_container(graph_cfg, resolve=True)
    serialized = repr(sorted(data.items())).encode()
    return hashlib.sha256(serialized).hexdigest()


def find_best_cfg(cfg, ParMETIS, env=None, cache_dir="parmetis_cfg", skip_search=False, mode="normal"):
    cfg.graph.config.steps = 256
    cfg.graph.env.change_duration = False
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cache_dir = Path(cache_dir) / f"{cfg.system.n_devices - 1}gpus" / mode

    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        graph_hash = hash_graph_cfg(cfg.graph)
        cache_file = cache_dir / f"{graph_hash}.pkl"

        if cache_file.exists():
            with cache_file.open("rb") as f:
                best_cfg = pickle.load(f)

            print(
                f"Using cached ParMETIS config " f"(graph={graph_hash[:8]}): " f"itr={best_cfg[0]}, ub={best_cfg[1]}, time={best_cfg[2]}",
                flush=True,
            )
        else:
            best_cfg = None
    else:
        best_cfg = None
        cache_file = None

    # Broadcast cache hit / miss
    best_cfg = comm.bcast(best_cfg, root=0)
    cache_file = comm.bcast(cache_file, root=0)

    return best_cfg
    # Obsolete
    if rank == 0:
        print("Finding best ParMETIS configuration...", flush=True)

    best_cfg = (None, None, float("inf"))  # (itr, ub, time)

    itr_list = [0.0001001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
    ub_list = [1.001, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]

    for itr in itr_list:
        for ub in ub_list:
            if rank == 0:
                temp = env.simulator.copy()
            else:
                temp = None

            comm.barrier()
            status = run_parmetis(
                sim=temp,
                cfg=cfg,
                unbalance=ub,
                itr=itr,
                best_time=best_cfg[2],
                n_compute_devices=cfg.system.n_devices - 1,
                ParMETIS=ParMETIS,
            )

            if not status:
                continue

            if rank == 0 and temp.time < best_cfg[2]:
                best_cfg = (itr, ub, temp.time)

            best_cfg = comm.bcast(best_cfg, root=0)

    # Save result using hash as filename
    if rank == 0:
        with cache_file.open("wb") as f:
            pickle.dump(best_cfg, f)

        print(
            f"Best ParMETIS config saved " f"(graph={graph_hash[:8]}): " f"itr={best_cfg[0]}, ub={best_cfg[1]}, time={best_cfg[2]}",
            flush=True,
        )

    return best_cfg


def run_parmetis_trial(cfg, ParMETIS, env, itr, ub, best_time):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        sim = env.simulator.copy()
    else:
        sim = None

    comm.barrier()

    status = run_parmetis(
        sim=sim,
        cfg=cfg,
        itr=itr,
        unbalance=ub,
        best_time=best_time,
        n_compute_devices=cfg.system.n_devices - 1,
        ParMETIS=ParMETIS,
    )

    if status == ParMETISState.FAILED:
        return None
    elif status == ParMETISState.TIMEOUT:
        if rank == 0:
            return best_time + 1
        else:
            return None
    elif status == ParMETISState.SUCCESS:
        if rank == 0:
            return sim.time
        return None
    else:
        raise ValueError("Unknown ParMETISState returned")


def find_best_cfg_optuna(
    cfg,
    ParMETIS,
    env: RuntimeEnv,
    cache_dir="parmetis_cfg",
    n_trials=100,
    skip_search=False,
    mode="optuna",
):
    # ---------------------------
    # 0. Setup and Caching
    # ---------------------------
    # Ensure consistent simulation steps for hashing/comparison
    cfg.graph.config.steps = 256
    cfg.graph.env.change_duration = False

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cache_dir = Path(cache_dir) / f"{cfg.system.n_devices - 1}gpus" / mode

    # Check for existing cache on Rank 0
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        graph_hash = hash_graph_cfg(cfg.graph)
        cache_file = cache_dir / f"{graph_hash}.pkl"

        if cache_file.exists():
            with cache_file.open("rb") as f:
                best_cfg = pickle.load(f)

            print(
                f"Using cached Optuna config (graph={graph_hash[:8]}): " f"itr={best_cfg[0]}, ub={best_cfg[1]}, time={best_cfg[2]}",
                flush=True,
            )
        else:
            best_cfg = None
    else:
        best_cfg = None
        cache_file = None

    # Broadcast cache hit/miss to all ranks
    best_cfg = comm.bcast(best_cfg, root=0)
    cache_file = comm.bcast(cache_file, root=0)

    # Return early if cache hit and we aren't forcing a new search
    if skip_search:
        return best_cfg
    if best_cfg is not None:
        return best_cfg

    # ---------------------------
    # 1. Optuna Initialization
    # ---------------------------
    if rank == 0:
        print("Finding best ParMETIS configuration (Optuna)...", flush=True)
        sampler = optuna.samplers.TPESampler()
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )
    else:
        study = None

    best_time = float("inf")

    # ---------------------------
    # 2. Optimization Loop
    # ---------------------------
    step = 0
    while step < n_trials:
        # Ask Optuna for params
        if rank == 0:
            trial = study.ask()
            itr = trial.suggest_categorical("itr", [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
            # itr = trial.suggest_float("itr", 1e-4, 1e6, log=True)
            ub = trial.suggest_float("ub", 1.0, float(cfg.system.n_devices - 1))
            print(f"Trial {step}: itr={itr}, ub={ub}, best={best_time}", flush=True)
        else:
            itr = ub = None

        # Synchronize params and current best_time
        itr = comm.bcast(itr, root=0)
        ub = comm.bcast(ub, root=0)
        best_time = comm.bcast(best_time, root=0)

        # Run MPI-synchronized trial
        time = run_parmetis_trial(
            cfg=cfg,
            ParMETIS=ParMETIS,
            env=env,
            itr=itr,
            ub=ub,
            best_time=best_time,
        )

        time = comm.bcast(time, root=0)

        # Report result to Optuna
        if rank == 0:
            if time is None:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                study.tell(trial, time)
                best_time = min(best_time, time)
                step += 1
        step = comm.bcast(step, root=0)

    # ---------------------------
    # 3. Finalize and Save
    # ---------------------------
    if rank == 0:
        best = study.best_trial
        itr = best.params["itr"]
        ub = best.params["ub"]
        best_time = best.value

        # Save result to cache
        best_cfg = (itr, ub, best_time)
        with cache_file.open("wb") as f:
            pickle.dump(best_cfg, f)

        print(
            f"Best Optuna config saved (graph={graph_hash[:8]}): " f"itr={itr}, ub={ub}, time={best_time}",
            flush=True,
        )
    else:
        itr = ub = best_time = None

    # Final broadcast of the best found configuration
    itr = comm.bcast(itr, root=0)
    ub = comm.bcast(ub, root=0)
    best_time = comm.bcast(best_time, root=0)

    return (itr, ub, best_time)
