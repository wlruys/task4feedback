from ast import pattern
import csv
import gc
import re
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from helper.graph import make_graph_builder
from helper.env import make_env
from helper.model import create_td_actor_critic_models, load_policy_from_checkpoint
from helper.algorithm import create_optimizer, create_lr_scheduler

from task4feedback.ml.algorithms.ppo import run_ppo, run_ppo_lstm
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.graphs.jacobi import JacobiRoundRobinMapper, LevelPartitionMapper, BlockCyclicMapper
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph

from hydra.experimental.callbacks import Callback
from hydra.core.utils import JobReturn
from omegaconf import DictConfig, open_dict
from pathlib import Path
import git
import os
from hydra.core.hydra_config import HydraConfig
from helper.run_name import make_run_name, cfg_hash
import torch
import numpy as np
import random
import pickle
from torchrl.envs import set_exploration_type, ExplorationType
from helper.parmetis import run_parmetis
from mpi4py import MPI
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import multiprocessing
import time


def run_parmetis_with_timeout(sim, cfg, unbalance, itr, timeout=60):
    """Runs ParMETIS with a timeout. Returns (success, time_value)."""
    if rank != 0:
        # Non-root ranks just call and return since they’ll block on MPI_Barrier
        try:
            run_parmetis(sim=sim, cfg=cfg, unbalance=unbalance, itr=itr)
            return True, None
        except Exception:
            return False, None

    # Root rank executes in a separate process
    def target(q):
        try:
            run_parmetis(sim=sim, cfg=cfg, unbalance=unbalance, itr=itr)
            q.put(("success", sim.time))
        except Exception as e:
            q.put(("error", str(e)))

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        print(f"⚠️ run_parmetis timed out after {timeout}s, killing process...", flush=True)
        p.terminate()
        p.join()
        return False, float("inf")  # Mark failure with very high time

    if not q.empty():
        status, result = q.get()
        if status == "success":
            return True, result
        else:
            print(f"ParMETIS failed with error: {result}", flush=True)
            return False, float("inf")

    return False, float("inf")


def count_flips(sim: SimulatorDriver, cfg):
    rt = sim.state.get_task_runtime()
    flips = 0
    for tid in range(64):
        for step in range(cfg.graph.config.steps):
            if step == 0:
                prev = rt.get_compute_task_mapped_device(tid)
            else:
                curr = rt.get_compute_task_mapped_device(tid + step * 64)
                if curr != prev:
                    flips += 1
                prev = curr
    return flips


def configure_training(cfg: DictConfig):
    n_samples = 20
    if not cfg.graph.env.change_priority and not cfg.graph.env.change_location and not cfg.graph.env.change_workload and not cfg.graph.env.change_duration:
        n_samples = 1

    def closest_ratio_string(value: float) -> str:
        mapping = {100: "100", 10: "10", 1: "1", 0.1: "0.1"}
        closest = min(mapping.keys(), key=lambda x: abs(value - x))
        return mapping[closest]

    interior_ratio = 595.5555555 / (cfg.graph.config.arithmetic_intensity)
    boundary_ratio = interior_ratio * cfg.graph.config.boundary_width * 4

    interior_ratio = closest_ratio_string(interior_ratio)
    boundary_ratio = closest_ratio_string(boundary_ratio)

    if OmegaConf.select(cfg, "graph.config.workload_args.traj_type") is not None:
        graph_name = cfg.graph.config.workload_args.traj_type
    else:
        graph_name = "static"

    if cfg.graph.env.change_duration:
        if cfg.graph.config.workload_args.traj_type == "circle":
            graph_name = "ncircle"
        elif cfg.graph.config.workload_args.traj_type == "corners":
            graph_name = "noise"
    if cfg.graph.config.steps > 256:
        graph_name = "l" + graph_name

    proceed = False
    if OmegaConf.select(cfg, "skipRL"):
        skipRL = True
    else:
        skipRL = False
    # saved_models/8x8x128_10-1-1_corners_74GB/5.286_D.pt
    if rank == 0:
        # check if useRL key is in cfg
        if skipRL:
            print("Skipping RL evaluation as per configuration.", flush=True)
            proceed = True
            files = []
            original_steps = cfg.graph.config.steps
            norm = False
            root_dir = Path(f"./{interior_ratio}-{boundary_ratio}-{graph_name}/")
            root_dir.mkdir(parents=True, exist_ok=True)
        else:
            skipRL = False
            original_steps = cfg.graph.config.steps
            root_dir = (
                Path(__file__).resolve().parent
                / "saved_models"
                / f"{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}_{interior_ratio}-{boundary_ratio}-1_{graph_name}_{int(cfg.system.mem/1e9)}GB"
            )
            files = list(root_dir.rglob("*.pt"))
            log_file = root_dir / "model_usage.log"
            results_file = root_dir / "model_eval_results.csv"
            final_file = root_dir / "results.csv"

            if not files:
                print(f"No saved models found in {root_dir}, exiting.")
                proceed = False
            else:
                print(f"Found {len(files)} models. Beginning empirical evaluation...", flush=True)

                model_scores = []
                logged_models = {}
                best_time_from_log = float("inf")
                best_model_from_log = None

                # Step 1: Try to recover past evaluations
                if log_file.exists():
                    with open(log_file, "r") as f:
                        for line in f:
                            if "Evaluated model:" in line:
                                try:
                                    # Example line: 2025-10-06T17:20:43.794698 - Evaluated model: 1.007_D.pt, mean_time=46334451.0000
                                    parts = line.strip().split("Evaluated model:")[1].split(", mean_time=")
                                    model_name = parts[0].strip()
                                    mean_time = float(parts[1])
                                    logged_models[model_name] = mean_time
                                    if mean_time < best_time_from_log:
                                        best_time_from_log = mean_time
                                        best_model_from_log = model_name
                                except Exception:
                                    continue
                    print(f"Loaded {len(logged_models)} logged models from log. Best so far: {best_model_from_log} ({best_time_from_log:.2f})", flush=True)
                else:
                    print("No previous evaluation log found.", flush=True)

                # Step 2: Set up normalization once
                graph_builder = make_graph_builder(cfg)
                norm_file = root_dir / f"norm.pkl"
                if norm_file.exists():
                    print(f"Loading normalization from {norm_file}")
                    norm = pickle.load(open(norm_file, "rb"))
                else:
                    print(f"Normalization file {norm_file} not found, creating new normalization")
                    env, norm = make_env(graph_builder=graph_builder, cfg=cfg, eval=True)
                    pickle.dump(norm, open(norm_file, "wb"))

                # Step 3: Evaluate only *new* models
                new_evals = []
                for ckpt_path in files:
                    model_name = ckpt_path.name
                    if model_name in logged_models:
                        print(f"Skipping {model_name}: already logged.")
                        continue

                    print(f"Evaluating new model {model_name}...", flush=True)

                    # Longer graphs
                    cfg.graph.config.steps = 256
                    if cfg.graph.config.workload_args.traj_type == "circle":
                        cfg.graph.config.workload_args.traj_specifics.max_angle = 20
                    graph_builder = make_graph_builder(cfg)

                    eval_times = []
                    graph_builder = make_graph_builder(cfg)
                    eval_env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=norm, eval=True)

                    feature_config = FeatureDimConfig.from_observer(eval_env.get_observer())
                    model, _, _ = create_td_actor_critic_models(cfg, feature_config)
                    loaded = load_policy_from_checkpoint(model, ckpt_path)
                    if not loaded:
                        print(f"Could not load policy from {model_name}, skipping.")
                        continue

                    model.eval()

                    for i in range(1):
                        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                            eval_env.rollout(max_steps=1000000, policy=model.actor, auto_reset=True)
                            eval_times.append(eval_env.simulator.time)

                    mean_time = float(np.mean(eval_times))
                    print(f"→ {model_name}: mean time = {mean_time:.2f}", flush=True)

                    # Store result in memory and append to log
                    logged_models[model_name] = mean_time
                    new_evals.append((model_name, ckpt_path, mean_time))

                    with open(log_file, "a") as f:
                        f.write(f"{datetime.now().isoformat()} - Evaluated model: {model_name}, mean_time={mean_time:.4f}\n")
                    gc.collect()

                # Step 4: Update or create results CSV
                with open(results_file, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Model", "MeanTime"])
                    for name, t in logged_models.items():
                        writer.writerow([name, t])

                # Step 5: Pick the best overall model (old or new)
                best_model_name, best_time = min(logged_models.items(), key=lambda x: x[1])
                best_file = next((f for f in files if f.name == best_model_name), None)

                if best_file is None:
                    print(f"Best model file {best_model_name} not found on disk, exiting.")
                    proceed = False
                else:
                    print(f"✅ Selected best model: {best_model_name} (mean time={best_time:.2f})", flush=True)
                    ckpt_path = best_file
                    if best_model_name == best_model_from_log:
                        print(f"Note: Best model is same as previously best logged model {best_model_from_log} ({best_time_from_log:.2f}), no new best found.")
                        proceed = False
                        if not final_file.exists():
                            proceed = True

                    else:
                        with open(log_file, "a") as f:
                            f.write(f"{datetime.now().isoformat()} - Best model: {best_model_name}, mean_time={best_time:.4f}\n")
                        proceed = True

    proceed = comm.bcast(proceed, root=0)
    if not proceed:
        print("Model already evaluated or no valid model found.")
        exit()

    if rank == 0:
        cfg.graph.config.steps = original_steps
        if cfg.graph.config.workload_args.traj_type == "circle":
            cfg.graph.config.workload_args.traj_specifics.max_angle = original_steps // 128
        graph_builder = make_graph_builder(cfg)
        eval_env = make_env(
            graph_builder=graph_builder,
            cfg=cfg,
            normalization=norm,
            eval=True,
        )
        for ckpt_path in files:
            if best_model_name == ckpt_path.name:
                feature_config = FeatureDimConfig.from_observer(eval_env.get_observer())
                model, _, _ = create_td_actor_critic_models(cfg, feature_config)
                loaded = load_policy_from_checkpoint(model, ckpt_path)
                if not loaded:
                    print(f"Could not load policy from {best_model_name}, exiting.")
                    comm.Abort(1)
                model.eval()
                break

    results = {}
    candidates = ["ParMETIS", "BlockCyclic(4x4)", "BlockCyclic(2x2)", "BlockCyclic(1x1)", "EFT", "RowCyclic"]
    if not skipRL:
        candidates.append("RL")

    for policy in candidates:
        results[policy] = {
            "times": [],
            "evictions": [],
            "flips": [],
        }

    # First find the best configuration for parmetis
    best_cfg = (None, None, float("inf"))  # (itr, ub, time)
    ub_cur = 1.0001
    for itr in [0.0001001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]:
        if rank == 0:
            temp = eval_env.simulator.copy()
        comm.barrier()
        run_parmetis(sim=temp if rank == 0 else None, cfg=cfg, unbalance=ub_cur, itr=itr)
        if rank == 0 and temp.time < best_cfg[2]:
            best_cfg = (itr, ub_cur, temp.time)
            print(f"New best ITR {itr} with time {temp.time}", flush=True)

    best_cfg = comm.bcast(best_cfg, root=0)
    # ub_list = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
    ub_list = [1.05, 1.1, 1.15, 1.2]
    for ub in ub_list:
        if rank == 0:
            temp = eval_env.simulator.copy()
        comm.barrier()
        status = run_parmetis(sim=(temp if rank == 0 else None), cfg=cfg, unbalance=ub, itr=best_cfg[0])
        if not status:
            break
        if rank == 0:
            print(f"Tried ub {ub:.2f} with time {temp.time}", flush=True)
            if temp.time < best_cfg[2]:
                # Improvement: accept move, keep direction, keep step
                ub_cur = ub
                best_cfg = (best_cfg[0], ub_cur, temp.time)
                print(f"New best ub {ub_cur:.2f} with time {temp.time}", flush=True)

    best_cfg = comm.bcast(best_cfg, root=0)

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        if rank == 0:
            cfg.graph.config.steps = 256
            if cfg.graph.config.workload_args.traj_type == "circle":
                cfg.graph.config.workload_args.traj_specifics.max_angle = 1 * 8
            graph_builder = make_graph_builder(cfg)
            eval_env = make_env(
                graph_builder=graph_builder,
                cfg=cfg,
                normalization=norm,
                eval=True,
            )

        for i in range(n_samples):
            gc.collect()
            if rank == 0:
                td = eval_env.reset()
                copy_sim = eval_env.simulator.copy()
                copy_sim.disable_external_mapper()
                copy_sim.run()
                results["EFT"]["times"].append(copy_sim.time)
                results["EFT"]["flips"].append(count_flips(copy_sim, cfg))
                results["EFT"]["evictions"].append(sum(list(copy_sim.total_eviction_movement())[1:]) / 1e9)
                copy_sim = eval_env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.run_until_external_mapping()

            run_parmetis(sim=copy_sim if rank == 0 else None, cfg=cfg, unbalance=best_cfg[1], itr=best_cfg[0])
            if rank == 0:
                results["ParMETIS"]["times"].append(copy_sim.time)
                results["ParMETIS"]["flips"].append(count_flips(copy_sim, cfg))
                results["ParMETIS"]["evictions"].append(sum(list(copy_sim.total_eviction_movement())[1:]) / 1e9)
                copy_sim = eval_env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.run_until_external_mapping()
                graph: DynamicJacobiGraph = copy_sim.input.graph
                copy_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=4, block_size=4, offset=1)
                copy_sim.run()
                results["BlockCyclic(4x4)"]["times"].append(copy_sim.time)
                results["BlockCyclic(4x4)"]["flips"].append(0)
                results["BlockCyclic(4x4)"]["evictions"].append(sum(list(copy_sim.total_eviction_movement())[1:]) / 1e9)

                copy_sim = eval_env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.run_until_external_mapping()
                graph: DynamicJacobiGraph = copy_sim.input.graph
                copy_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=4, block_size=2, offset=1)
                copy_sim.run()
                results["BlockCyclic(2x2)"]["times"].append(copy_sim.time)
                results["BlockCyclic(2x2)"]["flips"].append(0)
                results["BlockCyclic(2x2)"]["evictions"].append(sum(list(copy_sim.total_eviction_movement())[1:]) / 1e9)

                copy_sim = eval_env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.run_until_external_mapping()
                graph: DynamicJacobiGraph = copy_sim.input.graph
                copy_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=4, block_size=1, offset=1)
                copy_sim.run()
                results["BlockCyclic(1x1)"]["times"].append(copy_sim.time)
                results["BlockCyclic(1x1)"]["flips"].append(0)
                results["BlockCyclic(1x1)"]["evictions"].append(sum(list(copy_sim.total_eviction_movement())[1:]) / 1e9)

                copy_sim = eval_env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.run_until_external_mapping()
                graph: DynamicJacobiGraph = copy_sim.input.graph
                copy_sim.external_mapper = JacobiRoundRobinMapper(n_devices=4, setting=1, offset=1)
                copy_sim.run()
                results["RowCyclic"]["times"].append(copy_sim.time)
                results["RowCyclic"]["flips"].append(0)
                results["RowCyclic"]["evictions"].append(sum(list(copy_sim.total_eviction_movement())[1:]) / 1e9)

                if not skipRL:
                    td = eval_env.rollout(max_steps=1000000, policy=model.actor, auto_reset=False, tensordict=td)
                    results["RL"]["times"].append(eval_env.simulator.time)
                    results["RL"]["flips"].append(count_flips(eval_env.simulator, cfg))
                    results["RL"]["evictions"].append(sum(list(eval_env.simulator.total_eviction_movement())[1:]) / 1e9)
                for k, v in results.items():
                    print(f"Policy {k}:\tmean {np.mean(v['times'])}, std {np.std(v['times'])}", flush=True)

    if rank == 0:
        # Find the policy with the best mean time
        best_policy = None
        best_mean = float("inf")
        for k, v in results.items():
            if k == "RL":
                continue
            mean_time = np.mean(v["times"])
            if mean_time < best_mean:
                best_mean = mean_time
                best_policy = k
        if best_policy == "ParMETIS":
            best_policy = f"ParMETIS({best_cfg[0]},{best_cfg[1]})"
        with open(root_dir / "results.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(["Interior", "Boundary", "Graph", "Memory", "BestPolicyName", "BestPolicy", "RL"])
            writer.writerow(
                [
                    interior_ratio,
                    boundary_ratio,
                    graph_name,
                    f"{int(cfg.system.mem/1e9)}GB",
                    best_policy,
                    f"{int(np.mean(results['ParMETIS' if 'ParMETIS' in best_policy else best_policy]['times']))}",
                    f"{int(np.mean(results.get('RL', {}).get('times', [])))}" if not skipRL and "RL" in results else "N/A",
                ]
            )

        with open(root_dir / "detailed_results.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(["Policy", "Time", "Evictions(GB)", "Flips"])
            for k, v in results.items():
                writer.writerow(
                    [
                        f"ParMETIS({best_cfg[0]},{best_cfg[1]})" if "ParMETIS" in k else k,
                        f"{np.mean(v['times']):.2f}",
                        f"{np.mean(v['evictions']):.2f}",
                        f"{np.mean(v['flips']):.2f}",
                    ]
                )


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
