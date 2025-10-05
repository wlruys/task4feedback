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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class GitInfo(Callback):
    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        try:
            repo = git.Repo(search_parent_directories=True)
            outdir = Path(config.hydra.runtime.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "git_sha.txt").write_text(repo.head.commit.hexsha)
            (outdir / "git_dirty.txt").write_text(str(repo.is_dirty()))
            diff = repo.git.diff(None)
            (outdir / "git_diff.patch").write_text(diff)

            print(
                "Git SHA:",
                repo.head.commit.hexsha,
                " (dirty)" if repo.is_dirty() else " (clean)",
                flush=True,
            )

        except Exception as e:
            print(f"GitInfo callback failed: {e}")


def configure_training(cfg: DictConfig):
    # start_logger()
    # Attempt to load policy weights from a local checkpoint next to this file
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

    proceed = False
    # saved_models/8x8x128_10-1-1_corners_74GB/5.286_D.pt
    if rank == 0:
        root_dir = (
            Path(__file__).resolve().parent
            / "saved_models"
            / f"{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}_{interior_ratio}-{boundary_ratio}-1_{graph_name}_{int(cfg.system.mem/1e9)}GB"
        )

        files = list(root_dir.rglob("*.pt"))
        if len(files) == 0:
            print(f"No saved models found in {root_dir}, exiting.")
            proceed = False
        elif len(files) > 1:
            print(f"Multiple saved models found in {root_dir}, please specify a more specific policy or second_best. Found:")
            proceed = False

        ckpt_path = root_dir / f"{list(files)[0].name}"

    if rank == 0:
        cfg.feature.add_device_load = True
        cfg.feature.observer.prev_frames = 1
        norm_file = root_dir / f"norm.pkl"

        graph_builder = make_graph_builder(cfg)
        if norm_file.exists():
            print(f"Loading normalization from {norm_file}")
            norm = pickle.load(open(norm_file, "rb"))
            env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=norm, eval=True)
        else:
            print(f"Normalization file {norm_file} not found, creating new normalization")
            env, norm = make_env(graph_builder=graph_builder, cfg=cfg, eval=True)
            pickle.dump(norm, open(norm_file, "wb"))

        observer = env.get_observer()
        feature_config = FeatureDimConfig.from_observer(observer)
        model, _, _ = create_td_actor_critic_models(cfg, feature_config)

        loaded = load_policy_from_checkpoint(model, ckpt_path)
        if not loaded:
            print(f"Found {ckpt_path}, but not a compatible policy module to load into.")
            proceed = False
        else:
            proceed = True

    proceed = comm.bcast(proceed, root=0)
    if not proceed:
        print("No compatible model found, exiting.")
        exit()

    if rank == 0:
        eval_env = make_env(
            graph_builder=graph_builder,
            cfg=cfg,
            normalization=norm,
            eval=True,
        )

        model.eval()

    results = {}
    for policy in ["ParMETIS", "BlockCyclic(2x2)", "BlockCyclic(1x1)", "EFT", "RowCyclic", "RL"]:
        results[policy] = {
            "times": [],
        }

    # First find the best configuration for parmetis
    best_cfg = (None, None, float("inf"))  # (itr, ub, time)
    ub_cur = 1.01
    for itr in [0.0001001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        if rank == 0:
            temp = eval_env.simulator.copy()
        run_parmetis(sim=temp if rank == 0 else None, cfg=cfg, unbalance=ub_cur, itr=itr)
        if rank == 0 and temp.time < best_cfg[2]:
            best_cfg = (itr, ub_cur, temp.time)
            print(f"New best ITR {itr} with time {temp.time}", flush=True)

    best_cfg = comm.bcast(best_cfg, root=0)

    for ub in [1.0001, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
        if rank == 0:
            temp = eval_env.simulator.copy()
        run_parmetis(sim=(temp if rank == 0 else None), cfg=cfg, unbalance=ub, itr=best_cfg[0])
        if rank == 0:
            print(f"Tried ub {ub:.2f} with time {temp.time}", flush=True)
            if temp.time < best_cfg[2]:
                # Improvement: accept move, keep direction, keep step
                ub_cur = ub
                best_cfg = (best_cfg[0], ub_cur, temp.time)
                print(f"New best ub {ub_cur:.2f} with time {temp.time}", flush=True)

    best_cfg = comm.bcast(best_cfg, root=0)

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for i in range(n_samples):
            gc.collect()
            if rank == 0:
                td = eval_env.reset()
                eft_time = eval_env._get_baseline("EFT")
                results["EFT"]["times"].append(eft_time)
                copy_sim = eval_env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.run_until_external_mapping()

            run_parmetis(sim=copy_sim if rank == 0 else None, cfg=cfg, unbalance=best_cfg[1], itr=best_cfg[0])
            if rank == 0:
                results["ParMETIS"]["times"].append(copy_sim.time)
                copy_sim = eval_env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.run_until_external_mapping()
                graph: DynamicJacobiGraph = copy_sim.input.graph
                copy_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=4, block_size=2, offset=1)
                copy_sim.run()
                results["BlockCyclic(2x2)"]["times"].append(copy_sim.time)
                copy_sim = eval_env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.run_until_external_mapping()
                graph: DynamicJacobiGraph = copy_sim.input.graph
                copy_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=4, block_size=1, offset=0)
                copy_sim.run()
                results["BlockCyclic(1x1)"]["times"].append(copy_sim.time)
                copy_sim = eval_env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.run_until_external_mapping()
                graph: DynamicJacobiGraph = copy_sim.input.graph
                copy_sim.external_mapper = JacobiRoundRobinMapper(n_devices=4, setting=1, offset=1)
                copy_sim.run()
                results["RowCyclic"]["times"].append(copy_sim.time)

                td = eval_env.rollout(max_steps=100000, policy=model.actor, auto_reset=False, tensordict=td)
                results["RL"]["times"].append(eval_env.simulator.time)
                # for k, v in results.items():
                #     print(f"Policy {k}: times {v['times']}, mean {np.mean(v['times'])}, std {np.std(v['times'])}", flush=True)

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
        with open(root_dir / "results.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Interior", "Boundary", "Graph", "Memory", "BestPolicyName", "BestPolicy", "RL"])
            writer.writerow(
                [
                    interior_ratio,
                    boundary_ratio,
                    graph_name,
                    f"{int(cfg.system.mem/1e9)}GB",
                    best_policy,
                    f"{np.mean(results['EFT']['times'])/np.mean(results[best_policy]['times']):.2f}",
                    f"{np.mean(results['EFT']['times'])/np.mean(results['RL']['times']):.2f}",
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
