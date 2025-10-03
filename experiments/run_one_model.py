import csv
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


def parse_policy(policy_str: str):
    """
    Parse cfg.sweep.policy string into its components.
    Supports Oracle(k), ParMETIS(ub,itr), BlockCyclic, EFT.
    """
    # Regex to capture function-like calls
    match = re.match(r"([A-Za-z]+)\(([^)]*)\)", policy_str)

    if match:
        name = match.group(1)
        args_str = match.group(2)
        # Split args by comma, convert to int or float
        args = []
        for arg in args_str.split(","):
            arg = arg.strip()
            if arg.isdigit():
                args.append(int(arg))
            else:
                try:
                    args.append(float(arg))
                except ValueError:
                    args.append(arg)  # fallback as string
        return name, args
    else:
        # Just a string (BlockCyclic, EFT, etc.)
        return policy_str, []


def configure_training(cfg: DictConfig):
    # start_logger()
    # Attempt to load policy weights from a local checkpoint next to this file
    n_samples = 100
    if not cfg.graph.env.change_priority and not cfg.graph.env.change_location and not cfg.graph.env.change_workload and not cfg.graph.env.change_duration:
        n_samples = 1

    def closest_ratio_string(value: float) -> str:
        mapping = {10: "10", 1: "1", 0.1: "0.1"}
        closest = min(mapping.keys(), key=lambda x: abs(value - x))
        return mapping[closest]

    interior_ratio = 595.5555555 / (cfg.graph.config.arithmetic_intensity)
    boundary_ratio = interior_ratio * cfg.graph.config.boundary_width * 4

    interior_ratio = closest_ratio_string(interior_ratio)
    boundary_ratio = closest_ratio_string(boundary_ratio)

    best_policy, best_args = parse_policy(cfg.sweep.policy)
    second_policy, second_args = parse_policy(cfg.sweep.second_best)

    if OmegaConf.select(cfg, "graph.config.workload_args.traj_type") is not None:
        graph_name = cfg.graph.config.workload_args.traj_type
    else:
        graph_name = "static"
    if "Dilation" in cfg.network.layers.state._target_:
        if "Uncond" in cfg.network.layers.state._target_:
            network = "UncondCNN"
        else:
            network = "CNN"
    elif "Vector" in cfg.network.layers.state._target_:
        network = "Vector"
    elif "GNN" in cfg.network.layers.state._target_:
        network = "GNN"
    else:
        print(cfg.network.layers.state._target_)
        raise ValueError("Unknown network type in cfg.network.layers.state._target_")

    proceed = False
    results = []

    if rank == 0:
        root_dir = Path(__file__).resolve().parent / "saved_models" / f"8x8x128_{interior_ratio}-{boundary_ratio}-1_{graph_name}_{network}"

        saved_models_dir = root_dir / "models"
        norms_dir = root_dir / "norms"
        results_dir = root_dir / "results"

        root_dir.mkdir(parents=True, exist_ok=True)
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        norms_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Looking for models in {saved_models_dir}")

        pattern = re.compile(
            r"(?P<grid>\d+x\d+x\d+)_"  # grid
            r"(?P<interior>\d+(?:\.\d+)?)[\:\-]"  # interior
            r"(?P<boundary>\d+(?:\.\d+)?)[\:\-]1_"  # boundary
            r"(?P<traj_type>[^_]+)_"  # traj_type
            rf"CNN_{cfg.feature.observer.version}_"  # observer version
            r"Device(?P<device>\w+)_"  # device flag
            r"(?P<frames>\d+)Frames"  # frames
            r"(?:_.*)?"  # <-- allow dots in extra metadata
            r"\.pt$"  # extension
        )

        for file in saved_models_dir.rglob("*.pt"):
            print(f"Checking file {file.name}")
            match = pattern.match(file.name)
            if match:
                info = match.groupdict()
                info["path"] = str(file)
                results.append(info)

    results = comm.bcast(results, root=0)

    for item in results:
        if rank == 0:
            if interior_ratio != item["interior"] or boundary_ratio != item["boundary"]:
                raise ValueError(f"Loaded model with different ratio: {item['interior']}:{item['boundary']}:1 (expected {interior_ratio}:{boundary_ratio}:1)")
            if cfg.graph.config.workload_args.traj_type != item["traj_type"]:
                raise ValueError(f"Loaded model with different traj_type: {item['traj_type']} (expected {cfg.graph.config.workload_args.traj_type})")

            cfg.feature.add_device_load = item["device"] in ["1", "True", "true", "T", "t"]
            cfg.feature.observer.prev_frames = int(item["frames"])
            print(f"Running model: {item['path']} with observer version {cfg.feature.observer.version}, add_device_load={cfg.feature.add_device_load}, prev_frames={cfg.feature.observer.prev_frames}")

            model_name = f"8x8x128_{interior_ratio}-{boundary_ratio}-1_{cfg.graph.config.workload_args.traj_type}_CNN_{cfg.feature.observer.version}_Device{cfg.feature.add_device_load}_{cfg.feature.observer.prev_frames}Frames"

            ckpt_path = item["path"]
            norm_file = norms_dir / f"{model_name}_norm.pkl"

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
                break
            else:
                proceed = True

        proceed = comm.bcast(proceed, root=0)
        if not proceed:
            print("No compatible model found, exiting.")
            # print(f"Found {ckpt_path}, but not a compatible policy module to load into.")
            exit()

        if rank == 0:
            eval_env = make_env(
                graph_builder=graph_builder,
                cfg=cfg,
                normalization=norm,
                eval=True,
            )

            model.eval()
            eval_config = instantiate(cfg.eval)
            vsBest = []
            vsEFT = []
            vsSecond = []
            result_path = results_dir / "plain" / f"{model_name}_result.txt"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_string = ""

            def log_line(message: str) -> None:
                print(message)
                nonlocal result_string
                result_string += message + "\n"

            raw_rows = []  # collect raw data

            log_line("Starting evaluation run")
            log_line(f"Checkpoint: {ckpt_path}")

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            for i in range(n_samples):
                if rank == 0:
                    td = eval_env.reset()
                    eft_time = eval_env._get_baseline("EFT")
                    best_policy_sim = eval_env.simulator.copy()
                    best_policy_sim.enable_external_mapper()
                    best_policy_sim.run_until_external_mapping()

                    second_policy_sim = eval_env.simulator.copy()
                    second_policy_sim.enable_external_mapper()
                    second_policy_sim.run_until_external_mapping()

                if best_policy == "Oracle" and rank == 0:
                    graph = eval_env.get_graph()
                    graph.mincut_per_levels(
                        bandwidth=cfg.system.d2d_bw,
                        mode="metis",
                        offset=1,
                        level_chunks=best_args[0],
                    )
                    graph.align_partitions()
                    best_policy_sim.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
                    best_policy_sim.run()

                elif best_policy == "ParMETIS":
                    run_parmetis(sim=best_policy_sim if rank == 0 else None, cfg=cfg, unbalance=best_args[0], itr=best_args[1])

                elif best_policy == "BlockCyclic" and rank == 0:
                    print("Using BlockCyclic mapping")
                    graph = eval_env.get_graph()
                    best_policy_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=4, block_size=1, offset=1)
                    best_policy_sim.run()
                    print(f"BlockCyclic time: {best_policy_sim.time}")
                elif best_policy == "EFT" and rank == 0:
                    print("Using EFT as best policy")
                    # EFT is already computed
                    pass

                if second_policy == "Oracle" and rank == 0:
                    graph = eval_env.get_graph()
                    graph.mincut_per_levels(
                        bandwidth=cfg.system.d2d_bw,
                        mode="metis",
                        offset=1,
                        level_chunks=second_args[0],
                    )
                    graph.align_partitions()
                    second_policy_sim.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
                    second_policy_sim.run()

                elif second_policy == "ParMETIS":
                    run_parmetis(sim=second_policy_sim if rank == 0 else None, cfg=cfg, unbalance=second_args[0], itr=second_args[1])

                elif second_policy == "BlockCyclic" and rank == 0:
                    print("Using BlockCyclic mapping")
                    graph = eval_env.get_graph()
                    second_policy_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=4, block_size=1, offset=1)
                    second_policy_sim.run()
                    print(f"BlockCyclic time: {second_policy_sim.time}")
                elif second_policy == "EFT" and rank == 0:
                    print("Using EFT as second best policy")
                    # EFT is already computed
                    pass
                if rank == 0:
                    print("Running policy model")
                    td = eval_env.rollout(policy=model.actor, max_steps=10000, auto_reset=False, tensordict=td)
                    if best_policy == "EFT":
                        best_policy_time = eft_time
                    else:
                        best_policy_time = best_policy_sim.time
                    if second_policy == "EFT":
                        second_policy_time = eft_time
                    else:
                        second_policy_time = second_policy_sim.time
                    ml_time = td["observation", "aux", "time"][-1].item()
                    vsBest.append(best_policy_time / ml_time)
                    vsSecond.append(second_policy_time / ml_time)
                    vsEFT.append(eft_time / ml_time)

                    raw_rows.append(
                        {
                            "iter": i,
                            "eft_time": f"{eft_time:.3f}",
                            "best_policy_time": f"{best_policy_time:.3f}",
                            "ml_time": f"{ml_time:.3f}",
                            "vsEFT": f"{eft_time / ml_time:.3f}",
                            "vsBest": f"{best_policy_time / ml_time:.3f}",
                            "vsSecond": f"{second_policy_time / ml_time:.3f}",
                        }
                    )

                    log_line(
                        f"iter {i:03d} | eft_time={eft_time:.3f} "
                        f"| best_policy_time={best_policy_time:.3f} "
                        f"| ml_time={ml_time:.3f} "
                        f"| vsEFT={eft_time / ml_time:.2f}x "
                        f"| vsBest={best_policy_time / ml_time:.2f}x "
                        f"| vsSecond={second_policy_time / ml_time:.2f}x"
                    )
                    print(f"Completed {i+1}/100", flush=True)

        if rank == 0:
            # Compute summary
            eft_q1, eft_q2, eft_q3 = np.percentile(vsEFT, [25, 50, 75])
            policy_q1, policy_q2, policy_q3 = np.percentile(vsBest, [25, 50, 75])
            summary = {
                "traj_type": cfg.graph.config.workload_args.traj_type,
                "Interior": interior_ratio,
                "Boundary": boundary_ratio,
                "version": cfg.feature.observer.version,
                "eft_worst": f"{min(vsEFT):.3f}",
                "eft_q1": f"{eft_q1:.3f}",
                "eft_q2": f"{eft_q2:.3f}",
                "eft_q3": f"{eft_q3:.3f}",
                "eft_best": f"{max(vsEFT):.3f}",
                "policy_worst": f"{min(vsBest):.3f}",
                "policy_q1": f"{policy_q1:.3f}",
                "policy_q2": f"{policy_q2:.3f}",
                "policy_q3": f"{policy_q3:.3f}",
                "policy_best": f"{max(vsBest):.3f}",
                "second_worst": f"{min(vsSecond):.3f}",
                "second_q1": f"{np.percentile(vsSecond, 25):.3f}",
                "second_q2": f"{np.percentile(vsSecond, 50):.3f}",
                "second_q3": f"{np.percentile(vsSecond, 75):.3f}",
                "second_best": f"{max(vsSecond):.3f}",
            }

            log_line(("vsEFT quartiles:    " f"Worst={min(vsEFT):.2f}  " f"Q1={eft_q1:.2f}, Q2={eft_q2:.2f}, Q3={eft_q3:.2f}, " f"Best={max(vsEFT):.2f}"))
            log_line(("vsPolicy quartiles: " f"Worst={min(vsBest):.2f} " f"Q1={policy_q1:.2f}, Q2={policy_q2:.2f}, Q3={policy_q3:.2f}  " f"Best={max(vsBest):.2f}"))
            log_line(
                (
                    "vsSecond quartiles: "
                    f"Worst={min(vsSecond):.2f} "
                    f"Q1={np.percentile(vsSecond, 25):.2f}, Q2={np.percentile(vsSecond, 50):.2f}, "
                    f"Q3={np.percentile(vsSecond, 75):.2f}, Best={max(vsSecond):.2f}"
                )
            )
            log_line(f"Detailed results saved to {result_path}")

            print(f"Evaluation summary written to {result_path}")

            # --- Save raw data CSV ---
            raw_csv_path = results_dir / "raw" / f"{model_name}_raw.csv"
            raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
                writer.writeheader()
                writer.writerows(raw_rows)
            print(f"Raw results CSV saved to {raw_csv_path}")

            # --- Save summary CSV ---
            summary_csv_path = results_dir / f"{model_name}_summary.csv"
            with summary_csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
                writer.writeheader()
                writer.writerow(summary)
            print(f"Summary CSV saved to {summary_csv_path}")

            # --- Append to aggregated summary CSV in root_dir ---
            aggregated_csv_path = root_dir / "aggregated_summary.csv"
            write_header = not aggregated_csv_path.exists()
            with aggregated_csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(summary)
            print(f"Aggregated summary updated at {aggregated_csv_path}")


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
