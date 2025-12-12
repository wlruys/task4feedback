import torch
import random
import numpy as np
from pathlib import Path
import torchrl
from tqdm import trange
import pickle
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.run_name import make_folder_name
from tensordict import TensorDict

from task4feedback.experiment_helper.parmetis import query_parmetis, run_parmetis
from mpi4py import MPI
from task4feedback.logging import training

from task4feedback.fastsim2 import ParMETIS_wrapper

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import gc


def aggregate_expert_dataset(expert_dir: Path, output_path: Path):
    """
    Aggregate per-episode rollout files into a single dataset file.

    Result format:
        List[TensorDict], one TensorDict per episode
    """
    episode_files = sorted(expert_dir.glob("episode_*.pkl"))
    if len(episode_files) == 0:
        raise RuntimeError(f"No episodes found in {expert_dir}")

    dataset = []
    for ep_file in episode_files:
        with open(ep_file, "rb") as f:
            episode = pickle.load(f)
        dataset.append(episode)

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"📦 Aggregated {len(dataset)} episodes → {output_path}")


def collect_expert_rollouts(env, eft_env, cfg, n_episodes=10, max_steps=10000, save_dir=None):
    """
    Collect expert rollouts and save each episode as its own file:
    f"{cfg.seed}_{env.resets}.pkl".
    """
    if size != 4:
        raise ValueError(f"Expected 4 ranks, but got {size}. Please run with 4 ranks.")
    ParMETIS = ParMETIS_wrapper()

    checkpoint = 1

    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "parmetis").mkdir(parents=True, exist_ok=True)
        while True:
            if (save_dir / "parmetis" / f"episode_{cfg.seed}_{checkpoint}.pkl").exists():
                checkpoint += 1
            else:
                checkpoint -= 1
                break
        print(f"🚀 Starting collection of expert rollouts at episode {checkpoint}.")
        env.set_reset_counter(checkpoint)

    checkpoint = comm.bcast(checkpoint, root=0)

    if checkpoint < n_episodes:
        # First find the best configuration for parmetis
        best_cfg = (None, None, float("inf"))  # (itr, ub, time)
        ub_cur = 1.0001
        for itr in [0.0001001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]:
            if rank == 0:
                temp = env.simulator.copy()
            comm.barrier()
            status = run_parmetis(sim=temp if rank == 0 else None, cfg=cfg, unbalance=ub_cur, itr=itr, n_compute_devices=cfg.system.n_devices - 1, ParMETIS=ParMETIS)
            if rank == 0 and temp.time < best_cfg[2]:
                best_cfg = (itr, ub_cur, temp.time)
                print(f"New best ITR {itr} with time {temp.time}", flush=True)

        best_cfg = comm.bcast(best_cfg, root=0)
        ub_list = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
        for ub in ub_list:
            if rank == 0:
                temp = env.simulator.copy()
            comm.barrier()
            status = run_parmetis(sim=(temp if rank == 0 else None), cfg=cfg, unbalance=ub, itr=best_cfg[0], n_compute_devices=cfg.system.n_devices - 1, ParMETIS=ParMETIS)
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

        # Collect parmetis expert rollouts
        for ep in trange(checkpoint, n_episodes, desc="Collecting expert rollouts", disable=(rank != 0)):
            gc.collect()

            if rank == 0:
                td = env.reset()
                episode_data = []

            done = False
            action = None

            for step in range(max_steps):

                if rank == 0:
                    obs = td["observation"].clone()

                action, status = query_parmetis(
                    ParMETIS,
                    env,
                    cfg,
                    prev_mapping=action,
                    first_call=(step == 0),
                    offset=0,
                    itr=best_cfg[0],
                    unbalance=best_cfg[1],
                )

                action = comm.bcast(action, root=0)

                if rank == 0:
                    td["action"] = torch.tensor(action, dtype=torch.int32)
                    td_next = env.step(td)
                    episode_data.append(td_next.clone())

                    td = td_next["next"]
                    if td_next["next", "done"].any():
                        done = True

                done = comm.bcast(done, root=0)
                if done:
                    break

            if rank == 0:
                episode_tensor = torch.stack(episode_data, dim=0)
                reset_id = env.resets
                episode_path = save_dir / "parmetis" / f"episode_{cfg.seed}_{reset_id}.pkl"

                with open(episode_path, "wb") as f:
                    pickle.dump(episode_tensor, f)

                print(f"Saved episode to {episode_path} (len={episode_tensor.shape[0]})")

    if rank != 0:
        return None

    checkpoint = 1

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "eft").mkdir(parents=True, exist_ok=True)

    while True:
        if (save_dir / "eft" / f"episode_{cfg.seed}_{checkpoint}_eft.pkl").exists():
            checkpoint += 1
        else:
            checkpoint -= 1
            break

    print(f"Starting collection of expert rollouts at episode {checkpoint}.")
    eft_env.set_reset_counter(checkpoint)

    for ep in trange(checkpoint, n_episodes, desc="Collecting expert rollouts"):
        gc.collect()

        td = eft_env.reset()
        episode_data = []

        done = False
        action = None
        runtime = eft_env.simulator.state.get_task_runtime()
        candidate_workspace = torch.zeros(
            eft_env.simulator_factory[eft_env.active_idx].graph_spec.max_candidates,
            dtype=torch.int64,
        )

        for step in range(max_steps):

            obs = td["observation"].clone()

            sim_current = eft_env.simulator.copy()
            sim_current.disable_external_mapper()

            sim_current.set_steps(eft_env.simulator_factory[eft_env.active_idx].graph_spec.max_candidates)
            sim_current.run()
            sim_current.start_drain()
            sim_current.run()

            eft_env.simulator.get_mappable_candidates(candidate_workspace)

            action = []

            for i, id in enumerate(candidate_workspace):
                action.append(runtime.get_compute_task_mapped_device(id.item()))

            td["action"] = torch.tensor(action, dtype=torch.int32)
            td_next = eft_env.step(td)
            episode_data.append(td_next.clone())

            td = td_next["next"]
            if td_next["next", "done"].any():
                done = True

            if done:
                break

        episode_tensor = torch.stack(episode_data, dim=0)
        reset_id = eft_env.resets
        episode_path = save_dir / "eft" / f"episode_{cfg.seed}_{reset_id}_eft.pkl"

        with open(episode_path, "wb") as f:
            pickle.dump(episode_tensor, f)

        print(f"Saved episode to {episode_path} (len={episode_tensor.shape[0]})")

    parmetis_dir = save_dir / "parmetis"
    eft_dir = save_dir / "eft"

    aggregate_expert_dataset(
        parmetis_dir,
        save_dir / "parmetis.pkl",
    )

    aggregate_expert_dataset(
        eft_dir,
        save_dir / "eft.pkl",
    )

    return None


def save_dataset_and_config(dataset, cfg, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    cnt = 0
    dataset_path = save_dir / f"expert_dataset_{cfg.seed}.pkl"
    while dataset_path.exists():
        dataset_path = dataset_path.with_name(dataset_path.stem + f"_{cnt}" + dataset_path.suffix)
        cnt += 1
    config_path = save_dir / "config.yaml"

    # Save dataset
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Expert dataset saved to: {dataset_path}")

    # Save configuration (exact Hydra/OmegaConf)
    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Config saved to: {config_path}")


@hydra.main(config_path="conf", config_name="8x8x1024_dynamic_lcorners_cnn.yaml", version_base=None)
def main(cfg: DictConfig):
    if rank == 0:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        extend = cfg.extend
        num_samples = cfg.eval.samples

        folder_name, graph_name, interior_str, boundary_str = make_folder_name(cfg)

        save_dir = Path(f"dataset/{folder_name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        eval_norm_path = Path(f"evaluate/{folder_name}/norm.pkl")

        cfg.graph.config.steps *= extend
        if cfg.graph.config.workload_args.traj_type == "circle":
            cfg.graph.config.workload_args.traj_specifics.max_angle *= extend

        graph_builder = make_graph_builder(cfg)

        if eval_norm_path.exists():
            with open(eval_norm_path, "rb") as f:
                normalization = pickle.load(f)
            print(f"Loaded normalization from {eval_norm_path}")
            env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization, eval=True)
            eft_env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization, eval=True)
        else:
            normalization = None
            env, normalization = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization, eval=True)
            eft_env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization, eval=True)
        if eval_norm_path.exists() is False and normalization is not None:
            with open(eval_norm_path, "wb") as f:
                pickle.dump(normalization, f)
            print(f"Saved normalization to {eval_norm_path}")

    collect_expert_rollouts(
        env if rank == 0 else None,
        eft_env if rank == 0 else None,
        cfg,
        n_episodes=cfg.get("dataset_size", 2),
        max_steps=100000,
        save_dir=save_dir if rank == 0 else None,
    )


if __name__ == "__main__":
    main()
