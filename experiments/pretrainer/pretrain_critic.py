import os
import torch
import pickle
from pathlib import Path
from tqdm import tqdm

from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *
from task4feedback.ml.util import *

import hydra
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import create_td_actor_critic_models, load_policy_from_checkpoint
from task4feedback.logging import training


def load_all_episodes(save_dir: Path):
    """Load all episode_*.pkl files and concatenate into one big TensorDict [T_total, ...]."""
    temp_dir = save_dir / "parmetis"
    assert temp_dir.exists(), f"No temp episode directory found at {temp_dir}"

    episodes = []
    for epfile in sorted(temp_dir.glob("episode_*.pkl")):
        with open(epfile, "rb") as f:
            td = pickle.load(f)  # td is TensorDict [T, ...]
        episodes.append(td)
        if len(episodes) == 1024:
            break

    print(f"Loaded {len(episodes)} episodes.")
    dataset = torch.cat(episodes, dim=0)
    print(f"Final dataset shape: {dataset.batch_size}")
    return dataset


def compute_mc_returns(rewards: torch.Tensor, gamma: float):
    """
    Compute Monte-Carlo returns for each step in the trajectory.
    This works per-episode: rewards shape must be [T].
    """
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


# -----------------------------------------------------------
# Phase 2: Critic Pretraining
# -----------------------------------------------------------


def pretrain_critic(dataset: TensorDict, critic, cfg):
    """
    Offline critic training phase.
    Supports Option A (MC regression) and Option B (TD backup).

    Now prints actual values vs predicted values every epoch.
    """
    device = next(critic.parameters()).device
    dataset = dataset.to(device)

    gamma = 0.99
    lr = 3e-4
    epochs = 100
    batch_size = 256
    mode = "mc"  # "mc" or "td"

    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    # flatten dataset components
    rewards = dataset["next", "reward"].squeeze(-1)
    next_obs = dataset["next", "observation"]
    obs = dataset["observation"]
    done = dataset["next", "done"].squeeze(-1).float()

    # ---- Targets (MC or TD) ----
    if mode == "mc":
        returns = torch.zeros_like(rewards)
        G = 0.0
        for i in reversed(range(len(rewards))):
            if done[i] == 1:
                G = rewards[i]
            else:
                G = rewards[i] + gamma * G
            returns[i] = G
        target = returns
    else:
        with torch.no_grad():
            td_next = TensorDict(
                {"observation": next_obs},
                batch_size=next_obs.batch_size,
                device=device,
            )
            td_next = critic(td_next)
            next_v = td_next["state_value"].squeeze(-1)
            target = rewards + gamma * next_v * (1 - done)

    n = len(rewards)
    print(f"Critic pretraining dataset size = {n}")

    # ---- Training Loop ----
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        losses = []

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]

            batch_obs = obs[idx]
            batch_target = target[idx]

            td_batch = TensorDict(
                {"observation": batch_obs},
                batch_size=batch_obs.batch_size,
                device=device,
            )

            td_batch = critic(td_batch)
            pred = td_batch["state_value"].squeeze(-1)

            loss = torch.nn.functional.mse_loss(pred, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # ---- Debug Print (Actual vs Predicted) ----
        with torch.no_grad():
            sample_idx = torch.randint(0, n, size=(5,), device=device)  # pick 5 random points
            td_debug = TensorDict(
                {"observation": obs[sample_idx]},
                batch_size=[5],
                device=device,
            )
            td_debug = critic(td_debug)
            pred_debug = td_debug["state_value"].squeeze(-1)
            actual_debug = target[sample_idx]

            mae = torch.mean(torch.abs(pred_debug - actual_debug)).item()

            print(f"\n[Epoch {ep+1}/{epochs}] loss={sum(losses)/len(losses):.6f}")
            print("  Actual returns:  ", actual_debug.detach().cpu().numpy())
            print("  Predicted values:", pred_debug.detach().cpu().numpy())
            print(f"  MAE = {mae:.6f}\n")

    return critic


# -----------------------------------------------------------
# Hydra main
# -----------------------------------------------------------


@hydra.main(config_path="conf", config_name="8x8x1024_dynamic_lcorners_cnn.yaml", version_base=None)
def main(cfg: DictConfig):
    training.disabled = True

    # --- Load normalization + env (same as Phase 1) ---
    save_dir = Path("dataset/phase1_expert_data")
    norm_path = save_dir / "normalization.pkl"

    graph_builder = make_graph_builder(cfg)

    if norm_path.exists():
        with open(norm_path, "rb") as f:
            normalization = pickle.load(f)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization)
    else:
        raise ValueError(f"No normalization file found at {norm_path}. Please run Phase 1 first.")

    # --- Load critic ---
    print("Instantiating critic...")

    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    model, _, _ = create_td_actor_critic_models(cfg, feature_config)
    critic = model.critic

    # --- Load dataset ---
    print("Loading Phase 1 dataset...")
    dataset = load_all_episodes(save_dir)
    improvement = dataset["next"]["observation"]["aux"]["improvement"]  # shape [256000, 1]

    # --- Train critic ---
    critic = pretrain_critic(dataset, critic, cfg)

    # --- Save critic ---
    out_path = save_dir / f"critic_pretrained_{cfg.seed}.pt"
    torch.save(critic.state_dict(), out_path)
    print(f"Saved pretrained critic to {out_path}")


if __name__ == "__main__":
    main()
