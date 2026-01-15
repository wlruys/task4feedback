# =========================================================
# Imports
# =========================================================
import os
import math
import time
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from tensordict import TensorDict
from omegaconf import DictConfig, OmegaConf
import hydra

from task4feedback.experiment_helper.run_name import make_folder_name
from task4feedback.logging import training
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import (
    FeatureDimConfig,
    create_actor_flow_model,
)


# =========================================================
# (1) Load dataset
# =========================================================
def load_train_episodes(cfg, save_dir: Path, max_eps=1024):
    temp_dir = save_dir / cfg.eval.expert_path
    assert temp_dir.exists(), f"Dataset not found at {temp_dir}"

    epfiles = sorted(temp_dir.glob("episode_*.pkl"))
    train_eps = []

    for epfile in epfiles[:max_eps]:
        with open(epfile, "rb") as f:
            train_eps.append(pickle.load(f))

    dataset = torch.cat(train_eps, dim=0)
    print(f"Loaded dataset size: {dataset.batch_size}")
    return dataset


# =========================================================
# (2) LR scheduler
# =========================================================
def make_warmup_cosine_scheduler(
    optimizer,
    total_steps,
    warmup_steps,
    min_lr_ratio=0.05,
):
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================
# (3) Flow matching loss + optional v_pred
# =========================================================
def flow_matching_loss(
    actor,
    observation,
    expert_actions,  # [B, T]
    action_dim,
    device,
    return_v=False,
):
    """
    Conditional Flow Matching:
      x_t = (1 - t) * eps + t * x0
      v*  = x0 - eps
    """

    B, T = expert_actions.shape

    x0 = F.one_hot(expert_actions, num_classes=action_dim).float().to(device)
    eps = torch.randn_like(x0)
    t = torch.rand(B, 1, device=device)
    t_b = t[:, :, None]

    x_t = (1.0 - t_b) * eps + t_b * x0

    state_module = actor[0].module
    H, W = state_module.length, state_module.width

    x_t = x_t.view(B, H, W, action_dim).permute(0, 3, 1, 2)
    v_target = (x0 - eps).view(B, H, W, action_dim).permute(0, 3, 1, 2)

    td = TensorDict(
        {
            "observation": observation,
            "x_t": x_t,
            "t": t,
        },
        batch_size=[B],
        device=device,
    )

    v_pred = actor(td)["v"]
    loss = F.mse_loss(v_pred, v_target)

    if return_v:
        return loss, v_pred
    return loss


# =========================================================
# (4) Metrics from flow prediction
# =========================================================
def flow_metrics_from_v(v_pred, expert_actions):
    """
    Uses argmax over v_pred channels as discrete action proxy.
    """
    B, A, H, W = v_pred.shape
    T = H * W

    logits = v_pred.permute(0, 2, 3, 1).reshape(B, T, A)
    pred = logits.argmax(dim=-1)

    with torch.no_grad():
        slot_acc = (pred == expert_actions).float().mean()
        seq_acc = (pred == expert_actions).all(dim=-1).float().mean()

        probs = logits.softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

    return slot_acc, seq_acc, entropy


# =========================================================
# (5) Validation
# =========================================================


@torch.no_grad()
def sample_actions_from_flow(
    actor,
    observation,
    action_dim,
    *,
    num_steps=50,
    device,
):
    """
    observation: TensorDict or Tensor (batched)
    returns: LongTensor [B, T]
    """
    actor.eval()

    if hasattr(observation, "to"):
        observation = observation.to(device)

    B = observation.batch_size[0]
    state_module = actor[0].module
    H, W = state_module.length, state_module.width
    T = H * W

    # initial noise
    x = torch.randn(B, action_dim, H, W, device=device)

    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = ts[i].expand(B, 1)

        td = TensorDict(
            {
                "observation": observation,
                "x_t": x,
                "t": t,
            },
            batch_size=[B],
            device=device,
        )

        v = actor(td)["v"]
        x = x + dt * v

    # x ≈ x0 (one-hot-ish)
    logits = x.permute(0, 2, 3, 1).reshape(B, T, action_dim)
    actions = logits.argmax(dim=-1)

    return actions


@torch.no_grad()
def eval_flow_sampling(
    actor,
    obs,
    expert_actions,
    indices,
    action_dim,
    device,
    *,
    batch_size=64,
    num_steps=50,
):
    actor.eval()

    total_slots = 0
    slot_correct = 0
    seq_correct = 0

    for start in range(0, indices.numel(), batch_size):
        idx = indices[start : start + batch_size]

        batch_obs = obs[idx]
        if hasattr(batch_obs, "to"):
            batch_obs = batch_obs.to(device)

        batch_act = expert_actions[idx].to(device)

        pred = sample_actions_from_flow(
            actor,
            batch_obs,
            action_dim,
            num_steps=num_steps,
            device=device,
        )

        match = pred == batch_act

        total_slots += match.numel()
        slot_correct += match.sum().item()
        seq_correct += match.all(dim=-1).sum().item()

    return {
        "slot_acc": slot_correct / max(1, total_slots),
        "seq_acc": seq_correct / max(1, indices.numel()),
    }


# =========================================================
# (6) Pretrain actor with Flow Matching + logging
# =========================================================
def pretrain_actor_flow(
    cfg,
    train_dataset,
    actor,
    *,
    lr=1e-4,
    epochs=200,
    batch_size=256,
    weight_decay=1e-4,
    grad_clip=1.0,
    warmup_ratio=0.05,
    min_lr_ratio=0.05,
    val_ratio=0.05,
    seed=0,
    amp=True,
    save_dir=None,
    wandb_run=None,
):
    device = next(actor.parameters()).device
    actor.train()

    obs = train_dataset["observation"]
    expert_actions = train_dataset["action"].long()
    N = expert_actions.shape[0]
    action_dim = cfg.system.n_devices - 1

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    val_n = int(N * val_ratio)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(train_idx.numel() / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        total_steps,
        warmup_steps,
        min_lr_ratio,
    )

    scaler = GradScaler("cuda", enabled=amp and device.type == "cuda")

    best_val = float(0.0)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = save_dir / "flow_actor_best.pt"
        last_path = save_dir / "flow_actor_last.pt"

    global_step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        actor.train()
        perm = train_idx[torch.randperm(train_idx.numel())]

        epoch_slots = 0
        epoch_loss_sum = 0.0
        epoch_entropy_sum = 0.0
        epoch_slot_correct = 0
        epoch_seq_correct = 0
        epoch_seqs = 0

        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]

            batch_obs = obs[idx]
            if hasattr(batch_obs, "to"):
                batch_obs = batch_obs.to(device)

            batch_act = expert_actions[idx].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=amp and device.type == "cuda"):
                loss, v_pred = flow_matching_loss(
                    actor,
                    batch_obs,
                    batch_act,
                    action_dim,
                    device,
                    return_v=True,
                )

            slot_acc, seq_acc, entropy = flow_metrics_from_v(v_pred, batch_act)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            slots = batch_act.numel()
            epoch_slots += slots
            epoch_loss_sum += float(loss.item()) * slots
            epoch_entropy_sum += float(entropy.item()) * slots
            epoch_slot_correct += int(slot_acc.item() * slots)
            epoch_seq_correct += int(seq_acc.item() * batch_act.shape[0])
            epoch_seqs += batch_act.shape[0]

            global_step += 1

        train_loss = epoch_loss_sum / max(1, epoch_slots)
        train_entropy = epoch_entropy_sum / max(1, epoch_slots)
        train_slot_acc = epoch_slot_correct / max(1, epoch_slots)
        train_seq_acc = epoch_seq_correct / max(1, epoch_seqs)

        val_metrics = eval_flow_sampling(
            actor,
            obs,
            expert_actions,
            val_idx,
            action_dim,
            device,
            batch_size=batch_size,
            num_steps=50,
        )

        elapsed = (time.time() - t0) / 60
        print(
            f"Epoch {epoch:04d} | "
            f"train_loss={train_loss:.6f} "
            f"train_slot_acc={train_slot_acc*100:.2f}% "
            f"train_seq_acc={train_seq_acc*100:.2f}% | "
            f"val_slot_acc={val_metrics['slot_acc']*100:.2f}% "
            f"val_seq_acc={val_metrics['seq_acc']*100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f} min"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/slot_acc": val_metrics["slot_acc"],
                    "val/seq_acc": val_metrics["seq_acc"],
                    "train/lr_epoch_end": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

        if val_metrics["seq_acc"] > best_val:
            best_val = val_metrics["seq_acc"]
            if save_dir is not None:
                torch.save(actor.state_dict(), best_path)

        if save_dir is not None:
            torch.save(actor.state_dict(), last_path)

    return actor


# =========================================================
# (7) Hydra main
# =========================================================
@hydra.main(
    config_path="conf",
    config_name="8x8x1024_dynamic_lcorners_cnn.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    training.disabled = True

    folder_name, *_ = make_folder_name(cfg)
    orig_cwd = Path(hydra.utils.get_original_cwd())

    save_dir = orig_cwd / f"dataset/{folder_name}"
    norm_path = orig_cwd / f"norm/{folder_name}/{cfg.feature.observer.version}_norm.pkl"

    graph_builder = make_graph_builder(cfg)
    with open(norm_path, "rb") as f:
        normalization = pickle.load(f)

    env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization)
    observer = env.get_observer()

    feature_cfg = FeatureDimConfig.from_observer(observer)
    actor = create_actor_flow_model(cfg, feature_cfg)

    print("Loading dataset...")
    dataset = load_train_episodes(cfg, save_dir, max_eps=384)

    wandb_run = None
    if cfg.get("wandb", None) is not None and cfg.wandb.get("enabled", False):
        import wandb

        wandb_run = wandb.init(
            project="behavior_cloning",
            group=cfg.wandb.get("group", folder_name),
            name=cfg.wandb.get("name", f"{folder_name}-flow"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    actor = pretrain_actor_flow(
        cfg,
        train_dataset=dataset,
        actor=actor,
        lr=1e-3,
        epochs=1000,
        batch_size=256,
        save_dir=save_dir,
        wandb_run=wandb_run,
    )

    out_path = save_dir / "flow_actor_final.pt"
    torch.save(actor.state_dict(), out_path)
    print(f"Saved flow actor to {out_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
