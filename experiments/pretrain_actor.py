import os
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
from tensordict import TensorDict
from torch.amp import autocast, GradScaler
from task4feedback.experiment_helper.run_name import make_folder_name
from task4feedback.logging import training
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import create_td_actor_critic_models, FeatureDimConfig, create_td_actor_critic_discriminator_models

from task4feedback.ml.env import RuntimeEnv

import math
import time
import torch
import torch.nn.functional as F
from tensordict import TensorDict


# =========================================================
# (1) Load dataset (same shape as Phase 2)
# =========================================================
def load_train_episodes(cfg, save_dir: Path, max_eps=1024):
    temp_dir = save_dir / cfg.eval.expert_path
    assert temp_dir.exists(), f"Phase 1 dataset not found at {temp_dir}"

    epfiles = sorted(temp_dir.glob("episode_*.pkl"))
    total_eps = len(epfiles)

    num_train = max_eps

    print(f"Total episodes found: {total_eps}")

    # ---- Load training episodes ----
    train_eps = []
    for epfile in epfiles[:num_train]:
        with open(epfile, "rb") as f:
            train_eps.append(pickle.load(f))
    train_dataset = torch.cat(train_eps, dim=0)

    print(f"Train dataset size: {train_dataset.batch_size}")

    return train_dataset


def evaluate_exact_match(actor, obs, actions, batch_size=512):
    actor.eval()
    device = next(actor.parameters()).device

    total = 0
    correct = 0

    for i in range(0, obs.shape[0], batch_size):
        batch_obs = obs[i : i + batch_size].to(device)
        batch_act = actions[i : i + batch_size].to(device)

        with torch.no_grad():
            td = TensorDict({"observation": batch_obs}, batch_size=batch_obs.batch_size, device=device)
            logits = actor(td)["logits"]  # [B,64,num_classes]
            pred = logits.argmax(-1)  # [B,64]
            match = pred == batch_act  # [B,64]

        correct += match.sum().item()
        total += match.numel()

    acc = correct / total * 100
    return acc


# =========================================================
# (3) Behavior Cloning loss
# =========================================================


def bc_loss_and_metrics_from_logits(
    logits: torch.Tensor,  # [B, 64, A]
    expert_actions: torch.Tensor,  # [B, 64]
    label_smoothing: float = 0.0,
):
    """
    Returns:
      loss (per-slot mean CE),
      slot_acc (mean over B*64),
      seq_acc (mean over B; 1 if all 64 match),
      entropy (mean entropy over B*64)
    """
    B, T, A = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, A),
        expert_actions.reshape(B * T),
        reduction="mean",
        label_smoothing=label_smoothing,
    )

    with torch.no_grad():
        pred = logits.argmax(-1)  # [B,64]
        slot_acc = (pred == expert_actions).float().mean()

        seq_acc = (pred == expert_actions).all(dim=-1).float().mean()

        probs = logits.softmax(dim=-1)
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).mean()  # average over B*64

    return loss, slot_acc, seq_acc, entropy


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.05,  # final_lr = base_lr * min_lr_ratio
):
    """
    Linear warmup for warmup_steps, then cosine decay to min_lr_ratio.
    Implemented as a LambdaLR on top of the optimizer's initial LR.
    """

    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(step: int):
        # step starts at 0
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def eval_bc(
    actor,
    obs,  # TensorDict or Tensor (batched, indexable)
    expert_actions,  # Tensor [N, 64]
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
    amp: bool = True,
    label_smoothing: float = 0.0,
):
    actor.eval()

    total_slots = 0
    total_loss_sum = 0.0
    total_slot_correct = 0
    total_seq_correct = 0
    total_entropy_sum = 0.0

    use_amp = amp and (device.type == "cuda")

    for start in range(0, indices.numel(), batch_size):
        idx = indices[start : start + batch_size]

        batch_obs = obs[idx]
        if hasattr(batch_obs, "to"):  # TensorDict or Tensor
            batch_obs = batch_obs.to(device)

        batch_act = expert_actions[idx].to(device).long()

        with autocast("cuda", enabled=use_amp):
            td = TensorDict({"observation": batch_obs}, batch_size=[batch_act.shape[0]], device=device)
            logits = actor(td)["logits"]  # [B,64,A]
            loss, slot_acc, seq_acc, entropy = bc_loss_and_metrics_from_logits(logits, batch_act, label_smoothing=label_smoothing)

        # loss is mean over slots; weight by number of slots for global average
        slots = batch_act.numel()
        total_slots += slots
        total_loss_sum += float(loss.item()) * slots
        total_entropy_sum += float(entropy.item()) * slots

        pred = logits.argmax(-1)
        total_slot_correct += int((pred == batch_act).sum().item())
        total_seq_correct += int((pred == batch_act).all(dim=-1).sum().item())

    avg_loss = total_loss_sum / max(1, total_slots)
    avg_entropy = total_entropy_sum / max(1, total_slots)
    slot_acc = total_slot_correct / max(1, total_slots)
    seq_acc = total_seq_correct / max(1, indices.numel())

    return {
        "loss": avg_loss,
        "slot_acc": slot_acc,
        "seq_acc": seq_acc,
        "entropy": avg_entropy,
    }


# =========================================================
# (5) Complete actor pretraining loop
# =========================================================
def pretrain_actor(
    cfg,
    train_dataset,
    actor,
    *,
    lr: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.05,
    label_smoothing: float = 0.0,
    val_ratio: float = 0.05,
    seed: int = 0,
    amp: bool = True,
    log_every_steps: int = 50,
    log_examples_every_epochs: int = 5,
    save_dir: Path | None = None,
    wandb_run=None,  # pass wandb.run or None
):
    """
    BC pretraining with:
      - AdamW
      - warmup + cosine LR
      - AMP (optional)
      - grad clipping
      - train/val split
      - wandb logging
      - checkpoint best + last
    """

    device = next(actor.parameters()).device
    actor = actor.to(device)
    actor.train()

    # Keep dataset on CPU if possible; only move batches.
    obs = train_dataset["observation"]
    expert_actions = train_dataset["action"]  # [N,64]
    N = expert_actions.shape[0]
    print(f"BC dataset size: {N}")

    # --- split ---
    g = torch.Generator().manual_seed(seed)
    perm_all = torch.randperm(N, generator=g)
    val_n = int(N * val_ratio)
    val_idx = perm_all[:val_n]
    train_idx = perm_all[val_n:]
    print(f"Train samples: {train_idx.numel()} | Val samples: {val_idx.numel()}")

    # --- optimizer + scheduler ---
    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(train_idx.numel() / batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # --- AMP ---
    use_amp = amp and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    # --- checkpoints ---
    best_metric = -1.0
    best_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = save_dir / f"bc_actor_best_{cfg.feature.observer.version}_{cfg.eval.expert_path}.pt"
        last_path = save_dir / f"bc_actor_last_{cfg.feature.observer.version}_{cfg.eval.expert_path}.pt"

    global_step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        actor.train()

        # shuffle train indices each epoch
        perm = train_idx[torch.randperm(train_idx.numel(), device=train_idx.device)]

        # running sums weighted by slots for stable averaging
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

            batch_act = expert_actions[idx].to(device).long()

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                td = TensorDict({"observation": batch_obs}, batch_size=[batch_act.shape[0]], device=device)

                logits = actor(td)["logits"]  # [B,64,A]

                loss, slot_acc, seq_acc, entropy = bc_loss_and_metrics_from_logits(logits, batch_act, label_smoothing=label_smoothing)

            scaler.scale(loss).backward()

            # grad clip (after unscale)
            grad_norm = None
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # aggregate stats
            slots = batch_act.numel()
            epoch_slots += slots
            epoch_loss_sum += float(loss.item()) * slots
            epoch_entropy_sum += float(entropy.item()) * slots
            epoch_slot_correct += int(slot_acc.item() * slots)
            epoch_seq_correct += int(seq_acc.item() * batch_act.shape[0])
            epoch_seqs += int(batch_act.shape[0])

            # wandb step logging
            if wandb_run is not None and (global_step % log_every_steps == 0):
                lr_now = optimizer.param_groups[0]["lr"]
                log_dict = {
                    "train/step_loss": float(loss.item()),
                    "train/step_slot_acc": float(slot_acc.item()),
                    "train/step_seq_acc": float(seq_acc.item()),
                    "train/step_entropy": float(entropy.item()),
                    "train/lr": lr_now,
                    "train/epoch": epoch,
                }
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = float(grad_norm.item())
                wandb_run.log(log_dict, step=global_step)

            global_step += 1

        # --- epoch summaries ---
        train_loss = epoch_loss_sum / max(1, epoch_slots)
        train_entropy = epoch_entropy_sum / max(1, epoch_slots)
        train_slot_acc = epoch_slot_correct / max(1, epoch_slots)
        train_seq_acc = epoch_seq_correct / max(1, epoch_seqs)

        # --- validation ---
        val_metrics = eval_bc(
            actor,
            obs=obs,
            expert_actions=expert_actions,
            indices=val_idx,
            device=device,
            batch_size=batch_size,
            amp=amp,
            label_smoothing=label_smoothing,
        )

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_slot_acc={train_slot_acc*100:.2f}% train_seq_acc={train_seq_acc*100:.2f}% | "
            f"val_loss={val_metrics['loss']:.4f} val_slot_acc={val_metrics['slot_acc']*100:.2f}% val_seq_acc={val_metrics['seq_acc']*100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.3e} | {elapsed/60:.1f} min"
        )

        # wandb epoch logging
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/slot_acc": train_slot_acc,
                    "train/seq_acc": train_seq_acc,
                    "train/entropy": train_entropy,
                    "val/loss": val_metrics["loss"],
                    "val/slot_acc": val_metrics["slot_acc"],
                    "val/seq_acc": val_metrics["seq_acc"],
                    "val/entropy": val_metrics["entropy"],
                    "train/lr_epoch_end": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

        # --- checkpoints: pick best by val_seq_acc (strict), fallback val_slot_acc if you prefer ---
        score = float(val_metrics["seq_acc"])
        if score > best_metric:
            best_metric = score
            if best_path is not None:
                torch.save(actor.state_dict(), best_path)
                if wandb_run is not None:
                    wandb_run.log({"ckpt/best_val_seq_acc": best_metric, "ckpt/best_epoch": epoch}, step=global_step)

        if save_dir is not None:
            torch.save(actor.state_dict(), last_path)

        # --- optional: log a couple prediction examples ---
        if wandb_run is not None and (epoch % log_examples_every_epochs == 0) and val_idx.numel() > 0:
            ex = val_idx[torch.randint(0, val_idx.numel(), (2,))]
            batch_obs = obs[ex]
            if hasattr(batch_obs, "to"):
                batch_obs = batch_obs.to(device)
            batch_act = expert_actions[ex].to(device)

            with torch.no_grad(), autocast("cuda", enabled=use_amp):
                td = TensorDict({"observation": batch_obs}, batch_size=[batch_act.shape[0]], device=device)
                logits = actor(td)["logits"]
                pred = logits.argmax(-1)

            import wandb as _wandb

            table = _wandb.Table(columns=["epoch", "idx", "expert_actions", "pred_actions"])
            for k in range(ex.numel()):
                table.add_data(
                    int(epoch),
                    int(ex[k].item()),
                    batch_act[k].detach().cpu().tolist(),
                    pred[k].detach().cpu().tolist(),
                )
            wandb_run.log({"debug/examples": table}, step=global_step)

    return actor


# =========================================================
# (6) Hydra main — mirrors Phase 2 style
# =========================================================
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="8x8x1024_dynamic_lcorners_cnn.yaml", version_base=None)
def main(cfg: DictConfig):
    training.disabled = True

    folder_name, graph_name, interior_str, boundary_str = make_folder_name(cfg)

    # If Hydra changes cwd, this keeps paths relative to the repo root:
    orig_cwd = Path(hydra.utils.get_original_cwd())
    save_dir = orig_cwd / f"dataset/{folder_name}"
    norm_path = orig_cwd / f"norm/{folder_name}/{cfg.feature.observer.version}_norm.pkl"

    # --- Environment setup ---
    graph_builder = make_graph_builder(cfg)
    if norm_path.exists():
        with open(norm_path, "rb") as f:
            normalization = pickle.load(f)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization)
    else:
        raise ValueError(f"Missing normalization file at {norm_path}. Run Phase 1 first.")

    # --- Instantiate actor & critic ---
    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    model, _, _ = create_td_actor_critic_discriminator_models(cfg, feature_config)

    # checkpoint_path = "/home/cc/task4feedback_torchrl/experiments/outputs/workspace/scheduling/model_checkpoints/8w_256lvl_4gpu_noise_10-1_72GB/0.647_D_10000000.pt"
    # checkpoint = torch.load(checkpoint_path, weights_only=False)
    # if "extras" in checkpoint and "discriminator" in checkpoint["extras"]:
    #     print("Found Discriminator in 'extras'. Loading...")
    #     disc_state_dict = checkpoint["extras"]["discriminator"]

    #     # Load into the discriminator submodule
    #     # Note: If your model.discriminator has a different prefix than the checkpoint,
    #     # we use strict=False or manually strip prefixes.
    #     model.discriminator.load_state_dict(disc_state_dict)
    #     print("Discriminator loaded successfully.")
    # else:
    #     print("Key 'discriminator' not found in checkpoint['extras'].")
    #     # Debug: Print keys inside extras to find where it is
    #     if "extras" in checkpoint:
    #         print(f"Available keys in extras: {list(checkpoint['extras'].keys())}")

    # exit()
    actor = model.actor

    # --- Load dataset ---
    print("Loading dataset...")
    train_dataset = load_train_episodes(cfg, save_dir, max_eps=384)

    # --- W&B (optional) ---
    wandb_run = None
    if cfg.get("wandb", None) is not None and cfg.wandb.get("enabled", False):
        import wandb

        wandb_run = wandb.init(
            project="behavior_cloning",
            group=cfg.wandb.get("group", folder_name),
            name=cfg.wandb.get("name", f"{folder_name}-bc"),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(Path.cwd()),  # hydra output dir
        )
        wandb_run.log({"dataset/N": int(train_dataset.batch_size[0])})

    # --- Pretrain actor ---
    actor = pretrain_actor(
        cfg,
        train_dataset=train_dataset,
        actor=actor,
        lr=1e-3,  # with warmup+cosine, 1e-3 is a safer start than 1e-2
        epochs=1000,
        batch_size=256,
        weight_decay=1e-4,
        grad_clip=1.0,
        warmup_ratio=0.05,
        min_lr_ratio=0.05,
        label_smoothing=0.0,
        val_ratio=0.05,
        amp=True,
        log_every_steps=50,
        log_examples_every_epochs=5,
        save_dir=save_dir,  # saves bc_actor_best.pt & bc_actor_last.pt here
        wandb_run=wandb_run,
    )

    # --- Save final actor weights (state_dict only) ---
    out_path = save_dir / f"bc_actor_{cfg.feature.observer.version}_{cfg.eval.expert_path}.pt"
    torch.save(actor.state_dict(), out_path)
    print(f"Saved pretrained actor to {out_path}")

    if wandb_run is not None:
        wandb_run.save(str(out_path))
        wandb_run.finish()


if __name__ == "__main__":
    main()
