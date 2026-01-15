import os
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
from tensordict import TensorDict

import hydra
from omegaconf import DictConfig

from task4feedback.experiment_helper.run_name import make_folder_name
from task4feedback.logging import training
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import (
    create_td_actor_critic_models,
    FeatureDimConfig,
)
from task4feedback.experiment_helper.parmetis import query_parmetis, run_parmetis
from mpi4py import MPI
from task4feedback.fastsim2 import ParMETIS_wrapper
import gc

import math
import time
from omegaconf import OmegaConf
from torch.amp import autocast, GradScaler

import torch.nn.functional as F


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ============================================================
# Utility: Load original Phase 1 expert dataset
# ============================================================
def load_phase1_dataset(cfg, save_dir: Path, max_eps=384):
    temp_dir = save_dir / f"{cfg.eval.expert_path}"
    assert temp_dir.exists(), f"Phase 1 dataset not found at {temp_dir}"

    episodes = []
    for epfile in sorted(temp_dir.glob("episode_*.pkl")):
        with open(epfile, "rb") as f:
            td = pickle.load(f)
        episodes.append(td)
        if len(episodes) >= max_eps:
            break

    dataset = torch.cat(episodes, dim=0)
    print(f"[DAgger] Loaded Phase 1 expert dataset: {dataset.batch_size}")
    return dataset


# ============================================================
# (1) Collect actor rollouts + expert labels
# ============================================================
def collect_dagger_parmetis_data(ParMETIS, env, actor, cfg, best_cfg, max_steps=5000, max_episodes=32):
    new_data = []
    if rank == 0:
        actor.eval()
    for ep in tqdm(range(max_episodes), desc="DAgger Rollouts", disable=(rank != 0)):
        if rank == 0:
            td = env.reset()
        prev_mapping = None
        done = False
        temp_data = []
        while not done:
            for step in range(max_steps):
                # Actor predicts
                if rank == 0:
                    td_act = TensorDict({"observation": td["observation"]}, batch_size=td.batch_size)
                    td_act = actor(td_act)
                    logits = td_act["logits"]
                    actor_action = logits.argmax(-1).squeeze(0)  # [64]

                # Expert label

                expert_action, status = query_parmetis(
                    ParMETIS,
                    env,
                    cfg,
                    prev_mapping=prev_mapping,
                    first_call=(step == 0),
                    offset=0,
                    itr=best_cfg[0],
                    unbalance=best_cfg[1],
                )
                status = True
                if status is False:
                    if rank == 0:
                        env.reset()
                        print("ParMETIS failed during DAgger data collection. Retrying episode...", flush=True)
                        temp_data = []
                        prev_mapping = None
                        gc.collect()
                    break
                if rank == 0:
                    expert_action = torch.tensor(expert_action, dtype=torch.int64)
                    prev_mapping = actor_action.cpu().numpy()

                    # Step environment with *actor's* action
                    td["action"] = actor_action
                    td_next = env.step(td)

                    # Save expert label as ground truth
                    td_labeled = td_next.clone()
                    td_labeled["action"] = expert_action
                    temp_data.append(td_labeled)

                    td = td_next["next"]
                    done = td_next["next", "done"].any()
                done = comm.bcast(done, root=0)
                if done:
                    break
        if rank == 0:
            new_data.extend(temp_data)
    if rank == 0:
        new_data = torch.stack(new_data, dim=0)
        print(f"[DAgger] Collected new data: {new_data.batch_size}")

        return new_data
    else:
        return None


def collect_dagger_eft_data(env, actor, cfg, max_steps=5000, max_episodes=32):
    candidate_workspace = torch.zeros(
        env.simulator_factory[env.active_idx].graph_spec.max_candidates,
        dtype=torch.int64,
    )
    new_data = []
    if rank == 0:
        actor.eval()
    for ep in tqdm(range(max_episodes), desc="DAgger Rollouts"):
        td = env.reset()
        done = False
        temp_data = []
        while not done:
            for step in range(max_steps):
                # Actor predicts
                td_act = TensorDict({"observation": td["observation"]}, batch_size=td.batch_size)
                td_act = actor(td_act)
                logits = td_act["logits"]
                actor_action = logits.argmax(-1).squeeze(0)  # [64]

                # Expert label
                sim_reference = env.simulator.copy()
                sim_reference.disable_external_mapper()
                sim_reference.run()
                runtime = sim_reference.state.get_task_runtime()

                expert_action = []
                num_candidates = env.simulator.get_mappable_candidates(candidate_workspace)
                if num_candidates == 0:
                    print(env.simulator.time, "No candidates to map, enabling external mapper.")
                    exit()

                for id in candidate_workspace:
                    expert_action.append(runtime.get_compute_task_mapped_device(id.item()) - 1)

                expert_action = torch.tensor(expert_action, dtype=torch.int64)

                # Step environment with *actor's* action
                td["action"] = actor_action
                td_next = env.step(td)

                # Save expert label as ground truth
                td_labeled = td_next.clone()
                td_labeled["action"] = expert_action
                temp_data.append(td_labeled)

                td = td_next["next"]
                done = td_next["next", "done"].any()
                if done:
                    break
        new_data.extend(temp_data)
    new_data = torch.stack(new_data, dim=0)
    print(f"[DAgger] Collected new data: {new_data.batch_size}")

    return new_data


# ============================================================
# (2) Measure disagreement rate
# ============================================================
@torch.no_grad()
def compute_disagreement_rates(actor, dataset, device, sample_size=256, amp=True):
    obs = dataset["observation"]
    expert_actions = dataset["action"].long()

    N = obs.shape[0]
    if N > sample_size:
        idx = torch.randint(0, N, (sample_size,))
        obs = obs[idx]
        expert_actions = expert_actions[idx]

    actor.eval()
    use_amp = amp and (device.type == "cuda")

    batch_obs = obs.to(device) if hasattr(obs, "to") else obs
    batch_act = expert_actions.to(device).long()

    with autocast("cuda", enabled=use_amp):
        td = TensorDict({"observation": batch_obs}, batch_size=batch_obs.batch_size, device=device)
        logits = actor(td)["logits"]
        pred = logits.argmax(-1)

    slot_disagree = (pred != batch_act).float().mean().item()
    seq_disagree = (~(pred == batch_act).all(dim=-1)).float().mean().item()
    return slot_disagree, seq_disagree


# ============================================================
# (3) Behavior Cloning
# ============================================================
def bc_loss_and_metrics_from_logits(
    logits: torch.Tensor,  # [B, 64, A]
    expert_actions: torch.Tensor,  # [B, 64]
    label_smoothing: float = 0.0,
):
    B, T, A = logits.shape
    expert_actions = expert_actions.long()
    logits = logits.contiguous()
    expert_actions = expert_actions.contiguous()

    loss = F.cross_entropy(
        logits.view(B * T, A),
        expert_actions.view(B * T),
        reduction="mean",
        label_smoothing=label_smoothing,
    )

    with torch.no_grad():
        pred = logits.argmax(-1)  # [B,64]
        slot_acc = (pred == expert_actions).float().mean()
        seq_acc = (pred == expert_actions).all(dim=-1).float().mean()
        probs = logits.softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

    return loss, slot_acc, seq_acc, entropy


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.05,
):
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def retrain_actor(
    dataset,
    actor,
    *,
    epochs=5,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    warmup_ratio=0.05,
    min_lr_ratio=0.05,
    label_smoothing=0.0,
    val_ratio=0.00,
    amp=True,
    seed=0,
    wandb_run=None,
    iter_idx=None,  # DAgger iteration (for logging)
    log_every_steps=50,
):
    device = next(actor.parameters()).device
    actor.train()

    obs = dataset["observation"]  # keep on CPU
    actions = dataset["action"].long()
    N = actions.shape[0]

    # split
    g = torch.Generator().manual_seed(seed)
    perm_all = torch.randperm(N, generator=g)
    val_n = int(N * val_ratio)
    val_idx = perm_all[:val_n]
    train_idx = perm_all[val_n:]

    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(train_idx.numel() / batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = make_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=min_lr_ratio)

    use_amp = amp and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    global_step = 0
    t0 = time.time()

    def run_split(indices, train_mode: bool):
        nonlocal global_step
        if train_mode:
            actor.train()
        else:
            actor.eval()

        total_slots = 0
        loss_sum = 0.0
        ent_sum = 0.0
        slot_correct = 0
        seq_correct = 0
        seq_total = 0

        for start in range(0, indices.numel(), batch_size):
            idx = indices[start : start + batch_size]
            b_obs = obs[idx]
            b_act = actions[idx].long()

            b_obs = b_obs.to(device) if hasattr(b_obs, "to") else b_obs
            b_act = b_act.to(device).long()

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                td = TensorDict({"observation": b_obs}, batch_size=[b_act.shape[0]], device=device)
                logits = actor(td)["logits"]
                loss, slot_acc, seq_acc, entropy = bc_loss_and_metrics_from_logits(logits, b_act, label_smoothing=label_smoothing)

            if train_mode:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                if wandb_run is not None and (global_step % log_every_steps == 0):
                    wandb_run.log(
                        {
                            "bc/step_loss": float(loss.item()),
                            "bc/step_slot_acc": float(slot_acc.item()),
                            "bc/step_seq_acc": float(seq_acc.item()),
                            "bc/step_entropy": float(entropy.item()),
                            "bc/lr": optimizer.param_groups[0]["lr"],
                            "dagger/iter": iter_idx if iter_idx is not None else -1,
                        },
                        step=wandb_run.step if hasattr(wandb_run, "step") else None,
                    )
                global_step += 1

            slots = b_act.numel()
            total_slots += slots
            loss_sum += float(loss.item()) * slots
            ent_sum += float(entropy.item()) * slots
            # for exact counting:
            pred = logits.argmax(-1)
            slot_correct += int((pred == b_act).sum().item())
            seq_correct += int((pred == b_act).all(dim=-1).sum().item())
            seq_total += int(b_act.shape[0])

        return {
            "loss": loss_sum / max(1, total_slots),
            "entropy": ent_sum / max(1, total_slots),
            "slot_acc": slot_correct / max(1, total_slots),
            "seq_acc": seq_correct / max(1, seq_total),
        }

    print("[DAgger] Retraining actor on aggregated dataset...")
    for ep in range(1, epochs + 1):
        # shuffle train indices each epoch
        train_perm = train_idx[torch.randperm(train_idx.numel(), device=train_idx.device)]
        train_metrics = run_split(train_perm, train_mode=True)
        val_metrics = run_split(val_idx, train_mode=False) if val_idx.numel() > 0 else None

        elapsed = (time.time() - t0) / 60.0
        msg = (
            f"[DAgger][BC] ep {ep}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_slot_acc={train_metrics['slot_acc']*100:.2f}% "
            f"train_seq_acc={train_metrics['seq_acc']*100:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.3e} "
            f"({elapsed:.1f} min)"
        )
        if val_metrics is not None:
            msg += f" | val_loss={val_metrics['loss']:.4f} " f"val_slot_acc={val_metrics['slot_acc']*100:.2f}% " f"val_seq_acc={val_metrics['seq_acc']*100:.2f}%"
        print(msg)

        if wandb_run is not None:
            log = {
                "train/epoch": ep,
                "train/loss": train_metrics["loss"],
                "train/slot_acc": train_metrics["slot_acc"],
                "train/seq_acc": train_metrics["seq_acc"],
                "train/entropy": train_metrics["entropy"],
                "train/lr_epoch_end": optimizer.param_groups[0]["lr"],
                "dagger/iter": iter_idx if iter_idx is not None else -1,
            }
            if val_metrics is not None:
                log.update(
                    {
                        "train/val_loss": val_metrics["loss"],
                        "train/val_slot_acc": val_metrics["slot_acc"],
                        "train/val_seq_acc": val_metrics["seq_acc"],
                        "train/val_entropy": val_metrics["entropy"],
                    }
                )
            wandb_run.log(log)

    return actor


def sample_proportional_dataset(
    base_dataset,
    dagger_buffer,
    *,
    it: int,
    total_samples: int,
    warmup_iters: int = 10,
    max_online_frac: float = 0.8,  # <-- key fix
):
    """
    Returns a TensorDict with samples drawn proportionally from:
      - offline base_dataset
      - online dagger_buffer

    Guarantees:
      - offline data is NEVER dropped completely
      - online fraction saturates at max_online_frac
      - safe when dagger_buffer is empty or small
    """
    assert total_samples > 0
    assert 0.0 < max_online_frac < 1.0

    # -----------------------
    # Compute online fraction
    # -----------------------
    alpha = min(1.0, it / warmup_iters)
    alpha = min(alpha, max_online_frac)

    n_online = int(round(total_samples * alpha))
    n_offline = total_samples - n_online

    samples = []

    # -----------------------
    # Offline sampling
    # -----------------------
    if n_offline > 0:
        N_off = base_dataset.batch_size[0]

        # Sample WITH replacement if needed
        idx = torch.randint(0, N_off, (n_offline,), device=base_dataset.device)
        samples.append(base_dataset[idx])

    # -----------------------
    # Online sampling
    # -----------------------
    if n_online > 0 and len(dagger_buffer) > 0:
        online_dataset = torch.cat(dagger_buffer, dim=0)
        N_on = online_dataset.batch_size[0]

        idx = torch.randint(0, N_on, (n_online,), device=online_dataset.device)
        samples.append(online_dataset[idx])

    # -----------------------
    # Fallback safety
    # -----------------------
    if len(samples) == 0:
        # Should never happen, but be safe
        idx = torch.randint(0, base_dataset.batch_size[0], (total_samples,))
        return base_dataset[idx]

    return torch.cat(samples, dim=0)


# ============================================================
# (4) Improved DAgger Loop (with buffer + early stopping)
# ============================================================
@hydra.main(config_path="conf", config_name="8x8x1024_dynamic_lcorners_cnn.yaml", version_base=None)
def main(cfg: DictConfig):
    training.disabled = True
    env = None
    actor = None

    if cfg.eval.expert_path == "eft" and rank != 0:
        # EFT expert only runs on rank 0
        exit(0)
    elif cfg.eval.expert_path == "parmetis":
        ParMETIS = ParMETIS_wrapper()

    if rank == 0:
        # --------------------------
        # Setup and environment
        # --------------------------
        folder_name, graph_name, interior_str, boundary_str = make_folder_name(cfg)
        save_dir = Path(f"dataset/{folder_name}")
        norm_path = f"norm/{folder_name}/{cfg.feature.observer.version}_norm.pkl"

        wandb_run = None
        if cfg.get("wandb", None) is not None and cfg.wandb.get("enabled", False):
            import wandb

            wandb_run = wandb.init(
                project="behavior_cloning",
                group=folder_name,
                name=cfg.wandb.get("name", f"dagger_retrain"),
                tags=["dagger"],
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(Path.cwd()),  # hydra output dir
            )

        graph_builder = make_graph_builder(cfg)
        with open(norm_path, "rb") as f:
            normalization = pickle.load(f)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization, eval=True)
        env.reset()  # make it consistent with eval_env below
        eval_env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization, eval=True)
        # --------------------------
        # Build actor
        # --------------------------
        observer = env.get_observer()
        feature_config = FeatureDimConfig.from_observer(observer)
        model, _, _ = create_td_actor_critic_models(cfg, feature_config)
        actor = model.actor

        # Load pretrained actor from Phase 3
        actor_path = save_dir / f"bc_actor_best_{cfg.eval.expert_path}_{cfg.feature.observer.version}.pt"
        print(f"[DAgger] Loading pretrained actor: {actor_path}")
        ckpt = torch.load(actor_path, map_location="cpu")
        if "actor_state_dict" in ckpt:
            actor.load_state_dict(ckpt["actor_state_dict"])
        else:
            actor.load_state_dict(ckpt)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = actor.to(device)

        # --------------------------
        # Load Phase 1 expert dataset
        # --------------------------
        base_dataset = load_phase1_dataset(cfg, save_dir)
        aggregated = base_dataset.clone()

        best_seq_dis = float("inf")
        best_iter = -1
        best_path = save_dir / f"dagger_actor_best_{cfg.eval.expert_path}_{cfg.feature.observer.version}.pt"

    if cfg.eval.expert_path == "parmetis":
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

    # --------------------------
    # DAgger hyperparameters
    # --------------------------
    N_ITER = 1000
    BUFFER_K = 4  # keep last 4 DAgger datasets
    dagger_buffer = []  # sliding buffer
    prev_disagree = 1.0
    best_time = float("inf")
    early_stop_count = 0
    TOTAL_TRAIN_SAMPLES = 4096  # or scale with GPU memory

    # --------------------------
    # DAgger main loop
    # --------------------------

    done = False

    if rank == 0:
        train_dataset = aggregated
    for it in range(N_ITER):
        print(f"\n========== DAgger Iteration {it+1}/{N_ITER} ==========\n")

        # 1. Rollout actor, label with expert
        if cfg.eval.expert_path == "parmetis":
            dagger_data = collect_dagger_parmetis_data(ParMETIS, env, actor, cfg, best_cfg, max_steps=5000, max_episodes=4)
        elif cfg.eval.expert_path == "eft":
            dagger_data = collect_dagger_eft_data(env, actor, cfg, max_steps=5000, max_episodes=4)

        # 2. Add to buffer
        if rank == 0:
            eval_env.set_reset_counter(0)
            td = eval_env.rollout(policy=actor, max_steps=100000)
            eval_time = td["observation", "aux", "time"][-1].item()
            print(eval_time)

            # dagger_buffer.append(dagger_data)
            # if len(dagger_buffer) > BUFFER_K:
            #     dagger_buffer.pop(0)  # drop oldest

            # 3. Aggregated = expert data + last K dagger datasets

            # train_dataset = sample_proportional_dataset(
            #     base_dataset=base_dataset,
            #     dagger_buffer=dagger_buffer,
            #     it=it,
            #     total_samples=TOTAL_TRAIN_SAMPLES,
            #     warmup_iters=10,
            # )
            train_dataset = TensorDict.cat([train_dataset, dagger_data], dim=0)

            print(f"[DAgger] Train samples: " f"{train_dataset.batch_size[0]} " f"(alpha={min(1.0, it/10):.2f})")

            # 4. Retrain actor on expanded dataset
            actor = retrain_actor(
                train_dataset,
                actor,
                epochs=5,
                batch_size=512,
                lr=1e-3,  # (1e-2 was aggressive; warmup+cosine works best around 1e-3)
                weight_decay=1e-4,
                grad_clip=1.0,
                warmup_ratio=0.05,
                min_lr_ratio=0.05,
                label_smoothing=0.0,
                val_ratio=0.00,
                amp=True,
                seed=it,
                wandb_run=wandb_run,
                iter_idx=it,
            )

            # ---- DISAGREEMENT RATES ----

            # slot_dis_total, seq_dis_total = compute_disagreement_rates(actor, aggregated, device=device)
            # slot_dis_off, seq_dis_off = compute_disagreement_rates(actor, base_dataset, device=device)
            # slot_dis_new, seq_dis_new = compute_disagreement_rates(actor, dagger_data, device=device)

            # print(f"[DAgger] Slot Disagree Total:   {slot_dis_total:.4f} | Seq Disagree Total:   {seq_dis_total:.4f}")
            # print(f"[DAgger] Slot Disagree Offline: {slot_dis_off:.4f} | Seq Disagree Offline: {seq_dis_off:.4f}")
            # print(f"[DAgger] Slot Disagree New:     {slot_dis_new:.4f} | Seq Disagree New:     {seq_dis_new:.4f}")

            # ---------------------------------------
            # Save BEST DAgger actor (rank 0 only)
            # ---------------------------------------
            # if seq_dis_new < best_seq_dis:
            #     best_seq_dis = seq_dis_new
            #     best_iter = it

            #     torch.save(
            #         actor.state_dict(),
            #         best_path,
            #     )

            if eval_time < best_time:
                best_time = eval_time

                torch.save(
                    actor.state_dict(),
                    best_path,
                )

            #     print(f"[DAgger] ✅ New BEST actor saved at iter {it+1} " f"(seq_dis_new={seq_dis_new:.4f})")

            #     if wandb_run is not None:
            #         wandb_run.log(
            #             {
            #                 "best/iter": it,
            #                 "best/seq_dis_new": seq_dis_new,
            #                 "best/slot_dis_new": slot_dis_new,
            #             }
            #         )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "dagger/iter": it,
                        "data/size_base": int(train_dataset.batch_size[0]),
                        "eval/time": eval_time / best_cfg[2],
                        # "data/size_agg": int(aggregated.batch_size[0]),
                        # "data/size_new": int(dagger_data.batch_size[0]),
                        # "disagree/slot_total": slot_dis_total,
                        # "disagree/seq_total": seq_dis_total,
                        # "disagree/slot_offline": slot_dis_off,
                        # "disagree/seq_offline": seq_dis_off,
                        # "disagree/slot_new": slot_dis_new,
                        # "disagree/seq_new": seq_dis_new,
                    }
                )

            # 6. Early stopping check
            # if seq_dis_new >= prev_disagree * 0.95 and it > BUFFER_K:
            #     early_stop_count += 1
            # else:
            #     early_stop_count = 0

            # prev_disagree = seq_dis_new

            if early_stop_count >= 100:
                print(f"[DAgger] Early stopping triggered at iteration {it+1}.")
                done = True

            out_path = save_dir / f"dagger_actor_{cfg.eval.expert_path}_{cfg.feature.observer.version}.pt"
            torch.save(actor.state_dict(), out_path)
            print(f"[DAgger] Saved final DAgger actor to {out_path}")

        done = comm.bcast(done, root=0)
        if done:
            break

    # --------------------------
    # Save final actor
    # --------------------------
    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    if rank == 0:
        out_path = save_dir / f"dagger_actor_{cfg.eval.expert_path}_{cfg.feature.observer.version}.pt"
        torch.save(actor.state_dict(), out_path)
        print(f"[DAgger] Saved final DAgger actor to {out_path}")


if __name__ == "__main__":
    main()
