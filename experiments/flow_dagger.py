"""Flow-DAgger (likelihood-based Diff-DAgger for Flow Matching policies).

This script is meant to be a *minimal* bridge between:
  - your vanilla DAgger implementation (dagger.py)
  - your flow-matching BC pretraining (pretrain_actor_flow_match.py)

Core idea (per your sketch):
  - keep the interactive DAgger-style loop
  - keep flow-matching training exactly the same
  - replace the "diffusion loss" uncertainty with a likelihood score
      U(o,a) = -log p_θ(a | o)
    computed via the probability-flow ODE:
      d log p / dt = -div_x v_θ(x,t|o)

Notes / assumptions:
  - Action space is represented as an (H,W) grid (e.g. 8x8 => 64) with
    categorical device IDs in {0..A-1}. We use one-hot vectors as the
    continuous variable x at t=1.
  - The flow policy learns a conditional vector field v_θ(x,t|o).
  - Sampling uses Euler integration from t=0→1 starting from N(0,I).
  - Likelihood uses Euler integration from t=1→0 (invert ODE) plus
    Hutchinson trace estimator for divergence.

You can tune the defaults at the bottom under "Flow-DAgger hyperparameters".
If your Hydra config already contains a `flow_dagger` section, the script will
read from it and fall back to these defaults otherwise.
"""

# =========================================================
# Imports
# =========================================================
import gc
import math
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from tensordict import TensorDict

import hydra
from omegaconf import DictConfig, OmegaConf

from mpi4py import MPI

from task4feedback.experiment_helper.run_name import make_folder_name
from task4feedback.logging import training
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import FeatureDimConfig, create_actor_flow_model
from task4feedback.experiment_helper.parmetis import query_parmetis, run_parmetis
from task4feedback.fastsim2 import ParMETIS_wrapper


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# =========================================================
# Dataset loading (Phase-1 expert rollouts)
# =========================================================
def load_phase1_dataset(cfg, save_dir: Path, max_eps: int = 384):
    """Loads expert episodes saved as episode_*.pkl and returns a flat TensorDict."""
    temp_dir = save_dir / f"{cfg.eval.expert_path}"
    assert temp_dir.exists(), f"Expert dataset not found at {temp_dir}"

    episodes = []
    for epfile in sorted(temp_dir.glob("episode_*.pkl")):
        with open(epfile, "rb") as f:
            td = pickle.load(f)
        episodes.append(td)
        if len(episodes) >= max_eps:
            break

    dataset = torch.cat(episodes, dim=0)
    if rank == 0:
        print(f"[Flow-DAgger] Loaded expert dataset: {dataset.batch_size}")
    return dataset


# =========================================================
# Flow matching training
# =========================================================
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


def flow_matching_loss(
    actor,
    observation,
    expert_actions: torch.Tensor,  # [B, T]
    *,
    action_dim: int,
    device: torch.device,
    return_v: bool = False,
):
    """Conditional flow matching loss (linear interpolation)."""
    B, T = expert_actions.shape

    x0 = F.one_hot(expert_actions, num_classes=action_dim).float().to(device)  # [B,T,A]
    eps = torch.randn_like(x0)
    t = torch.rand(B, 1, device=device)
    t_b = t[:, :, None]

    x_t = (1.0 - t_b) * eps + t_b * x0

    # reshape to (B, A, H, W)
    state_module = actor[0].module
    H, W = state_module.length, state_module.width
    x_t = x_t.view(B, H, W, action_dim).permute(0, 3, 1, 2).contiguous()
    v_target = (x0 - eps).view(B, H, W, action_dim).permute(0, 3, 1, 2).contiguous()

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


def train_flow_actor(
    cfg,
    dataset,
    actor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.05,
    amp: bool = True,
    seed: int = 0,
    log_every_steps: int = 100,
    wandb_run=None,
    tag: str = "train",
):
    """(Re)train the flow-matching actor on a flat TensorDict dataset."""
    device = next(actor.parameters()).device
    actor.train()

    obs = dataset["observation"]
    expert_actions = dataset["action"].long()
    N = expert_actions.shape[0]
    action_dim = cfg.system.n_devices - 1

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = max(1, math.ceil(N / batch_size))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = make_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=min_lr_ratio)

    use_amp = amp and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    global_step = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        # shuffle each epoch
        perm = perm[torch.randperm(perm.numel(), device=perm.device)]
        loss_sum = 0.0
        n_samples = 0

        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]
            batch_obs = obs[idx]
            if hasattr(batch_obs, "to"):
                batch_obs = batch_obs.to(device)
            batch_act = expert_actions[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss = flow_matching_loss(
                    actor,
                    batch_obs,
                    batch_act,
                    action_dim=action_dim,
                    device=device,
                    return_v=False,
                )

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += float(loss.item()) * idx.numel()
            n_samples += int(idx.numel())

            if wandb_run is not None and (global_step % log_every_steps == 0):
                wandb_run.log(
                    {
                        f"{tag}/step_loss": float(loss.item()),
                        f"{tag}/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
            global_step += 1

        epoch_loss = loss_sum / max(1, n_samples)
        elapsed = (time.time() - t0) / 60.0
        if rank == 0:
            print(f"[Flow-DAgger][FM] ep {ep}/{epochs} loss={epoch_loss:.6f} lr={optimizer.param_groups[0]['lr']:.2e} ({elapsed:.1f} min)")

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{tag}/epoch": ep,
                    f"{tag}/epoch_loss": epoch_loss,
                    f"{tag}/lr_epoch_end": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

    return actor


# =========================================================
# Flow sampling (policy execution)
# =========================================================
@torch.no_grad()
def sample_actions_from_flow(
    actor,
    observation,
    *,
    action_dim: int,
    num_steps: int,
    device: torch.device,
):
    """Euler integrate x'=v(x,t|o) from t=0→1 and return discrete actions via argmax."""
    actor.eval()
    if hasattr(observation, "to"):
        observation = observation.to(device)

    B = observation.batch_size[0]
    state_module = actor[0].module
    H, W = state_module.length, state_module.width
    T = H * W

    x = torch.randn(B, action_dim, H, W, device=device)
    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    dt = 1.0 / float(num_steps)

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

    logits = x.permute(0, 2, 3, 1).reshape(B, T, action_dim)
    actions = logits.argmax(dim=-1)
    return actions


# =========================================================
# Likelihood via probability-flow ODE (uncertainty)
# =========================================================
def _standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    """Log N(z;0,I) per batch element."""
    B = z.shape[0]
    d = z[0].numel()
    z2 = z.reshape(B, -1).pow(2).sum(dim=1)
    return -0.5 * (z2 + d * math.log(2.0 * math.pi))


def _actions_to_onehot_grid(
    actions: torch.Tensor,  # [B,T]
    *,
    action_dim: int,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert discrete actions to one-hot x_1 in shape [B, A, H, W]."""
    B, T = actions.shape
    assert T == H * W, f"Expected T==H*W ({T} vs {H}*{W})"
    x1 = F.one_hot(actions.long(), num_classes=action_dim).to(dtype=dtype)
    x1 = x1.view(B, H, W, action_dim).permute(0, 3, 1, 2).contiguous()
    return x1.to(device)


def probability_flow_nll(
    actor,
    observation,
    actions: torch.Tensor,  # [B,T]
    *,
    action_dim: int,
    num_steps: int,
    hutchinson_samples: int,
    device: torch.device,
    eps_type: str = "rademacher",  # or "gaussian"
) -> torch.Tensor:
    """Compute -log p_θ(a|o) using probability-flow ODE + Hutchinson divergence.

    This treats the one-hot action grid as the continuous endpoint x(t=1).

    Returns:
        nll: Tensor [B]
    """
    actor.eval()
    if hasattr(observation, "to"):
        observation = observation.to(device)

    # infer grid dims from model
    state_module = actor[0].module
    H, W = state_module.length, state_module.width

    x = _actions_to_onehot_grid(actions, action_dim=action_dim, H=H, W=W, device=device, dtype=torch.float32)
    B = x.shape[0]

    dt = 1.0 / float(num_steps)
    delta_logp = torch.zeros(B, device=device, dtype=torch.float32)  # ≈ ∫ div v dt

    # We need gradients w.r.t. x for divergence, but NOT w.r.t. parameters.
    # autograd.grad will compute only what it needs.
    with torch.enable_grad():
        # Integrate backwards: t=1 → 0
        for i in range(num_steps):
            t_scalar = 1.0 - i * dt
            t = torch.full((B, 1), t_scalar, device=device, dtype=torch.float32)

            x = x.detach().requires_grad_(True)
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

            # Hutchinson trace estimator for divergence: tr(J) ≈ epsᵀ J eps
            div = torch.zeros(B, device=device, dtype=torch.float32)
            for s in range(max(1, int(hutchinson_samples))):
                if eps_type == "gaussian":
                    eps = torch.randn_like(x)
                else:
                    # Rademacher (+/-1) has lower variance for many problems
                    eps = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)

                v_eps = (v * eps).reshape(B, -1).sum(dim=1)  # [B]
                grad = torch.autograd.grad(
                    v_eps,
                    x,
                    grad_outputs=torch.ones_like(v_eps),
                    create_graph=False,
                    retain_graph=(s < hutchinson_samples - 1),
                )[0]
                div = div + (grad * eps).reshape(B, -1).sum(dim=1)

            div = div / float(max(1, int(hutchinson_samples)))

            # Accumulate ∫ div v dt (this equals log p0 - log p1)
            delta_logp = delta_logp + div.detach() * dt

            # Backward Euler step: x_{t-dt} = x_t - dt * v
            x = (x - dt * v.detach()).detach()

    # base log prob at t=0
    logp0 = _standard_normal_logprob(x)
    logp1 = logp0 - delta_logp
    nll = -logp1
    return nll


@torch.no_grad()
def compute_uncertainty_threshold(
    actor,
    dataset,
    *,
    alpha: float,
    action_dim: int,
    device: torch.device,
    batch_size: int,
    num_steps: int,
    hutchinson_samples: int,
    max_points: Optional[int] = None,
):
    """Compute U_train and τ = quantile(U_train, alpha).

    For speed, you can limit the number of points using `max_points`.
    """
    N = dataset.batch_size[0]
    if max_points is not None and N > max_points:
        # TensorDict.device can be None; sampling indices on CPU is always safe.
        idx = torch.randint(0, N, (max_points,), device="cpu")
        ds = dataset[idx]
    else:
        ds = dataset

    obs = ds["observation"]
    acts = ds["action"].long()

    U_list = []
    # IMPORTANT: probability_flow_nll uses autograd; keep batches small.
    for start in tqdm(range(0, acts.shape[0], batch_size), desc="Compute U_train", disable=(rank != 0)):
        b_obs = obs[start : start + batch_size]
        b_act = acts[start : start + batch_size]
        nll = probability_flow_nll(
            actor,
            b_obs,
            b_act,
            action_dim=action_dim,
            num_steps=num_steps,
            hutchinson_samples=hutchinson_samples,
            device=device,
        )
        U_list.append(nll.detach().cpu())
        # free graph-related memory aggressively
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    U_train = torch.cat(U_list, dim=0)
    tau = torch.quantile(U_train, float(alpha)).item()
    return tau, U_train


# =========================================================
# Expert action helpers
# =========================================================
def get_expert_action_eft(env):
    """EFT expert: read mapping from runtime for current simulator state."""
    candidate_workspace = torch.zeros(
        env.simulator_factory[env.active_idx].graph_spec.max_candidates,
        dtype=torch.int64,
    )

    sim_reference = env.simulator.copy()
    sim_reference.disable_external_mapper()
    sim_reference.run()
    runtime = sim_reference.state.get_task_runtime()

    expert_action = []
    num_candidates = env.simulator.get_mappable_candidates(candidate_workspace)
    if num_candidates == 0:
        raise RuntimeError("No candidates to map (EFT expert).")

    for task_id in candidate_workspace:
        expert_action.append(runtime.get_compute_task_mapped_device(task_id.item()) - 1)

    return torch.tensor(expert_action, dtype=torch.int64)


def run_flow_dagger_episode(
    *,
    env,
    actor,
    cfg,
    tau: float,
    patience_k: int,
    action_dim: int,
    device: torch.device,
    ode_steps_sample: int,
    ode_steps_likelihood: int,
    hutchinson_samples: int,
    ParMETIS=None,
    best_cfg=None,  # (itr, ub, time) for ParMETIS
    max_decisions: int = 5000,
):
    """Run one episode with Flow-DAgger switching.

    Returns (rank0): TensorDict of newly collected expert transitions (may be None).
    """
    if rank == 0:
        td = env.reset()
        control = "ROBOT"
        violation_count = 0
        prev_mapping = None
        env_steps = 0
        temp_data = []
        done = False
    else:
        control = None
        violation_count = None
        prev_mapping = None
        env_steps = 0
        td = None
        done = False

    for decision_step in range(max_decisions):
        # -------------------------------------------------
        # Broadcast done/control so all ranks stay in sync
        # -------------------------------------------------
        if rank == 0:
            payload = (done, control, env_steps)
        else:
            payload = None
        done, control, env_steps = comm.bcast(payload, root=0)
        if done:
            break

        # -------------------------------------------------
        # EXPERT control
        # -------------------------------------------------
        if control == "EXPERT":
            if cfg.eval.expert_path == "parmetis":
                if ParMETIS is None:
                    raise RuntimeError("ParMETIS wrapper must be provided for parmetis expert.")
                expert_action, status = query_parmetis(
                    ParMETIS,
                    env,
                    cfg,
                    prev_mapping=prev_mapping,
                    first_call=(env_steps == 0),
                    offset=0,
                    itr=best_cfg[0],
                    unbalance=best_cfg[1],
                )

                if rank == 0:
                    # Put expert action on the same device as the env tensordict (safer).
                    env_dev = getattr(td, "device", None)
                    expert_action = torch.tensor(expert_action, dtype=torch.int64)
                    if env_dev is not None:
                        expert_action = expert_action.to(env_dev)
            else:
                raise ValueError(f"Unknown expert_path: {cfg.eval.expert_path}")

            if rank == 0:
                td["action"] = expert_action
                td_next = env.step(td)

                td_labeled = td_next.clone()
                td_labeled["action"] = expert_action.to(dtype=torch.int64)
                temp_data.append(td_labeled)

                prev_mapping = expert_action.detach().cpu().numpy()
                td = td_next["next"]
                done = bool(td_next["next", "done"].any().item())
                env_steps += 1

            continue

        # -------------------------------------------------
        # ROBOT control
        # -------------------------------------------------
        if rank == 0:
            # 1) sample action from flow
            obs = td["observation"]
            robot_action = sample_actions_from_flow(
                actor,
                obs,
                action_dim=action_dim,
                num_steps=ode_steps_sample,
                device=device,
            ).squeeze(0)

            # 2) compute uncertainty (negative log-likelihood)
            U = probability_flow_nll(
                actor,
                obs,
                robot_action.unsqueeze(0),
                action_dim=action_dim,
                num_steps=ode_steps_likelihood,
                hutchinson_samples=hutchinson_samples,
                device=device,
            ).item()

            if U > tau:
                violation_count += 1
            else:
                violation_count = 0

            # 3) switch decision
            if violation_count >= patience_k:
                control = "EXPERT"
                # NOTE: we do *not* step the env on the triggering decision.
                # Next decision will be expert on the same state.
                continue

            # 4) execute robot action
            # Move to env's device if needed.
            env_dev = getattr(td, "device", None)
            a_env = robot_action.to(dtype=torch.int64)
            if env_dev is not None and a_env.device != env_dev:
                a_env = a_env.to(env_dev)
            td["action"] = a_env
            td_next = env.step(td)
            prev_mapping = robot_action.detach().cpu().numpy()
            td = td_next["next"]
            done = bool(td_next["next", "done"].any().item())
            env_steps += 1

    if rank == 0:
        if len(temp_data) == 0:
            return None
        return torch.stack(temp_data, dim=0)
    return None


# =========================================================
# Hydra main
# =========================================================
@hydra.main(config_path="conf", config_name="8x8x1024_dynamic_lcorners_cnn.yaml", version_base=None)
def main(cfg: DictConfig):
    training.disabled = True

    # EFT expert only runs on rank0
    if cfg.eval.expert_path == "eft" and rank != 0:
        return

    # --------------------------
    # Setup
    # --------------------------
    folder_name, *_ = make_folder_name(cfg)
    orig_cwd = Path(hydra.utils.get_original_cwd())
    save_dir = orig_cwd / f"dataset/{folder_name}"
    norm_path = orig_cwd / f"norm/{folder_name}/{cfg.feature.observer.version}_norm.pkl"

    # Optional wandb
    wandb_run = None
    if rank == 0 and cfg.get("wandb", None) is not None and cfg.wandb.get("enabled", False):
        import wandb

        wandb_run = wandb.init(
            project="behavior_cloning",
            group=cfg.wandb.get("group", folder_name),
            name=cfg.wandb.get("name", f"{folder_name}-flow-dagger"),
            tags=["flow_dagger"],
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(Path.cwd()),
        )

    # --------------------------
    # Env (rank0 only)
    # --------------------------
    env = None
    if rank == 0:
        graph_builder = make_graph_builder(cfg)
        with open(norm_path, "rb") as f:
            normalization = pickle.load(f)

        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization, eval=True)
        env.reset()

    # --------------------------
    # Actor (flow policy)
    # --------------------------
    if rank == 0:
        observer = env.get_observer()
        feature_cfg = FeatureDimConfig.from_observer(observer)
        actor = create_actor_flow_model(cfg, feature_cfg)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = actor.to(device)
    else:
        actor = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_dim = cfg.system.n_devices - 1

    # --------------------------
    # Expert tuning (ParMETIS)
    # --------------------------
    ParMETIS = None
    best_cfg = (None, None, None)
    if cfg.eval.expert_path == "parmetis":
        ParMETIS = ParMETIS_wrapper()
        best_cfg = (None, None, float("inf"))  # (itr, ub, time)
        ub_cur = 1.0001

        # scan itr
        for itr in [0.0001001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]:
            if rank == 0:
                temp = env.simulator.copy()
            else:
                temp = None
            comm.barrier()
            status = run_parmetis(
                sim=temp,
                cfg=cfg,
                unbalance=ub_cur,
                itr=itr,
                n_compute_devices=cfg.system.n_devices - 1,
                ParMETIS=ParMETIS,
            )
            if rank == 0 and status and temp.time < best_cfg[2]:
                best_cfg = (itr, ub_cur, temp.time)
                print(f"[Flow-DAgger][ParMETIS] New best itr={itr} time={temp.time}", flush=True)

        best_cfg = comm.bcast(best_cfg, root=0)

        # scan ub
        for ub in [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]:
            if rank == 0:
                temp = env.simulator.copy()
            else:
                temp = None
            comm.barrier()
            status = run_parmetis(
                sim=temp,
                cfg=cfg,
                unbalance=ub,
                itr=best_cfg[0],
                n_compute_devices=cfg.system.n_devices - 1,
                ParMETIS=ParMETIS,
            )
            if not status:
                break
            if rank == 0 and temp.time < best_cfg[2]:
                ub_cur = ub
                best_cfg = (best_cfg[0], ub_cur, temp.time)
                print(f"[Flow-DAgger][ParMETIS] New best ub={ub_cur:.2f} time={temp.time}", flush=True)

        best_cfg = comm.bcast(best_cfg, root=0)

    # --------------------------
    # Load initial expert dataset
    # --------------------------
    if rank == 0:
        base_dataset = load_phase1_dataset(cfg, save_dir, max_eps=384)
        aggregated = base_dataset.clone()

    # --------------------------
    # Flow-DAgger hyperparameters
    # --------------------------
    # You can set these in your YAML as:
    # flow_dagger:
    #   alpha: 0.95
    #   patience_k: 3
    #   initial_train_epochs: 200
    #   retrain_epochs: 50
    #   batch_size: 256
    #   lr: 1e-3
    #   ode_steps_sample: 50
    #   ode_steps_likelihood: 10
    #   hutchinson_samples: 1
    #   baseline_batch_size: 8
    #   baseline_max_points: 1024
    #   episodes: 50
    fd = cfg.get("flow_dagger", {})
    alpha = float(fd.get("alpha", 0.95))
    patience_k = int(fd.get("patience_k", 3))

    initial_train_epochs = int(fd.get("initial_train_epochs", 200))
    retrain_epochs = int(fd.get("retrain_epochs", 50))
    train_batch_size = int(fd.get("batch_size", 256))
    lr = float(fd.get("lr", 1e-3))

    ode_steps_sample = int(fd.get("ode_steps_sample", 50))
    ode_steps_likelihood = int(fd.get("ode_steps_likelihood", 10))
    hutchinson_samples = int(fd.get("hutchinson_samples", 1))

    baseline_batch_size = int(fd.get("baseline_batch_size", 8))
    baseline_max_points = fd.get("baseline_max_points", 1024)
    baseline_max_points = int(baseline_max_points) if baseline_max_points is not None else None

    episodes = int(fd.get("episodes", 50))
    max_decisions = int(fd.get("max_decisions", 5000))

    # --------------------------
    # 1) Initial training
    # --------------------------
    if rank == 0:
        # Optionally resume from a checkpoint if it exists
        ckpt_path = save_dir / f"flow_actor_best_{cfg.feature.observer.version}.pt"
        if ckpt_path.exists():
            print(f"[Flow-DAgger] Loading pretrained flow actor: {ckpt_path}")
            actor.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            ckpt_path = save_dir / f"flow_actor_flow_dagger_last.pt"
            print("[Flow-DAgger] No pretrained flow actor found; training from scratch.")
            actor = train_flow_actor(
                cfg,
                aggregated,
                actor,
                epochs=initial_train_epochs,
                batch_size=train_batch_size,
                lr=lr,
                amp=True,
                seed=0,
                wandb_run=wandb_run,
                tag="init",
            )
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(actor.state_dict(), ckpt_path)
            print(f"[Flow-DAgger] Saved initial flow actor to {ckpt_path}")

        # 1.2) Compute uncertainty baseline and threshold
        tau, U_train = compute_uncertainty_threshold(
            actor,
            aggregated,
            alpha=alpha,
            action_dim=action_dim,
            device=device,
            batch_size=baseline_batch_size,
            num_steps=ode_steps_likelihood,
            hutchinson_samples=hutchinson_samples,
            max_points=baseline_max_points,
        )
        print(f"[Flow-DAgger] Initial τ (alpha={alpha}) = {tau:.4f} (computed on {U_train.numel()} points)")
    else:
        tau = None

    # broadcast threshold to all ranks
    tau = comm.bcast(tau, root=0)

    # --------------------------
    # 2) Interactive learning loop
    # --------------------------
    for ep in range(episodes):
        if rank == 0:
            print(f"\n========== Flow-DAgger Episode {ep+1}/{episodes} ==========")

        new_data = run_flow_dagger_episode(
            env=env,
            actor=actor,
            cfg=cfg,
            ParMETIS=ParMETIS,
            tau=tau,
            patience_k=patience_k,
            action_dim=action_dim,
            device=device,
            ode_steps_sample=ode_steps_sample,
            ode_steps_likelihood=ode_steps_likelihood,
            hutchinson_samples=hutchinson_samples,
            best_cfg=best_cfg,
            max_decisions=max_decisions,
        )

        if rank == 0:
            if new_data is not None:
                aggregated = TensorDict.cat([aggregated, new_data.cpu()], dim=0)
                print(f"[Flow-DAgger] Added {new_data.batch_size[0]} expert-labeled steps. Dataset now {aggregated.batch_size[0]}.")
            else:
                print("[Flow-DAgger] No expert takeover this episode (no new labels).")

            # 3) Policy update (retrain / finetune)
            actor = train_flow_actor(
                cfg,
                aggregated,
                actor,
                epochs=retrain_epochs,
                batch_size=train_batch_size,
                lr=lr,
                amp=True,
                seed=ep + 1,
                wandb_run=wandb_run,
                tag="retrain",
            )

            # 3.2) Recompute tau
            tau, U_train = compute_uncertainty_threshold(
                actor,
                aggregated,
                alpha=alpha,
                action_dim=action_dim,
                device=device,
                batch_size=baseline_batch_size,
                num_steps=ode_steps_likelihood,
                hutchinson_samples=hutchinson_samples,
                max_points=baseline_max_points,
            )
            print(f"[Flow-DAgger] Updated τ = {tau:.4f} (alpha={alpha}, points={U_train.numel()})")

            # save
            out_path = save_dir / "flow_actor_flow_dagger_last.pt"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(actor.state_dict(), out_path)
            print(f"[Flow-DAgger] Saved actor checkpoint to {out_path}")

        # broadcast tau so all ranks are consistent
        tau = comm.bcast(tau if rank == 0 else None, root=0)

    if rank == 0 and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
