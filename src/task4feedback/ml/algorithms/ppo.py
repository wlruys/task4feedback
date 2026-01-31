from pathlib import Path
from ..models import *
from ..util import *
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import (
    SliceSampler,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage, TensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE, VTrace
from torchrl.objectives.utils import ValueEstimators
from tensordict import TensorDict
import wandb
import torch
import torch.nn.functional as F
from torchrl._utils import compile_with_warmup
from .base import AlgorithmConfig, LoggingConfig
from ..base import ActorCriticModule
from omegaconf import OmegaConf
from task4feedback.logging import training
import time
from torchrl.collectors.utils import split_trajectories
from task4feedback.ml.util import log_parameter_and_gradient_norms
import math


def joint_stats(td, ppo):
    with torch.no_grad():
        prev_lp = td["sample_log_prob"].squeeze(-1)  # [N]
        cur_lp, dist, _ = ppo._get_cur_log_prob(td)
        cur_lp = cur_lp.squeeze(-1)  # [N]
        act = td["action"]  # [N, 64]
        logits = td["logits"]  # [N, 64, 4]

        # Recompute joint log-prob explicitly via per-head log_softmax (+ gather)
        logp_heads = F.log_softmax(logits, dim=-1)  # [N, 64, 4]
        gathered = logp_heads.gather(-1, act.unsqueeze(-1)).squeeze(-1)  # [N, 64]
        joint_lp_explicit = gathered.sum(-1)  # [N]

        print("\n=== LOG-PROB STATS ===")
        for name, x in [
            ("prev_lp (stored)", prev_lp),
            ("cur_lp (dist)", cur_lp),
            ("cur_lp (explicit)", joint_lp_explicit),
        ]:
            x = x.detach()
            print(f"{name:20s} mean={x.mean():8.3f} std={x.std():8.3f} " f"min={x.min():8.3f} max={x.max():8.3f}")

        x = td["observation", "nodes", "tasks", "attr"]  # [B, 600] ideally; if it's [600], fix your batch shaping first
        x = x.detach()
        print("\n=== OBSERVATION STATS ===")
        print(f"obs: {x}")
        print(f"obs shape: {x.shape}")
        print(f"obs mean={x.mean():8.3f} std={x.std():8.3f}")
        print(f"obs min={x.min():8.3f} max={x.max():8.3f}")
        print(f"obs numel={x.numel()} nan={torch.isnan(x).sum()} inf={torch.isinf(x).sum()}")
        print(f"obs unique={torch.unique(x)}")

        # Show me the row where the max value is
        max_val = x.max()
        max_pos = (x == max_val).nonzero(as_tuple=False)
        print(f"obs max position: {max_pos}")
        if max_pos.shape[0] < 100:
            for pos in max_pos:
                b, i, z = pos
                print(f"obs[{b}, {i}, :] = {x[b, i, :]}")

        # KL approximations
        kl_approx = prev_lp - cur_lp  # [N]
        kl_approx_explicit = prev_lp - joint_lp_explicit
        print("\n=== KL APPROX (sample-wise) ===")
        for name, x in [
            ("kl_approx", kl_approx),
            ("kl_approx_explicit", kl_approx_explicit),
        ]:
            x = x.detach()
            print(f"{name:20s} mean={x.mean():8.3f} std={x.std():8.3f} " f"min={x.min():8.3f} max={x.max():8.3f}")

        # Logit scale diagnostics
        l = logits.detach()
        print(f"\nlogits shape: {l.shape}")
        lmax = l.abs().amax().item()
        per_head_span = l.max(dim=-1).values - l.min(dim=-1).values  # [N, 64]
        print("\n=== LOGIT SCALE ===")
        print(f"|logits|_max = {lmax:.1f}; span per head: mean={per_head_span.mean():.2f} " f"max={per_head_span.max():.2f}")


def _get_step_primitives(flat_td: TensorDict) -> torch.Tensor:
    """Return per-step primitive counts (n_candidates), clamped to at least 0 for progress."""
    try:
        n_candidates = flat_td["observation", "aux", "candidates", "count"].view(-1)
    except KeyError as exc:
        raise KeyError("Missing observation->aux->candidates->count for milestone accounting.") from exc
    if n_candidates.ndim != 1:
        n_candidates = n_candidates.view(-1)
    return n_candidates.to(dtype=torch.int64).clamp_min(0)


def _value_loss_per_sample(values: torch.Tensor, targets: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "l1":
        return (values - targets).abs()
    if loss_type in ("smooth_l1", "huber"):
        return F.smooth_l1_loss(values, targets, reduction="none")
    # Default to L2 / MSE
    return (values - targets).pow(2)


def _masked_critic_loss(
    batch: TensorDict,
    loss_module: ClipPPOLoss,
    ppo_config: "PPOConfig",
) -> Optional[torch.Tensor]:
    """Compute critic loss only on milestone boundaries when a mask is present."""
    if ("milestone_boundary" not in batch.keys()) or ("value_target" not in batch.keys()):
        return None
    if ppo_config.val_coef is None:
        return None

    values = loss_module.critic_network(batch)["state_value"].view(-1)
    targets = batch["value_target"].view(-1).to(dtype=values.dtype)
    mask = batch["milestone_boundary"].view(-1).to(dtype=values.dtype)
    if mask.sum() <= 0:
        return values.new_tensor(0.0)

    if ppo_config.clip_vloss and ("state_value" in batch.keys()):
        old_values = batch["state_value"].view(-1).detach()
        delta = (values - old_values).clamp(-ppo_config.clip_eps, ppo_config.clip_eps)
        clipped = old_values + delta
        loss_unclipped = _value_loss_per_sample(values, targets, ppo_config.value_norm)
        loss_clipped = _value_loss_per_sample(clipped, targets, ppo_config.value_norm)
        value_loss = torch.max(loss_unclipped, loss_clipped)
    else:
        value_loss = _value_loss_per_sample(values, targets, ppo_config.value_norm)

    return ppo_config.val_coef * (value_loss * mask).sum() / mask.sum()


def _milestone_regularization_scale(batch: TensorDict) -> Optional[torch.Tensor]:
    """Scale entropy/KL to be per-milestone instead of per-step."""
    if "milestone_avg_block_len" in batch.keys():
        avg_len = batch["milestone_avg_block_len"].mean()
        return 1.0 / avg_len.clamp_min(1.0)
    if "milestone_boundary" in batch.keys():
        mask = batch["milestone_boundary"].view(-1).to(dtype=torch.float32)
        blocks = mask.sum()
        if blocks > 0:
            avg_len = mask.numel() / blocks
            return mask.new_tensor(1.0) / avg_len.clamp_min(1.0)
    return None


def _flatten_valid_batch(batch: TensorDict) -> TensorDict:
    """Flatten batch and drop padded entries when valid_mask is present."""
    if "valid_mask" not in batch.keys():
        return batch
    flat = batch.reshape(-1)
    valid = flat["valid_mask"].view(-1)
    if valid.all():
        return flat
    if valid.any():
        return flat[valid]
    return flat


def _pad_tensor_time(x: torch.Tensor, pad: int, value) -> torch.Tensor:
    if pad <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[0] = pad
    pad_t = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_t], dim=0)


def _reshape_time_to_blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    # x: [T, ...] with T % block == 0
    T = x.shape[0]
    B = T // block
    return x.view(B, block, *x.shape[1:])


@torch.no_grad()
def pack_primitives_to_macros(
    td: TensorDict,
    milestone: int,
    *,
    time_dim: int = 0,
    pad_done_to_true: bool = True,
) -> TensorDict:
    """
    Pack primitive-step rollouts into fixed-length milestone blocks.

    Input td shape: [T] (time)
    Output macro_td shape: [B] where each leaf is shaped [B, milestone, ...]
      and macro_td["valid_mask"] is [B, milestone] (True for real primitive steps).
    """
    assert td.ndim >= 1 and td.shape[time_dim] == td.shape[0], "assumes time dim is 0"
    if milestone <= 0:
        return td

    T = td.shape[0]
    B = int(math.ceil(T / milestone))
    T_pad = B * milestone
    pad = T_pad - T
    device = td.device

    valid_mask = torch.zeros((T_pad,), dtype=torch.bool, device=device)
    valid_mask[:T] = True

    tdp = td.clone()

    # Pad tensor leaves that are time-major
    done_key = None
    if ("next", "terminated") in tdp.keys(True):
        done_key = ("next", "terminated")
    elif ("next", "done") in tdp.keys(True):
        done_key = ("next", "done")

    for key in tdp.keys(True):
        leaf = tdp.get(key)
        if not torch.is_tensor(leaf):
            continue
        if leaf.shape[0] != T:
            continue
        if done_key is not None and key == done_key:
            pad_value = True if pad_done_to_true else False
        else:
            pad_value = False if leaf.dtype == torch.bool else 0.0
        tdp.set(key, _pad_tensor_time(leaf, pad, pad_value))

    macro = TensorDict({}, batch_size=[B], device=device)

    def blockify(key):
        macro.set(key, _reshape_time_to_blocks(tdp.get(key), milestone))

    for key in tdp.keys(True):
        leaf = tdp.get(key)
        if not torch.is_tensor(leaf):
            continue
        if leaf.shape[0] != T_pad:
            continue
        blockify(key)

    macro.set("valid_mask", _reshape_time_to_blocks(valid_mask, milestone))
    return macro


@torch.no_grad()
def compute_macro_gae(
    macro_td: TensorDict,
    critic: torch.nn.Module,
    milestone: int,
    gamma: float,
    lmbda: float,
) -> TensorDict:
    """
    Compute GAE on macro transitions:
      S_t = first observation in block
      S_{t+1} = last next_observation in block
      R_t = sum of rewards in block (no internal discount)
      done_t = any(done) within the valid part of the block

    Adds:
      macro_td["macro_advantage"]    [B, 1]
      macro_td["macro_value_target"] [B, 1]
    """
    if milestone <= 0:
        return macro_td

    device = macro_td.device
    valid = macro_td["valid_mask"]  # [B, milestone]
    obs0 = macro_td["observation"][:, 0]
    next_obs = macro_td[("next", "observation")]
    rew = macro_td[("next", "reward")]
    if rew.ndim == 2:
        rew = rew.unsqueeze(-1)

    done_key = None
    if ("next", "terminated") in macro_td.keys(True):
        done_key = ("next", "terminated")
    elif ("next", "done") in macro_td.keys(True):
        done_key = ("next", "done")

    if done_key is None:
        done_any = torch.zeros((macro_td.shape[0], 1), dtype=torch.bool, device=device)
    else:
        done = macro_td.get(done_key)
        if done.ndim == 2:
            done = done.unsqueeze(-1)
        done_any = ((done.squeeze(-1) & valid).any(dim=1, keepdim=True))

    not_done = (~done_any).to(dtype=torch.float32)

    obs1 = next_obs[:, -1]

    V0 = critic(TensorDict({"observation": obs0}, batch_size=[obs0.shape[0]], device=device))["state_value"]
    V1 = critic(TensorDict({"observation": obs1}, batch_size=[obs1.shape[0]], device=device))["state_value"]

    R = (rew.squeeze(-1) * valid.to(dtype=rew.dtype)).sum(dim=1, keepdim=True)

    # One gamma step per block.
    delta = R + gamma * not_done * V1 - V0

    B = delta.shape[0]
    A = torch.zeros_like(delta)
    gae = torch.zeros((1,), device=device, dtype=delta.dtype)
    for t in reversed(range(B)):
        gae = delta[t] + gamma * lmbda * not_done[t] * gae
        A[t] = gae

    VT = A + V0

    macro_td.set("macro_advantage", A)
    macro_td.set("macro_value_target", VT)
    return macro_td


class MacroReplayPPOLoss(torch.nn.Module):
    """
    PPO loss where replay unit is a macro item, but objective is token-level.

    Expects batch leaves shaped:
      observation:        [MB, milestone, ...]
      action:             [MB, milestone, ...]
      sample_log_prob:    [MB, milestone] (or action_log_prob)
      valid_mask:         [MB, milestone]
      macro_advantage:    [MB, 1]
      macro_value_target: [MB, 1]
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        clip_eps: float,
        ent_coef: float,
        val_coef: float,
        normalize_advantage: bool = True,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.val_coef = val_coef
        self.normalize_advantage = normalize_advantage

    def forward(self, batch: TensorDict) -> TensorDict:
        device = batch.device
        valid = batch["valid_mask"]  # [MB, milestone]
        MB, X = valid.shape

        adv = batch["macro_advantage"]  # [MB,1] or [MB,X]
        vt = batch["macro_value_target"]  # [MB,1] or [MB,X]
        if adv.ndim == 2 and adv.shape[1] == X:
            adv_tok = adv
        else:
            adv_tok = adv.expand(MB, X).contiguous()
        if vt.ndim == 2 and vt.shape[1] == X:
            vt_tok = vt
        else:
            vt_tok = vt.expand(MB, X).contiguous()

        if self.normalize_advantage:
            v = adv_tok[valid]
            adv_tok = (adv_tok - v.mean()) / (v.std().clamp_min(1e-8))

        obs = batch["observation"]
        if isinstance(obs, TensorDict):
            obs_flat = TensorDict({}, batch_size=[MB * X], device=device)
            for key in obs.keys(True):
                leaf = obs.get(key)
                if not torch.is_tensor(leaf):
                    continue
                if leaf.shape[0] != MB or leaf.shape[1] != X:
                    raise RuntimeError(
                        f"Expected observation leaf {key} to have leading shape ({MB}, {X}), "
                        f"got {tuple(leaf.shape)}."
                    )
                obs_flat.set(key, leaf.reshape(MB * X, *leaf.shape[2:]))
        else:
            if obs.shape[0] != MB or obs.shape[1] != X:
                raise RuntimeError(
                    f"Expected observation to have leading shape ({MB}, {X}), got {tuple(obs.shape)}."
                )
            obs_flat = obs.reshape(MB * X, *obs.shape[2:])

        act_unflat = batch["action"]
        if act_unflat.shape[0] != MB or act_unflat.shape[1] != X:
            raise RuntimeError(
                f"Expected action to have leading shape ({MB}, {X}), got {tuple(act_unflat.shape)}."
            )
        act = act_unflat.reshape(MB * X, *act_unflat.shape[2:])

        flat = TensorDict({"observation": obs_flat, "action": act}, batch_size=[MB * X], device=device)
        dist = self.actor.get_dist(flat)
        logp = dist.log_prob(act)
        if logp.ndim > 1 and logp.shape[-1] == 1:
            logp = logp.squeeze(-1)
        if act_unflat.ndim > 2 and logp.shape == act.shape:
            logp = logp.sum(dim=-1)
        logp = logp.view(MB, X)

        if "sample_log_prob" in batch.keys():
            old_logp = batch["sample_log_prob"]
        else:
            old_logp = batch["action_log_prob"]
        if old_logp.ndim > 1 and old_logp.shape[-1] == 1:
            old_logp = old_logp.squeeze(-1)
        if act_unflat.ndim > 2 and old_logp.shape == act_unflat.shape:
            old_logp = old_logp.sum(dim=-1)
        old_logp = old_logp.view(MB, X)

        ratio = (logp - old_logp).exp()
        surr1 = ratio * adv_tok
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_tok
        policy_loss_tok = -torch.minimum(surr1, surr2)
        policy_loss = policy_loss_tok[valid].mean()

        vpred = self.critic(TensorDict({"observation": obs_flat}, batch_size=[MB * X], device=device))["state_value"]
        vpred = vpred.view(MB, X)
        value_loss_tok = 0.5 * (vpred - vt_tok).pow(2)
        value_loss = value_loss_tok[valid].mean()

        ent = dist.entropy()
        if ent.ndim > 1:
            ent = ent.sum(dim=-1)
        ent = ent.view(MB, X)
        entropy = ent[valid].mean()

        ratio_valid = ratio[valid].detach()
        kl_approx = (old_logp - logp)[valid].mean().detach()
        clip_fraction = (ratio_valid.gt(1.0 + self.clip_eps) | ratio_valid.lt(1.0 - self.clip_eps)).float().mean()
        ess = (ratio_valid.sum().pow(2) / ratio_valid.pow(2).sum().clamp_min(1e-8)).detach()

        loss_entropy = -self.ent_coef * entropy
        loss_critic = self.val_coef * value_loss
        total = policy_loss + loss_critic + loss_entropy

        return TensorDict(
            {
                "loss_total": total,
                "loss_objective": policy_loss,
                "loss_critic": loss_critic,
                "loss_entropy": loss_entropy,
                "entropy": entropy,
                "kl_approx": kl_approx,
                "clip_fraction": clip_fraction,
                "ESS": ess,
            },
            batch_size=[],
            device=device,
        )

@torch.no_grad()
def pack_milestone_blocks(
    flat_td: TensorDict,
    milestone: int,
) -> TensorDict:
    """
    Pack primitive-step rollouts into milestone-sized macroaction blocks.

    Each block aggregates env steps until the accumulated primitive count reaches `milestone`
    (or a trajectory terminates). Blocks are padded to the max block length with a valid_mask.

    Note: milestone_ticks is always 1 per block (simplified assumption).
    """
    if milestone <= 0:
        return flat_td

    device = flat_td.device

    # Split into individual trajectories using TorchRL's built-in function
    traj_list = split_trajectories(flat_td, trajectory_key=("collector", "traj_ids"))

    if ("next", "done") in flat_td.keys(True):
        done_key = ("next", "done")
    elif ("next", "terminated") in flat_td.keys(True):
        done_key = ("next", "terminated")
    else:
        raise KeyError("Missing next->done (or next->terminated) required for milestone blocking.")

    all_blocks = []

    # Process each trajectory independently
    for traj_td in traj_list:
        if traj_td.shape[0] == 0:
            continue

        step_primitives = _get_step_primitives(traj_td)
        dones = traj_td[done_key].view(-1)

        start_pos = 0
        primitive_acc = 0

        for pos in range(traj_td.shape[0]):
            n_prims = int(step_primitives[pos].item())
            primitive_acc += n_prims
            done_now = bool(dones[pos].item())

            if done_now or primitive_acc >= milestone:
                # Extract block from start_pos to pos+1
                block_td = traj_td[start_pos : pos + 1]
                all_blocks.append(block_td)

                primitive_acc = primitive_acc % milestone
                start_pos = pos + 1

        # Handle partial block at end of trajectory
        if start_pos < traj_td.shape[0]:
            block_td = traj_td[start_pos:]
            all_blocks.append(block_td)

    if not all_blocks:
        return TensorDict({}, batch_size=[0], device=device)

    # Pad blocks to uniform width
    num_blocks = len(all_blocks)
    max_block_len = max(block.shape[0] for block in all_blocks)

    valid_mask = torch.zeros((num_blocks, max_block_len), dtype=torch.bool, device=device)

    # Create padded result TensorDict
    block_td = TensorDict({}, batch_size=[num_blocks, max_block_len], device=device)

    # Get all keys from first block to determine structure
    sample_block = all_blocks[0]
    for key in sample_block.keys(True, leaves_only=True):
        leaf = sample_block.get(key)
        if not torch.is_tensor(leaf):
            continue

        # Create padded tensor for this key
        pad_value = False if leaf.dtype == torch.bool else 0
        padded_shape = (num_blocks, max_block_len, *leaf.shape[1:])
        out = torch.full(padded_shape, pad_value, dtype=leaf.dtype, device=device)

        # Fill in actual values from each block
        for b, block in enumerate(all_blocks):
            block_len = block.shape[0]
            out[b, :block_len] = block.get(key)

        block_td.set(key, out)

    # Set valid_mask
    for b, block in enumerate(all_blocks):
        block_len = block.shape[0]
        valid_mask[b, :block_len] = True

    block_td.set("valid_mask", valid_mask)
    return block_td


@torch.no_grad()
def compute_milestone_gae_flat(
    flat_td: TensorDict,
    critic: torch.nn.Module,
    milestone: int,
    gamma: float,
    lmbda: float,
    scale_advantage_by_block: bool = True,
) -> TensorDict:
    """
    Compute milestone-time GAE on variable-length blocks (in primitive steps),
    then broadcast advantages/value targets back to primitive steps.

    Each env step contributes `n_candidates` primitive steps. Milestones tick every
    `milestone` primitive steps, independent of n_candidates.
    """
    if milestone <= 0:
        return flat_td

    device = flat_td.device
    traj_ids = flat_td.get(("collector", "traj_ids"), None)
    if traj_ids is None:
        raise KeyError("Missing collector->traj_ids required for milestone GAE.")
    traj_ids = traj_ids.view(-1)

    rewards = flat_td.get(("next", "reward")).view(-1)
    dones = flat_td.get(("next", "done")).view(-1)
    step_primitives = _get_step_primitives(flat_td)

    advantage = torch.zeros_like(rewards, dtype=torch.float32, device=device)
    value_target = torch.zeros_like(rewards, dtype=torch.float32, device=device)
    milestone_boundary = torch.zeros_like(rewards, dtype=torch.bool, device=device)
    milestone_block_len = torch.zeros_like(rewards, dtype=torch.float32, device=device)
    total_steps = 0
    total_blocks = 0

    # Compute per-trajectory to avoid cross-episode leakage.
    for traj in traj_ids.unique():
        idx = (traj_ids == traj).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        idx = idx.sort().values

        traj_rewards = rewards[idx]
        traj_dones = dones[idx]
        traj_primitives = step_primitives[idx]

        cumsum = torch.cumsum(traj_rewards, dim=0)

        def sum_range(a: int, b: int) -> torch.Tensor:
            if a == 0:
                return cumsum[b]
            return cumsum[b] - cumsum[a - 1]

        start_positions: list[int] = []
        end_positions: list[int] = []
        block_lens: list[int] = []
        block_rewards: list[torch.Tensor] = []
        block_dones: list[torch.Tensor] = []

        start_pos = 0
        primitive_acc = 0

        for pos in range(idx.numel()):
            primitive_acc += int(traj_primitives[pos].item())
            done_now = bool(traj_dones[pos].item())

            if done_now or primitive_acc >= milestone:
                start_positions.append(start_pos)
                end_positions.append(pos)
                block_lens.append(pos - start_pos + 1)
                block_rewards.append(sum_range(start_pos, pos))
                block_dones.append(traj_dones[pos])

                primitive_acc = primitive_acc % milestone
                start_pos = pos + 1

        # Final partial block (bootstrap if not done)
        if start_pos < idx.numel():
            end_pos = idx.numel() - 1
            start_positions.append(start_pos)
            end_positions.append(end_pos)
            block_lens.append(end_pos - start_pos + 1)
            block_rewards.append(sum_range(start_pos, end_pos))
            block_dones.append(traj_dones[end_pos])

        if not block_rewards:
            continue

        start_indices = idx[torch.tensor(start_positions, device=device)]
        end_indices = idx[torch.tensor(end_positions, device=device)]

        start_td = flat_td[start_indices]
        next_obs = flat_td["next", "observation"][end_indices]
        next_td = TensorDict({"observation": next_obs}, batch_size=[len(end_indices)], device=device)

        V0 = critic(start_td)["state_value"].view(-1)
        V1 = critic(next_td)["state_value"].view(-1)

        R = torch.stack(block_rewards).view(-1)
        done_b = torch.stack(block_dones).view(-1)
        not_done = (~done_b).to(dtype=V0.dtype)

        delta = R + gamma * not_done * V1 - V0

        A = torch.zeros_like(delta)
        gae = torch.zeros(1, device=device, dtype=delta.dtype)
        for t in reversed(range(delta.numel())):
            gae = delta[t] + gamma * lmbda * not_done[t] * gae
            A[t] = gae

        VT = A + V0

        for b, (s_pos, e_pos, blen) in enumerate(zip(start_positions, end_positions, block_lens)):
            step_idx = idx[s_pos : e_pos + 1]
            adv_val = A[b]
            if scale_advantage_by_block and blen > 0:
                adv_val = adv_val / float(blen)
            advantage[step_idx] = adv_val
            value_target[step_idx] = VT[b]
            milestone_block_len[step_idx] = float(blen)

        milestone_boundary[start_indices] = True
        total_steps += idx.numel()
        total_blocks += len(block_lens)

    avg_block_len = float(total_steps) / float(total_blocks) if total_blocks > 0 else 1.0
    flat_td.set("advantage", advantage.unsqueeze(-1))
    flat_td.set("value_target", value_target.unsqueeze(-1))
    flat_td.set("milestone_boundary", milestone_boundary.unsqueeze(-1))
    flat_td.set("milestone_block_len", milestone_block_len.unsqueeze(-1))
    flat_td.set(
        "milestone_avg_block_len",
        torch.full_like(rewards, avg_block_len, dtype=torch.float32).unsqueeze(-1),
    )
    return flat_td


@dataclass
class PPOConfig(AlgorithmConfig):
    implementation: str = "torchrl"
    graphs_per_collection: int = 10
    states_per_collection: int = 1920
    minibatch_size: int = 250
    epochs_per_collection: int = 4
    num_collections: int = 1000
    workers: int = 1
    clip_eps: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    val_coef: float = 0.5
    max_grad_norm: float = 0.5
    threads_per_worker: int = 1
    collect_device: str = "cpu"
    update_device: str = "cpu"
    storing_device: str = "cpu"
    gamma: float = 1
    lmbda: float = 0.99
    normalize_advantage: bool = False
    value_norm: str = "l2"
    compile_policy: bool = False
    compile_update: bool = False
    compile_advantage: bool = False
    collector: str = "multi_sync"  # "sync" or "multi_sync"
    sample_slices: bool = True  # if using lstm, whether slices are used instead of episodes
    slice_len: int = 16  # length of slices for LSTM, only used if sample_slices is True
    rollout_steps: int = 250  # rollout length in milestones
    advantage_type: str = "gae"  # "gae" or "vtrace"
    bagged_policy: str = "uniform"
    timeout: int = 60 * 60 * 24  # 1 day
    milestone: int = 32  # primitive steps per milestone; 0 disables milestone-time GAE
    milestone_scale_advantage: bool = True  # normalize advantage by steps per milestone block


def should_log(
    n_updates: int,
    logging_config: Optional[LoggingConfig],
) -> bool:
    """Check if we should log based on the current update count and logging configuration."""
    if logging_config is None:
        return False
    return n_updates % logging_config.stats_interval == 0


def should_eval(
    n_updates: int,
    eval_config: Optional[EvaluationConfig],
) -> bool:
    """Check if we should evaluate based on the current update count and logging configuration."""
    if eval_config is None:
        return False
    return eval_config.eval_interval > 0 and n_updates % eval_config.eval_interval == 0


def should_checkpoint(
    n_updates: int,
    logging_config: Optional[LoggingConfig],
) -> bool:
    """Check if we should checkpoint based on the current update count and logging configuration."""
    if logging_config is None:
        return False
    return n_updates % logging_config.checkpoint_interval == 0


def log_training_metrics(
    flattened_data: TensorDict,
    tensordict_data: TensorDict,
    loss: Dict[str, torch.Tensor],
    loss_module: ClipPPOLoss,
    optimizer: torch.optim.Optimizer,
    n_updates: int,
    i: int,
    n_samples: int,
) -> None:
    """Log training metrics to wandb."""
    with torch.no_grad():
        rewards = flattened_data["next", "reward"]
        improvements = flattened_data["next", "observation", "aux", "improvement"]
        valid_improvement_mask = torch.isfinite(improvements) & (improvements > -100)
        valid_improvements = improvements[valid_improvement_mask]
        valid_times = flattened_data["next", "observation", "aux", "time"][valid_improvement_mask]
        valid_times = valid_times.to(torch.float32)

        # Calculate improvement metrics
        if valid_improvements.numel() > 0:
            avg_improvement = valid_improvements.mean().item()
            max_improvement = valid_improvements.max().item()
            min_improvement = valid_improvements.min().item()
            avg_time = valid_times.mean().item()
            min_time = valid_times.min().item()
            max_time = valid_times.max().item()

            if valid_improvements.numel() > 1:
                std_improvement = valid_improvements.std().item()
            else:
                std_improvement = None

        # Calculate reward metrics
        if rewards.numel() > 0:
            avg_reward = rewards.mean().item()

            if rewards.numel() > 1:
                std_reward = rewards.std().item()
            else:
                std_reward = None

        # Calculate advantage and value target metrics
        advantage_mean = flattened_data["advantage"].mean().item()
        advantage_std = flattened_data["advantage"].std().item()
        value_target_mean = flattened_data["value_target"].mean().item()
        value_target_std = flattened_data["value_target"].std().item()

        explained_variance = None
        if "state_value" in flattened_data.keys() and "value_target" in flattened_data.keys():
            values = flattened_data["state_value"].squeeze(-1)
            targets = flattened_data["value_target"].squeeze(-1)
            if targets.numel() > 1:
                var_targets = targets.var(unbiased=False)
                if torch.isfinite(var_targets) and var_targets > 1e-8:
                    residual_var = (targets - values).var(unbiased=False)
                    explained_variance = (1.0 - (residual_var / var_targets)).item()

        # Get gradient and parameter norms
        post_clip_norms = log_parameter_and_gradient_norms(loss_module)

        # Base log payload
        log_payload = {
            **post_clip_norms,
            "batch/n_updates": n_updates,
            "batch/n_collections": i + 1,
            "batch/avg_reward": avg_reward,
            "batch/n_samples": n_samples,
            "batch/policy_loss": loss["loss_objective"].item(),
            "batch/critic_loss": loss["loss_critic"].item(),
            "batch/entropy_loss": loss["loss_entropy"].item(),
            "batch/entropy": loss["entropy"].item(),
            "batch/kl_approx": loss["kl_approx"].item(),
            "batch/clip_fraction": loss["clip_fraction"].item(),
            "batch/ESS": loss["ESS"].item(),
            "batch/advantage_mean": advantage_mean,
            "batch/advantage_std": advantage_std,
            "batch/mean_value_target": value_target_mean,
            "batch/std_value_target": value_target_std,
            "batch/lr": optimizer.param_groups[0]["lr"],
        }

        if std_reward is not None:
            log_payload["batch/std_reward"] = std_reward

        if explained_variance is not None:
            log_payload["batch/explained_variance"] = explained_variance

        # Add improvement metrics if available
        if valid_improvements.numel() > 0:
            log_payload.update(
                {
                    "batch/mean_improvement": avg_improvement,
                    "batch/max_improvement": max_improvement,
                    "batch/min_improvement": min_improvement,
                    "batch/mean_time": avg_time,
                    "batch/max_time": max_time,
                    "batch/min_time": min_time,
                }
            )
            if std_improvement is not None:
                log_payload["batch/std_improvement"] = std_improvement

            training.info(f"Average training improvement: {avg_improvement}")

        training.info(f"Average entropy {loss['entropy'].item()}")
        wandb.log(log_payload)


def run_ppo(
    actor_critic_module: ActorCriticModule,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    eval_config: Optional[EvaluationConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    seed: int = 0,
    eval_location = None,
):
    if logging_config is not None and (logging_frequency := logging_config.stats_interval):
        wandb.define_metric("batch/n_updates")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_updates")
        wandb.define_metric("batch/n_collections", step_metric="batch/n_updates")
        wandb.define_metric("batch/*", step_metric="batch/n_updates")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("param_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("eval/*", step_metric="batch/n_updates")

    print("Using PPO with config:", OmegaConf.to_yaml(ppo_config))
    if ppo_config.milestone > 0 and ppo_config.advantage_type != "gae":
        raise ValueError("Milestone GAE currently supports only advantage_type='gae'.")
    if ppo_config.milestone > 0 and ppo_config.advantage_type != "gae":
        raise ValueError("Milestone GAE currently supports only advantage_type='gae'.")

    eval_envs = make_eval_envs(env_constructors)
    max_tasks = max([env.size() for env in eval_envs])
    max_graph_size = max_tasks
    max_candidates = max([env.simulator_factory[0].graph_spec.max_candidates for env in eval_envs])

    rollout_env_steps = 0
    if ppo_config.rollout_steps > 0:
        if ppo_config.milestone <= 0:
            raise ValueError("rollout_steps is in milestones; set a positive milestone size.")
        rollout_env_steps = milestones_to_env_steps(ppo_config.rollout_steps, ppo_config.milestone, max_candidates)
        max_tasks = rollout_env_steps

    max_states_per_collection = ppo_config.graphs_per_collection * max_tasks
    max_macro_per_collection = (
        int(math.ceil(max_states_per_collection / ppo_config.milestone))
        if ppo_config.milestone > 0
        else max_states_per_collection
    )

    if ppo_config.advantage_type == "gae":
        training.info("Using GAE for advantage estimation")
        advantage_module = GAE(
            gamma=ppo_config.gamma,
            lmbda=ppo_config.lmbda,
            value_network=actor_critic_module.critic,
            average_gae=False,
            device=ppo_config.update_device,
            vectorized=(False if ppo_config.compile_advantage else True),
            deactivate_vmap=True,
        )
    elif ppo_config.advantage_type == "vtrace":
        training.info("Using VTrace for advantage estimation")
        advantage_module = VTrace(
            gamma=ppo_config.gamma,
            value_network=actor_critic_module.critic,
            actor_network=actor_critic_module.actor,
            device=ppo_config.update_device,
        )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            max_size=max_macro_per_collection if ppo_config.milestone > 0 else max_states_per_collection,
            device=ppo_config.update_device,
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=ppo_config.minibatch_size,
    )

    def env_workers():
        return [env_constructors[i % len(env_constructors)] for i in range(ppo_config.graphs_per_collection)]

    if ppo_config.collector == "multi_sync":
        collector = MultiSyncDataCollector(
            env_workers(),
            actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            cat_results="stack",
            reset_at_each_iter=False if rollout_env_steps > 0 else True,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            num_threads=ppo_config.workers,
            compile_policy=({"mode": "reduce-overhead"} if ppo_config.compile_policy else None),
        )
    elif ppo_config.collector == "sync":
        collector = SyncDataCollector(
            env_workers()[0],
            policy=actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            reset_at_each_iter=False if rollout_env_steps > 0 else True,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            compile_policy=({"mode": "reduce-overhead"} if ppo_config.compile_policy else None),
        )
    else:
        raise ValueError(f"Unknown collector type: {ppo_config.collector}. " "Use 'sync' or 'multi_sync'.")

    collector.set_seed(seed)

    if ppo_config.milestone > 0:
        loss_module = MacroReplayPPOLoss(
            actor=actor_critic_module.actor,
            critic=actor_critic_module.critic,
            clip_eps=ppo_config.clip_eps,
            ent_coef=ppo_config.ent_coef,
            val_coef=ppo_config.val_coef,
            normalize_advantage=ppo_config.normalize_advantage,
        )
    else:
        loss_module = ClipPPOLoss(
            actor_network=actor_critic_module.actor,
            critic_network=actor_critic_module.critic,
            clip_epsilon=ppo_config.clip_eps,
            entropy_bonus=True,
            entropy_coeff=ppo_config.ent_coef,
            critic_coeff=ppo_config.val_coef,
            loss_critic_type=ppo_config.value_norm,
            clip_value=ppo_config.clip_vloss,
            normalize_advantage=ppo_config.normalize_advantage,
        )

        if ppo_config.advantage_type == "gae":
            loss_module.make_value_estimator(ValueEstimators.GAE)
        elif ppo_config.advantage_type == "vtrace":
            loss_module.make_value_estimator(ValueEstimators.VTrace)

    if optimizer is None:
        optimizer = torch.optim.Adam(loss_module.parameters())
    else:
        optimizer = optimizer(loss_module.parameters())
    training.info(f"Using optimizer: {optimizer}")

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer)
        training.info(f"Using learning rate scheduler: {lr_scheduler}")

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    def update(batch, loss_module, optimizer, ppo_config):
        loss_vals = loss_module(batch)
        loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

        if loss_vals["kl_approx"] > 0.8:
            training.warning(f"High KL divergence detected: {loss_vals['kl_approx'].item()}")
            training.warning("Skipping gradient update to maintain training stability.")
            optimizer.zero_grad()
            return loss_vals

        # joint_stats(batch, loss_module)

        optimizer.zero_grad()
        loss_value.backward()

        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=ppo_config.max_grad_norm)

        optimizer.step()

        return loss_vals

    if ppo_config.compile_advantage and ppo_config.milestone <= 0:
        advantage_module = compile_with_warmup(advantage_module, mode="reduce-overhead", warmup=8)

    if ppo_config.compile_update:
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    states_per_collection = min(ppo_config.states_per_collection, max_states_per_collection)
    if ppo_config.milestone > 0:
        states_per_collection_macro = int(math.ceil(states_per_collection / ppo_config.milestone))
        n_batch = max(1, states_per_collection_macro // ppo_config.minibatch_size)
        if ppo_config.minibatch_size > states_per_collection_macro:
            training.warning(
                f"Minibatch size <{ppo_config.minibatch_size}> is larger than macro samples per collection <{states_per_collection_macro}>. "
            )
    else:
        n_batch = max(1, states_per_collection // ppo_config.minibatch_size)
        if ppo_config.minibatch_size > states_per_collection:
            training.warning(
                f"Minibatch size <{ppo_config.minibatch_size}> is larger than states per collection <{states_per_collection}>. "
            )

    training.info(
        f"Running PPO training with {ppo_config.num_collections} collections, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{n_batch} batches per epoch, "
        f"{ppo_config.workers} workers."
    )

    training.info(f"Max tasks per graph: {max_graph_size}, max candidates per task: {max_candidates}")

    max_performance = 0.0
    if should_eval(0, eval_config):
        training.info("Running initial evaluation before training")
        metrics = run_evaluation(collector.policy, eval_envs, eval_config, 0, eval_location=eval_location)

    training.info("Starting PPO training loop")

    start_t = time.perf_counter()
    n_updates = 0
    n_samples = 0
    n_collections = 0
    for i, tensordict_data in enumerate(collector):
        n_collections += 1
        replay_buffer.empty()

        if i >= ppo_config.num_collections:
            break

        collector.policy.eval()

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        updates_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
        seconds_per_update = elapsed_time / (i + 1) if (i + 1) > 0 else 0

        training.info(f"Collection {i + 1}/{ppo_config.num_collections}, " f"Collections/s: {updates_per_second:.2f}, " f"ms/Update: {seconds_per_update * 1000:.2f}")

        tensordict_data = tensordict_data.to(ppo_config.update_device, non_blocking=True)

        adv_start_t = time.perf_counter()
        with torch.no_grad():
            # Redistribute Rewards
            if ppo_config.bagged_policy == "uniform":
                redistribute_rewards_uniform(tensordict_data)

            if ppo_config.milestone > 0:
                flattened_data = tensordict_data.reshape(-1)
                macro_data = pack_primitives_to_macros(flattened_data, milestone=ppo_config.milestone)
                macro_data = compute_macro_gae(
                    macro_data,
                    critic=actor_critic_module.critic,
                    milestone=ppo_config.milestone,
                    gamma=ppo_config.gamma,
                    lmbda=ppo_config.lmbda,
                )

                valid = macro_data["valid_mask"]
                MB, X = valid.shape
                adv_tok = macro_data["macro_advantage"].expand(MB, X)
                vt_tok = macro_data["macro_value_target"].expand(MB, X)
                valid_flat = valid.reshape(-1)
                flattened_data.set("advantage", adv_tok.reshape(-1)[valid_flat].unsqueeze(-1))
                flattened_data.set("value_target", vt_tok.reshape(-1)[valid_flat].unsqueeze(-1))
            else:
                print("DEFAULT")
                advantage_module(tensordict_data)
                flattened_data = tensordict_data.reshape(-1)

        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")
        samples_in_collection = macro_data.shape[0] if ppo_config.milestone > 0 else flattened_data.shape[0]
        n_samples += samples_in_collection

        # print("SANITY CHECK OF SIZES IN OBSERVATION")
        # print("observation shape", flattened_data["observation"].shape)
        # print("action shape", flattened_data["action"].shape)
        # print("reward shape", flattened_data["next", "reward"].shape)
        # print("done shape", flattened_data["next", "done"].shape)
        # print("logits shape", flattened_data["logits"].shape)

        # print("keys", flattened_data.keys())

        # if max_candidates > 1:
        #    flattened_data["advantage"] = flattened_data["advantage"].expand(
        #        -1, max_candidates
        #    )
        #    flattened_data["advantage"] = flattened_data["advantage"].unsqueeze(-1)
        #
        #    flattened_data["value_target"] = flattened_data["value_target"].expand(
        #        -1, max_candidates
        #    )
        #    flattened_data["value_target"] = flattened_data["value_target"].unsqueeze(-1)

        # flattened_data["reward"] = flattened_data["next", "reward"].expand(
        #     -1, max_candidates
        # )

        # flattened_data["reward"] = flattened_data["reward"].unsqueeze(-1)

        # flattened_data["done"] = flattened_data["next", "done"].expand(
        #     -1, max_candidates
        # )
        # flattened_data["done"] = flattened_data["done"].unsqueeze(-1)

        # print("advantage shape", flattened_data["advantage"].shape)
        # print("value target shape", flattened_data["value_target"].shape)
        # print("reward shape", flattened_data["next", "reward"].shape)
        # print("done shape", flattened_data["next", "done"].shape)
        if ppo_config.milestone > 0:
            replay_buffer.extend(macro_data)
        else:
            replay_buffer.extend(flattened_data)

        update_start_t = time.perf_counter()
        actor_net = loss_module.actor_network if hasattr(loss_module, "actor_network") else loss_module.actor
        critic_net = loss_module.critic_network if hasattr(loss_module, "critic_network") else loss_module.critic
        actor_net.train()
        critic_net.train()
        for j in range(ppo_config.epochs_per_collection):
            if ppo_config.milestone > 0:
                buffer_len = len(replay_buffer)
                if buffer_len <= 0:
                    continue
                effective_batch = min(ppo_config.minibatch_size, buffer_len)
                n_batch_local = max(1, buffer_len // effective_batch)
            else:
                effective_batch = ppo_config.minibatch_size
                n_batch_local = n_batch

            for k in range(n_batch_local):
                n_updates += 1
                batch = replay_buffer.sample(effective_batch)
                batch = batch.to(ppo_config.update_device, non_blocking=True)
                loss = update(batch, loss_module, optimizer, ppo_config)

                if should_log(n_updates, logging_config):
                    log_training_metrics(
                        flattened_data,
                        tensordict_data,
                        loss,
                        loss_module,
                        optimizer,
                        n_updates,
                        i,
                        n_samples,
                    )

        actor_net = loss_module.actor_network if hasattr(loss_module, "actor_network") else loss_module.actor
        collector.update_policy_weights_(TensorDict.from_module(actor_net).to(ppo_config.collect_device))
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if should_eval(n_collections, eval_config=eval_config):
            collector.policy.eval()
            metrics = run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples, eval_location=eval_location)
            if eval_config.pickle_path is not None:
                if metrics[f"eval/DETERMINISTIC"]["mean_vsEFT"] > max_performance:
                    max_performance = metrics[f"eval/DETERMINISTIC"]["mean_vsEFT"]
                    training.info(f"New max performance: {max_performance:.4f}. Saving checkpoint.")
                    if logging_config.best_policy_dir is not None:
                        critic_net = loss_module.critic_network if hasattr(loss_module, "critic_network") else loss_module.critic
                        save_checkpoint(
                            n_collections,
                            policy_module=collector.policy,
                            value_module=critic_net,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            filename=f"{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_{max_performance:.3f}_{seed}.pt",
                            checkpoint_dir=logging_config.best_policy_dir,
                        )

        if should_checkpoint(n_collections, logging_config):
            training.info(f"Checkpointing at collection {n_collections}")
            critic_net = loss_module.critic_network if hasattr(loss_module, "critic_network") else loss_module.critic
            save_checkpoint(
                n_collections,
                policy_module=collector.policy,
                value_module=critic_net,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        if elapsed_time > ppo_config.timeout:
            training.warning(f"Timeout reached after {elapsed_time:.2f} seconds. Stopping training.")
            break

    if eval_config is not None and eval_config.eval_interval > 0:
        training.info("Running final evaluation after training")
        run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples, eval_location=eval_location)

    critic_net = loss_module.critic_network if hasattr(loss_module, "critic_network") else loss_module.critic
    save_checkpoint(n_collections, policy_module=collector.policy, value_module=critic_net, optimizer=optimizer, lr_scheduler=lr_scheduler)

    collector.shutdown()


def run_ppo_lstm(
    actor_critic_module: ActorCriticModule,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    eval_config: Optional[EvaluationConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    seed: int = 0,
    eval_location = None,
):
    if logging_config is not None and (logging_frequency := logging_config.stats_interval):
        wandb.define_metric("batch/n_updates")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_updates")
        wandb.define_metric("batch/n_collections", step_metric="batch/n_updates")
        wandb.define_metric("batch/*", step_metric="batch/n_updates")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("param_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("eval/*", step_metric="batch/n_updates")

    print("Using PPO with config:", OmegaConf.to_yaml(ppo_config))

    eval_envs = make_eval_envs(env_constructors)
    max_tasks = max([env.size() for env in eval_envs])
    max_candidates = max([env.simulator_factory[0].graph_spec.max_candidates for env in eval_envs])

    print(f"Max tasks in env constructors: {max_tasks}")

    rollout_env_steps = 0
    if ppo_config.rollout_steps > 0:
        if ppo_config.milestone <= 0:
            raise ValueError("rollout_steps is in milestones; set a positive milestone size.")
        rollout_env_steps = milestones_to_env_steps(ppo_config.rollout_steps, ppo_config.milestone, max_candidates)
        max_tasks = rollout_env_steps

    max_states_per_collection = ppo_config.graphs_per_collection * max_tasks

    if ppo_config.advantage_type == "gae":
        training.info("Using GAE for advantage estimation")
        advantage_module = GAE(
            gamma=ppo_config.gamma,
            lmbda=ppo_config.lmbda,
            value_network=actor_critic_module.critic,
            average_gae=False,
            device=ppo_config.update_device,
            deactivate_vmap=True,
        )

    elif ppo_config.advantage_type == "vtrace":
        training.info("Using VTrace for advantage estimation")
        advantage_module = VTrace(
            gamma=ppo_config.gamma,
            lmbda=ppo_config.lmbda,
            value_network=actor_critic_module.critic,
            actor_network=actor_critic_module.actor,
            device=ppo_config.update_device,
            deactivate_vmap=True,
        )

    if ppo_config.milestone > 0 and ppo_config.sample_slices:
        training.warning("Milestone macro replay is incompatible with slice sampling; disabling sample_slices.")
        ppo_config.sample_slices = False

    max_macro_per_collection = (
        int(math.ceil(max_states_per_collection / ppo_config.milestone))
        if ppo_config.milestone > 0
        else max_states_per_collection
    )

    if ppo_config.sample_slices:
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=max_states_per_collection,
                device=ppo_config.update_device,
            ),
            sampler=SliceSampler(
                strict_length=True,
                slice_len=ppo_config.slice_len,
                traj_key=("collector", "traj_ids"),
            ),
            batch_size=ppo_config.minibatch_size,
        )
        num_slices = ppo_config.minibatch_size // ppo_config.slice_len
    else:
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=max_macro_per_collection if ppo_config.milestone > 0 else max_states_per_collection,
                device=ppo_config.update_device,
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=ppo_config.minibatch_size,
        )
        num_slices = ppo_config.minibatch_size

    def env_workers():
        return [env_constructors[i % len(env_constructors)] for i in range(ppo_config.workers)]

    print(f"Creating collector with {ppo_config.workers} workers")

    if ppo_config.collector == "multi_sync":
        collector = MultiSyncDataCollector(
            env_workers(),
            actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            cat_results="stack",
            reset_at_each_iter=False if rollout_env_steps > 0 else True,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            compile_policy=({"mode": "reduce-overhead"} if ppo_config.compile_policy else None),
        )
    elif ppo_config.collector == "sync":
        collector = SyncDataCollector(
            env_workers()[0],
            policy=actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            reset_at_each_iter=False if rollout_env_steps > 0 else True,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
        )
    else:
        raise ValueError(f"Unknown collector type: {ppo_config.collector}. " "Use 'sync' or 'multi_sync'.")

    collector.set_seed(seed)

    if ppo_config.milestone > 0:
        loss_module = MacroReplayPPOLoss(
            actor=actor_critic_module.actor,
            critic=actor_critic_module.critic,
            clip_eps=ppo_config.clip_eps,
            ent_coef=ppo_config.ent_coef,
            val_coef=ppo_config.val_coef,
            normalize_advantage=ppo_config.normalize_advantage,
        )
    else:
        loss_module = ClipPPOLoss(
            actor_network=actor_critic_module.actor,
            critic_network=actor_critic_module.critic,
            clip_epsilon=ppo_config.clip_eps,
            entropy_bonus=True,
            entropy_coeff=ppo_config.ent_coef,
            critic_coeff=ppo_config.val_coef,
            loss_critic_type=ppo_config.value_norm,
            clip_value=ppo_config.clip_vloss,
            normalize_advantage=ppo_config.normalize_advantage,
        )

        if ppo_config.advantage_type == "gae":
            loss_module.make_value_estimator(ValueEstimators.GAE)
        elif ppo_config.advantage_type == "vtrace":
            loss_module.make_value_estimator(ValueEstimators.VTrace)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            loss_module.parameters(),
            lr=3e-4,
            eps=1e-5,
        )
    else:
        optimizer = optimizer(loss_module.parameters())

    print(f"Using optimizer: {optimizer}")

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer)
        print(f"Using learning rate scheduler: {lr_scheduler}")

    def update(batch, i, j, k):
        if ppo_config.sample_slices:
            batch = batch.reshape(num_slices, -1)

        loss_vals = loss_module(batch)
        loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

        if loss_vals["kl_approx"] > 0.8:
            training.warning(f"High KL divergence detected: {loss_vals['kl_approx'].item()}")
            training.warning("Skipping gradient update to maintain training stability.")
            optimizer.zero_grad()
            return loss_vals

        optimizer.zero_grad()
        loss_value.backward()

        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=ppo_config.max_grad_norm)

        optimizer.step()

        return loss_vals

    if ppo_config.compile_advantage and ppo_config.milestone <= 0:
        advantage_module = compile_with_warmup(advantage_module, mode="reduce-overhead", warmup=8)

    if ppo_config.compile_update:
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    states_per_collection = min(ppo_config.states_per_collection, max_states_per_collection)

    if ppo_config.milestone > 0:
        states_per_collection_macro = int(math.ceil(states_per_collection / ppo_config.milestone))
        n_batch = max(1, states_per_collection_macro // ppo_config.minibatch_size)
    elif ppo_config.sample_slices:
        n_batch = max(1, states_per_collection // ppo_config.minibatch_size)
    else:
        n_batch = max(1, ppo_config.graphs_per_collection // ppo_config.minibatch_size)

    training.info(
        f"Starting PPO-LSTM training with {ppo_config.num_collections} collections, "
        f"sample_slices={ppo_config.sample_slices}, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{n_batch} batches per epoch, "
        f"{ppo_config.workers} workers.",
    )

    # Initial evaluation
    if should_eval(0, eval_config):
        training.info("Running initial evaluation before training")
        run_evaluation(collector.policy, eval_envs, eval_config, 0, 0, 0, eval_location=eval_location)

    start_t = time.perf_counter()

    n_updates = 0
    n_samples = 0
    n_collections = 0
    for i, tensordict_data in enumerate(collector):
        n_collections += 1
        replay_buffer.empty()

        if i >= ppo_config.num_collections:
            break

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        updates_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0

        training.info(
            f"Collection {i + 1}/{ppo_config.num_collections}, " f"Collections/s: {updates_per_second:.2f}",
        )

        tensordict_data = tensordict_data.to(ppo_config.update_device, non_blocking=True)

        adv_start_t = time.perf_counter()
        with torch.no_grad():
            if ppo_config.milestone > 0:
                flattened_data = tensordict_data.reshape(-1)
                macro_data = pack_primitives_to_macros(flattened_data, milestone=ppo_config.milestone)
                macro_data = compute_macro_gae(
                    macro_data,
                    critic=actor_critic_module.critic,
                    milestone=ppo_config.milestone,
                    gamma=ppo_config.gamma,
                    lmbda=ppo_config.lmbda,
                )

                valid = macro_data["valid_mask"]
                MB, X = valid.shape
                adv_tok = macro_data["macro_advantage"].expand(MB, X)
                vt_tok = macro_data["macro_value_target"].expand(MB, X)
                valid_flat = valid.reshape(-1)
                flattened_data.set("advantage", adv_tok.reshape(-1)[valid_flat].unsqueeze(-1))
                flattened_data.set("value_target", vt_tok.reshape(-1)[valid_flat].unsqueeze(-1))
            else:
                advantage_module(tensordict_data)
                flattened_data = tensordict_data.reshape(-1)
        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")

        if ppo_config.milestone > 0:
            replay_buffer.extend(macro_data)
        elif ppo_config.sample_slices:
            if max_candidates > 1:
                flattened_data["advantage"] = flattened_data["advantage"].expand(-1, max_candidates)
                flattened_data["advantage"] = flattened_data["advantage"].unsqueeze(-1)

            replay_buffer.extend(flattened_data)
        else:
            if max_candidates > 1:
                tensordict_data["advantage"] = tensordict_data["advantage"].expand(-1, max_candidates)
                tensordict_data["advantage"] = tensordict_data["advantage"].unsqueeze(-1)
            replay_buffer.extend(tensordict_data)

        n_samples += macro_data.shape[0] if ppo_config.milestone > 0 else flattened_data.shape[0]

        update_start_t = time.perf_counter()
        for j in range(ppo_config.epochs_per_collection):
            if ppo_config.milestone > 0:
                buffer_len = len(replay_buffer)
                if buffer_len <= 0:
                    continue
                effective_batch = min(ppo_config.minibatch_size, buffer_len)
                n_batch_local = max(1, buffer_len // effective_batch)
            else:
                effective_batch = ppo_config.minibatch_size
                n_batch_local = n_batch

            for k in range(n_batch_local):
                n_updates += 1
                batch, info = replay_buffer.sample(effective_batch, return_info=True)
                batch = batch.to(ppo_config.update_device, non_blocking=True)
                loss = update(batch, i, j, k)

                if should_log(n_updates, logging_config):
                    log_training_metrics(
                        flattened_data,
                        tensordict_data,
                        loss,
                        loss_module,
                        optimizer,
                        n_updates,
                        i,
                        n_samples,
                    )

        actor_net = loss_module.actor_network if hasattr(loss_module, "actor_network") else loss_module.actor
        collector.update_policy_weights_(TensorDict.from_module(actor_net).to(ppo_config.collect_device))
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if should_eval(n_collections, eval_config=eval_config):
            run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples, eval_location=eval_location)

        if should_checkpoint(n_collections, logging_config):
            training.info(f"Checkpointing at update: {n_updates}")
            critic_net = loss_module.critic_network if hasattr(loss_module, "critic_network") else loss_module.critic
            save_checkpoint(n_updates, policy_module=collector.policy, value_module=critic_net, optimizer=optimizer, lr_scheduler=lr_scheduler)

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        if elapsed_time > ppo_config.timeout:
            training.warning(f"Timeout reached after {elapsed_time:.2f} seconds. Stopping training.")
            break

    if eval_config is not None and eval_config.eval_interval > 0:
        training.info("Running final evaluation after training")
        run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples, eval_location=eval_location)

    critic_net = loss_module.critic_network if hasattr(loss_module, "critic_network") else loss_module.critic
    save_checkpoint(
        n_collections,
        policy_module=collector.policy,
        value_module=critic_net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    collector.shutdown()
