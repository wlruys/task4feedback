from tensordict import TensorDict
import torch

def compute_advantage(td: TensorDict):
    with torch.no_grad():
        state_values = td["state_value"].view(-1)
        traj_ids = td["collector", "traj_ids"].view(-1)
        rewards = td["next", "reward"].view(-1)
        # Sum the rewards along each trajectory
        # td["value_target"] = cumulative reward at each step
        cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
        for traj in traj_ids.unique():
            mask = traj_ids == traj
            traj_rewards = rewards[mask]
            # each element = sum of rewards from that step to the end of the trajectory
            traj_cum_rewards = torch.flip(
                torch.cumsum(torch.flip(traj_rewards, dims=[0]), dim=0), dims=[0]
            )
            cumulative_rewards[mask] = traj_cum_rewards.to(torch.float32)

        td["value_target"] = cumulative_rewards.unsqueeze(1)
        td["advantage"] = cumulative_rewards - state_values
    return td

def redistribute_rewards_uniform(td: TensorDict) -> TensorDict:
    """
    Redistribute each non‐zero (bagged) reward uniformly across the
    preceding zero‐reward steps of the same trajectory.
    """
    # [workers, time_dim, 1]
    rewards = td.get(("next", "reward"))
    # [workers, time_dim]
    traj_ids = td.get(("collector", "traj_ids"))
    device = rewards.device

    W, T, _ = rewards.shape
    new_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=device)

    with torch.no_grad():
        for w in range(W):
            # shape: [time_dim]
            local_r = rewards[w, :, 0]
            local_traj = traj_ids[w]
            buffer = torch.zeros_like(local_r, dtype=torch.float32, device=device)

            for traj in local_traj.unique():
                mask = local_traj == traj
                if not mask.any():
                    continue

                # pull out just this trajectory’s rewards
                traj_r = local_r[mask]  # shape: [n_steps_for_traj]

                # reverse so that we can group “backwards from each nonzero”
                r_rev = traj_r.flip(0)
                # group‐id increases by 1 at each nonzero
                groups = torch.cumsum(r_rev != 0, dim=0)
                # number of distinct groups = max_group_id + 1
                n_groups = int(groups.max().item()) + 1

                # accumulate sums and counts per group
                sum_per_group = torch.zeros(n_groups, device=device)
                cnt_per_group = torch.zeros(n_groups, device=device)
                sum_per_group.scatter_add_(0, groups, r_rev)
                cnt_per_group.scatter_add_(0, groups, torch.ones_like(r_rev))

                # compute the uniform share for each reversed position
                avg_per_group = sum_per_group / cnt_per_group
                share_rev = avg_per_group[groups]  # shape: same as r_rev

                # flip back to original order
                share = share_rev.flip(0)
                # write into the buffer at the masked positions
                buffer[mask] = share

            # assign back into the [workers, time_dim, 1] tensor
            new_rewards[w, :, 0] = buffer

        # overwrite the tensordict’s reward field
        td.set(("next", "reward"), new_rewards)

    return td

# def compute_gae(tensordict_data, gamma=1, lam=0.99):
#     with torch.no_grad():
#         value = tensordict_data["state_value"].view(-1)
#         reward = tensordict_data["next", "reward"].view(-1)
#         done = tensordict_data["next", "done"].view(-1)
#         traj_ids = tensordict_data["collector", "traj_ids"].view(-1)

#         advantage = torch.zeros_like(value)
#         value_target = torch.zeros_like(value)

#         for traj_id in traj_ids.unique():
#             mask = traj_ids == traj_id
#             traj_value = value[mask]
#             traj_reward = reward[mask]
#             traj_done = done[mask]
#             traj_advantage = torch.zeros_like(traj_value)
#             traj_value_target = torch.zeros_like(traj_value)

#             gae = 0.0
#             T = len(traj_value)

#             # Compute next values for TD error calculation
#             next_values = torch.zeros_like(traj_value)
#             for t in range(T - 1):
#                 if not traj_done[
#                     t
#                 ]:  # If not done, next value is the next state's value
#                     next_values[t] = traj_value[t + 1]
#                 # If done, next value remains 0 (already initialized)
#             # Last step's next value is 0 (episode ends)

#             # Compute GAE backwards through the trajectory
#             for t in reversed(range(T)):
#                 if traj_done[t]:
#                     # Episode terminates, next value is 0
#                     delta = traj_reward[t] - traj_value[t]
#                     gae = delta
#                 else:
#                     # Normal step
#                     delta = traj_reward[t] + gamma * next_values[t] - traj_value[t]
#                     gae = delta + gamma * lam * gae

#                 traj_advantage[t] = gae
#                 traj_value_target[t] = gae + traj_value[t]

#             advantage[mask] = traj_advantage
#             value_target[mask] = traj_value_target

#         tensordict_data["advantage"] = advantage.unsqueeze(1)
#         tensordict_data["value_target"] = value_target.unsqueeze(1)

#         return tensordict_data


def compute_gae(tensordict_data, gamma: float = 0.99, lam: float = 0.95):
    """
    Vectorized GAE computation that handles multiple trajectories efficiently.

    Args:
        tensordict_data: TensorDict containing:
            - "state_value": State values [batch_size, 1] or [batch_size]
            - ("next", "reward"): Rewards [batch_size, 1] or [batch_size]
            - ("next", "done"): Done flags [batch_size, 1] or [batch_size]
            - ("collector", "traj_ids"): Trajectory IDs [batch_size, 1] or [batch_size]
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        tensordict_data: Updated with "advantage" and "value_target"
    """
    with torch.no_grad():
        # Extract and flatten tensors
        values = tensordict_data["state_value"].squeeze(-1)  # [batch_size]
        rewards = tensordict_data["next", "reward"].squeeze(-1)  # [batch_size]
        dones = tensordict_data["next", "done"].squeeze(-1)  # [batch_size]
        traj_ids = tensordict_data["collector", "traj_ids"].squeeze(-1)  # [batch_size]

        batch_size = values.shape[0]
        device = values.device

        # Initialize outputs
        advantages = torch.zeros_like(values)
        value_targets = torch.zeros_like(values)

        # Get unique trajectory IDs and their positions
        unique_traj_ids = torch.unique(traj_ids)

        # Process each trajectory
        for traj_id in unique_traj_ids:
            # Get mask for current trajectory
            traj_mask = traj_ids == traj_id
            traj_indices = torch.where(traj_mask)[0]

            # Extract trajectory data
            traj_values = values[traj_mask]
            traj_rewards = rewards[traj_mask]
            traj_dones = dones[traj_mask]

            T = len(traj_values)

            # Compute next values for this trajectory
            next_values = torch.zeros_like(traj_values)
            # For all steps except the last, next value is the next state's value if not done
            next_values[:-1] = traj_values[1:] * (~traj_dones[:-1])
            # Last step always has next_value = 0 (episode boundary or terminal)

            # Compute TD residuals
            deltas = traj_rewards + gamma * next_values - traj_values

            # Compute GAE backwards through trajectory
            traj_advantages = torch.zeros_like(traj_values)
            gae = 0.0

            for t in reversed(range(T)):
                # GAE formula: A_t = δ_t + γλ(1-done_t)A_{t+1}
                gae = deltas[t] + gamma * lam * gae * (~traj_dones[t])
                traj_advantages[t] = gae

            # Store results back to full arrays
            advantages[traj_mask] = traj_advantages
            value_targets[traj_mask] = traj_advantages + traj_values

        # Update tensordict with computed advantages and value targets
        tensordict_data["advantage"] = advantages.unsqueeze(-1)
        tensordict_data["value_target"] = value_targets.unsqueeze(-1)

        return tensordict_data


def logits_to_action(logits: torch.Tensor, action):
    probs = torch.distributions.Categorical(logits=logits)
    return probs.log_prob(action), probs.entropy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_parameter_and_gradient_norms(model):
    """Log parameter and gradient norms to wandb"""
    param_norms = {}
    grad_norms = {}

    total_param_norm = 0.0
    total_grad_norm = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_norm = param.detach().norm().item()
            param_norms[f"param_norm/{name}"] = param_norm
            total_param_norm += param_norm**2

            if param.grad is not None:
                grad_norm = param.grad.detach().norm().item()
                grad_norms[f"grad_norm/{name}"] = grad_norm
                total_grad_norm += grad_norm**2

    total_param_norm = total_param_norm**0.5
    total_grad_norm = total_grad_norm**0.5

    return {
        **param_norms,
        **grad_norms,
        "param_norm/total": total_param_norm,
        "grad_norm/total": total_grad_norm,
    }


# def evaluate_policy(
#     policy,
#     eval_env_fn: Callable,
#     max_steps: int = 10000,
#     num_episodes: int = 1,
#     step=0,
# ) -> Dict[str, float]:
#     episode_rewards = []
#     completion_times = []
#     episode_returns = []
#     std_rewards = []

#     for i in range(num_episodes):
#         env = eval_env_fn()
#         with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
#             tensordict = env.rollout(
#                 max_steps=max_steps,
#                 policy=policy,
#             )

#         if "next" in tensordict and "reward" in tensordict["next"]:
#             rewards = tensordict["next", "reward"]
#             avg_reward = rewards.mean().item()
#             std_reward = rewards.std().item()
#             returns = tensordict["next", "reward"].sum().item()
#         else:
#             returns = 0.0
#             avg_non_zero_reward = 0.0
#             std_rewards = 0.0

#         episode_returns.append(returns)
#         episode_rewards.append(avg_reward)
#         std_rewards.append(std_reward)

#         if hasattr(env, "simulator") and hasattr(env.simulator, "time"):
#             completion_time = env.simulator.time
#             completion_times.append(completion_time)

#             if i == 0 and completion_time > 0:
#                 max_frames = 400
#                 time_interval = int(completion_time / max_frames)

#                 title = f"network_eval_{step}_{i}"
#                 print(title)
#                 animate_mesh_graph(
#                     env,
#                     time_interval=time_interval,
#                     show=False,
#                     title=title,
#                     figsize=(4, 4),
#                     dpi=50,
#                     bitrate=50,
#                 )

#                 if wandb.run.dir is None:
#                     path = "."
#                 else:
#                     path = wandb.run.dir

#                 video_path = os.path.join(path, title + ".mp4")

#                 wandb.log(
#                     {
#                         "eval/animation": wandb.Video(
#                             video_path,
#                             caption=title,
#                             format="mp4",
#                         )
#                     }
#                 )

#     # Create metrics dictionary
#     metrics = {
#         "eval/mean_return": sum(episode_rewards) / max(len(episode_rewards), 1),
#         # "eval/std_return": np.std(episode_rewards) if len(episode_rewards) > 1 else 0,
#         "eval/mean_reward": sum(episode_rewards) / max(len(episode_rewards), 1),
#         # "eval/std_mean_reward": np.std(episode_rewards)
#         # if len(episode_rewards) > 1
#         # else 0,
#         # "eval/std_std_reward": np.std(std_rewards) if len(std_rewards) > 1 else 0,
#         "eval/mean_std_reward": sum(std_rewards) / max(len(std_rewards), 1),
#     }

#     # Add completion time metrics if available
#     if completion_times:
#         metrics["eval/mean_completion_time"] = sum(completion_times) / len(
#             completion_times
#         )
#         # metrics["eval/min_completion_time"] = min(completion_times)
#         # metrics["eval/max_completion_time"] = max(completion_times)

#     return metrics
