from tensordict import TensorDict
import torch
from torchrl.envs.transforms import Transform


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
