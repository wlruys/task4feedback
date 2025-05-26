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


def compute_gae(tensordict_data, gamma=1, lam=0.99):
    with torch.no_grad():
        value = tensordict_data["state_value"].view(-1)
        reward = tensordict_data["next", "reward"].view(-1)
        done = tensordict_data["next", "done"].view(-1)
        traj_ids = tensordict_data["collector", "traj_ids"].view(-1)

        advantage = torch.zeros_like(value)

        for traj_id in traj_ids.unique():
            mask = traj_ids == traj_id
            traj_value = value[mask]
            traj_reward = reward[mask]
            traj_done = done[mask]
            traj_advantage = torch.zeros_like(traj_value)

            gae = 0.0
            T = len(traj_value)

            for t in reversed(range(T)):
                if traj_done[t]:
                    c = 0
                    nv = 0
                else:
                    c = 1
                    nv = traj_value[t + 1] if t + 1 < T else 0

                delta = traj_reward[t] + gamma * nv * c - traj_value[t]
                gae = delta + gamma * lam * c * gae
                traj_advantage[t] = gae

                if traj_done[t] and t < T - 1:
                    print(
                        "Warning: done flag is set but not at the end of the trajectory.",
                        t,
                        T,
                    )

            advantage[mask] = traj_advantage

        value_target = advantage + value
        tensordict_data["advantage"] = advantage.unsqueeze(1)
        tensordict_data["value_target"] = value_target.unsqueeze(1)

        return tensordict_data


def logits_to_action(logits: torch.Tensor, action):
    probs = torch.distributions.Categorical(logits=logits)
    return probs.log_prob(action), probs.entropy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
