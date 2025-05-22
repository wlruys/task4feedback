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


def compute_gae(tensordict_data, gamma=1, lam=0.90):
    with torch.no_grad():
        # critic(tensordict_data)
        # critic(tensordict_data["next"])

        value = tensordict_data["state_value"]
        # next_value = tensordict_data["next", "state_value"]
        reward = tensordict_data["next", "reward"]
        done = tensordict_data["next", "done"]

        print(reward.view(-1))
        print(done.view(-1))

        advantage = torch.zeros_like(value)
        gae = 0.0
        T = reward.shape[0]
        for t in reversed(range(T)):
            if done[t]:
                c = 0
                nv = 0
            else:
                c = 1
                nv = value[t + 1]

            delta = reward[t] + gamma * nv * c - value[t]
            gae = delta + gamma * lam * c * gae
            advantage[t] = gae

        value_target = advantage + value
        tensordict_data["advantage"] = advantage
        tensordict_data["value_target"] = value_target
        return tensordict_data


def logits_to_action(logits: torch.Tensor, action):
    probs = torch.distributions.Categorical(logits=logits)
    return probs.log_prob(action), probs.entropy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
