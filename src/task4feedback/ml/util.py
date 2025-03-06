from tensordict import TensorDict
import torch
from torchrl.envs.transforms import Transform


def compute_advantage(td: TensorDict):
    with torch.no_grad():
        state_values = td["state_value"].view(-1)

        traj_ids = td["collector", "traj_ids"].view(-1)
        rewards = td["next", "reward"].view(-1)

        # Sum the rewards along each trajectory
        # td["returns"] = cumulative reward at each step
        cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
        for traj in traj_ids.unique():
            mask = traj_ids == traj
            traj_rewards = rewards[mask]
            # Compute cumulative sum in reverse order so that each element
            # contains the sum of rewards from that step to the end of the trajectory
            traj_cum_rewards = torch.flip(
                torch.cumsum(torch.flip(traj_rewards, dims=[0]), dim=0), dims=[0]
            )
            cumulative_rewards[mask] = traj_cum_rewards.to(torch.float32)

        # print(td["reward_to_go"])
        td["returns"] = cumulative_rewards.unsqueeze(1)
        # print(td["returns"])
        td["advantage"] = cumulative_rewards - state_values
    return td


def logits_to_action(logits: torch.Tensor, action):
    probs = torch.distributions.Categorical(logits=logits)
    return probs.log_prob(action), probs.entropy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
