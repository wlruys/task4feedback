from .models import *
from .util import *
from dataclasses import dataclass
from typing import Callable
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torch_geometric.data import HeteroData, Batch
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor
import tensordict
from tensordict.nn import (
    TensorDictModule,
    ProbabilisticTensorDictModule,
    TensorDictSequential,
)
import torchrl
import torch_geometric
import aim
from aim.pytorch import track_gradients_dists, track_params_dists
from torchrl.envs.transforms import Reward2GoTransform
from torch_geometric.loader import DataLoader


@dataclass
class PPOConfig:
    states_per_collection: int = 1000
    minibatch_size: int = 250
    num_epochs_per_collection: int = 4
    num_collections: int = 1000
    workers: int = 4
    seed: int = 0
    lr: float = 2.5e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    val_coef: float = 0.5
    max_grad_norm: float = 0.5


def run_ppo_cleanrl_no_rb(
    actor_critic_base: nn.Module, make_env: Callable[[], EnvBase], config: PPOConfig
):
    _actor_critic_td = HeteroDataWrapper(actor_critic_base)

    _actor_critic_module = TensorDictModule(
        _actor_critic_td,
        in_keys=["observation"],
        out_keys=["logits", "state_value"],
    )

    actor_critic = ProbabilisticActor(
        _actor_critic_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        cache_dist=True,
        return_log_prob=True,
    )

    collector = MultiSyncDataCollector(
        [make_env for _ in range(config.workers)],
        actor_critic,
        frames_per_batch=config.states_per_collection,
        reset_at_each_iter=True,
        cat_results=0,
        # replay_buffer=replay_buffer,
    )
    out_seed = collector.set_seed(config.seed)

    optimizer = torch.optim.Adam(actor_critic_base.parameters(), lr=config.lr)

    for i, td in enumerate(collector):
        print("Collection:", i)
        with torch.no_grad():
            td = compute_advantage(td)

        state = []
        for l in range(len(td)):
            state.append(observation_to_heterodata(td[l]["observation"]))
            state[l]["action"] = td[l]["action"]
            state[l]["sample_log_prob"] = td[l]["sample_log_prob"]
            state[l]["state_value"] = td[l]["state_value"]
            state[l]["advantage"] = td[l]["advantage"]
            state[l]["returns"] = td[l]["returns"]
            state[l]["task_counts"] = td[l]["observation"]["nodes"]["tasks"]["count"]

        for j in range(config.num_epochs_per_collection):
            loader = DataLoader(state, batch_size=config.minibatch_size, shuffle=True)

            for j, batch in enumerate(loader):
                # print(batch, batch["task_counts"])
                new_logits, new_value = actor_critic_base.forward(
                    batch, batch["task_counts"].unsqueeze(-1)
                )

                new_logits = new_logits.view(-1)
                new_value = new_value.view(-1)

                sample_logprob = batch["sample_log_prob"].detach().view(-1)
                sample_value = batch["state_value"].detach().view(-1)
                sample_advantage = batch["advantage"].detach().view(-1)
                sample_returns = batch["returns"].detach().view(-1)
                sample_action = batch["action"].detach().view(-1)

                new_logprob, new_entropy = logits_to_action(new_logits, sample_action)
                new_logprob = new_logprob.view(-1)
                new_entropy = new_entropy.view(-1)

                with torch.no_grad():
                    print("Average Return:", sample_returns.mean())

                # Policy Loss
                logratio = new_logprob.view(-1) - sample_logprob.detach().view(-1)
                ratio = logratio.exp().view(-1)

                policy_loss_1 = sample_advantage * ratio.view(-1)
                policy_loss_2 = sample_advantage * torch.clamp(
                    ratio, 1 - config.clip_eps, 1 + config.clip_eps
                )
                policy_loss = -1 * torch.min(policy_loss_1, policy_loss_2).mean()

                # Value Loss
                v_loss_unclipped = (new_value - sample_returns) ** 2
                v_clipped = sample_value + torch.clamp(
                    new_value - sample_value, -config.clip_eps, config.clip_eps
                )
                v_loss_clipped = (v_clipped - sample_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                v_loss = 0.5 * v_loss_max

                # Entropy Loss
                entropy_loss = new_entropy.mean()

                loss = (
                    policy_loss
                    + config.val_coef * v_loss
                    - config.ent_coef * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    actor_critic_base.parameters(), config.max_grad_norm
                )
                optimizer.step()
        collector.update_policy_weights_()


def run_ppo_cleanrl(
    actor_critic_base: nn.Module, make_env: Callable[[], EnvBase], config: PPOConfig
):
    # r2g = Reward2GoTransform(gamma=1, out_keys=["reward_to_go"])
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.states_per_collection),
        sampler=SamplerWithoutReplacement(),
        # transform=r2g,
    )

    _actor_critic_td = HeteroDataWrapper(actor_critic_base)

    _actor_critic_module = TensorDictModule(
        _actor_critic_td,
        in_keys=["observation"],
        out_keys=["logits", "state_value"],
    )

    actor_critic = ProbabilisticActor(
        _actor_critic_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        cache_dist=True,
        return_log_prob=True,
    )

    collector = MultiSyncDataCollector(
        [make_env for _ in range(config.workers)],
        actor_critic,
        frames_per_batch=config.states_per_collection,
        reset_at_each_iter=True,
        cat_results=0,
        # replay_buffer=replay_buffer,
    )
    out_seed = collector.set_seed(config.seed)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.lr)

    for i, td in enumerate(collector):
        with torch.no_grad():
            td = compute_advantage(td)

        replay_buffer.extend(td.reshape(-1))

        for j in range(config.num_epochs_per_collection):
            n_batches = len(replay_buffer) // config.minibatch_size

            for k in range(n_batches):
                batch = replay_buffer.sample(config.minibatch_size)
                eval_batch = batch.clone()

                sample_logprob = batch["sample_log_prob"].detach().view(-1)
                sample_value = batch["state_value"].detach().view(-1)
                sample_advantage = batch["advantage"].detach().view(-1)
                sample_returns = batch["returns"].detach().view(-1)
                sample_action = batch["action"].detach().view(-1)

                eval_batch = actor_critic.forward(eval_batch)

                new_logprob, new_entropy = logits_to_action(
                    eval_batch["logits"], sample_action
                )
                new_logprob = new_logprob.view(-1)
                new_entropy = new_entropy.view(-1)

                new_value = eval_batch["state_value"].view(-1)

                with torch.no_grad():
                    print("Average Return:", sample_returns.mean())

                # Policy Loss
                logratio = new_logprob.view(-1) - sample_logprob.detach().view(-1)
                ratio = logratio.exp().view(-1)

                policy_loss_1 = sample_advantage * ratio.view(-1)
                policy_loss_2 = sample_advantage * torch.clamp(
                    ratio, 1 - config.clip_eps, 1 + config.clip_eps
                )
                policy_loss = -1 * torch.min(policy_loss_1, policy_loss_2).mean()

                # Value Loss
                v_loss_unclipped = (new_value - sample_returns) ** 2
                v_clipped = sample_value + torch.clamp(
                    new_value - sample_value, -config.clip_eps, config.clip_eps
                )
                v_loss_clipped = (v_clipped - sample_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                v_loss = 0.5 * v_loss_max

                # Entropy Loss
                entropy_loss = new_entropy.mean()

                loss = (
                    policy_loss
                    + config.val_coef * v_loss
                    - config.ent_coef * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    actor_critic.parameters(), config.max_grad_norm
                )
                optimizer.step()


def ppo_torchrl(
    actor_critic_base: nn.Module, make_env: Callable[[], EnvBase], config: PPOConfig
):
    # Implement version using torchrl built ins

    # Set up the actor-critic modules
    _actor_crtic_td = HeteroDataWrapper(actor_critic_base)

    module_action = TensorDictModule(
        _actor_crtic_td,
        in_keys=["observation"],
        out_keys=["logits", "state_value"],
    )

    td_module_action = ProbabilisticActor(
        module_action,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        cache_dist=True,
        return_log_prob=True,
    )

    # Set up the advantage module
    advantage_module = GAE(
        gamma=1.0,
        lmbda=0.95,
        value_network=TensorDictModule(
            lambda td: td["state_value"],
            in_keys=["state_value"],
            out_keys=["state_value"],
        ),
        average_gae=False,
    )

    # Set up collector
    collector = MultiSyncDataCollector(
        [make_env for _ in range(config.workers)],
        td_module_action,
        frames_per_batch=config.states_per_collection,
        reset_at_each_iter=True,
        cat_results=0,
    )
    collector.set_seed(config.seed)

    # Set up replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.states_per_collection),
        sampler=SamplerWithoutReplacement(),
    )

    # Set up loss module
    loss_module = ClipPPOLoss(
        actor_network=td_module_action,
        critic_network=TensorDictModule(
            lambda td: td["state_value"],
            in_keys=["state_value"],
            out_keys=["state_value"],
        ),
        clip_epsilon=config.clip_eps,
        entropy_bonus=True,
        entropy_coef=config.ent_coef,
        critic_coef=config.val_coef,
        loss_critic_type="l2",
    )

    # Set up optimizer
    optimizer = torch.optim.Adam(actor_critic_base.parameters(), lr=config.lr)

    # Set up logging
    aim_run = aim.Run(experiment="ppo-torchrl")
    track_params_dists(aim_run, actor_critic_base)

    epoch_idx = 0

    # Main training loop
    for i, tensordict_data in enumerate(collector):
        print(f"Collection: {i}")

        # Compute advantages
        with torch.no_grad():
            advantage_module(tensordict_data)

        # Track rewards
        episode_reward = tensordict_data["next", "reward"].mean().item()
        aim_run.track(episode_reward, name="reward/average", step=i)

        # Log non-zero rewards
        non_zero_rewards = tensordict_data["next", "reward"][
            tensordict_data["next", "reward"] != 0
        ]
        if len(non_zero_rewards) > 0:
            avg_non_zero_reward = non_zero_rewards.mean().item()
            aim_run.track(avg_non_zero_reward, name="reward/average_non_zero", step=i)
            print(f"Average non-zero reward: {avg_non_zero_reward}")

        # Extend replay buffer
        replay_buffer.extend(tensordict_data.reshape(-1))

        # Training epochs
        for j in range(config.num_epochs_per_collection):
            print(f"Epoch: {j}")

            # Calculate number of minibatches
            n_batches = config.states_per_collection // config.minibatch_size

            batch_loss_objective = 0
            batch_loss_critic = 0
            batch_loss_entropy = 0
            batch_loss_total = 0

            # Update policy on minibatches
            for k in range(n_batches):
                # Sample minibatch
                subdata = replay_buffer.sample(config.minibatch_size)

                # Compute losses
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Update parameters
                optimizer.zero_grad()
                loss_value.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor_critic_base.parameters(), max_norm=config.max_grad_norm
                )
                optimizer.step()

                # Track losses
                batch_loss_objective += loss_vals["loss_objective"].item()
                batch_loss_critic += loss_vals["loss_critic"].item()
                batch_loss_entropy += loss_vals["loss_entropy"].item()
                batch_loss_total += loss_value.item()

            # Log average losses for this epoch
            avg_loss_objective = batch_loss_objective / n_batches
            avg_loss_critic = batch_loss_critic / n_batches
            avg_loss_entropy = batch_loss_entropy / n_batches
            avg_loss_total = batch_loss_total / n_batches

            aim_run.track(avg_loss_objective, name="loss/objective", step=epoch_idx)
            aim_run.track(avg_loss_critic, name="loss/critic", step=epoch_idx)
            aim_run.track(avg_loss_entropy, name="loss/entropy", step=epoch_idx)
            aim_run.track(avg_loss_total, name="loss/total", step=epoch_idx)

            epoch_idx += 1

        # Stop if we've reached the desired number of collections
        if i >= config.num_collections - 1:
            break

    # Clean up
    collector.shutdown()
    aim_run.close()

    return actor_critic_base
