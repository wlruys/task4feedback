from .models import *
from .util import *
from dataclasses import dataclass
from typing import Callable
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import (
    ProbabilisticActor,
    ValueOperator,
    ActorCriticWrapper,
    LSTMModule,
    MLP,
    QValueModule,
    EGreedyModule,
    LSTM,
)
from torchrl.envs import TransformedEnv, set_exploration_type, ExplorationType
from torchrl.envs.transforms import StepCounter, TrajCounter, Compose, InitTracker

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch_geometric.loader import DataLoader
import copy
from tensordict import TensorDictBase, TensorDict
import wandb
import os
from task4feedback.ml import FeatureDimConfig, LayerConfig
from tensordict.nn import TensorDictSequential as Sequential


@dataclass
class PPOConfig:
    states_per_collection: int = 1920
    minibatch_size: int = 250
    num_epochs_per_collection: int = 4
    num_collections: int = 1000
    workers: int = 1
    seed: int = 0
    lr: float = 2.5e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.001
    val_coef: float = 0.5
    max_grad_norm: float = 0.5
    threads_per_worker: int = 1
    train_device: str = "cpu"
    gae_gamma: float = 1
    gae_lmbda: float = 0.1


def run_ppo_cleanrl_no_rb(
    actor_critic_base: nn.Module, make_env: Callable[[], EnvBase], config: PPOConfig
):
    """
    I don't know if this one works. Use the others for now.
    """
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
        cache_dist=False,
        return_log_prob=True,
    )

    collector = MultiSyncDataCollector(
        [make_env for _ in range(config.workers)],
        actor_critic,
        frames_per_batch=config.states_per_collection,
        reset_at_each_iter=True,
        cat_results=0,
        policy_device="cpu",
        env_device="cpu",
        # storing_device=config.train_device,
        # replay_buffer=replay_buffer,
    )
    out_seed = collector.set_seed(config.seed)

    actor_critic_base_t = copy.deepcopy(actor_critic_base)
    actor_critic_base_t = actor_critic_base_t.to(config.train_device)

    optimizer = torch.optim.Adam(actor_critic_base_t.parameters(), lr=config.lr)

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
            state[l]["value_target"] = td[l]["value_target"]
            state[l]["task_counts"] = td[l]["observation"]["nodes"]["tasks"]["count"]

        for j in range(config.num_epochs_per_collection):
            loader = DataLoader(state, batch_size=config.minibatch_size, shuffle=True)

            for j, batch in enumerate(loader):
                batch = batch.to(config.train_device, non_blocking=True)
                new_logits, new_value = actor_critic_base_t(
                    batch, batch["task_counts"].unsqueeze(-1)
                )

                new_logits = new_logits.view(-1)
                new_value = new_value.view(-1)

                sample_logprob = batch["sample_log_prob"].detach().view(-1)
                sample_value = batch["state_value"].detach().view(-1)
                sample_advantage = batch["advantage"].detach().view(-1)
                sample_returns = batch["value_target"].detach().view(-1)
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
                    actor_critic_base_t.parameters(), config.max_grad_norm
                )
                optimizer.step()

        collector.policy.module[0].module.network.load_state_dict(
            actor_critic_base_t.state_dict()
        )
        collector.update_policy_weights_(TensorDict.from_module(collector.policy))


def run_ppo_cleanrl(
    actor_critic_base: nn.Module, make_env: Callable[[], EnvBase], config: PPOConfig
):
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.states_per_collection),
        sampler=SamplerWithoutReplacement(),
        pin_memory=True,
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
        cache_dist=False,
        return_log_prob=True,
    )

    collector = MultiSyncDataCollector(
        [make_env for _ in range(config.workers)],
        actor_critic,
        frames_per_batch=config.states_per_collection,
        reset_at_each_iter=True,
        cat_results=0,
        # storing_device=config.train_device,
        env_device="cpu",
        policy_device="cpu",
        # replay_buffer=replay_buffer,
    )
    out_seed = collector.set_seed(config.seed)

    # Create a copy of the actor_critic model to be used for training
    actor_critic_t = copy.deepcopy(actor_critic)
    actor_critic_t = actor_critic_t.to(config.train_device)

    optimizer = torch.optim.Adam(actor_critic_t.parameters(), lr=config.lr)

    for i, td in enumerate(collector):
        print(f"Collection: {i}")

        with torch.no_grad():
            td = compute_advantage(td)

            td["record_state_value"] = td["state_value"].clone()
            td["record_log_prob"] = td["sample_log_prob"].clone()
            td["record_logits"] = td["logits"].clone()

        replay_buffer.extend(td.reshape(-1))

        for j in range(config.num_epochs_per_collection):
            n_batches = len(replay_buffer) // config.minibatch_size
            # actor_critic_t = actor_critic.to(config.train_device)

            for k in range(n_batches):
                batch = replay_buffer.sample(config.minibatch_size)
                batch = batch.to(config.train_device, non_blocking=True)

                sample_logprob = batch["record_log_prob"].detach().view(-1)
                sample_value = batch["record_state_value"].detach().view(-1)
                sample_advantage = batch["advantage"].detach().view(-1)
                sample_returns = batch["value_target"].detach().view(-1)
                sample_action = batch["action"].detach().view(-1)

                eval_batch = actor_critic_t(batch)

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
                    actor_critic_t.parameters(), config.max_grad_norm
                )
                optimizer.step()

        collector.policy.load_state_dict(actor_critic_t.state_dict())
        collector.update_policy_weights_(TensorDict.from_module(collector.policy))
    collector.shutdown()


def run_ppo_torchrl(
    actor_critic_base: nn.Module,
    make_env: Callable[[], EnvBase],
    config: PPOConfig,
    model_name: str = "model",
):
    _actor_td = HeteroDataWrapper(actor_critic_base.actor, device=config.train_device)
    _critic_td = HeteroDataWrapper(actor_critic_base.critic, device=config.train_device)

    module_action = TensorDictModule(
        _actor_td,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    td_module_action = ProbabilisticActor(
        module_action,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        cache_dist=False,
        return_log_prob=True,
    )

    td_critic_module = ValueOperator(
        module=_critic_td,
        in_keys=["observation"],
    )

    td_module_action = td_module_action.to(config.train_device)
    td_critic_module = td_critic_module.to(config.train_device)
    train_actor_network = copy.deepcopy(td_module_action).to(config.train_device)
    train_critic_network = copy.deepcopy(td_critic_module).to(config.train_device)
    model = torch.nn.ModuleList([train_actor_network, train_critic_network])
    collector = MultiSyncDataCollector(
        [make_env for _ in range(config.workers)],
        td_module_action,
        frames_per_batch=config.states_per_collection,
        reset_at_each_iter=True,
        cat_results=0,
        device=config.train_device,
        env_device="cpu",
    )
    out_seed = collector.set_seed(config.seed)
    print(f"Seed: {out_seed}")

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=config.states_per_collection, device=config.train_device
        ),
        sampler=SamplerWithoutReplacement(),
        pin_memory=torch.cuda.is_available(),
    )

    advantage_module = GAE(
        gamma=config.gae_gamma,
        lmbda=config.gae_lmbda,
        value_network=model[1],
        average_gae=False,
        device=config.train_device,
    )

    loss_module = ClipPPOLoss(
        actor_network=model[0],
        critic_network=model[1],
        clip_epsilon=config.clip_eps,
        entropy_bonus=True,
        entropy_coef=config.ent_coef,
        critic_coef=config.val_coef,
        loss_critic_type="l2",
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

    for i, tensordict_data in enumerate(collector):
        if (i + 1) % 50 == 0:
            if wandb.run.dir is None:
                path = "."
            else:
                path = wandb.run.dir
            torch.save(
                model.state_dict(),
                os.path.join(wandb.run.dir, model_name + f"_{i + 1}.pth"),
            )
        if i >= config.num_collections:
            break
        print(f"Collection: {i}")
        tensordict_data = tensordict_data.to(config.train_device, non_blocking=True)

        with torch.no_grad():
            advantage_module(tensordict_data)

        non_zero_rewards = tensordict_data["next", "reward"]
        improvements = tensordict_data["next", "observation", "aux", "improvement"]
        mask = improvements > -1.5
        filtered_improvements = improvements[mask]
        if filtered_improvements.numel() > 0:
            avg_improvement = filtered_improvements.mean()
        if len(non_zero_rewards) > 0:
            avg_non_zero_reward = non_zero_rewards.mean().item()
            print(
                f"Average reward: {avg_non_zero_reward}, Average Improvement: {avg_improvement}"
            )

        replay_buffer.extend(tensordict_data.reshape(-1))

        for j in range(config.num_epochs_per_collection):
            n_batches = config.states_per_collection // config.minibatch_size

            for k in range(n_batches):
                subdata = replay_buffer.sample(config.minibatch_size)
                subdata.to(config.train_device)

                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                optimizer.zero_grad()
                loss_value.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=config.max_grad_norm
                )
                optimizer.step()

        # Update the policy
        collector.policy.load_state_dict(loss_module.actor_network.state_dict())
        collector.update_policy_weights_(TensorDict.from_module(collector.policy))
        wandb.log(
            {
                "Average Return": avg_non_zero_reward,
                "Average Improvement": avg_improvement,
                "loss_objective": loss_vals["loss_objective"].item(),
                "loss_critic": loss_vals["loss_critic"].item(),
                "loss_entropy": loss_vals["loss_entropy"].item(),
                "loss_total": loss_value.item(),
                "grad_norm": grad_norm,
            },
        )

    collector.shutdown()


class ActorWrapper(HeteroDataWrapper):
    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)
        data = self._convert_to_heterodata(obs, is_batch, actions=actions)
        data = data.to(self.device)
        # Compute task embeddings from the hetero-GAT network
        task_embeddings = self.network(data)
        # Extract candidate embedding based on batch presence
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]
        # Return the candidate embedding wrapped in a dict with the expected key 'embed'
        return {"embed": candidate_embedding}


class CriticEmbedWrapper(HeteroDataWrapper):
    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)
        data = self._convert_to_heterodata(obs, is_batch, actions=actions)
        data = data.to(self.device)
        # Compute task embeddings from the hetero-GAT network
        task_embeddings = self.network(data)

        # Extract candidate embedding based on batch presence
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None

        # Aggregate node embeddings to get one embedding per graph/sample
        # This ensures the resulting embedding tensor has shape [batch_size, feature_dim]
        pooled_embeddings = global_mean_pool(task_embeddings, task_batch)
        # Return the pooled embedding wrapped in a dict with the expected key 'embed'
        return {"embed": pooled_embeddings}


class CriticHeadWrapper(nn.Module):
    def __init__(self, critic: nn.Module, device: str):
        super().__init__()
        self.critic = critic
        self.device = device

    def forward(self, obs: TensorDict):
        v = self.critic(obs)
        # v = global_mean_pool(v, obs["task_batch"])
        return {"state_value": v}


def compute_gae(tensordict_data, critic, gamma=0.99, lam=0.95):
    with torch.no_grad():
        critic(tensordict_data)
        critic(tensordict_data["next"])

    value = tensordict_data["state_value"]
    next_value = tensordict_data["next", "state_value"]
    reward = tensordict_data["next", "reward"]
    done = tensordict_data["next", "done"]

    advantage = torch.zeros_like(value)
    gae = 0.0
    T = reward.shape[0]
    for t in reversed(range(T)):
        if done[t]:
            c = 0
        else:
            c = 1
        delta = reward[t] + gamma * next_value[t] * c - value[t]
        gae = delta + gamma * lam * c * gae
        advantage[t] = gae

    value_target = advantage + value
    tensordict_data["advantage"] = advantage
    tensordict_data["value_target"] = value_target
    return tensordict_data


def run_recurrent_ppo_torchrl_single(
    feature_config,
    layer_config,
    n_devices,
    make_env,
    config,
    model_name: str = "model",
):
    env = make_env()

    # Actor modules ...
    module_action_gat = TensorDictModule(
        ActorWrapper(
            HeteroGAT1Layer(feature_config, layer_config), device=config.train_device
        ),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    module_action_lstm = LSTMModule(
        input_size=module_action_gat(env.reset())["embed"].shape[-1],
        hidden_size=128,
        device=config.train_device,
        in_key="embed",
        out_key="embed",
    )
    module_action_fc = TensorDictModule(
        OutputHead(128, layer_config.hidden_channels, n_devices),
        in_keys=["embed"],
        out_keys=["logits"],
    )
    td_module_action = ProbabilisticActor(
        module=Sequential(module_action_gat, module_action_lstm, module_action_fc),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        cache_dist=False,
        return_log_prob=True,
    )

    module_critic_gat = TensorDictModule(
        CriticEmbedWrapper(
            HeteroGAT1Layer(feature_config, layer_config), device=config.train_device
        ),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    module_critic_lstm = LSTMModule(
        input_size=module_critic_gat(env.reset())["embed"].shape[-1],
        hidden_size=128,
        device=config.train_device,
        in_key="embed",
        out_key="embed",
    )
    module_critic_fc = TensorDictModule(
        CriticHeadWrapper(
            OutputHead(128, layer_config.hidden_channels, 1), device=config.train_device
        ),
        in_keys=["embed"],
        out_keys=["state_value"],
    )

    def transformed_env():
        env = make_env()
        env = TransformedEnv(env, Compose(StepCounter(), TrajCounter(), InitTracker()))
        env.append_transform(module_critic_lstm.make_tensordict_primer())
        return env

    td_critic_module = Sequential(
        module_critic_gat, module_critic_lstm, module_critic_fc
    )

    td_module_action = td_module_action.to(config.train_device)
    td_critic_module = td_critic_module.to(config.train_device)

    train_actor_network = copy.deepcopy(td_module_action).to(config.train_device)
    train_critic_network = copy.deepcopy(td_critic_module).to(config.train_device)
    model = torch.nn.ModuleList([train_actor_network, train_critic_network])

    collector = SyncDataCollector(
        transformed_env(),
        td_module_action,
        frames_per_batch=config.states_per_collection,
        total_frames=config.states_per_collection * config.num_collections,
        split_trajs=True,
        device=config.train_device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=config.states_per_collection, device=config.train_device
        ),
        sampler=SamplerWithoutReplacement(),
        pin_memory=torch.cuda.is_available(),
    )

    loss_module = ClipPPOLoss(
        actor_network=model[0],
        critic_network=model[1],
        clip_epsilon=config.clip_eps,
        entropy_bonus=bool(config.ent_coef),
        entropy_coef=config.ent_coef,
        critic_coef=config.val_coef,
        loss_critic_type="l2",
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

    for i, tensordict_data in enumerate(collector):
        if (i + 1) % 50 == 0:
            if wandb.run.dir is None:
                path = "."
            else:
                path = wandb.run.dir
            torch.save(
                model.state_dict(),
                os.path.join(wandb.run.dir, model_name + f"_{i + 1}.pth"),
            )
        if i >= config.num_collections:
            break
        print(f"Collection: {i}")
        tensordict_data = tensordict_data.to(config.train_device, non_blocking=True)

        with torch.no_grad():
            tensordict_data = tensordict_data.reshape(-1)
            compute_gae(
                tensordict_data,
                train_critic_network,
                gamma=config.gae_gamma,
                lam=config.gae_lmbda,
            )

        non_zero_rewards = tensordict_data["next", "reward"]
        improvements = tensordict_data["next", "observation", "aux", "improvement"]
        mask = improvements > -1.5
        filtered_improvements = improvements[mask]
        if filtered_improvements.numel() > 0:
            avg_improvement = filtered_improvements.mean()
        if len(non_zero_rewards) > 0:
            avg_non_zero_reward = non_zero_rewards.mean().item()
            print(
                f"Average reward: {avg_non_zero_reward}, "
                f"Average Improvement: {avg_improvement}"
            )

        replay_buffer.extend(tensordict_data)

        for j in range(config.num_epochs_per_collection):
            n_batches = config.states_per_collection // config.minibatch_size
            for k in range(n_batches):
                subdata = replay_buffer.sample(config.minibatch_size)
                subdata.to(config.train_device)

                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                optimizer.zero_grad()
                loss_value.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=config.max_grad_norm
                )
                optimizer.step()

        collector.policy.load_state_dict(loss_module.actor_network.state_dict())
        collector.update_policy_weights_(TensorDict.from_module(collector.policy))

        wandb.log(
            {
                "Average Return": avg_non_zero_reward,
                "Average Improvement": avg_improvement,
                "loss_objective": loss_vals["loss_objective"].item(),
                "loss_critic": loss_vals["loss_critic"].item(),
                "loss_entropy": loss_vals["loss_entropy"].item(),
                "loss_total": loss_value.item(),
                "grad_norm": grad_norm,
            }
        )

    collector.shutdown()


# def run_recurrnt_ppo_torchrl(
#     feature_config: FeatureDimConfig,
#     layer_config: LayerConfig,
#     n_devices: int,
#     make_env: Callable[[], EnvBase],
#     config: PPOConfig,
#     model_name: str = "model",
# ):
#     gat_output_dim = layer_config.hidden_channels * 3 + feature_config.task_feature_dim
#     env = make_env()

#     module_action_gat = TensorDictModule(
#         ActorWrapper(
#             HeteroGAT1Layer(feature_config, layer_config), device=config.train_device
#         ),
#         in_keys=["observation"],
#         out_keys=["embed"],
#     )
#     module_action_lstm = LSTMModule(
#         input_size=module_action_gat(env.reset())["embed"].shape[-1],
#         hidden_size=128,
#         device=config.train_device,
#         in_key="embed",
#         out_key="embed",
#     )
#     module_action_fc = TensorDictModule(
#         OutputHead(128, layer_config.hidden_channels, n_devices),
#         in_keys=["embed"],
#         out_keys=["logits"],
#     )
#     td_module_action = ProbabilisticActor(
#         module=Sequential(
#             module_action_gat,
#             module_action_lstm,
#             module_action_fc,
#         ),
#         in_keys=["logits"],
#         out_keys=["action"],
#         distribution_class=torch.distributions.Categorical,
#         cache_dist=False,
#         return_log_prob=True,
#     )


#     module_critic_gat = TensorDictModule(
#         CriticEmbedWrapper(
#             HeteroGAT1Layer(feature_config, layer_config), device=config.train_device
#         ),
#         in_keys=["observation"],
#         out_keys=["embed"],
#     )
#     module_critic_lstm = LSTMModule(
#         input_size=module_critic_gat(env.reset())["embed"].shape[-1],
#         hidden_size=128,
#         device=config.train_device,
#         in_key="embed",
#         out_key="embed",
#     )

#     module_critic_fc = TensorDictModule(
#         CriticHeadWrapper(
#             OutputHead(128, layer_config.hidden_channels, 1),
#             device=config.train_device,
#         ),
#         in_keys=["embed"],
#         out_keys=["state_value"],
#     )

#     def transformed_env():
#         env = make_env()
#         env = TransformedEnv(env, Compose(StepCounter(), TrajCounter(), InitTracker()))
#         env.append_transform(module_critic_lstm.make_tensordict_primer())
#         return env

#     td_critic_module = Sequential(module_critic_gat, module_critic_lstm, module_critic_fc)

#     td_module_action = td_module_action.to(config.train_device)
#     td_critic_module = td_critic_module.to(config.train_device)
#     train_actor_network = copy.deepcopy(td_module_action).to(config.train_device)
#     train_critic_network = copy.deepcopy(td_critic_module).to(config.train_device)
#     model = torch.nn.ModuleList([train_actor_network, train_critic_network])

#     collector = MultiSyncDataCollector(
#         [transformed_env for _ in range(config.workers)],
#         td_module_action,
#         frames_per_batch=config.states_per_collection,
#         reset_at_each_iter=True,
#         cat_results=0,
#         device=config.train_device,
#         env_device="cpu",
#     )
#     collector = SyncDataCollector(
#         transformed_env(),
#         td_module_action,
#         frames_per_batch=config.states_per_collection,
#         total_frames=config.states_per_collection,
#         split_trajs=False,
#         device=config.train_device,
#     )
#     # out_seed = collector.set_seed(config.seed)
#     # print(f"Seed: {out_seed}")

#     replay_buffer = ReplayBuffer(
#         storage=LazyTensorStorage(
#             max_size=config.states_per_collection, device=config.train_device
#         ),
#         sampler=SamplerWithoutReplacement(),
#         pin_memory=torch.cuda.is_available(),
#     )

#     advantage_module = GAE(
#         gamma=config.gae_gamma,
#         lmbda=config.gae_lmbda,
#         value_network=model[1],
#         average_gae=False,
#         device=config.train_device,
#     )

#     loss_module = ClipPPOLoss(
#         actor_network=model[0],
#         critic_network=model[1],
#         clip_epsilon=config.clip_eps,
#         entropy_bonus=bool(config.ent_coef),
#         entropy_coef=config.ent_coef,
#         critic_coef=config.val_coef,
#         loss_critic_type="l2",
#     )

#     optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

#     for i, tensordict_data in enumerate(collector):
#         if (i + 1) % 50 == 0:
#             if wandb.run.dir is None:
#                 path = "."
#             else:
#                 path = wandb.run.dir
#             torch.save(
#                 model.state_dict(),
#                 os.path.join(wandb.run.dir, model_name + f"_{i + 1}.pth"),
#             )
#         if i >= config.num_collections:
#             break
#         print(f"Collection: {i}")
#         tensordict_data = tensordict_data.to(config.train_device, non_blocking=True)

#         with torch.no_grad():
#             advantage_module(tensordict_data)

#         non_zero_rewards = tensordict_data["next", "reward"]
#         improvements = tensordict_data["next", "observation", "aux", "improvement"]
#         mask = improvements > -1.5
#         filtered_improvements = improvements[mask]
#         if filtered_improvements.numel() > 0:
#             avg_improvement = filtered_improvements.mean()
#         if len(non_zero_rewards) > 0:
#             avg_non_zero_reward = non_zero_rewards.mean().item()
#             print(
#                 f"Average reward: {avg_non_zero_reward}, Average Improvement: {avg_improvement}"
#             )

#         replay_buffer.extend(tensordict_data.unsqueeze(0).to_tensordict())

#         for j in range(config.num_epochs_per_collection):
#             n_batches = config.states_per_collection // 1

#             for k in range(n_batches):
#                 subdata = replay_buffer.sample(1)
#                 print(subdata.shape)
#                 subdata.to(config.train_device)

#                 loss_vals = loss_module(subdata)
#                 loss_value = (
#                     loss_vals["loss_objective"]
#                     + loss_vals["loss_critic"]
#                     + loss_vals["loss_entropy"]
#                 )

#                 optimizer.zero_grad()
#                 loss_value.backward()
#                 grad_norm = torch.nn.utils.clip_grad_norm_(
#                     loss_module.parameters(), max_norm=config.max_grad_norm
#                 )
#                 optimizer.step()

#         # Update the policy
#         collector.policy.load_state_dict(loss_module.actor_network.state_dict())
#         collector.update_policy_weights_(TensorDict.from_module(collector.policy))
#         wandb.log(
#             {
#                 "Average Return": avg_non_zero_reward,
#                 "Average Improvement": avg_improvement,
#                 "loss_objective": loss_vals["loss_objective"].item(),
#                 "loss_critic": loss_vals["loss_critic"].item(),
#                 "loss_entropy": loss_vals["loss_entropy"].item(),
#                 "loss_total": loss_value.item(),
#                 "grad_norm": grad_norm,
#             },
#         )

#     collector.shutdown()
