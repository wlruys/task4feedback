from .models import *
from .util import *
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any, Tuple
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule, GRUModule
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import StepCounter, TrajCounter, Compose, InitTracker
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch_geometric.loader import DataLoader
import copy
from tensordict import TensorDictBase, TensorDict
import wandb
import os
import numpy as np
import torch
from torchrl.envs import set_exploration_type, ExplorationType
from task4feedback.graphs.mesh.plot import *
from tensordict.nn import TensorDictSequential as Sequential
from .ppo import evaluate_policy


@dataclass
class RNNPPOConfig:
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
    train_device: str = "cpu"
    gae_gamma: float = 1
    gae_lmbda: float = 0.1
    rnn_model: str = "LSTM"  # "LSTM" or "GRU"
    hidden_size: int = 128
    eval_interval: int = 50
    eval_episodes: int = 1


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


class ActorWrapper(HeteroDataWrapper):
    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)
        data, task_count, data_count = self._convert_to_heterodata(
            obs, is_batch, actions=actions
        )
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
        data, task_count, data_count = self._convert_to_heterodata(
            obs, is_batch, actions=actions
        )
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


class HeteroDataWrapperNoDevice(HeteroDataWrapper):
    def __init__(
        self,
        network: nn.Module,
        device: Optional[str] = "cpu",
    ):
        super().__init__(network, device)

    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)
        data, task_count, data_count = self._convert_to_heterodata(
            obs, is_batch, actions=actions
        )
        # print(data)
        data = data.to(self.device)
        # Compute task embeddings from the hetero-GAT network
        candidate_embedding = self.network(data, (task_count, data_count))
        return {"embed": candidate_embedding}


class _DataTaskGAT(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(_DataTaskGAT, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.data_task_gat = DataTaskGAT(feature_config, layer_config)
        data_task_dim = self.data_task_gat.output_dim

        self.task_task_gat = TaskTaskGAT2Layer(
            data_task_dim, feature_config, layer_config
        )

        self.unrolled_device_layer = UnRolledDeviceLayer(
            feature_config, layer_config, n_devices
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        task_embeddings = self.data_task_gat(data)
        task_embeddings = self.task_task_gat(task_embeddings, data)

        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        device_batch = data["devices"].batch if isinstance(data, Batch) else None

        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]

        device_features = self.unrolled_device_layer(data)
        device_features = device_features.squeeze(0)

        counts_0 = torch.clip(counts[0], min=1)
        global_embedding = global_add_pool(task_embeddings, task_batch)
        global_embedding = torch.div(global_embedding, counts_0)

        global_embedding = global_embedding.squeeze(0)

        candidate_embedding = torch.cat(
            [candidate_embedding, global_embedding, device_features], dim=-1
        )

        return candidate_embedding


def run_rnn_ppo_torchrl(
    feature_config: FeatureDimConfig,
    layer_config: LayerConfig,
    make_env,
    config: RNNPPOConfig,
    model_name: str = "model",
    model_path: str = None,
    eval_env_fn: Optional[Callable[[], EnvBase]] = None,
):

    env = make_env()
    if config.rnn_model == "LSTM":
        RNNModule = LSTMModule
    elif config.rnn_model == "GRU":
        RNNModule = GRUModule
    else:
        raise ValueError(f"Invalid RNN model: {config.rnn_model}")
    # Actor modules ...
    module_action_gat = TensorDictModule(
        ActorWrapper(
            HeteroGAT1Layer(feature_config, layer_config), device=config.train_device
        ),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    module_action_lstm = RNNModule(
        input_size=module_action_gat(env.reset())["embed"].shape[-1],
        hidden_size=config.hidden_size,
        device=config.train_device,
        in_key="embed",
        out_key="embed",
    )
    module_action_fc = TensorDictModule(
        OldOutputHead(config.hidden_size, layer_config.hidden_channels, 4),
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
    module_critic_lstm = RNNModule(
        input_size=module_critic_gat(env.reset())["embed"].shape[-1],
        hidden_size=config.hidden_size,
        device=config.train_device,
        in_key="embed",
        out_key="embed",
    )
    module_critic_fc = TensorDictModule(
        CriticHeadWrapper(
            OldOutputHead(config.hidden_size, layer_config.hidden_channels, 1),
            device=config.train_device,
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

    collector = MultiSyncDataCollector(
        [transformed_env for _ in range(config.workers)],
        model[0],
        frames_per_batch=config.states_per_collection,
        total_frames=config.states_per_collection * config.num_collections,
        split_trajs=True,
        reset_at_each_iter=True,
        # cat_results=0,
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
        batch_size=config.minibatch_size,
        prefetch=config.states_per_collection // config.minibatch_size,
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

    if eval_env_fn is None:
        eval_env_fn = transformed_env
    if config.eval_interval > 0:
        eval_metrics = evaluate_policy(
            policy=model[0],
            eval_env_fn=eval_env_fn,
            num_episodes=config.eval_episodes,
            step=0,
        )
        eval_metrics["eval/step"] = 0
        wandb.log(eval_metrics)

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

        # Run evaluation at specified intervals
        if config.eval_interval > 0 and (i + 1) % config.eval_interval == 0:
            eval_metrics = evaluate_policy(
                policy=model[0],
                eval_env_fn=eval_env_fn,
                num_episodes=config.eval_episodes,
                step=i + 1,
            )
            eval_metrics["eval/step"] = i + 1
            wandb.log(eval_metrics)

        if i >= config.num_collections:
            break
        print(f"Collection: {i}")
        tensordict_data = tensordict_data.to(config.train_device, non_blocking=True)

        with torch.no_grad():
            tensordict_data = tensordict_data.reshape(-1)
            compute_gae(
                tensordict_data,
                model[1],
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
                "collect_loss/step": i,
                "collect_loss/mean_nonzero_reward": avg_non_zero_reward,
                "collect_loss/loss_objective": loss_vals["loss_objective"].item(),
                "collect_loss/average_improvement": avg_improvement.item(),
                "collect_loss/std_improvement": filtered_improvements.std().item(),
                "collect_loss/std_return": tensordict_data["value_target"].std().item(),
                "collect_loss/mean_return": tensordict_data["value_target"]
                .mean()
                .item(),
                "collect_loss/loss_critic": loss_vals["loss_critic"].item(),
                "collect_loss/loss_entropy": loss_vals["loss_entropy"].item(),
                "collect_loss/loss_total": loss_value.item(),
                "collect_loss/grad_norm": grad_norm,
                "collect_loss/advantage_mean": tensordict_data["advantage"]
                .mean()
                .item(),
                "collect_loss/advantage_std": tensordict_data["advantage"].std().item(),
            },
        )

    collector.shutdown()


def run_rnn_ppo_torchrl_noDevice(
    feature_config: FeatureDimConfig,
    layer_config: LayerConfig,
    make_env,
    config: RNNPPOConfig,
    model_name: str = "model",
    model_path: str = None,
    eval_env_fn: Optional[Callable[[], EnvBase]] = None,
):
    wandb.define_metric("collect_loss/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("collect_loss/*", step_metric="collect_loss/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

    env = make_env()
    if config.rnn_model == "LSTM":
        RNNModule = LSTMModule
    elif config.rnn_model == "GRU":
        RNNModule = GRUModule
    else:
        raise ValueError(f"Invalid RNN model: {config.rnn_model}")
    # Actor modules ...
    module_action_gat = TensorDictModule(
        HeteroDataWrapperNoDevice(
            _DataTaskGAT(feature_config, layer_config, 5),
            device=config.train_device,
        ),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    module_action_lstm = RNNModule(
        input_size=module_action_gat(env.reset())["embed"].shape[-1],
        hidden_size=config.hidden_size,
        device=config.train_device,
        in_key="embed",
        out_key="embed",
    )
    module_action_fc = TensorDictModule(
        OldOutputHead(config.hidden_size, layer_config.hidden_channels, 4, logits=True),
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
        HeteroDataWrapperNoDevice(
            _DataTaskGAT(feature_config, layer_config, 5), device=config.train_device
        ),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    module_critic_lstm = RNNModule(
        input_size=module_critic_gat(env.reset())["embed"].shape[-1],
        hidden_size=config.hidden_size,
        device=config.train_device,
        in_key="embed",
        out_key="embed",
    )
    module_critic_fc = TensorDictModule(
        OldOutputHead(
            config.hidden_size, layer_config.hidden_channels, 1, logits=False
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

    collector = MultiSyncDataCollector(
        [transformed_env for _ in range(config.workers)],
        model[0],
        frames_per_batch=config.states_per_collection,
        total_frames=config.states_per_collection * config.num_collections,
        split_trajs=True,
        reset_at_each_iter=True,
        # cat_results=0,
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
        batch_size=config.minibatch_size,
        prefetch=config.states_per_collection // config.minibatch_size,
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

    if eval_env_fn is None:
        eval_env_fn = transformed_env
    if config.eval_interval > 0:
        eval_metrics = evaluate_policy(
            policy=model[0],
            eval_env_fn=eval_env_fn,
            num_episodes=config.eval_episodes,
            step=0,
        )
        eval_metrics["eval/step"] = 0
        wandb.log(eval_metrics)

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

        # Run evaluation at specified intervals
        if config.eval_interval > 0 and (i + 1) % config.eval_interval == 0:
            eval_metrics = evaluate_policy(
                policy=model[0],
                eval_env_fn=eval_env_fn,
                num_episodes=config.eval_episodes,
                step=i + 1,
            )
            eval_metrics["eval/step"] = i + 1
            wandb.log(eval_metrics)

        if i >= config.num_collections:
            break
        print(f"Collection: {i}")
        tensordict_data = tensordict_data.to(config.train_device, non_blocking=True)

        with torch.no_grad():
            tensordict_data = tensordict_data.reshape(-1)
            compute_gae(
                tensordict_data,
                model[1],
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
                "collect_loss/step": i,
                "collect_loss/mean_nonzero_reward": avg_non_zero_reward,
                "collect_loss/loss_objective": loss_vals["loss_objective"].item(),
                "collect_loss/average_improvement": avg_improvement.item(),
                "collect_loss/std_improvement": filtered_improvements.std().item(),
                "collect_loss/std_return": tensordict_data["value_target"].std().item(),
                "collect_loss/mean_return": tensordict_data["value_target"]
                .mean()
                .item(),
                "collect_loss/loss_critic": loss_vals["loss_critic"].item(),
                "collect_loss/loss_entropy": loss_vals["loss_entropy"].item(),
                "collect_loss/loss_total": loss_value.item(),
                "collect_loss/grad_norm": grad_norm,
                "collect_loss/advantage_mean": tensordict_data["advantage"]
                .mean()
                .item(),
                "collect_loss/advantage_std": tensordict_data["advantage"].std().item(),
            },
        )

    collector.shutdown()
