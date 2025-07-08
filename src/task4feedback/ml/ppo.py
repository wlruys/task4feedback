from .models import *
from .util import *
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any, Tuple
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage, TensorStorage
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
import tensordict


@dataclass
class PPOConfig:
    states_per_collection: int = 1920
    minibatch_size: int = 250
    num_epochs_per_collection: int = 4
    num_collections: int = 5000
    workers: int = 1
    seed: int = 0
    lr: float = 2.5e-4
    clip_eps: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    val_coef: float = 0.5
    max_grad_norm: float = 0.5
    threads_per_worker: int = 1
    train_device: str = "cpu"
    collect_device: str = "cpu"
    update_device: str = "cuda:0"
    gae_gamma: float = 1
    gae_lmbda: float = 0.99
    normalize_advantage: bool = True
    value_norm: str = "smooth_l1"
    eval_interval: int = 10  # Evaluate every N collections
    eval_episodes: int = 1  # Number of episodes to evaluate


def log_parameter_and_gradient_norms(model):
    """Log parameter and gradient norms to wandb"""
    param_norms = {}
    grad_norms = {}

    # Log overall model norm
    total_param_norm = 0.0
    total_grad_norm = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Calculate parameter norm
            param_norm = param.detach().norm().item()
            param_norms[f"param_norm/{name}"] = param_norm
            total_param_norm += param_norm**2

            # Calculate gradient norm if gradient exists
            if param.grad is not None:
                grad_norm = param.grad.detach().norm().item()
                grad_norms[f"grad_norm/{name}"] = grad_norm
                total_grad_norm += grad_norm**2

    # Calculate total norms
    total_param_norm = total_param_norm**0.5
    total_grad_norm = total_grad_norm**0.5

    return {
        **param_norms,
        **grad_norms,
        "param_norm/total": total_param_norm,
        "grad_norm/total": total_grad_norm,
    }


def evaluate_policy(
    policy,
    eval_env_fn: Callable,
    max_steps: int = 10000,
    num_episodes: int = 1,
    step=0,
) -> Dict[str, float]:
    episode_rewards = []
    completion_times = []
    episode_returns = []
    std_rewards = []

    for i in range(num_episodes):
        env = eval_env_fn()
        with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
            env._reset()
            tensordict = env.rollout(
                max_steps=max_steps,
                policy=policy,
            )

        if "next" in tensordict and "reward" in tensordict["next"]:
            rewards = tensordict["next", "reward"]
            avg_reward = rewards.mean().item()
            std_reward = rewards.std().item()
            returns = tensordict["next", "reward"].sum().item()
        else:
            returns = 0.0
            avg_non_zero_reward = 0.0
            std_rewards = 0.0

        episode_returns.append(returns)
        episode_rewards.append(avg_reward)
        std_rewards.append(std_reward)

        # Extract completion time if available
        if hasattr(env, "simulator") and hasattr(env.simulator, "time"):
            completion_time = env.simulator.time
            completion_times.append(completion_time)

            if i == 0 and completion_time > 0:
                # Animate the first environment
                max_frames = 400
                time_interval = int(completion_time / max_frames)

                title = f"network_eval_{step}_{i}"
                print(title)
                animate_mesh_graph(
                    env,
                    time_interval=time_interval,
                    show=False,
                    title=title,
                    figsize=(2, 2),
                    dpi=20,
                    bitrate=50,
                )

                if wandb.run.dir is None:
                    path = "."
                else:
                    path = wandb.run.dir

                video_path = os.path.join(path, title + ".mp4")

                wandb.log(
                    {
                        "eval/animation": wandb.Video(
                            video_path,
                            caption=title,
                        )
                    }
                )

    # Create metrics dictionary
    metrics = {
        "eval/mean_return": sum(episode_rewards) / max(len(episode_rewards), 1),
        # "eval/std_return": np.std(episode_rewards) if len(episode_rewards) > 1 else 0,
        "eval/mean_reward": sum(episode_rewards) / max(len(episode_rewards), 1),
        # "eval/std_mean_reward": np.std(episode_rewards)
        # if len(episode_rewards) > 1
        # else 0,
        # "eval/std_std_reward": np.std(std_rewards) if len(std_rewards) > 1 else 0,
        "eval/mean_std_reward": sum(std_rewards) / max(len(std_rewards), 1),
    }

    # Add completion time metrics if available
    if completion_times:
        metrics["eval/mean_completion_time"] = sum(completion_times) / len(
            completion_times
        )
        # metrics["eval/min_completion_time"] = min(completion_times)
        # metrics["eval/max_completion_time"] = max(completion_times)

    return metrics


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


def run_ppo_torchrl(
    actor_critic_base: nn.Module,
    make_env: Callable[[], EnvBase],
    config: PPOConfig,
    model_name: str = "model",
    model_path: str = None,
    eval_env_fn: Optional[Callable[[], EnvBase]] = None,
    do_rollout: bool = False,
):
    wandb.define_metric("batch_loss/step")
    wandb.define_metric("collect_loss/step")
    wandb.define_metric("batch_loss/*", step_metric="batch_loss/step")
    wandb.define_metric("grad_norm/*", step_metric="batch_loss/step")
    wandb.define_metric("param_norm/*", step_metric="batch_loss/step")
    wandb.define_metric("collect_loss/*", step_metric="collect_loss/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    env = make_env()
    num_candidates = env.simulator_factory.graph_spec.max_candidates
    print(f"{num_candidates} candidates per action")
    _actor_td = actor_critic_base.actor
    _critic_td = actor_critic_base.critic
    print(actor_critic_base)
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

    # Create evaluation environment if not provided
    if eval_env_fn is None:
        eval_env_fn = make_env

    # if model_path:
    #     ckpts = torch.load(model_path)  # this is a list or tuple of two dicts
    #     model[0].load_state_dict(ckpts[0])  # only load the first one
    #     print("Loaded model[0] from path:", model_path)
    #     # model.load_state_dict(torch.load(model_path))
    #     # print("Loaded model from path:", model_path)
    if model_path:
        full_sd = torch.load(model_path)
        sd0 = {
            k.replace("module0.", ""): v
            for k, v in full_sd.items()
            if k.startswith("module0.")
        }
        model[0].load_state_dict(sd0, strict=False)
        print("Loaded only model[0] parameters")
    model.eval()
    if do_rollout:

        def rollout_env():
            env = make_env()
            env.set_policy(model[0])
            return env

        _make_env = rollout_env
    else:
        _make_env = make_env

    collector = MultiSyncDataCollector(
        [_make_env for _ in range(config.workers)],
        model[0],
        frames_per_batch=config.states_per_collection,
        reset_at_each_iter=True,
        cat_results=0,
        device=config.train_device,
        env_device="cpu",
    )
    out_seed = collector.set_seed(config.seed)
    print("Collector seed:", out_seed)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=config.states_per_collection, device=config.train_device
        ),
        sampler=SamplerWithoutReplacement(),
        pin_memory=torch.cuda.is_available(),
        prefetch=4,
        batch_size=config.minibatch_size,
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
        loss_critic_type=config.value_norm,
        clip_value=config.clip_vloss,
        normalize_advantage=config.normalize_advantage,
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

    # Run initial evaluation
    if config.eval_interval > 0:
        eval_metrics = evaluate_policy(
            policy=model[0],
            eval_env_fn=eval_env_fn,
            num_episodes=config.eval_episodes,
            step=0,
        )
        eval_metrics["eval/step"] = 0
        wandb.log(eval_metrics)
    prev_max_vsoptimal = 1.0
    for i, tensordict_data in enumerate(collector):
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
            advantage_module(tensordict_data)
            if num_candidates > 1:
                tensordict_data["advantage"] = tensordict_data["advantage"].expand(
                    -1, num_candidates
                )
                tensordict_data["advantage"] = tensordict_data["advantage"].unsqueeze(
                    -1
                )
            non_zero_rewards = tensordict_data["next", "reward"]
            improvements = tensordict_data["next", "observation", "aux", "improvement"]
            vsoptimals = tensordict_data["next", "observation", "aux", "vsoptimal"]
            mask = improvements > -1.5
            filtered_improvements = improvements[mask]
            if filtered_improvements.numel() > 0:
                avg_improvement = filtered_improvements.mean()
            mask = vsoptimals > -1.5
            filtered_vsoptimals = vsoptimals[mask]
            if filtered_vsoptimals.numel() > 0:
                avg_vsoptimal = filtered_vsoptimals.mean()

            if len(non_zero_rewards) > 0:
                avg_non_zero_reward = non_zero_rewards.mean().item()
                std_rewards = non_zero_rewards.std().item()
                print(f"Average reward: {avg_non_zero_reward}")
                print(f"Average improvement: {avg_improvement}")
                print(f"Average vs Optimal: {avg_vsoptimal}")

        replay_buffer.extend(tensordict_data.reshape(-1))

        # Training loop
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

                # Log pre-clipping gradient norms
                pre_clip_norms = log_parameter_and_gradient_norms(loss_module)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=config.max_grad_norm
                )

                # Log post-clipping gradient norms
                post_clip_norms = log_parameter_and_gradient_norms(loss_module)
                post_clip_norms = {
                    f"post_clip_{k}": v
                    for k, v in post_clip_norms.items()
                    if "grad_norm" in k
                }

                optimizer.step()

                # Log norms for this batch
                step = (
                    i * config.num_epochs_per_collection * n_batches + j * n_batches + k
                )
                wandb.log(
                    {
                        **pre_clip_norms,
                        **post_clip_norms,
                        "batch_loss/step": step,
                        "batch_loss/objective": loss_vals["loss_objective"].item(),
                        "batch_loss/critic": loss_vals["loss_critic"].item(),
                        "batch_loss/entropy": loss_vals["entropy"].item(),
                        "batch_loss/total": loss_value.item(),
                        "batch_loss/kl_approx": loss_vals["kl_approx"].item(),
                        "batch_loss/clip_fraction": loss_vals["clip_fraction"].item(),
                        # "batch_loss/value_clip_fraction": loss_vals[
                        #    "value_clip_fraction"
                        # ].item(),
                        "batch_loss/ESS": loss_vals["ESS"].item(),
                    },
                )

        # Update the policy
        collector.policy.load_state_dict(loss_module.actor_network.state_dict())
        collector.update_policy_weights_(TensorDict.from_module(collector.policy))
        wandb.log(
            {
                "collect_loss/step": i,
                "collect_loss/mean_nonzero_reward": avg_non_zero_reward,
                "collect_loss/std_nonzero_reward": std_rewards,
                "collect_loss/loss_objective": loss_vals["loss_objective"].item(),
                "collect_loss/average_improvement": avg_improvement.item(),
                "collect_loss/average_vsoptimal": avg_vsoptimal.item(),
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
        replay_buffer.empty()

        if (i + 1) % 100 == 0:
            if wandb.run.dir is None:
                path = "."
            else:
                path = wandb.run.dir
            torch.save(
                model.state_dict(),
                os.path.join(path, model_name + f"_{i+1}.pth"),
            )

        if avg_vsoptimal >= prev_max_vsoptimal + 0.01:
            if wandb.run.dir is None:
                path = "."
            else:
                path = wandb.run.dir
            torch.save(
                model.state_dict(),
                os.path.join(path, "max_" + model_name + f"_{avg_vsoptimal}.pth"),
            )
            prev_max_vsoptimal = avg_vsoptimal

    if config.eval_interval > 0 and i % config.eval_interval == 0:
        eval_metrics = evaluate_policy(
            policy=td_module_action,
            eval_env_fn=eval_env_fn,
            num_episodes=config.eval_episodes,
            step=config.num_collections,
        )
        eval_metrics["eval/step"] = config.num_collections
        wandb.log(eval_metrics)

    # save final network
    if wandb.run.dir is None:
        path = "."
    else:
        path = wandb.run.dir
    torch.save(
        model.state_dict(),
        os.path.join(path, model_name + f"_{config.num_collections}.pth"),
    )
    # save final optimizer state
    torch.save(
        optimizer.state_dict(),
        os.path.join(
            path, "optimizer_" + model_name + f"_{config.num_collections}.pth"
        ),
    )

    collector.shutdown()
    wandb.finish()


class ConcatCNNRNN(nn.Module):
    def forward(self, cnn_embed, rnn_embed):
        # assumes last dim is feature dim
        return torch.cat([cnn_embed, rnn_embed], dim=-1)


class ConcatObservation(nn.Module):
    def __init__(self, concat_progress: bool = False):
        super().__init__()
        self.concat_progress = concat_progress

    def forward(self, cnn_embed, observation):
        devices = observation["devices"]
        devices = devices.flatten(start_dim=-2, end_dim=-1)
        if self.concat_progress:
            progress = observation["aux", "progress"].unsqueeze(-1)
            return torch.cat([cnn_embed, devices, progress], dim=-1)
        return torch.cat([cnn_embed, devices], dim=-1)


class Concat3Embeds(nn.Module):
    def __init__(self, add_progress: bool = True, add_device: bool = False):
        super().__init__()
        self.add_progress = add_progress
        self.add_device = add_device

    def forward(self, cnn_embed, rnn_embed, observation):
        parts = [cnn_embed, rnn_embed]

        if self.add_device:
            devices = observation["devices"]
            devices = devices.flatten(start_dim=-2, end_dim=-1)
            parts.append(devices)

        if self.add_progress:
            progress = observation["aux", "progress"].unsqueeze(-1)
            parts.append(progress)

        return torch.cat(parts, dim=-1)


def run_rnn_ppo_torchrl(
    actor_critic_base: ActorCriticCNNBase,
    _make_env: Callable[[], EnvBase],
    config: PPOConfig,
    model_name: str = "model",
    model_path: str = None,
    eval_env_fn: Optional[Callable[[], EnvBase]] = None,
    rnn_model: str = "LSTM",
    rnn_hidden_size: int = 64,
):
    wandb.define_metric("batch_loss/step")
    wandb.define_metric("collect_loss/step")
    wandb.define_metric("batch_loss/*", step_metric="batch_loss/step")
    wandb.define_metric("grad_norm/*", step_metric="batch_loss/step")
    wandb.define_metric("param_norm/*", step_metric="batch_loss/step")
    wandb.define_metric("collect_loss/*", step_metric="collect_loss/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

    _actor_cnn = actor_critic_base.actor_cnn
    _critic_cnn = actor_critic_base.critic_cnn

    env = _make_env()
    if rnn_model == "LSTM":
        RNNModule = LSTMModule
    elif rnn_model == "GRU":
        RNNModule = GRUModule
    else:
        raise ValueError(f"Invalid RNN model: {rnn_model}")

    module_action_list = []

    module_action_cnn = TensorDictModule(
        _actor_cnn,
        in_keys=["observation"],
        out_keys=["actor_cnn_embed"],
    )
    module_action_cnn_obs = TensorDictModule(
        ConcatObservation(),
        in_keys=["actor_cnn_embed", "observation"],
        out_keys=["actor_cnn_concat"],
    )

    obs_size = module_action_cnn_obs(module_action_cnn(env.reset()))[
        "actor_cnn_concat"
    ].shape[-1]

    print(
        "Module Action CNN Output Size:",
        module_action_cnn(env.reset())["actor_cnn_embed"].shape,
        "Concat Size:",
        module_action_cnn_obs(module_action_cnn(env.reset()))["actor_cnn_concat"].shape,
    )

    module_action_rnn = RNNModule(
        input_size=obs_size,
        hidden_size=rnn_hidden_size,
        in_key="actor_cnn_concat",
        out_key="actor_rnn_embed",
    )
    # module_action_concat = TensorDictModule(
    #     Concat3Embeds(
    #         add_progress=False,
    #         add_device=env.simulator.observer.device_feature_dim != 0,
    #     ),
    #     in_keys=["actor_cnn_embed", "actor_rnn_embed", "observation"],
    #     out_keys=["actor_concat_embed"],
    # )

    module_action_fc = TensorDictModule(
        OutputHead(
            rnn_hidden_size,
            # + obs_size
            # + env.simulator.observer.device_feature_dim
            # * (actor_critic_base.n_devices - 1),
            actor_critic_base.layer_config.hidden_channels,
            actor_critic_base.n_devices - 1,
            num_layers=1,
            logits=True,
        ),
        in_keys=["actor_rnn_embed"],
        out_keys=["logits"],
    )

    td_module_action = ProbabilisticActor(
        # module=Sequential(
        #     module_action_cnn, module_action_rnn, module_action_concat, module_action_fc
        # ),
        module=Sequential(
            module_action_cnn,
            module_action_cnn_obs,
            module_action_rnn,
            module_action_fc,
        ),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        cache_dist=True,
        return_log_prob=True,
        default_interaction_type=tensordict.nn.InteractionType.RANDOM,
    )

    module_critic_cnn = TensorDictModule(
        _critic_cnn,
        in_keys=["observation"],
        out_keys=["critic_cnn_embed"],
    )

    module_critic_cnn_obs = TensorDictModule(
        ConcatObservation(concat_progress=True),
        in_keys=["critic_cnn_embed", "observation"],
        out_keys=["critic_cnn_concat"],
    )

    obs_size = module_critic_cnn_obs(module_critic_cnn(env.reset()))[
        "critic_cnn_concat"
    ].shape[-1]
    print(
        "Module Critic CNN Output Size:",
        module_critic_cnn(env.reset())["critic_cnn_embed"].shape,
        "Concat Size:",
        module_critic_cnn_obs(module_critic_cnn(env.reset()))[
            "critic_cnn_concat"
        ].shape,
    )
    module_critic_rnn = RNNModule(
        input_size=obs_size,
        hidden_size=rnn_hidden_size,
        in_key="critic_cnn_concat",
        out_key="critic_rnn_embed",
    )

    # module_critic_concat = TensorDictModule(
    #     Concat3Embeds(
    #         add_progress=True, add_device=env.simulator.observer.device_feature_dim != 0
    #     ),
    #     in_keys=["critic_cnn_embed", "critic_rnn_embed", "observation"],
    #     out_keys=["critic_concat_embed"],
    # )

    module_critic_fc = TensorDictModule(
        OutputHead(
            rnn_hidden_size,
            # + obs_size
            # + 1
            # + env.simulator.observer.device_feature_dim
            # * (actor_critic_base.n_devices - 1),
            actor_critic_base.layer_config.hidden_channels,
            1,  # Single value output
            num_layers=1,
            logits=False,
        ),
        in_keys=["critic_rnn_embed"],
        out_keys=["state_value"],
    )
    td_critic_module = Sequential(
        module_critic_cnn, module_critic_cnn_obs, module_critic_rnn, module_critic_fc
    )

    def make_env():
        env = _make_env()
        env.append_transform(module_action_rnn.make_tensordict_primer())
        env.append_transform(module_critic_rnn.make_tensordict_primer())
        return env

    print(td_module_action)
    print(td_critic_module)
    td_module_action = td_module_action.to(config.train_device)
    td_critic_module = td_critic_module.to(config.train_device)
    train_actor_network = copy.deepcopy(td_module_action).to(config.train_device)
    train_critic_network = copy.deepcopy(td_critic_module).to(config.train_device)
    model = torch.nn.ModuleList([train_actor_network, train_critic_network])
    model.eval()
    # Create evaluation environment if not provided
    if eval_env_fn is None:
        eval_env_fn = make_env

    if model_path:
        model.load_state_dict(torch.load(model_path))
        print("Loaded model from path:", model_path)

    collector = MultiSyncDataCollector(
        [make_env for _ in range(config.workers)],
        model[0],
        frames_per_batch=config.states_per_collection,
        total_frames=config.states_per_collection * config.num_collections,
        reset_at_each_iter=True,
        split_trajs=True,
        # cat_results=0,
        device=config.train_device,
        env_device="cpu",
    )
    out_seed = collector.set_seed(config.seed)
    print("Collector seed:", out_seed)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=config.states_per_collection, device=config.train_device
        ),
        sampler=SamplerWithoutReplacement(),
        pin_memory=torch.cuda.is_available(),
        prefetch=1,
        batch_size=config.minibatch_size,
    )

    advantage_module = GAE(
        gamma=config.gae_gamma,
        lmbda=config.gae_lmbda,
        value_network=model[1],
        average_gae=False,
        device=config.train_device,
        deactivate_vmap=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=model[0],
        critic_network=model[1],
        clip_epsilon=config.clip_eps,
        entropy_bonus=True,
        entropy_coef=config.ent_coef,
        critic_coef=config.val_coef,
        loss_critic_type=config.value_norm,
        clip_value=config.clip_vloss,
        normalize_advantage=config.normalize_advantage,
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

    # Run initial evaluation
    if config.eval_interval > 0:
        eval_metrics = evaluate_policy(
            policy=model[0],
            eval_env_fn=eval_env_fn,
            num_episodes=config.eval_episodes,
            step=0,
        )
        eval_metrics["eval/step"] = 0
        wandb.log(eval_metrics)
    prev_max_vsoptimal = 1.0
    for i, tensordict_data in enumerate(collector):
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
            advantage_module(tensordict_data)

            non_zero_rewards = tensordict_data["next", "reward"]
            improvements = tensordict_data["next", "observation", "aux", "improvement"]
            vsoptimals = tensordict_data["next", "observation", "aux", "vsoptimal"]
            mask = improvements > -1.5
            filtered_improvements = improvements[mask]
            if filtered_improvements.numel() > 0:
                avg_improvement = filtered_improvements.mean()
            mask = vsoptimals > -1.5
            filtered_vsoptimals = vsoptimals[mask]
            if filtered_vsoptimals.numel() > 0:
                avg_vsoptimal = filtered_vsoptimals.mean()

            if len(non_zero_rewards) > 0:
                avg_non_zero_reward = non_zero_rewards.mean().item()
                std_rewards = non_zero_rewards.std().item()
                print(f"Average reward: {avg_non_zero_reward}")
                print(f"Average improvement: {avg_improvement}")
                print(f"Average vs Optimal: {avg_vsoptimal}")

        replay_buffer.extend(tensordict_data.reshape(-1))

        # Training loop
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

                # Log pre-clipping gradient norms
                pre_clip_norms = log_parameter_and_gradient_norms(loss_module)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=config.max_grad_norm
                )

                # Log post-clipping gradient norms
                post_clip_norms = log_parameter_and_gradient_norms(loss_module)
                post_clip_norms = {
                    f"post_clip_{k}": v
                    for k, v in post_clip_norms.items()
                    if "grad_norm" in k
                }

                optimizer.step()

                # Log norms for this batch
                step = (
                    i * config.num_epochs_per_collection * n_batches + j * n_batches + k
                )
                wandb.log(
                    {
                        **pre_clip_norms,
                        **post_clip_norms,
                        "batch_loss/step": step,
                        "batch_loss/objective": loss_vals["loss_objective"].item(),
                        "batch_loss/critic": loss_vals["loss_critic"].item(),
                        "batch_loss/entropy": loss_vals["entropy"].item(),
                        "batch_loss/total": loss_value.item(),
                        "batch_loss/kl_approx": loss_vals["kl_approx"].item(),
                        "batch_loss/clip_fraction": loss_vals["clip_fraction"].item(),
                        # "batch_loss/value_clip_fraction": loss_vals[
                        #    "value_clip_fraction"
                        # ].item(),
                        "batch_loss/ESS": loss_vals["ESS"].item(),
                    },
                )

        # Update the policy
        collector.policy.load_state_dict(loss_module.actor_network.state_dict())
        collector.update_policy_weights_(TensorDict.from_module(collector.policy))
        wandb.log(
            {
                "collect_loss/step": i,
                "collect_loss/mean_nonzero_reward": avg_non_zero_reward,
                "collect_loss/std_nonzero_reward": std_rewards,
                "collect_loss/loss_objective": loss_vals["loss_objective"].item(),
                "collect_loss/average_improvement": avg_improvement.item(),
                "collect_loss/average_vsoptimal": avg_vsoptimal.item(),
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
        replay_buffer.empty()

        if (i + 1) % 100 == 0:
            if wandb.run.dir is None:
                path = "."
            else:
                path = wandb.run.dir
            torch.save(
                model.state_dict(),
                os.path.join(path, model_name + f"_{i+1}.pth"),
            )

        if avg_vsoptimal >= prev_max_vsoptimal + 0.01:
            if wandb.run.dir is None:
                path = "."
            else:
                path = wandb.run.dir
            torch.save(
                model.state_dict(),
                os.path.join(path, "max_" + model_name + f"_{avg_vsoptimal}.pth"),
            )
            prev_max_vsoptimal = avg_vsoptimal

    if config.eval_interval > 0 and i % config.eval_interval == 0:
        eval_metrics = evaluate_policy(
            policy=td_module_action,
            eval_env_fn=eval_env_fn,
            num_episodes=config.eval_episodes,
            step=config.num_collections,
        )
        eval_metrics["eval/step"] = config.num_collections
        wandb.log(eval_metrics)

    # save final network
    if wandb.run.dir is None:
        path = "."
    else:
        path = wandb.run.dir
    torch.save(
        model.state_dict(),
        os.path.join(path, model_name + f"_{config.num_collections}.pth"),
    )
    # save final optimizer state
    torch.save(
        optimizer.state_dict(),
        os.path.join(
            path, "optimizer_" + model_name + f"_{config.num_collections}.pth"
        ),
    )

    collector.shutdown()
    wandb.finish()
