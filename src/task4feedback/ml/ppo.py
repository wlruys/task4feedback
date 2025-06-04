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
from torchrl._utils import compile_with_warmup
from torch.profiler import profile, record_function, ProfilerActivity
from torchrl.objectives.utils import ValueEstimators


@dataclass
class PPOConfig:
    states_per_collection: int = 1920
    minibatch_size: int = 250
    num_epochs_per_collection: int = 6
    num_collections: int = 1000
    workers: int = 1
    seed: int = 0
    lr: float = 2.5e-4
    clip_eps: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    val_coef: float = 0.5
    max_grad_norm: float = 0.5
    threads_per_worker: int = 1
    collect_device: str = "cpu"
    update_device: str = "cpu"
    gae_gamma: float = 1
    gae_lmbda: float = 0.99
    normalize_advantage: bool = True
    value_norm: str = "l2"
    eval_interval: int = 10
    eval_episodes: int = 1


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

        if hasattr(env, "simulator") and hasattr(env.simulator, "time"):
            completion_time = env.simulator.time
            completion_times.append(completion_time)

            if i == 0 and completion_time > 0:
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
                            format="mp4",
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


def run_ppo_cleanrl_no(
    actor_critic_base: nn.Module,
    make_env: Callable[[], EnvBase],
    config: PPOConfig,
    model_name: str = "model",
    model_path: str = None,
    eval_env_fn: Optional[Callable[[], EnvBase]] = None,
    do_rollout: bool = False,
):
    # actor_critic_base = torch.compile(actor_critic_base)
    # _actor_critic_td = HeteroDataWrapper(actor_critic_base)
    actor_critic_base = actor_critic_base.to(config.collect_device)

    wandb.define_metric("batch_loss/step")
    wandb.define_metric("collect_loss/step")
    wandb.define_metric("batch_loss/*", step_metric="batch_loss/step")
    wandb.define_metric("collect_loss/*", step_metric="collect_loss/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

    _actor_critic_module = TensorDictModule(
        actor_critic_base,
        in_keys=["observation"],
        out_keys=["logits", "state_value"],
    )
    _actor_critic_module = _actor_critic_module.to(config.collect_device)

    actor_critic = ProbabilisticActor(
        _actor_critic_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        cache_dist=False,
        return_log_prob=True,
    )

    actor_critic = actor_critic.to(config.collect_device)

    # actor_critic = torch.compile(actor_critic)

    collector = MultiSyncDataCollector(
        [make_env for _ in range(config.workers)],
        actor_critic,
        frames_per_batch=config.states_per_collection,
        reset_at_each_iter=True,
        env_device="cpu",
        policy_device=config.collect_device,
        # compile_policy=True,
        cat_results=0,
        # use_buffers=True,
        trust_policy=True,
    )

    out_seed = collector.set_seed(config.seed)

    actor_critic_base_t = copy.deepcopy(actor_critic_base)
    actor_critic_base_t = actor_critic_base_t.to(config.update_device)

    optimizer = torch.optim.Adam(actor_critic_base_t.parameters(), lr=config.lr)

    if eval_env_fn is None:
        eval_env_fn = make_env

    if config.eval_interval > 0:
        eval_metrics = evaluate_policy(
            policy=actor_critic,
            eval_env_fn=eval_env_fn,
            num_episodes=config.eval_episodes,
            step=0,
        )
        eval_metrics["eval/step"] = 0
        wandb.log(eval_metrics)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    def update_fn(step, new_logits, new_value, batch, actor_critic_base_t, optimizer):
        # new_logits = new_logits.view(-1)
        new_value = new_value.view(-1)

        sample_logprob = batch["sample_log_prob"].detach()
        sample_value = batch["state_value"].detach().view(-1)
        sample_advantage = batch["advantage"].detach().view(-1)
        sample_returns = batch["value_target"].detach().view(-1)
        sample_action = batch["action"].detach()

        new_logprob, new_entropy = logits_to_action(new_logits, sample_action)

        new_logprob = new_logprob.view(-1)
        new_entropy = new_entropy.view(-1)

        # Policy Loss
        logratio = new_logprob.view(-1) - sample_logprob.detach().view(-1)
        ratio = logratio.exp().view(-1)

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clip_fraction = ((torch.abs(ratio - 1) > config.clip_eps).float()).mean()
            ess = 1 - torch.sum(
                torch.exp(logratio) * (1 - torch.exp(logratio))
            ).mean() / (torch.sum(torch.exp(logratio)) + 1e-8)

        if config.normalize_advantage:
            sample_advantage = (sample_advantage - sample_advantage.mean()) / (
                sample_advantage.std() + 1e-8
            )

        policy_loss_1 = sample_advantage * ratio.view(-1)
        policy_loss_2 = sample_advantage * torch.clamp(
            ratio, 1 - config.clip_eps, 1 + config.clip_eps
        )
        policy_loss = -1 * torch.min(policy_loss_1, policy_loss_2).mean()

        # Value Loss
        v_loss_unclipped = (new_value - sample_returns) ** 2

        if config.clip_vloss:
            v_clipped = sample_value + torch.clamp(
                new_value - sample_value, -config.clip_eps, config.clip_eps
            )
            v_loss_clipped = (v_clipped - sample_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss_max = v_loss_unclipped.mean()
        v_loss = 0.5 * v_loss_max

        # Entropy Loss
        entropy_loss = new_entropy.mean()

        loss = policy_loss + config.val_coef * v_loss - config.ent_coef * entropy_loss

        # LOG GRADIENTS
        optimizer.zero_grad()
        loss.backward()
        pre_clip_norms = log_parameter_and_gradient_norms(actor_critic_base_t)
        nn.utils.clip_grad_norm_(actor_critic_base_t.parameters(), config.max_grad_norm)
        post_clip_norms = log_parameter_and_gradient_norms(actor_critic_base_t)
        optimizer.step()

        wandb.log(
            {
                **pre_clip_norms,
                **post_clip_norms,
                "batch_loss/step": step,
                "batch_loss/objective": policy_loss.item(),
                "batch_loss/critic": v_loss.item(),
                "batch_loss/scaled_critic": config.val_coef * v_loss.item(),
                "batch_loss/scaled_entropy": config.ent_coef * entropy_loss.item(),
                "batch_loss/entropy": entropy_loss.item(),
                "batch_loss/total": loss.item(),
                "batch_loss/kl_approx": approx_kl.item(),
                "batch_loss/clip_fraction": clip_fraction.item(),
                "batch_loss/ESS": ess.item(),
            }
        )

    def repack_td(td):
        state = td["observation", "hetero_data"]

        # print("TD length:", len(td))
        # print("State length:", len(state))

        obs_actions = td["action"]
        obs_sample_log_prob = td["sample_log_prob"]
        obs_state_value = td["state_value"]
        obs_advantage = td["advantage"]
        obs_value_target = td["value_target"]
        # obs_task_counts = td["observation", "nodes", "tasks", "count"]
        # obs_data_counts = td["observation", "nodes", "data", "count"]

        for l in range(len(state)):
            # print(state, l)
            _obs = state[l]
            _obs["action"] = obs_actions[l]
            _obs["sample_log_prob"] = obs_sample_log_prob[l]
            _obs["state_value"] = obs_state_value[l]
            _obs["advantage"] = obs_advantage[l]
            _obs["value_target"] = obs_value_target[l]
            # _obs["tasks_count"] = obs_task_counts[l]
            # _obs["data_count"] = obs_data_counts[l]
            # state[l] = _obs.to(config.update_device)
        return state

    # repack_td = torch.compile(repack_td)
    # update_fn = torch.compile(update_fn, mode="reduce-overhead")

    # with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:

    collect_start_t = time.perf_counter()
    for i, td in enumerate(collector):
        if i >= config.num_collections:
            break

        collect_break_t = time.perf_counter()
        print("Collection time:", collect_break_t - collect_start_t)

        print("Collection:", i)

        start_t = time.perf_counter()
        td = td.to(config.update_device, non_blocking=True)
        end_t = time.perf_counter()
        print("Move to device time:", end_t - start_t)

        with torch.no_grad():
            start_t = time.perf_counter()
            td = compute_gae(td, gamma=config.gae_gamma, lam=config.gae_lmbda)
            # td = compute_advantage(td)
            end_t = time.perf_counter()
            print("Advantage computation time:", end_t - start_t)

            # Record average improvement in wandb
            non_zero_rewards = td["next", "reward"]
            improvements = td["next", "observation", "aux", "improvement"]
            mask = improvements > -1.5
            filtered_improvements = improvements[mask]
            if filtered_improvements.numel() > 0:
                avg_improvement = filtered_improvements.mean()
            else:
                avg_improvement = 0.0
            if len(non_zero_rewards) > 0:
                avg_non_zero_reward = non_zero_rewards.mean().item()
                std_rewards = non_zero_rewards.std().item()
                print(f"Average improvement: {avg_improvement}")

        start_t = time.perf_counter()
        state = repack_td(td)
        # torch.cuda.synchronize()
        end_t = time.perf_counter()
        print("Repacking Time:", end_t - start_t)

        if i % 50 == 0:
            # Save the model in the wandb directory
            if wandb.run.dir is None:
                path = "."
            else:
                path = wandb.run.dir
            torch.save(
                actor_critic_base_t.state_dict(),
                os.path.join(path, "model_" + f"{i}.pth"),
            )

        if config.eval_interval > 0 and (i + 1) % config.eval_interval == 0:
            eval_metrics = evaluate_policy(
                policy=actor_critic,
                eval_env_fn=eval_env_fn,
                num_episodes=config.eval_episodes,
                step=i + 1,
            )
            eval_metrics["eval/step"] = i + 1
            wandb.log(eval_metrics)

        for j in range(config.num_epochs_per_collection):
            loader = DataLoader(state, batch_size=config.minibatch_size, shuffle=True)

            for k, batch in enumerate(loader):
                batch = batch.to(config.update_device, non_blocking=True)

                start_t = time.perf_counter()
                new_logits, new_value = actor_critic_base_t(batch)

                step = (i * config.num_epochs_per_collection) + j * len(loader) + k

                update_fn(
                    step, new_logits, new_value, batch, actor_critic_base_t, optimizer
                )

                end_t = time.perf_counter()

        collector.policy.module[0].module.load_state_dict(
            actor_critic_base_t.state_dict()
        )
        collector.update_policy_weights_(TensorDict.from_module(collector.policy))
        collect_start_t = time.perf_counter()

        print("Update policy time:", collect_start_t - collect_break_t)
        wandb.log(
            {
                "collect_loss/step": i,
                "collect_loss/mean_nonzero_reward": avg_non_zero_reward,
                "collect_loss/std_nonzero_reward": std_rewards,
                "collect_loss/std_improvement": filtered_improvements.std().item(),
                "collect_loss/average_improvement": avg_improvement.item(),
                "collect_loss/mean_return": td["value_target"].mean().item(),
                "collect_loss/std_return": td["value_target"].std().item(),
                "collect_loss/advantage_mean": td["advantage"].mean().item(),
                "collect_loss/advantage_std": td["advantage"].std().item(),
            },
        )

    # Final evaluation
    if config.eval_interval > 0:
        eval_metrics = evaluate_policy(
            policy=actor_critic,
            eval_env_fn=eval_env_fn,
            num_episodes=config.eval_episodes,
            step=config.num_collections,
        )
        eval_metrics["eval/step"] = config.num_collections
        wandb.log(eval_metrics)

    if wandb.run.dir is None:
        path = "."
    else:
        path = wandb.run.dir
    torch.save(
        actor_critic_base_t.state_dict(),
        os.path.join(path, model_name + "_final.pth"),
    )

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    collector.shutdown()
    wandb.finish()


def run_ppo_cleanrl(
    actor_critic_base: nn.Module, make_env: Callable[[], EnvBase], config: PPOConfig
):
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.states_per_collection),
        sampler=SamplerWithoutReplacement(),
        pin_memory=torch.cuda.is_available(),
        batch_size=config.minibatch_size,
        prefetch=4,
    )

    # actor_critic_base = torch.compile(actor_critic_base)

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

    test_make = [make_env for _ in range(config.workers)]

    collector = MultiSyncDataCollector(
        [make_env for _ in range(config.workers)],
        actor_critic,
        frames_per_batch=config.states_per_collection,
        reset_at_each_iter=True,
        cat_results=0,
        env_device="cpu",
        policy_device=config.collect_device,
    )
    out_seed = collector.set_seed(config.seed)

    # Create a copy of the actor_critic model to be used for training
    actor_critic_t = copy.deepcopy(actor_critic)
    actor_critic_t = actor_critic_t.to(config.update_device)
    actor_critic_t.module[0].module.device = config.update_device

    optimizer = torch.optim.Adam(actor_critic_t.parameters(), lr=config.lr)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    def update_fn(batch, actor_critic_t, optimizer):
        batch = batch.to(config.update_device, non_blocking=True)

        sample_logprob = batch["record_log_prob"].detach().view(-1)
        sample_value = batch["record_state_value"].detach().view(-1)
        sample_advantage = batch["advantage"].detach().view(-1)
        sample_returns = batch["value_target"].detach().view(-1)
        sample_action = batch["action"].detach().view(-1)

        eval_batch = batch

        new_logprob, new_entropy = logits_to_action(eval_batch["logits"], sample_action)
        new_logprob = new_logprob.view(-1)
        new_entropy = new_entropy.view(-1)

        new_value = eval_batch["state_value"].view(-1)

        # Policy Loss
        logratio = new_logprob.view(-1) - sample_logprob.detach().view(-1)
        ratio = logratio.exp().view(-1)

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clip_fraction = ((torch.abs(ratio - 1) > config.clip_eps).float()).mean()
            print("Clip fraction:", clip_fraction.item())

        if config.normalize_advantage:
            sample_advantage = (sample_advantage - sample_advantage.mean()) / (
                sample_advantage.std() + 1e-8
            )

        policy_loss_1 = sample_advantage * ratio.view(-1)
        policy_loss_2 = sample_advantage * torch.clamp(
            ratio, 1 - config.clip_eps, 1 + config.clip_eps
        )
        policy_loss = -1 * torch.min(policy_loss_1, policy_loss_2).mean()

        # Value Loss
        v_loss_unclipped = (new_value - sample_returns) ** 2
        if config.clip_vloss:
            v_clipped = sample_value + torch.clamp(
                new_value - sample_value, -config.clip_eps, config.clip_eps
            )
            v_loss_clipped = (v_clipped - sample_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss_max = v_loss_unclipped.mean()
        v_loss = 0.5 * v_loss_max

        # Entropy Loss
        entropy_loss = new_entropy.mean()

        loss = policy_loss + config.val_coef * v_loss - config.ent_coef * entropy_loss

        wandb.log(
            {
                "batch_loss/objective": policy_loss.item(),
                "batch_loss/critic": v_loss.item(),
                "batch_loss/entropy": entropy_loss.item(),
                "batch_loss/total": loss.item(),
                "batch_loss/kl_approx": approx_kl.item(),
                "batch_loss/clip_fraction": clip_fraction.item(),
            }
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic_t.parameters(), config.max_grad_norm)
        optimizer.step()
        norms = log_parameter_and_gradient_norms(actor_critic_t)
        wandb.log(norms)

    # update_fn = torch.compile(update_fn)

    start_collect_t = time.perf_counter()
    for i, td in enumerate(collector):
        end_collect_t = time.perf_counter()
        print("Collection time:", end_collect_t - start_collect_t)
        print(f"Collection: {i}")

        start_t = time.perf_counter()
        with torch.no_grad():
            td = compute_advantage(td)

            td["record_state_value"] = td["state_value"].clone()
            td["record_log_prob"] = td["sample_log_prob"].clone()
            td["record_logits"] = td["logits"].clone()

        replay_buffer.extend(td.reshape(-1))
        end_t = time.perf_counter()
        print("Advantage computation time:", end_t - start_t)

        sort_by_keyword = "cpu_time_total"
        start_batch_t = time.perf_counter()
        for j in range(config.num_epochs_per_collection):
            n_batches = len(replay_buffer) // config.minibatch_size
            # actor_critic_t = actor_critic.to(config.update_device)

            for k in range(n_batches):
                start_t = time.perf_counter()

                batch = replay_buffer.sample(config.minibatch_size)
                batch = actor_critic_t(batch)

                # with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                #     with record_function("model_update"):
                update_fn(batch, actor_critic_t, optimizer)

                end_t = time.perf_counter()

                # print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=100))

                print("Batch training time:", end_t - start_t)
        end_batch_t = time.perf_counter()
        print("Total batch training time:", end_batch_t - start_batch_t)

        collector.policy.load_state_dict(actor_critic_t.state_dict())
        collector.update_policy_weights_(TensorDict.from_module(collector.policy))
        start_collect_t = time.perf_counter()
    collector.shutdown()


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

    # _actor_td = HeteroDataWrapper(actor_critic_base.actor)
    # _critic_td = HeteroDataWrapper(actor_critic_base.critic)

    _actor_td = actor_critic_base.actor
    _critic_td = actor_critic_base.critic

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

    td_module_action = td_module_action.to(config.collect_device)
    td_critic_module = td_critic_module.to(config.collect_device)
    train_actor_network = copy.deepcopy(td_module_action).to(config.update_device)
    train_critic_network = copy.deepcopy(td_critic_module).to(config.update_device)
    model = torch.nn.ModuleList([train_actor_network, train_critic_network])

    # Create evaluation environment if not provided
    if eval_env_fn is None:
        eval_env_fn = make_env

    if model_path:
        model.load_state_dict(torch.load(model_path))
        print("Loaded model from path:", model_path)

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
        device=config.collect_device,
        env_device="cpu",
    )
    out_seed = collector.set_seed(config.seed)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=config.states_per_collection, device=config.update_device
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
        device=config.update_device,
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

    def update(subdata, i, j, k):
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
            f"post_clip_{k}": v for k, v in post_clip_norms.items() if "grad_norm" in k
        }

        optimizer.step()

        # Log norms for this batch
        step = i * config.num_epochs_per_collection * n_batches + j * n_batches + k
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

    for i, tensordict_data in enumerate(collector):
        if (i + 1) % 20 == 0:
            if wandb.run.dir is None:
                path = "."
            else:
                path = wandb.run.dir
            torch.save(
                model.state_dict(),
                os.path.join(path, model_name + f"_{i + 1}.pth"),
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
        tensordict_data = tensordict_data.to(config.update_device, non_blocking=True)

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
                std_rewards = non_zero_rewards.std().item()
                print(f"Average reward: {avg_non_zero_reward}")
                print(f"Average improvement: {avg_improvement}")

        replay_buffer.extend(tensordict_data.reshape(-1))

        # Training loop
        for j in range(config.num_epochs_per_collection):
            n_batches = config.states_per_collection // config.minibatch_size

            for k in range(n_batches):
                subdata = replay_buffer.sample(config.minibatch_size)
                subdata.to(config.update_device, non_blocking=True)

                update(subdata, i, j, k)

        # Update the policy
        collector.policy.load_state_dict(loss_module.actor_network.state_dict())
        collector.update_policy_weights_(TensorDict.from_module(collector.policy))
        wandb.log(
            {
                "collect_loss/step": i,
                "collect_loss/mean_nonzero_reward": avg_non_zero_reward,
                "collect_loss/std_nonzero_reward": std_rewards,
                "collect_loss/average_improvement": avg_improvement.item(),
                "collect_loss/std_improvement": filtered_improvements.std().item(),
                "collect_loss/std_return": tensordict_data["value_target"].std().item(),
                "collect_loss/mean_return": tensordict_data["value_target"]
                .mean()
                .item(),
                "collect_loss/advantage_mean": tensordict_data["advantage"]
                .mean()
                .item(),
                "collect_loss/advantage_std": tensordict_data["advantage"].std().item(),
            },
        )

    # Final evaluation
    if config.eval_interval > 0:
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
