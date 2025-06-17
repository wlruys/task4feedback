from ..models import *
from ..util import *
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any, Tuple
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import (
    ReplayBuffer,
    SliceSampler,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage, TensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule, GRUModule
from torchrl.envs import TransformedEnv
import copy
from tensordict import TensorDict
import wandb
import os
import torch
from torchrl._utils import compile_with_warmup
from .base import AlgorithmConfig, LoggingConfig
from ..base import ActorCriticModule
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from task4feedback.logging import training
import time
from torchrl.envs import ParallelEnv
from torchrl.collectors.utils import split_trajectories


@dataclass
class PPOConfig(AlgorithmConfig):
    implementation: str = "torchrl"
    graphs_per_collection: int = 10
    states_per_collection: int = 1920
    minibatch_size: int = 250
    epochs_per_collection: int = 4
    num_collections: int = 1000
    workers: int = 1
    clip_eps: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    val_coef: float = 0.5
    max_grad_norm: float = 0.5
    threads_per_worker: int = 1
    collect_device: str = "cpu"
    update_device: str = "cpu"
    storing_device: str = "cpu"
    gae_gamma: float = 1
    gae_lmbda: float = 0.99
    normalize_advantage: bool = False
    value_norm: str = "l2"
    compile_policy: bool = False
    compile_update: bool = False
    compile_advantage: bool = False
    using_lstm: bool = False
    seed: int = 0
    collector: str = "multi_sync"  # "sync" or "multi_sync"
    sample_slices: bool = (
        True  # if using lstm, whether slices are used instead of episodes
    )
    slice_len: int = 16  # length of slices for LSTM, only used if sample_slices is True


def run_ppo(
    actor_critic_module: ActorCriticModule,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    opt_cfg: Optional[DictConfig] = None,
    lr_cfg: Optional[DictConfig] = None,
):
    if logging_config is not None and (
        logging_frequency := logging_config.stats_interval
    ):
        wandb.define_metric("batch_loss/step")
        wandb.define_metric("collect_loss/step")
        wandb.define_metric("batch_loss/*", step_metric="batch_loss/step")
        wandb.define_metric("grad_norm/*", step_metric="batch_loss/step")
        wandb.define_metric("param_norm/*", step_metric="batch_loss/step")
        wandb.define_metric("collect_loss/*", step_metric="collect_loss/step")
        wandb.define_metric("eval/*", step_metric="eval/step")

    print("Using PPO with config:", OmegaConf.to_yaml(ppo_config))

    max_tasks = max([make_env().size() for make_env in env_constructors])
    print(f"Max tasks in env constructors: {max_tasks}")

    max_states_per_collection = ppo_config.graphs_per_collection * max_tasks

    advantage_module = GAE(
        gamma=ppo_config.gae_gamma,
        lmbda=ppo_config.gae_lmbda,
        value_network=actor_critic_module.critic,
        average_gae=False,
        device=ppo_config.update_device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=max_states_per_collection,
            device=ppo_config.update_device,
        ),
        sampler=SamplerWithoutReplacement(),
        # pin_memory=torch.cuda.is_available(),
        prefetch=4,
        batch_size=ppo_config.minibatch_size,
        # shared=True,
        # transform=advantage_module,
    )

    def env_workers():
        return [
            env_constructors[i % len(env_constructors)]
            for i in range(ppo_config.workers)
        ]

    print(f"Creating collector with {ppo_config.workers} workers")

    if ppo_config.collector == "multi_sync":
        collector = MultiSyncDataCollector(
            env_workers(),
            actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            # cat_results="stack",
            cat_results=None,
            reset_at_each_iter=False,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            # replay_buffer=replay_buffer,
            # extend_buffer=True,
            # postproc=advantage_module,
            # compile with reduced overhead
            compile_policy={"mode": "reduce-overhead"}
            if ppo_config.compile_policy
            else None,
        )
    elif ppo_config.collector == "sync":
        collector = SyncDataCollector(
            env_workers()[0],
            policy=actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            reset_at_each_iter=False,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            # replay_buffer=replay_buffer,
            # extend_buffer=True,
            # postproc=advantage_module,
        )
    else:
        raise ValueError(
            f"Unknown collector type: {ppo_config.collector}. "
            "Use 'sync' or 'multi_sync'."
        )

    collector.set_seed(ppo_config.seed)

    loss_module = ClipPPOLoss(
        actor_network=actor_critic_module.actor,
        critic_network=actor_critic_module.critic,
        clip_epsilon=ppo_config.clip_eps,
        entropy_bonus=True,
        entropy_coef=ppo_config.ent_coef,
        critic_coef=ppo_config.val_coef,
        loss_critic_type=ppo_config.value_norm,
        clip_value=ppo_config.clip_vloss,
        normalize_advantage=ppo_config.normalize_advantage,
    )

    # optimizer = instantiate(opt_cfg, params=loss_module.parameters())
    optimizer = torch.optim.AdamW(
        loss_module.parameters(),
        lr=opt_cfg.lr if opt_cfg is not None else 3e-4,
        eps=opt_cfg.eps if opt_cfg is not None else 1e-5,
    )

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    if lr_cfg is not None:
        lr_scheduler = instantiate(lr_cfg, optimizer=optimizer)
    else:
        lr_scheduler = None

    def update(batch, i, j, k):
        loss_vals = loss_module(batch)
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        optimizer.zero_grad()
        loss_value.backward()

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        return loss_vals

    if ppo_config.compile_advantage:
        advantage_module = compile_with_warmup(
            advantage_module, mode="reduce-overhead", warmup=8
        )

    if ppo_config.compile_update:
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    states_per_collection = min(
        ppo_config.states_per_collection, max_states_per_collection
    )
    n_batch = states_per_collection // ppo_config.minibatch_size

    training.info(
        f"Starting PPO training with {ppo_config.num_collections} collections, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{n_batch} batches per collection, "
        f"{ppo_config.workers} workers."
    )

    start_t = time.perf_counter()

    for i, tensordict_data in enumerate(collector):
        replay_buffer.empty()

        if i >= ppo_config.num_collections:
            break

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        updates_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0

        training.info(
            f"Collection {i + 1}/{ppo_config.num_collections}, "
            f"Collections/s: {updates_per_second:.2f}",
        )

        tensordict_data = tensordict_data.to(
            ppo_config.update_device, non_blocking=True
        )

        print(f"Processing collection {i + 1} with {tensordict_data.shape} shape")

        adv_start_t = time.perf_counter()
        with torch.no_grad():
            # Compute advantages
            advantage_module(tensordict_data)
        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")

        replay_buffer.extend(tensordict_data.reshape(-1))

        update_start_t = time.perf_counter()
        for j in range(ppo_config.epochs_per_collection):
            for k in range(n_batch):
                batch = replay_buffer.sample(ppo_config.minibatch_size)
                batch.to(ppo_config.update_device, non_blocking=True)
                loss = update(batch, i, j, k)

            collector.update_policy_weights_(
                TensorDict.from_module(loss_module.actor_network).to(
                    ppo_config.collect_device
                )
            )
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")


def run_ppo_lstm(
    actor_critic_module: ActorCriticModule,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    opt_cfg: Optional[DictConfig] = None,
    lr_cfg: Optional[DictConfig] = None,
):
    if logging_config is not None and (
        logging_frequency := logging_config.stats_interval
    ):
        wandb.define_metric("batch_loss/step")
        wandb.define_metric("collect_loss/step")
        wandb.define_metric("batch_loss/*", step_metric="batch_loss/step")
        wandb.define_metric("grad_norm/*", step_metric="batch_loss/step")
        wandb.define_metric("param_norm/*", step_metric="batch_loss/step")
        wandb.define_metric("collect_loss/*", step_metric="collect_loss/step")
        wandb.define_metric("eval/*", step_metric="eval/step")

    print("Using PPO with config:", OmegaConf.to_yaml(ppo_config))

    max_tasks = max([make_env().size() for make_env in env_constructors])
    print(f"Max tasks in env constructors: {max_tasks}")

    max_states_per_collection = ppo_config.graphs_per_collection * max_tasks

    advantage_module = GAE(
        gamma=ppo_config.gae_gamma,
        lmbda=ppo_config.gae_lmbda,
        value_network=actor_critic_module.critic,
        average_gae=False,
        device=ppo_config.update_device,
        deactivate_vmap=True,
    )

    if ppo_config.sample_slices:
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=max_states_per_collection,
                device=ppo_config.update_device,
            ),
            sampler=SliceSampler(
                strict_length=True,
                slice_len=ppo_config.slice_len,
                traj_key=("collector", "traj_ids"),
            ),
            batch_size=ppo_config.minibatch_size,
        )
        num_slices = ppo_config.minibatch_size // ppo_config.slice_len
    else:
        ppo_config.minibatch_size = 2
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=max_states_per_collection,
                device=ppo_config.update_device,
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=ppo_config.minibatch_size,
        )
        num_slices = ppo_config.minibatch_size

    def env_workers():
        return [
            env_constructors[i % len(env_constructors)]
            for i in range(ppo_config.workers)
        ]

    print(f"Creating collector with {ppo_config.workers} workers")

    if ppo_config.collector == "multi_sync":
        collector = MultiSyncDataCollector(
            env_workers(),
            actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            cat_results="stack",
            reset_at_each_iter=False,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            # replay_buffer=replay_buffer,
            # extend_buffer=True,
            # postproc=advantage_module,
            # compile with reduced overhead
            compile_policy={"mode": "reduce-overhead"}
            if ppo_config.compile_policy
            else None,
        )
    elif ppo_config.collector == "sync":
        collector = SyncDataCollector(
            env_workers()[0],
            policy=actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            reset_at_each_iter=False,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            # replay_buffer=replay_buffer,
            # extend_buffer=True,
            # postproc=advantage_module,
        )
    else:
        raise ValueError(
            f"Unknown collector type: {ppo_config.collector}. "
            "Use 'sync' or 'multi_sync'."
        )

    collector.set_seed(ppo_config.seed)

    loss_module = ClipPPOLoss(
        actor_network=actor_critic_module.actor,
        critic_network=actor_critic_module.critic,
        clip_epsilon=ppo_config.clip_eps,
        entropy_bonus=True,
        entropy_coef=ppo_config.ent_coef,
        critic_coef=ppo_config.val_coef,
        loss_critic_type=ppo_config.value_norm,
        clip_value=ppo_config.clip_vloss,
        normalize_advantage=ppo_config.normalize_advantage,
    )

    # optimizer = instantiate(opt_cfg, params=loss_module.parameters())
    optimizer = torch.optim.AdamW(
        loss_module.parameters(),
        lr=opt_cfg.lr if opt_cfg is not None else 3e-4,
        eps=opt_cfg.eps if opt_cfg is not None else 1e-5,
    )

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    if lr_cfg is not None:
        lr_scheduler = instantiate(lr_cfg, optimizer=optimizer)
    else:
        lr_scheduler = None

    def update(batch, i, j, k):
        if ppo_config.sample_slices:
            batch = batch.reshape(num_slices, -1)
            # print(batch.shape)

        loss_vals = loss_module(batch)
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        optimizer.zero_grad()
        loss_value.backward()

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        return loss_vals

    if ppo_config.compile_advantage:
        advantage_module = compile_with_warmup(
            advantage_module, mode="reduce-overhead", warmup=8
        )

    if ppo_config.compile_update:
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    states_per_collection = min(
        ppo_config.states_per_collection, max_states_per_collection
    )

    if ppo_config.sample_slices:
        n_batch = max(1, states_per_collection // ppo_config.minibatch_size)
    else:
        n_batch = max(1, ppo_config.graphs_per_collection // ppo_config.minibatch_size)

    training.info(
        f"Starting PPO-LSTM training with {ppo_config.num_collections} collections, "
        f"sample_slices={ppo_config.sample_slices}, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{n_batch} batches per collection, "
        f"{ppo_config.workers} workers.",
    )

    start_t = time.perf_counter()

    for i, tensordict_data in enumerate(collector):
        replay_buffer.empty()

        if i >= ppo_config.num_collections:
            break

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        updates_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0

        training.info(
            f"Collection {i + 1}/{ppo_config.num_collections}, "
            f"Collections/s: {updates_per_second:.2f}",
        )

        tensordict_data = tensordict_data.to(
            ppo_config.update_device, non_blocking=True
        )

        print(f"Processing collection {i + 1} with {tensordict_data.shape} shape")

        adv_start_t = time.perf_counter()
        with torch.no_grad():
            # Compute advantages
            advantage_module(tensordict_data)
        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")

        # print("Done", tensordict_data["next", "done"].shape)
        # print("Done Shape", tensordict_data["next", "done"].shape)

        if ppo_config.sample_slices:
            replay_buffer.extend(tensordict_data.reshape(-1))
        else:
            # tensordict_data = split_trajectories(
            #     tensordict_data.reshape(-1),
            #     trajectory_key=("collector", "traj_ids"),
            #     done_key=("next", "truncated"),
            # )
            # print("Split trajectories shape:", tensordict_data.shape)
            # print(tensordict_data["action"])
            replay_buffer.extend(tensordict_data)

        update_start_t = time.perf_counter()
        for j in range(ppo_config.epochs_per_collection):
            for k in range(n_batch):
                batch, info = replay_buffer.sample(
                    ppo_config.minibatch_size, return_info=True
                )

                # print("Sampling batch", batch.shape)

                batch.to(ppo_config.update_device, non_blocking=True)
                loss = update(batch, i, j, k)

            collector.update_policy_weights_(
                TensorDict.from_module(loss_module.actor_network).to(
                    ppo_config.collect_device
                )
            )
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")


# def run_ppo_torchrl(
#     actor_model: ProbabilisticActor,
#     critic_model: ValueOperator,
#     make_env: Callable[[], EnvBase],
# ):
#     wandb.define_metric("batch_loss/step")
#     wandb.define_metric("collect_loss/step")
#     wandb.define_metric("batch_loss/*", step_metric="batch_loss/step")
#     wandb.define_metric("grad_norm/*", step_metric="batch_loss/step")
#     wandb.define_metric("param_norm/*", step_metric="batch_loss/step")
#     wandb.define_metric("collect_loss/*", step_metric="collect_loss/step")
#     wandb.define_metric("eval/*", step_metric="eval/step")

#     # _actor_td = HeteroDataWrapper(actor_critic_base.actor)
#     # _critic_td = HeteroDataWrapper(actor_critic_base.critic)

#     _actor_td = actor_critic_base.actor
#     _critic_td = actor_critic_base.critic

#     module_action = TensorDictModule(
#         _actor_td,
#         in_keys=["observation"],
#         out_keys=["logits"],
#     )

#     td_module_action = ProbabilisticActor(
#         module_action,
#         in_keys=["logits"],
#         out_keys=["action"],
#         distribution_class=torch.distributions.Categorical,
#         cache_dist=False,
#         return_log_prob=True,
#     )

#     td_critic_module = ValueOperator(
#         module=_critic_td,
#         in_keys=["observation"],
#     )

#     td_module_action = td_module_action.to(config.collect_device)
#     td_critic_module = td_critic_module.to(config.collect_device)
#     train_actor_network = copy.deepcopy(td_module_action).to(config.update_device)
#     train_critic_network = copy.deepcopy(td_critic_module).to(config.update_device)
#     model = torch.nn.ModuleList([train_actor_network, train_critic_network])

#     # Create evaluation environment if not provided
#     if eval_env_fn is None:
#         eval_env_fn = make_env

#     if model_path:
#         model.load_state_dict(torch.load(model_path))
#         print("Loaded model from path:", model_path)

#     if do_rollout:

#         def rollout_env():
#             env = make_env()
#             env.set_policy(model[0])
#             return env

#         _make_env = rollout_env
#     else:
#         _make_env = make_env

#     collector = MultiSyncDataCollector(
#         [_make_env for _ in range(config.workers)],
#         model[0],
#         frames_per_batch=config.states_per_collection,
#         reset_at_each_iter=True,
#         cat_results=0,
#         device=config.collect_device,
#         env_device="cpu",
#     )
#     out_seed = collector.set_seed(config.seed)

#     replay_buffer = ReplayBuffer(
#         storage=LazyTensorStorage(
#             max_size=config.states_per_collection, device=config.update_device
#         ),
#         sampler=SamplerWithoutReplacement(),
#         pin_memory=torch.cuda.is_available(),
#         prefetch=4,
#         batch_size=config.minibatch_size,
#     )

#     advantage_module = GAE(
#         gamma=config.gae_gamma,
#         lmbda=config.gae_lmbda,
#         value_network=model[1],
#         average_gae=False,
#         device=config.update_device,
#         vectorized=False,
#     )

#     loss_module = ClipPPOLoss(
#         actor_network=model[0],
#         critic_network=model[1],
#         clip_epsilon=config.clip_eps,
#         entropy_bonus=True,
#         entropy_coef=config.ent_coef,
#         critic_coef=config.val_coef,
#         loss_critic_type=config.value_norm,
#         clip_value=config.clip_vloss,
#         normalize_advantage=config.normalize_advantage,
#     )

#     optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

#     def update(subdata, i, j, k):
#         loss_vals = loss_module(subdata)
#         loss_value = (
#             loss_vals["loss_objective"]
#             + loss_vals["loss_critic"]
#             + loss_vals["loss_entropy"]
#         )

#         optimizer.zero_grad()
#         loss_value.backward()

#         # Log pre-clipping gradient norms
#         pre_clip_norms = log_parameter_and_gradient_norms(loss_module)

#         grad_norm = torch.nn.utils.clip_grad_norm_(
#             loss_module.parameters(), max_norm=config.max_grad_norm
#         )

#         # Log post-clipping gradient norms
#         post_clip_norms = log_parameter_and_gradient_norms(loss_module)
#         post_clip_norms = {
#             f"post_clip_{k}": v for k, v in post_clip_norms.items() if "grad_norm" in k
#         }

#         optimizer.step()

#         # Log norms for this batch
#         step = i * config.num_epochs_per_collection * n_batches + j * n_batches + k
#         wandb.log(
#             {
#                 **pre_clip_norms,
#                 **post_clip_norms,
#                 "batch_loss/step": step,
#                 "batch_loss/objective": loss_vals["loss_objective"].item(),
#                 "batch_loss/critic": loss_vals["loss_critic"].item(),
#                 "batch_loss/entropy": loss_vals["entropy"].item(),
#                 "batch_loss/total": loss_value.item(),
#                 "batch_loss/kl_approx": loss_vals["kl_approx"].item(),
#                 "batch_loss/clip_fraction": loss_vals["clip_fraction"].item(),
#                 # "batch_loss/value_clip_fraction": loss_vals[
#                 #    "value_clip_fraction"
#                 # ].item(),
#                 "batch_loss/ESS": loss_vals["ESS"].item(),
#             },
#         )

#     # advantage_module = torch.compile(advantage_module)
#     # update = torch.compile(update, mode="reduce-overhead")

#     # Run initial evaluation
#     if config.eval_interval > 0:
#         eval_metrics = evaluate_policy(
#             policy=model[0],
#             eval_env_fn=eval_env_fn,
#             num_episodes=config.eval_episodes,
#             step=0,
#         )
#         eval_metrics["eval/step"] = 0
#         wandb.log(eval_metrics)

#     for i, tensordict_data in enumerate(collector):
#         replay_buffer.empty()

#         if (i + 1) % 20 == 0:
#             if wandb.run.dir is None:
#                 path = "."
#             else:
#                 path = wandb.run.dir
#             torch.save(
#                 model.state_dict(),
#                 os.path.join(path, model_name + f"_{i + 1}.pth"),
#             )

#         # Run evaluation at specified intervals
#         if config.eval_interval > 0 and (i + 1) % config.eval_interval == 0:
#             eval_metrics = evaluate_policy(
#                 policy=model[0],
#                 eval_env_fn=eval_env_fn,
#                 num_episodes=config.eval_episodes,
#                 step=i + 1,
#             )
#             eval_metrics["eval/step"] = i + 1
#             wandb.log(eval_metrics)

#         if i >= config.num_collections:
#             break

#         print(f"Collection: {i}")
#         tensordict_data = tensordict_data.to(config.update_device, non_blocking=True)

#         with torch.no_grad():
#             # print(f"Computing advantages for collection {i}")
#             # print("tensordict_data keys:", tensordict_data.keys(True))

#             advantage_module(tensordict_data)

#             non_zero_rewards = tensordict_data["next", "reward"]
#             improvements = tensordict_data["next", "observation", "aux", "improvement"]
#             mask = improvements > -1.5
#             filtered_improvements = improvements[mask]
#             if filtered_improvements.numel() > 0:
#                 avg_improvement = filtered_improvements.mean()

#             if len(non_zero_rewards) > 0:
#                 avg_non_zero_reward = non_zero_rewards.mean().item()
#                 std_rewards = non_zero_rewards.std().item()
#                 print(f"Average reward: {avg_non_zero_reward}")
#                 print(f"Average improvement: {avg_improvement}")

#         replay_buffer.extend(tensordict_data.reshape(-1))

#         # Training loop
#         for j in range(config.num_epochs_per_collection):
#             # n_batches = config.states_per_collection // config.minibatch_size
#             n_batches = (80 * 8) // config.minibatch_size

#             for k in range(n_batches):
#                 subdata = replay_buffer.sample(config.minibatch_size)
#                 subdata.to(config.update_device, non_blocking=True)

#                 update(subdata, i, j, k)

#         # Update the policy
#         collector.policy.load_state_dict(loss_module.actor_network.state_dict())
#         collector.update_policy_weights_(TensorDict.from_module(collector.policy))
#         wandb.log(
#             {
#                 "collect_loss/step": i,
#                 "collect_loss/mean_nonzero_reward": avg_non_zero_reward,
#                 "collect_loss/std_nonzero_reward": std_rewards,
#                 "collect_loss/average_improvement": avg_improvement.item(),
#                 "collect_loss/std_improvement": filtered_improvements.std().item(),
#                 "collect_loss/std_return": tensordict_data["value_target"].std().item(),
#                 "collect_loss/mean_return": tensordict_data["value_target"]
#                 .mean()
#                 .item(),
#                 "collect_loss/advantage_mean": tensordict_data["advantage"]
#                 .mean()
#                 .item(),
#                 "collect_loss/advantage_std": tensordict_data["advantage"].std().item(),
#             },
#         )

#     # Final evaluation
#     if config.eval_interval > 0:
#         eval_metrics = evaluate_policy(
#             policy=td_module_action,
#             eval_env_fn=eval_env_fn,
#             num_episodes=config.eval_episodes,
#             step=config.num_collections,
#         )
#         eval_metrics["eval/step"] = config.num_collections
#         wandb.log(eval_metrics)

#     # save final network
#     if wandb.run.dir is None:
#         path = "."
#     else:
#         path = wandb.run.dir
#     torch.save(
#         model.state_dict(),
#         os.path.join(path, model_name + f"_{config.num_collections}.pth"),
#     )
#     # save final optimizer state
#     torch.save(
#         optimizer.state_dict(),
#         os.path.join(
#             path, "optimizer_" + model_name + f"_{config.num_collections}.pth"
#         ),
#     )

#     collector.shutdown()
#     wandb.finish()
