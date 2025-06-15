from ..models import *
from ..util import *
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
    gae_gamma: float = 1
    gae_lmbda: float = 0.99
    normalize_advantage: bool = False
    value_norm: str = "l2"
    compile_policy: bool = False
    compile_update: bool = False
    compile_advantage: bool = False
    using_lstm: bool = False
    seed: int = 0


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

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=max_states_per_collection,
            device=ppo_config.update_device,
        ),
        sampler=SamplerWithoutReplacement(),
        pin_memory=torch.cuda.is_available(),
        prefetch=4,
        batch_size=ppo_config.minibatch_size,
    )

    def env_workers():
        return [
            env_constructors[i % len(env_constructors)]
            for i in range(ppo_config.workers)
        ]

    collector = MultiSyncDataCollector(
        env_workers(),
        actor_critic_module.actor,
        frames_per_batch=max_states_per_collection,
        reset_at_each_iter=True,
        cat_results=0,
        device=ppo_config.collect_device,
        env_device="cpu",
    )
    out_seed = collector.set_seed(ppo_config.seed)

    advantage_module = GAE(
        gamma=ppo_config.gae_gamma,
        lmbda=ppo_config.gae_lmbda,
        value_network=actor_critic_module.critic,
        average_gae=False,
        device=ppo_config.update_device,
    )

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

    optimizer = instantiate(opt_cfg, params=loss_module.parameters())

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

        return {
            "loss_objective": loss_vals["loss_objective"].item(),
            "loss_critic": loss_vals["loss_critic"].item(),
            "loss_entropy": loss_vals["entropy"].item(),
        }

    for i, tensordict_data in enumerate(collector):
        replay_buffer.empty()

        if i >= ppo_config.num_collections:
            break

        with torch.no_grad():
            # Compute advantages
            advantage_module(tensordict_data)

        replay_buffer.extend(tensordict_data.reshape(-1))

        for j in range(ppo_config.epochs_per_collection):
            n_batch = ppo_config.states_per_collection // ppo_config.minibatch_size

            for k in range(n_batch):
                batch = replay_buffer.sample(ppo_config.minibatch_size)
                batch.to(ppo_config.update_device, non_blocking=True)
                loss = update(batch, i, j, k)

                training.info(
                    f"Collection {i}, Epoch {j}, Batch {k}: "
                    f"Loss Objective: {loss['loss_objective']:.4f}, "
                    f"Loss Critic: {loss['loss_critic']:.4f}, "
                    f"Loss Entropy: {loss['loss_entropy']:.4f}"
                )


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
