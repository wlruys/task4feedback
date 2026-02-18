import time
from typing import Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage, RandomSampler
from torchrl.envs import EnvBase
from torchrl.objectives import DiscreteSACLoss
from torchrl.objectives.common import LossModule
from ..compile_utils import _safe_compile_with_warmup
from task4feedback.logging import training
from task4feedback.ml.rl_utils import log_parameter_and_gradient_norms
from ..interface import Algorithm
from .config import SACConfig


class SACAlgorithm(Algorithm):
    def __init__(self, config: SACConfig):
        super().__init__(config)
        self.config = config
        self.num_actions = None
        self.num_heads = None

    def initialize(
        self,
        model: torch.nn.Module,
        device: torch.device,
        env_constructors: Optional[List[Callable[[], EnvBase]]] = None,
    ):
        if env_constructors:
            dummy_env = env_constructors[0]()
            training.info(f"Action spec: {dummy_env.action_spec}")
            self.num_heads = dummy_env.max_candidates
            self.num_actions = dummy_env.n_devices
            training.info(f"Using multihead with {self.num_heads} heads")
            dummy_env.close()

    def make_loss_module(self, model: torch.nn.Module) -> DiscreteSACLoss:
        model.validate_for_algorithm("sac")

        actor = model.policy
        qvalue = model.qvalue

        if getattr(self.config, "compile_loss_networks", False):
            actor = _safe_compile_with_warmup(actor, warmup=8, mode="default")
            qvalue = _safe_compile_with_warmup(qvalue, warmup=8, mode="default")

        # - Q-network outputs (batch, num_heads, num_actions)
        # - Distribution.probs is (batch, num_heads, num_actions)
        # - Actions are (batch, num_heads)
        loss_module = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=self.config.num_qvalue_nets,
            loss_function=self.config.loss_function,
            alpha_init=self.config.alpha_init,
            target_entropy="auto",
            target_entropy_weight=0.05,
            num_actions=self.num_actions,
            action_space="categorical",
            reduction="sum",
        )
        return loss_module

    def make_replay_buffer(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[ReplayBuffer]:
        return TensorDictReplayBuffer(
            storage=LazyTensorStorage(self.config.replay_buffer_size, device=device),
            sampler=RandomSampler(),
            batch_size=batch_size,
        )
    
    @staticmethod
    def _get_loss_metrics(loss_out: TensorDict) -> Dict[str, float]:
        return {
            f"batch/{k}": v.mean().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in loss_out.items()
        }

    def process_batch(self, batch: TensorDict, device: torch.device) -> TensorDict:
        batch = batch.to(device, non_blocking=True)
        
        # Expand rewards, done, terminated, truncated to match the number of heads
        # We want to go from [batch, 1] to [batch, num_heads, 1]
        for key in ["reward", "done", "terminated", "truncated"]:
            for prefix in [("next",), ()]:
                full_key = prefix + (key,)
                val = batch.get(full_key, None)
                if val is not None:
                    if val.ndim == 1:
                        batch[full_key] = val.unsqueeze(-1).unsqueeze(1).expand(-1, self.num_heads, -1)
                    elif val.ndim == 2 and val.shape[1] == 1:
                        batch[full_key] = val.unsqueeze(1).expand(-1, self.num_heads, -1)
        
        return batch

    def _compute_collection_metrics(self, flattened_data: TensorDict) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        with torch.no_grad():
            rewards = flattened_data["next", "reward"]

            if ("next", "observation", "aux", "improvement") in flattened_data.keys(
                include_nested=True
            ):
                improvements = flattened_data["next", "observation", "aux", "improvement"]
                valid_mask = torch.isfinite(improvements) & (improvements > -100)
                valid_improvements = improvements[valid_mask]

                if valid_improvements.numel() > 0:
                    metrics["batch/mean_improvement"] = valid_improvements.mean().item()
                    metrics["batch/max_improvement"] = valid_improvements.max().item()
                    metrics["batch/min_improvement"] = valid_improvements.min().item()
                    if valid_improvements.numel() > 1:
                        metrics["batch/std_improvement"] = valid_improvements.std().item()
                    metrics["batch/n_completed"] = float(valid_improvements.numel())

                    if ("next", "observation", "aux", "time") in flattened_data.keys(
                        include_nested=True
                    ):
                        times = flattened_data["next", "observation", "aux", "time"]
                        valid_times = times[valid_mask].to(torch.float32)
                        if valid_times.numel() > 0:
                            metrics["batch/mean_time"] = valid_times.mean().item()
                            metrics["batch/min_time"] = valid_times.min().item()
                            metrics["batch/max_time"] = valid_times.max().item()
                else:
                    training.debug(
                        "No episodes completed in collection batch (total steps: %d). "
                        "Consider increasing frames_per_batch if this persists.",
                        improvements.numel()
                    )
                    metrics["batch/n_completed"] = 0.0

            if rewards.numel() > 0:
                metrics["batch/mean_reward"] = rewards.mean().item()
                if rewards.numel() > 1:
                    metrics["batch/std_reward"] = rewards.std().item()

        return metrics

    def update(
        self,
        loss_module: DiscreteSACLoss,
        batch: TensorDict,
        optimizer: torch.optim.Optimizer,
        target_net_updater: Optional[Any] = None,
    ) -> Dict[str, float]:        
        loss_td = loss_module(batch)
        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        loss = actor_loss + q_loss + alpha_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), self.config.max_grad_norm)
        optimizer.step()
        
        if target_net_updater:
            target_net_updater.step()
            
        metrics = self._get_loss_metrics(loss_td)
        metrics["batch/lr"] = optimizer.param_groups[0]["lr"]
        if hasattr(loss_module, "target_entropy"):
            metrics["batch/target_entropy"] = float(loss_module.target_entropy.item())
        if "td_error" in batch.keys():
            td_error = batch["td_error"]
            if isinstance(td_error, torch.Tensor) and td_error.numel() > 0:
                metrics["batch/td_error_mean"] = td_error.mean().item()
                if td_error.numel() > 1:
                    metrics["batch/td_error_std"] = td_error.std().item()
        if getattr(self.config, "log_norms", False):
            metrics.update(log_parameter_and_gradient_norms(loss_module))

        return metrics

    def train_step(
        self,
        batch: TensorDict,
        loss_module: LossModule,
        optimizer: torch.optim.Optimizer,
        replay_buffer: Optional[ReplayBuffer] = None,
        target_net_updater: Optional[Any] = None,
        device: Optional[torch.device] = None,
        n_collections: int = 0,
        n_updates: int = 0,
        n_samples: int = 0,
    ) -> List[Dict[str, float]]:
        
        if replay_buffer is None:
            raise ValueError("SAC requires a replay buffer.")
            
        flattened_data = batch.reshape(-1)
        collection_metrics = self._compute_collection_metrics(flattened_data)

        replay_buffer.extend(flattened_data)
        
        metrics_list = []
        
        if device is None:
            device = self.config.update_device

        update_start = time.perf_counter()
        for _ in range(self.config.updates_per_collection):
            sub_batch = replay_buffer.sample()
            sub_batch = self.process_batch(sub_batch, device)
            
            metrics = self.update(loss_module, sub_batch, optimizer, target_net_updater)
            metrics_list.append(metrics)
            print(f"len metrics_list: {len(metrics_list)}")

        update_elapsed = time.perf_counter() - update_start
        if metrics_list:
            shared_metrics = {
                **collection_metrics,
                "timing/update_seconds": update_elapsed,
                "batch/effective_batch_size": float(self.config.batch_size),
                "batch/replay_buffer_size": float(len(replay_buffer)),
            }
            for metrics in metrics_list:
                metrics.update(shared_metrics)

        return metrics_list
