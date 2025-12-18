from typing import Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage, RandomSampler
from torchrl.envs import EnvBase
from torchrl.objectives import DiscreteSACLoss
from torchrl.objectives.common import LossModule
from ..compile_utils import _safe_compile_with_warmup
from task4feedback.logging import training
from ..interface import Algorithm
from .config import SACConfig


class SACAlgorithm(Algorithm):
    def __init__(self, config: SACConfig):
        super().__init__(config)
        self.config = config
        self.num_actions = None
        self.num_heads = None
        self.use_multihead = False

    def initialize(
        self,
        model: torch.nn.Module,
        device: torch.device,
        env_constructors: Optional[List[Callable[[], EnvBase]]] = None,
    ):
        if env_constructors:
            dummy_env = env_constructors[0]()
            training.debug(f"Action spec: {dummy_env.action_spec}")

            # Try to infer num_actions and detect multi-discrete action spaces
            action_spec = dummy_env.action_spec

            # Prefer explicit action count from the underlying space (e.g., CategoricalBox.n)
            if hasattr(action_spec, "space") and hasattr(action_spec.space, "n"):
                self.num_actions = action_spec.space.n
            elif hasattr(action_spec, "n"):
                self.num_actions = action_spec.n
            elif hasattr(action_spec, "shape") and len(action_spec.shape) > 0:
                # Fallback: treat the trailing dimension as action count
                self.num_actions = action_spec.shape[-1]

            # Detect multi-head structure from the spec shape (e.g., shape=[num_heads])
            if hasattr(action_spec, "shape") and len(action_spec.shape) > 0:
                if len(action_spec.shape) == 2:
                    # shape=(num_heads, num_actions_per_head)
                    self.num_heads = action_spec.shape[0]
                    self.use_multihead = self.num_heads > 1
                elif len(action_spec.shape) == 1:
                    # shape=(num_heads,)
                    self.num_heads = action_spec.shape[0]
                    self.use_multihead = self.num_heads > 1

            if self.num_actions is None:
                training.warning("Could not infer num_actions from action spec")
            if self.num_heads is None:
                self.num_heads = 1

            training.info(f"Inferred num_actions: {self.num_actions}, num_heads: {self.num_heads}, use_multihead: {self.use_multihead}")
            dummy_env.close()

    def make_loss_module(self, model: torch.nn.Module) -> DiscreteSACLoss:
        training.debug(f"Making loss module with num_actions={self.num_actions}, use_multihead={self.use_multihead}")
        # Validate model has required components
        model.validate_for_algorithm("sac")

        # Direct access with standard names
        actor = model.policy
        qvalue = model.qvalue

        if actor is None or qvalue is None:
            raise ValueError("SAC requires both policy and qvalue networks")

        if getattr(self.config, "compile_loss_networks", False):
            actor = _safe_compile_with_warmup(actor, warmup=8, mode="default")
            qvalue = _safe_compile_with_warmup(qvalue, warmup=8, mode="default")

        # Configure target entropy
        target_entropy = self.config.target_entropy
        if target_entropy == "auto":
            import math
            # Use per-head entropy target; masked/padded heads contribute zero.
            target_entropy = -math.log(self.num_actions)
            if self.use_multihead:
                training.info(
                    f"Auto target entropy (per head) for multi-head policy: {target_entropy:.4f}"
                )

        # TorchRL's DiscreteSACLoss should work with multi-head distributions
        # as long as:
        # - Q-network outputs (batch, num_heads, num_actions)
        # - Distribution.probs is (batch, num_heads, num_actions)
        # - Actions are (batch, num_heads)
        # The loss computation handles the multi-head dimension naturally via
        # broadcasting and reduction operations.
        loss_module = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=self.config.num_qvalue_nets,
            loss_function=self.config.loss_function,
            alpha_init=self.config.alpha_init,
            target_entropy=target_entropy,
            num_actions=self.num_actions,
            action_space="categorical",  # Works for both single and multi-head
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

    def process_batch(self, batch: TensorDict, device: torch.device) -> TensorDict:
        batch = batch.to(device, non_blocking=True)
        mask = batch.get(("observation", "aux", "candidate_mask"), None)
        mask_b = mask.to(device=device, dtype=torch.bool) if mask is not None else None
        
        action = batch.get("action")

        # TorchRL's DiscreteSACLoss supports weighting via the prioritized replay
        # `priority_weight` key. Use the candidate mask to exclude padded heads
        # from actor/Q/alpha losses.
        if mask_b is not None and action.ndim > 1 and mask_b.shape == action.shape and mask_b.any():
            batch.set("priority_weight", mask_b.to(dtype=torch.float32))

        # Expand reward/done for multi-head categorical actions if needed
        if action.ndim > 1:
            C = action.shape[1]
            for key in ["reward", "done", "terminated"]:
                val = batch.get(("next", key))
                if val.ndim == 1:
                    # (B,) -> (B, 1)
                    val = val.unsqueeze(-1)
                if val.ndim == 2:
                    # (B, 1) -> (B, C, 1) or (B, C) -> (B, C, 1)
                    if val.shape[1] == 1:
                        val = val.expand(-1, C).unsqueeze(-1)
                    elif val.shape[1] == C:
                        val = val.unsqueeze(-1)

                if mask_b is not None:
                    head_mask = mask_b.unsqueeze(-1)
                    if key == "reward":
                        val = val * head_mask.to(val.dtype)
                    else:
                        # Treat padded heads as terminal to avoid bootstrapping on junk.
                        val = val.to(torch.bool) | (~head_mask)

                batch.set(("next", key), val)
                    
        return batch

    def update(
        self,
        loss_module: DiscreteSACLoss,
        batch: TensorDict,
        optimizer: torch.optim.Optimizer,
        target_net_updater: Optional[Any] = None,
    ) -> Dict[str, float]:
        # Debug
        training.debug(f"Batch keys: {batch.keys()}")
        if "sample_log_prob" in batch.keys():
            training.debug(f"sample_log_prob shape: {batch['sample_log_prob'].shape}")
        
        loss_td = loss_module(batch)
        loss_sum = sum(loss for key, loss in loss_td.items() if key.startswith("loss"))
        
        optimizer.zero_grad()
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), self.config.max_grad_norm)
        optimizer.step()
        
        if target_net_updater:
            target_net_updater.step()
            
        return {k: v.item() for k, v in loss_td.items()}

    def train_step(
        self,
        batch: TensorDict,
        loss_module: LossModule,
        optimizer: torch.optim.Optimizer,
        replay_buffer: Optional[ReplayBuffer] = None,
        target_net_updater: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> List[Dict[str, float]]:
        
        if replay_buffer is None:
            raise ValueError("SAC requires a replay buffer.")
            
        # Add to buffer
        replay_buffer.extend(batch.reshape(-1))
        
        metrics_list = []
        
        if device is None:
            device = self.config.update_device

        for _ in range(self.config.updates_per_collection):
            sub_batch = replay_buffer.sample()
            sub_batch = self.process_batch(sub_batch, device)
            
            metrics = self.update(loss_module, sub_batch, optimizer, target_net_updater)
            metrics_list.append(metrics)
            
        return metrics_list
