from typing import Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import EnvBase
from torchrl.objectives import DQNLoss
from torchrl.objectives.common import LossModule

from task4feedback.logging import training
from task4feedback.ml.actors import EpsilonGreedyQValueActor
from ..compile_utils import _safe_compile_with_warmup
from ..interface import Algorithm
from .config import DQNConfig


class DQNAlgorithm(Algorithm):
    def __init__(self, config: DQNConfig):
        super().__init__(config)
        self.config = config
        self._collection_actor: Optional[EpsilonGreedyQValueActor] = None

    def make_loss_module(self, model: torch.nn.Module) -> DQNLoss:
        # Validate model has required components
        model.validate_for_algorithm("dqn")

        # Direct access with standard name
        qvalue = model.qvalue

        if qvalue is None:
            raise ValueError("DQN requires qvalue network")

        if getattr(self.config, "compile_loss_networks", False):
            qvalue = _safe_compile_with_warmup(qvalue)

        loss_module = DQNLoss(
            value_network=qvalue,
            loss_function=self.config.loss_function,
            delay_value=True,
        )
        loss_module.make_value_estimator(gamma=self.config.gamma)
        return loss_module

    def make_replay_buffer(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[ReplayBuffer]:
        return TensorDictReplayBuffer(
            storage=LazyTensorStorage(self.config.replay_buffer_size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=batch_size,
        )

    def process_batch(self, batch: TensorDict, device: torch.device) -> TensorDict:
        batch = batch.to(device, non_blocking=True)
        return batch

    def get_collection_policy(
        self,
        model: torch.nn.Module,
        device: torch.device,
        mode: str = "train",
    ) -> torch.nn.Module:
        actor_kind = getattr(self.config, "collector_actor", "qvalue_epsilon_greedy")
        if actor_kind in ("qvalue_epsilon_greedy", "epsilon_greedy_qvalue"):
            if model.qvalue is None:
                raise ValueError("DQN collector requires a qvalue network.")
            if self._collection_actor is None:
                self._collection_actor = EpsilonGreedyQValueActor(
                    model.qvalue,
                    qvalue_key=("action_value",),
                    action_key=("action",),
                    mask_key=("observation", "aux", "candidate_mask"),
                    eps_init=self.config.eps_init,
                    eps_end=self.config.eps_end,
                    eps_decay=self.config.eps_decay,
                )
            self._collection_actor.train(mode == "train")
            return self._collection_actor

        return super().get_collection_policy(model, device, mode)

    def update(
        self,
        loss_module: DQNLoss,
        batch: TensorDict,
        optimizer: torch.optim.Optimizer,
        target_net_updater: Optional[Any] = None,
    ) -> Dict[str, float]:
        loss_td = loss_module(batch)
        loss_sum = sum(loss for key, loss in loss_td.items() if key.startswith("loss"))
        
        optimizer.zero_grad()
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), self.config.max_grad_norm)
        optimizer.step()
        
        if target_net_updater:
            target_net_updater.step()

        metrics = {k: v.item() for k, v in loss_td.items()}
        if self._collection_actor is not None:
            metrics["exploration/epsilon"] = float(self._collection_actor.epsilon)
        return metrics

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
            raise ValueError("DQN requires a replay buffer.")
            
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
