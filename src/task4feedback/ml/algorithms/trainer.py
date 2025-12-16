import time
import functools
from typing import Callable, List, Optional, Dict, Any

import torch
import wandb
from tensordict import TensorDict
from torchrl.collectors import DataCollectorBase
from torchrl.envs import EnvBase
from torchrl.objectives.common import LossModule
from torchrl.objectives import SoftUpdate

from task4feedback.logging import training
from task4feedback.ml.eval import EvaluationConfig, make_eval_envs, run_evaluation
from task4feedback.ml.rl_utils import save_checkpoint
from .base import (
    AlgorithmConfig,
    LoggingConfig,
    load_best_performance,
    save_best_performance,
    should_log,
    should_eval,
    should_checkpoint,
)
from .interface import Algorithm


class Trainer:
    def __init__(
        self,
        algorithm: Algorithm,
        model: torch.nn.Module,
        env_constructors: List[Callable[[], EnvBase]],
        alg_config: AlgorithmConfig,
        logging_config: LoggingConfig,
        eval_config: EvaluationConfig,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        seed: int = 0,
        eval_location: str = "cpu",
    ):
        self.algorithm = algorithm
        self.model = model
        self.env_constructors = env_constructors
        self.alg_config = alg_config
        self.logging_config = logging_config
        self.eval_config = eval_config
        self.seed = seed
        self.eval_location = eval_location

        # Instantiate optimizer if it's a partial
        if isinstance(optimizer, functools.partial):
            self.optimizer = optimizer(params=self.model.parameters())
        else:
            self.optimizer = optimizer

        # Instantiate scheduler if it's a partial
        if isinstance(lr_scheduler, functools.partial):
            self.lr_scheduler = lr_scheduler(optimizer=self.optimizer)
        else:
            self.lr_scheduler = lr_scheduler

        # Devices
        self.collect_device = torch.device(getattr(alg_config, "collect_device", "cpu"))
        self.update_device = torch.device(getattr(alg_config, "update_device", "cpu"))
        self.storing_device = torch.device(getattr(alg_config, "storing_device", "cpu"))

    def train(self):
        training.info(f"Starting training with {self.algorithm.__class__.__name__}")

        # 0. Initialize Algorithm
        self.algorithm.initialize(self.model, self.update_device, self.env_constructors)

        # 1. Setup Loss Module
        loss_module = self.algorithm.make_loss_module(self.model)
        loss_module.to(self.update_device)

        # Target Network Updater (only for off-policy algorithms)
        target_net_updater = None
        if self.alg_config.type == "off_policy" and hasattr(self.alg_config, "target_update_polyak"):
            target_net_updater = SoftUpdate(loss_module, eps=self.alg_config.target_update_polyak)

        # 2. Setup Replay Buffer
        replay_buffer = self.algorithm.make_replay_buffer(
            batch_size=getattr(self.alg_config, "batch_size", 256),
            device=self.storing_device,
        )

        # 3. Setup Collector
        # Determine policy for collection (Actor for PPO/SAC, Q-value for DQN exploration)
        # This logic might need to be in the algorithm or passed in.
        # Extract policy from model using standard attribute names
        # Try policy first (for PPO/SAC), then qvalue (for DQN), then use whole model
        policy = self.model.policy or self.model.qvalue or self.model
        
        collector = self.algorithm.make_collector(
            self.env_constructors,
            policy,
            self.collect_device,
            self.storing_device,
        )

        # 4. Setup Evaluation
        eval_env_fns = self.env_constructors if isinstance(self.env_constructors, list) else [self.env_constructors]
        eval_envs = make_eval_envs(eval_env_fns)
        best_performance = load_best_performance(self.logging_config.best_policy_dir)
        if (self.eval_config.best_metric_mode or "max").lower() == "min" and best_performance == 0.0:
            best_performance = float("inf")

        # 5. Training Loop
        start_time = time.time()
        n_updates = 0
        
        # Determine max collections
        num_collections = getattr(self.alg_config, "num_collections", None)

        try:
            for i, data in enumerate(collector):
                if num_collections is not None and i >= num_collections:
                    training.info(f"Reached maximum number of collections: {num_collections}")
                    break

                # Unified training step
                metrics_list = self.algorithm.train_step(
                    batch=data,
                    loss_module=loss_module,
                    optimizer=self.optimizer,
                    replay_buffer=replay_buffer,
                    target_net_updater=target_net_updater,
                    device=self.update_device,
                )
                
                current_buffer_size = len(replay_buffer) if replay_buffer is not None else 0
                
                for metrics in metrics_list:
                    n_updates += 1
                    if should_log(n_updates, self.logging_config):
                        metrics["batch/n_updates"] = n_updates
                        metrics["batch/n_collections"] = i + 1
                        if current_buffer_size > 0:
                            metrics["batch/buffer_size"] = current_buffer_size
                        if wandb.run is not None:
                            wandb.log(metrics, step=n_updates)

                # Evaluation
                if should_eval(n_updates, self.eval_config):
                    # Set policy to eval mode
                    policy.eval()
                    with torch.no_grad():
                        eval_metrics = run_evaluation(
                            policy,
                            eval_envs,
                            self.eval_config,
                            n_collections=i + 1,
                            n_updates=n_updates,
                        )
                    # Set policy back to train mode
                    policy.train()

                    best_metric_key = self.eval_config.best_metric or "eval/DETERMINISTIC/vs_baseline/mean_vs_EFT"
                    best_mode = (self.eval_config.best_metric_mode or "max").lower()
                    current_metric = eval_metrics.get(best_metric_key)
                    if current_metric is not None:
                        improved = current_metric < best_performance if best_mode == "min" else current_metric > best_performance
                        if improved:
                            best_performance = current_metric
                            save_best_performance(
                                self.logging_config.best_policy_dir,
                                best_performance,
                                self.logging_config.best_policy_name,
                                metric_name=best_metric_key,
                            )
                            # TODO: Fix save_checkpoint call to match actual function signature
                            # save_checkpoint(
                            #     self.model,
                            #     self.logging_config.best_policy_dir,
                            #     self.logging_config.best_policy_name,
                            # )

                    if wandb.run is not None:
                        wandb.log(eval_metrics, step=n_updates)

                # Checkpoint
                if should_checkpoint(n_updates, self.logging_config):
                    # TODO: Fix save_checkpoint call to match actual function signature
                    pass
                    # save_checkpoint(
                    #     self.model,
                    #     self.logging_config.best_policy_dir,
                    #     self.logging_config.best_policy_name
                    # )
                
                # Timeout
                timeout = getattr(self.alg_config, "timeout", 60 * 60 * 24)
                if time.time() - start_time > timeout:
                    training.info("Training timed out")
                    break
        finally:
            collector.shutdown()
