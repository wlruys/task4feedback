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
from task4feedback.ml.rl_utils import save_checkpoint, warmup_lazy_modules, has_uninitialized_params
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

        self._optimizer_spec = optimizer
        self._lr_scheduler_spec = lr_scheduler

        self.optimizer = None if isinstance(optimizer, functools.partial) else optimizer
        self.lr_scheduler = None if isinstance(lr_scheduler, functools.partial) else lr_scheduler

        # Devices
        self.collect_device = torch.device(getattr(alg_config, "collect_device", "cpu"))
        self.update_device = torch.device(getattr(alg_config, "update_device", "cpu"))
        self.storing_device = torch.device(getattr(alg_config, "storing_device", "cpu"))

    def _setup_wandb_logging(self) -> None:
        """Configure wandb metric stepping and optional model watching."""
        if wandb.run is None:
            return

        wandb.define_metric("batch/n_collections")
        wandb.define_metric("batch/n_updates", step_metric="batch/n_collections")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_collections")
        wandb.define_metric("batch/*", step_metric="batch/n_collections")
        wandb.define_metric("timing/*", step_metric="batch/n_collections")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_collections")
        wandb.define_metric("param_norm/*", step_metric="batch/n_collections")
        wandb.define_metric("eval/*", step_metric="batch/n_collections")

        if getattr(self.logging_config, "watch_model", False):
            try:
                wandb.watch(
                    self.model,
                    log=self.logging_config.watch_log_mode,
                    log_freq=self.logging_config.watch_log_freq,
                )
            except Exception as exc:
                training.warning("Failed to enable wandb.watch: %s", exc)

    def _maybe_add_missing_optimizer_params(self, module: torch.nn.Module) -> None:
        """Ensure the optimizer tracks all trainable parameters in `module`."""
        if self.optimizer is None:
            return
        try:
            existing = {
                id(param)
                for group in self.optimizer.param_groups
                for param in group.get("params", [])
            }
        except Exception:
            return

        missing = [
            param
            for param in module.parameters()
            if param.requires_grad and id(param) not in existing
        ]
        if not missing:
            return

        base_group = {}
        if self.optimizer.param_groups:
            base_group = {
                k: v
                for k, v in self.optimizer.param_groups[0].items()
                if k != "params"
            }
        self.optimizer.add_param_group({**base_group, "params": missing})
        training.info("Added %d loss-module parameters to optimizer.", len(missing))

    def _make_sac_param_groups(
        self,
        loss_module: LossModule,
    ) -> Optional[List[Dict[str, Any]]]:
        actor_params = None
        qvalue_params = None
        alpha_params = None

        if hasattr(loss_module, "actor_network_params"):
            actor_params = list(loss_module.actor_network_params.parameters())
        if hasattr(loss_module, "qvalue_network_params"):
            qvalue_params = list(loss_module.qvalue_network_params.parameters())

        log_alpha = getattr(loss_module, "log_alpha", None)
        if isinstance(log_alpha, torch.nn.Parameter) and log_alpha.requires_grad:
            alpha_params = [log_alpha]

        if not actor_params or not qvalue_params:
            return None

        param_groups: List[Dict[str, Any]] = [
            {"params": actor_params, "name": "actor"},
            {"params": qvalue_params, "name": "qvalue"},
        ]
        if alpha_params:
            param_groups.append({"params": alpha_params, "name": "alpha"})

        initial_lr = None
        if isinstance(self._optimizer_spec, functools.partial):
            initial_lr = self._optimizer_spec.keywords.get("lr", None)
        if initial_lr is None:
            initial_lr = getattr(self.alg_config, "lr", None)
        if initial_lr is not None:
            for group in param_groups:
                group["lr"] = initial_lr

        return param_groups

    def train(self):
        training.info(f"Starting training with {self.algorithm.__class__.__name__}")

        self.algorithm.initialize(self.model, self.update_device, self.env_constructors)
        self._setup_wandb_logging()

        if has_uninitialized_params(self.model) and self.env_constructors:
            warmup_env = None
            try:
                warmup_env = self.env_constructors[0]()
                warmup_lazy_modules(self.model, warmup_env, warmup_steps=2)
            except Exception as exc:
                training.warning("Lazy parameter warmup failed: %s", exc)
            finally:
                if warmup_env is not None:
                    warmup_env.close()

        loss_module = self.algorithm.make_loss_module(self.model)
        loss_module.to(self.update_device)

        # Create optimizer/scheduler after loss module exists
        if self.optimizer is None:
            if not isinstance(self._optimizer_spec, functools.partial):
                raise TypeError("Trainer requires an optimizer or a functools.partial optimizer factory.")
            param_groups = None
            if getattr(self.alg_config, "name", None) == "sac":
                param_groups = self._make_sac_param_groups(loss_module)
            if param_groups:
                self.optimizer = self._optimizer_spec(params=param_groups)
            else:
                self.optimizer = self._optimizer_spec(params=loss_module.parameters())
        else:
            self._maybe_add_missing_optimizer_params(loss_module)

        if self.lr_scheduler is None and isinstance(self._lr_scheduler_spec, functools.partial):
            self.lr_scheduler = self._lr_scheduler_spec(optimizer=self.optimizer)

        # Target Network Updater (if needed)
        target_net_updater = None
        if self.alg_config.type == "off_policy" and hasattr(self.alg_config, "target_update_polyak"):
            target_net_updater = SoftUpdate(loss_module, eps=self.alg_config.target_update_polyak)

        replay_buffer = self.algorithm.make_replay_buffer(
            batch_size=getattr(self.alg_config, "batch_size", 256),
            device=self.storing_device,
        )

        policy = self.algorithm.get_collection_policy(
            self.model,
            device=self.collect_device,
            mode="train",
        )

        try:
            policy_device = next(policy.parameters()).device
        except StopIteration:
            policy_device = self.collect_device
        
        collector = self.algorithm.make_collector(
            self.env_constructors,
            policy,
            policy_device,
            self.storing_device,
            seed=self.seed,
        )

        eval_env_fns = self.env_constructors if isinstance(self.env_constructors, list) else [self.env_constructors]
        eval_envs = make_eval_envs(eval_env_fns)
        best_performance = load_best_performance(self.logging_config.best_policy_dir)
        if (self.eval_config.best_metric_mode or "max").lower() == "min" and best_performance == 0.0:
            best_performance = float("inf")

        start_time = time.time()
        n_updates = 0
        n_collections = 0
        n_samples = 0

        num_collections = getattr(self.alg_config, "num_collections", None)

        try:
            for i, data in enumerate(collector):
                n_collections += 1

                if num_collections is not None and i >= num_collections:
                    training.info(f"Reached maximum number of collections: {num_collections}")
                    break

                try:
                    n_samples += data.reshape(-1).shape[0]
                except Exception:
                    training.warning("Could not determine number of samples in collected data.")
                    pass

                metrics_list = self.algorithm.train_step(
                    batch=data,
                    loss_module=loss_module,
                    optimizer=self.optimizer,
                    replay_buffer=replay_buffer,
                    target_net_updater=target_net_updater,
                    device=self.update_device,
                    n_collections=n_collections,
                    n_updates=n_updates,
                    n_samples=n_samples,
                )

                collector.update_policy_weights_()

                current_buffer_size = len(replay_buffer) if replay_buffer is not None else 0
                n_updates += len(metrics_list)

                if should_log(n_collections, self.logging_config) and metrics_list:
                    log_payload = dict(metrics_list[-1])
                    log_payload["batch/n_updates"] = n_updates
                    log_payload["batch/n_collections"] = n_collections
                    log_payload["batch/n_samples"] = n_samples
                    if current_buffer_size > 0:
                        log_payload["batch/buffer_size"] = current_buffer_size
                    if wandb.run is not None:
                        wandb.log(log_payload, step=n_collections)

                if should_eval(n_collections, self.eval_config):
                    policy.eval()
                    with torch.inference_mode():
                        eval_metrics = run_evaluation(
                            policy,
                            eval_envs,
                            self.eval_config,
                            n_collections=n_collections,
                            n_updates=n_updates,
                            n_samples=n_samples,
                            eval_location=self.eval_location,
                        )
                        if wandb.run is not None and eval_metrics is not None:
                            wandb.log(eval_metrics, step=n_collections)
                    policy.train()

                # Timeout guard (useful for final logging/checkpointing on long runs)
                timeout = getattr(self.alg_config, "timeout", 60 * 60 * 24)
                if time.time() - start_time > timeout:
                    training.info("Training timed out")
                    break
        finally:
            collector.shutdown()
