from .definitions import *
from task4feedback.ml.models import *
from task4feedback.ml import ActorCriticModule
from typing import Callable
import hydra
import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDictBase, unravel_key
from collections.abc import Sequence
import tensordict.nn as td_nn
from tensordict.nn import TensorDictModuleBase, dispatch, CompositeDistribution, InteractionType, TensorDictModule, TensorDictModuleWrapper, TensorDictSequential
from tensordict import TensorDict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import ExplorationType
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule, GRUModule
from pathlib import Path
from .logging_helpers import get_helper_logger
from tensordict.utils import expand_as_right, NestedKey
from torch.distributions import Categorical, Independent, constraints  

logger = get_helper_logger(__name__)



def _timing_options(cfg: DictConfig) -> dict:
    return {
        "logits": OmegaConf.select(cfg, "timing.logits", default=False),
        "actions": OmegaConf.select(cfg, "timing.actions", default=False),
        "logit_key": OmegaConf.select(cfg, "timing.logit_key", default="logit_inference_time_s"),
        "action_key": OmegaConf.select(cfg, "timing.action_key", default="action_inference_time_s"),
        "store_in_td": OmegaConf.select(cfg, "timing.store_in_tensordict", default=True),
        "sync_cuda": OmegaConf.select(cfg, "timing.sync_cuda", default=False),
        "log": OmegaConf.select(cfg, "timing.log", default=False),
        "conversion_key": OmegaConf.select(cfg, "timing.conversion_key", default="data_conversion_time_s"),
        "subtract_conversion": OmegaConf.select(cfg, "timing.subtract_conversion", default=True),
    }


def MultiHeadCategorical(**kwargs):
    base = torch.distributions.Categorical(**kwargs)
    return torch.distributions.Independent(base, 1)


class MultiHeadCategoricalMasked(Independent):
    """
    Multi-head categorical with dynamic head masking.
    """

    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    has_rsample = False

    def __init__(
        self,
        logits: Tensor | None = None,
        probs: Tensor | None = None,
        head_mask: Tensor | None = None,
        inactive_action: int = 0,
        validate_args: bool | None = None,
    ) -> None:
        base = Categorical(logits=logits, probs=probs, validate_args=validate_args)
        super().__init__(base, reinterpreted_batch_ndims=1, validate_args=validate_args)
        
        self._inactive = inactive_action
        self._logits = base.logits
        self._device = self._logits.device
        self._batch_shape = base.batch_shape
        
        # Coerce mask once
        if head_mask is None:
            self._mask = None  # Fast path: no masking
        else:
            self._mask = head_mask.to(device=self._device, dtype=torch.bool)
            if self._mask.shape != self._batch_shape:
                self._mask = self._mask.expand(self._batch_shape)

    def _expand_mask(self, shape: tuple[int, ...]) -> Tensor:
        """Expand mask to sample shape."""
        m = self._mask
        lead = len(shape) - m.ndim
        if lead > 0:
            m = m.view((1,) * lead + m.shape).expand(shape)
        return m

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        out = self.base_dist.sample(sample_shape)
        if self._mask is None:
            return out
        mask = self._expand_mask(out.shape) if out.shape != self._mask.shape else self._mask
        return out.masked_fill_(~mask, self._inactive)

    @property
    def mode(self) -> Tensor:
        m = self._logits.argmax(dim=-1)
        if self._mask is None:
            return m
        return m.masked_fill_(~self._mask, self._inactive)

    @property
    def deterministic_sample(self) -> Tensor:
        return self.mode

    @property
    def mean(self) -> Tensor:
        m = self.base_dist.mean
        if self._mask is None:
            return m
        return m.masked_fill(~self._mask, float(self._inactive))

    def log_prob(self, value: Tensor) -> Tensor:
        if self._mask is None:
            return self.base_dist.log_prob(value).sum(dim=-1)
        mask = self._expand_mask(value.shape) if value.shape != self._mask.shape else self._mask
        # Use 0 for masked positions
        lp = self.base_dist.log_prob(value.masked_fill(~mask, 0))
        return lp.mul_(mask.float()).sum(dim=-1)

    def entropy(self) -> Tensor:
        ent = self.base_dist.entropy()
        if self._mask is None:
            return ent.sum(dim=-1)
        return ent.mul_(self._mask.float()).sum(dim=-1)

def create_td_actor_critic_models(cfg: DictConfig, feature_cfg: FeatureDimConfig) -> tuple[nn.Module, nn.Module, LSTMModule | None]:
    # New DAG-based model configs expose a top-level Hydra builder target.
    model_cfg = OmegaConf.select(cfg, "models", default=None)
    if model_cfg is None:
        model_cfg = OmegaConf.select(cfg, "network", default=None)  # legacy name

    if model_cfg is not None and OmegaConf.select(model_cfg, "_target_", default=None) is not None:
        add_device_load = cfg.feature.get("add_device_load", False)
        length = OmegaConf.select(cfg.graph.config, "length", default=None)
        width = OmegaConf.select(cfg.graph.config, "width", default=None)

        runtime_base = dict(
            cfg=cfg,
            feature_config=feature_cfg,
            width=width,
            length=length,
            n_devices=cfg.system.n_devices,
            action_dim=cfg.system.n_devices - 1,
        )
        actor_runtime = dict(runtime_base)
        critic_runtime = dict(runtime_base)
        actor_runtime["output_dim"] = int(runtime_base["action_dim"])
        # Critic should always be conditioned on progress features.
        critic_add_progress_cfg = OmegaConf.select(model_cfg, "critic.add_progress", default=True)
        if critic_add_progress_cfg is False:
            logger.warning(
                "Config sets models.critic.add_progress=false, but critic networks always use progress features."
            )
        critic_runtime.update(
            add_progress=True,
            add_device_load=add_device_load,
            output_dim=1,
        )

        model, reference, lstm_mod = instantiate(
            model_cfg,
            runtime=runtime_base,
            actor_runtime=actor_runtime,
            critic_runtime=critic_runtime,
            _recursive_=False,
        )
        return model, reference, lstm_mod

    raise ValueError("cfg.models (or legacy cfg.network) is missing _target_; old schema no longer supported.")


def load_policy_from_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> bool:
    """Load a policy module state_dict from `ckpt_path` into `model`.

    The checkpoint may be either a full training checkpoint with a `policy_module`
    key (as saved by `save_checkpoint`) or a raw state_dict for the policy itself.
    Returns True if parameters were loaded; False otherwise.

    If model fingerprints are available in the checkpoint, validates that the
    architecture matches before loading.
    """
    # We trust local checkpoints produced by our code; allow full unpickling.
    # If you are loading an untrusted checkpoint, you may wish to set weights_only=True for safety.
    try:
        # We trust local checkpoints produced by our code; allow full unpickling.
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", ckpt_path, e)
        return False

    # Determine which state_dict to use
    state_dict = None
    if isinstance(obj, dict) and "policy_module" in obj and isinstance(obj["policy_module"], dict):
        state_dict = obj["policy_module"]

        # Validate fingerprint if available
        if "model_fingerprints" in obj:
            from task4feedback.ml.rl_utils import compute_model_fingerprint

            # Find policy module in model
            candidate_attrs = ["policy_module", "policy", "actor", "pi", "actor_net"]
            policy_module = None
            for attr in candidate_attrs:
                if hasattr(model, attr):
                    m = getattr(model, attr)
                    if isinstance(m, torch.nn.Module):
                        policy_module = m
                        break

            if policy_module is None:
                policy_module = model

            current_fingerprint = compute_model_fingerprint(policy_module)
            saved_fingerprint = obj["model_fingerprints"].get("policy")

            if saved_fingerprint and current_fingerprint != saved_fingerprint:
                logger.warning(
                    "Model fingerprint mismatch: checkpoint=%s current=%s. "
                    "This checkpoint may be from a different architecture. "
                    "Proceeding with load_state_dict(strict=False)...",
                    saved_fingerprint,
                    current_fingerprint,
                )

    elif isinstance(obj, dict):
        # Heuristic: treat as a raw state_dict if values are tensors
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj

    if state_dict is None:
        logger.warning(
            "Checkpoint at %s does not contain a recognizable policy state_dict.",
            ckpt_path,
        )
        return False

    # Common attribute names for the policy head/module
    candidate_attrs = ["policy_module", "policy", "actor", "pi", "actor_net"]
    target_module = None
    for attr in candidate_attrs:
        if hasattr(model, attr):
            m = getattr(model, attr)
            if isinstance(m, torch.nn.Module):
                target_module = m
                break

    # Fallback: try loading into the model itself
    if target_module is None:
        target_module = model

    try:
        missing, unexpected = target_module.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded policy weights from %s into %s (missing=%s, unexpected=%s).",
            ckpt_path,
            target_module.__class__.__name__,
            len(missing),
            len(unexpected),
        )
        return True
    except Exception as e:
        logger.warning(
            "Failed to load policy weights into target module: %s", e
        )
        return False
