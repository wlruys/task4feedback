from task4feedback.ml.models import *
from task4feedback.ml import ActorCriticModule
from typing import Callable
import hydra
import torch
import torch.nn as nn
import tensordict.nn as td_nn
from tensordict import TensorDict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from torchrl.envs import ExplorationType
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule, GRUModule
from pathlib import Path
from task4feedback.graphs.jacobi import get_length_from_config
from torch.distributions import Categorical, Independent, constraints  



class MultiHeadCategoricalMasked(Independent):
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    has_rsample = False

    def __init__(self, *, logits=None, probs=None, head_mask=None, inactive_action: int = -2, validate_args=None):
        base = Categorical(logits=logits, probs=probs, validate_args=validate_args)
        super().__init__(base, reinterpreted_batch_ndims=1, validate_args=validate_args)

        device = base.logits.device if base.logits is not None else base.probs.device
        self._inactive_action = int(inactive_action)

        D = base.logits.shape[-1]

        #print("NEW MultiHeadCategoricalMasked")
        #print(f"HEAD MASK", head_mask)

        if head_mask is None:
            head_mask = torch.ones(base.batch_shape, dtype=torch.bool, device=device)
        else:
            head_mask = head_mask.to(device).to(torch.bool)
            try:
                head_mask = head_mask.expand(base.batch_shape)
            except RuntimeError as e:
                raise ValueError(f"head_mask with shape {head_mask.shape} cannot be broadcast to distribution batch shape {base.batch_shape}") from e
        self._head_mask = head_mask

    def _broadcast_to(self, t: torch.Tensor, target_shape) -> torch.Tensor:
        #print("MASKED MULTIHEAD CATEGORICAL SAMPLE")
        #print(self._head_mask)
        mask = self._head_mask
        lead = len(target_shape) - len(mask.shape)
        if lead < 0:
            raise RuntimeError("Target has fewer dims than head_mask.")
        return mask.view((1,) * lead + mask.shape).expand(target_shape)

    def sample(self, sample_shape=torch.Size()):
        out = self.base_dist.sample(sample_shape)
        mask = self._broadcast_to(out, out.shape)       
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        #print(f"Mask Shape: {mask.shape}, Output Shape: {out.shape}")
        #print(f"Sample Mask: {mask}")
        out = torch.where(mask, out, torch.full_like(out, self._inactive_action))
        #print(f"MultiHeadCategoricalMasked sample: {out}")
        return out
    
    @property
    def mode(self):
        m = self.base_dist.logits.argmax(dim=-1)
        mask = self._broadcast_to(m, m.shape)
        out = torch.where(mask, m, torch.full_like(m, self._inactive_action))
        #print(f"MultiHeadCategoricalMasked mode: {out}")
        #print(f"Mode Mask: {mask}")
        return out
    
    @property
    def mean(self):
        mean = self.base_dist.mean
        mask = self._broadcast_to(mean, mean.shape)
        return torch.where(mask, mean, torch.full_like(mean, self._inactive_action))
    
    @property
    def deterministic_sample(self):
        return self.mode
    
    def log_prob(self, value):
        mask = self._broadcast_to(value, value.shape)      
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        value = torch.where(mask, value, torch.full_like(value, self._inactive_action))
        per_head = self.base_dist.log_prob(value)
        return (per_head * mask.to(per_head.dtype)).sum(dim=-1)
    
    def entropy(self):
        per_head = self.base_dist.entropy()                
        mask = self._broadcast_to(per_head, per_head.shape)
        return (per_head * mask.to(per_head.dtype)).sum(dim=-1)
    

    def with_mask(self, head_mask: torch.Tensor):
        logits = getattr(self.base_dist, "logits", None)
        probs  = getattr(self.base_dist, "probs", None)
        kwargs = {}
        if logits is not None:
            kwargs["logits"] = logits
        elif probs is not None:
            kwargs["probs"] = probs
        return type(self)(head_mask=head_mask,
                          inactive_action=self._inactive_action,
                          **kwargs)


def MultiHeadCategorical(*, head_mask=None, inactive_action: int = -2, **kwargs):
    return MultiHeadCategoricalMasked(head_mask=head_mask,
                                      inactive_action=inactive_action, **kwargs)

# def MultiHeadCategorical(**kwargs):
#     base = torch.distributions.Categorical(**kwargs)
#     return torch.distributions.Independent(base, 1)

def create_actor_critic_models(cfg: DictConfig, feature_cfg: FeatureDimConfig) -> nn.Module:
    layers = cfg.network.layers
    add_device_load = cfg.feature.get("add_device_load", False)


    state_layer = layers.state
    actor_layer = layers.actor
    critic_layer = layers.critic

    policy_state_module = instantiate(
        state_layer,
        feature_config=feature_cfg,
        add_device_load=add_device_load,
        _recursive_=False,
    )

    policy_output_module = instantiate(
        actor_layer,
        input_dim=policy_state_module.output_dim,
        output_dim=cfg.system.config.n_devices,
        _recursive_=False,
    )

    policy_module = nn.Sequential(policy_state_module, policy_output_module)

    critic_state_module = instantiate(
        state_layer,
        feature_config=feature_cfg,
        add_progress=cfg.network.critic.add_progress,
        add_device_load=add_device_load,
        _recursive_=False,
    )

    value_output_module = instantiate(
        critic_layer,
        input_dim=critic_state_module.output_dim,
        output_dim=1,
        _recursive_=False,
    )

    value_module = nn.Sequential(critic_state_module, value_output_module)

    return ActorCriticModule(policy_module, value_module)


def create_td_actor_critic_models(cfg: DictConfig, feature_cfg: FeatureDimConfig) -> tuple[nn.Module, nn.Module, LSTMModule | None]:

    graph_config = instantiate(cfg.graph.config)
    add_device_load = cfg.feature.get("add_device_load", False)

    lstm_mod = None
    layers = cfg.network.layers

    state_layer = layers.state
    actor_layer = layers.actor
    critic_layer = layers.critic
    actor_layers = []
    if hasattr(state_layer, "width") and hasattr(state_layer, "length"):
        print("Using rectangular state layer")
        policy_state_module = instantiate(
            state_layer,
            width=graph_config.n,
            add_device_load=add_device_load,
            n_devices=cfg.system.n_devices,
            length=get_length_from_config(graph_config),
            feature_config=feature_cfg,
            _recursive_=False,
        )
    else:
        policy_state_module = instantiate(
            state_layer,
            add_device_load=add_device_load,
            n_devices=cfg.system.n_devices,
            feature_config=feature_cfg,
            _recursive_=False,
        )

    if hasattr(policy_state_module, "output_keys"):
        state_output_keys = policy_state_module.output_keys
        print(f"State output keys: {state_output_keys}")
    else:
        state_output_keys = ["embed"]

    _td_policy_state = td_nn.TensorDictModule(
        policy_state_module,
        in_keys=["observation"],
        out_keys=state_output_keys,
    )
    actor_layers.append(_td_policy_state)
    output_dim = policy_state_module.output_dim
    print(f"Policy state output dim: {output_dim}")

    if "lstm" in layers:
        print("Using LSTM layer for actor")
        actor_lstm_layer = instantiate(
            layers.lstm,
            input_size=policy_state_module.output_dim,
        )
        output_dim = layers.lstm.hidden_size
        actor_layers.append(actor_lstm_layer)
        lstm_mod = actor_lstm_layer
        print(f"  LSTM hidden size: {layers.lstm.hidden_size}, Output dim: {output_dim}")

    if hasattr(actor_layer, "width") and hasattr(actor_layer, "length"):
        print("Using rectangular actor layer for actor")
        policy_output_module = instantiate(
            actor_layer,
            width=graph_config.n,
            length=get_length_from_config(graph_config),
            input_dim=output_dim,
            output_dim=cfg.system.n_devices - 1,
            _recursive_=False,
        )
    else:
        policy_output_module = instantiate(
            actor_layer,
            input_dim=output_dim,
            output_dim=cfg.system.n_devices - 1,
            _recursive_=False,
        )

    if hasattr(policy_output_module, "input_keys"):
        actor_input_keys = policy_output_module.input_keys
        print(f"Actor input keys: {actor_input_keys}")
    else:
        actor_input_keys = ["embed"]

    actor_input_keys = ["observation"] + actor_input_keys

    _td_policy_output = td_nn.TensorDictModule(
        policy_output_module,
        in_keys=actor_input_keys,
        out_keys=["logits"],
    )
    actor_layers.append(_td_policy_output)

    policy_module = td_nn.TensorDictSequential(*actor_layers, inplace=True)

    probabilistic_policy = ProbabilisticActor(
        module=policy_module,
        in_keys={"logits" : "logits", "head_mask" : ("observation", "aux", "candidate_mask")},
        out_keys=["action"],
        distribution_class=MultiHeadCategoricalMasked,
        distribution_kwargs={"inactive_action": 0},
        return_log_prob=True,
    )

    critic_layers = []
    reference_layers = []

    if hasattr(state_layer, "width") and hasattr(state_layer, "length"):
        critic_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            add_progress=cfg.network.critic.add_progress,
            add_device_load=add_device_load,
            n_devices=cfg.system.n_devices,
            _recursive_=False,
            width=graph_config.n,
            length=get_length_from_config(graph_config),
        )
        reference_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            add_progress=cfg.network.critic.add_progress,
            add_device_load=add_device_load,
            n_devices=cfg.system.n_devices,
            _recursive_=False,
            width=graph_config.n,
            length=get_length_from_config(graph_config),
        )
    else:
        critic_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            add_device_load=add_device_load,
            n_devices=cfg.system.n_devices,
            add_progress=cfg.network.critic.add_progress,
            _recursive_=False,
        )
        reference_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            add_device_load=add_device_load,
            n_devices=cfg.system.n_devices,
            add_progress=cfg.network.critic.add_progress,
            _recursive_=False,
        )

    _td_critic_state = td_nn.TensorDictModule(
        critic_state_module,
        in_keys=["observation"],
        out_keys=state_output_keys,
    )
    _td_reference_state = td_nn.TensorDictModule(
        critic_state_module,
        in_keys=["observation"],
        out_keys=state_output_keys,
    )

    output_dim = critic_state_module.output_dim
    critic_layers.append(_td_critic_state)
    reference_layers.append(_td_reference_state)

    if "lstm" in layers:
        critic_lstm_layer = instantiate(
            layers.lstm,
            input_size=critic_state_module.output_dim,
        )
        output_dim = layers.lstm.hidden_size
        critic_layers.append(critic_lstm_layer)

    critic_output_module = instantiate(
        critic_layer,
        input_dim=output_dim,
        add_device_load=add_device_load,
        n_devices=cfg.system.n_devices,
        add_progress=cfg.network.critic.add_progress,
        output_dim=1,
        _recursive_=False,
    )
    reference_output_module = instantiate(
        critic_layer,
        input_dim=output_dim,
        add_device_load=add_device_load,
        n_devices=cfg.system.n_devices,
        add_progress=cfg.network.critic.add_progress,
        output_dim=8,
        _recursive_=False,
    )

    if "input_keys" in cfg.network.critic:
        critic_input_keys = cfg.network.critic.input_keys
        print(f"Critic input keys: {critic_input_keys}")
    else:
        critic_input_keys = state_output_keys

    critic_input_keys = ["observation"] + critic_input_keys

    _td_critic_output = td_nn.TensorDictModule(
        critic_output_module,
        in_keys=critic_input_keys,
        out_keys=["state_value"],
    )
    _td_reference_output = td_nn.TensorDictModule(
        reference_output_module,
        in_keys=critic_input_keys,
        out_keys=["reference_state"],
    )
    critic_layers.append(_td_critic_output)
    reference_layers.append(_td_reference_output)

    critic_module = td_nn.TensorDictSequential(*critic_layers, inplace=True)
    reference_module = td_nn.TensorDictSequential(*reference_layers, inplace=True)
    value_operator = critic_module

    return ActorCriticModule(probabilistic_policy, value_operator), reference_module, lstm_mod


def load_policy_from_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> bool:
    """Load a policy module state_dict from `ckpt_path` into `model`.

    The checkpoint may be either a full training checkpoint with a `policy_module`
    key (as saved by `save_checkpoint`) or a raw state_dict for the policy itself.
    Returns True if parameters were loaded; False otherwise.
    """
    # We trust local checkpoints produced by our code; allow full unpickling.
    # If you are loading an untrusted checkpoint, you may wish to set weights_only=True for safety.
    try:
        # We trust local checkpoints produced by our code; allow full unpickling.
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        return False

    # Determine which state_dict to use
    state_dict = None
    if isinstance(obj, dict) and "policy_module" in obj and isinstance(obj["policy_module"], dict):
        state_dict = obj["policy_module"]
    elif isinstance(obj, dict):
        # Heuristic: treat as a raw state_dict if values are tensors
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj

    if state_dict is None:
        print(f"Checkpoint at {ckpt_path} does not contain a recognizable policy state_dict.")
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
        print(f"Loaded policy weights from {ckpt_path} into " f"{target_module.__class__.__name__} (missing={len(missing)}, unexpected={len(unexpected)}).")
        return True
    except Exception as e:
        print(f"Failed to load policy weights into target module: {e}")
        return False
