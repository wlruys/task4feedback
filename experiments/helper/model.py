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

def create_actor_critic_models(
    cfg: DictConfig, feature_cfg: FeatureDimConfig
) -> nn.Module:
    layers = cfg.network.layers

    state_layer = layers.state
    actor_layer = layers.actor
    critic_layer = layers.critic

    policy_state_module = instantiate(
        state_layer,
        feature_config=feature_cfg,
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


def create_td_actor_critic_models(
    cfg: DictConfig, feature_cfg: FeatureDimConfig
) -> tuple[nn.Module, LSTMModule | None]:
    lstm_mod = None
    layers = cfg.network.layers

    state_layer = layers.state
    actor_layer = layers.actor
    critic_layer = layers.critic
    actor_layers = []
    if hasattr(state_layer, "width"):
        policy_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            _recursive_=False,
            width=cfg.graph.config.n,
        )
    else:
        policy_state_module = instantiate(
            state_layer,
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

    if "lstm" in layers:
        actor_lstm_layer = instantiate(
            layers.lstm,
            input_size=policy_state_module.output_dim,
        )
        output_dim = layers.lstm.hidden_size
        actor_layers.append(actor_lstm_layer)
        lstm_mod = actor_lstm_layer
    if hasattr(actor_layer, "width"):
        policy_output_module = instantiate(
            actor_layer,
            input_dim=output_dim,
            output_dim=cfg.system.n_devices - 1,
            _recursive_=False,
            width=cfg.graph.config.n,
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

    _td_policy_output = td_nn.TensorDictModule(
        policy_output_module,
        in_keys=actor_input_keys,
        out_keys=["logits"],
    )
    actor_layers.append(_td_policy_output)

    policy_module = td_nn.TensorDictSequential(*actor_layers, inplace=True)

    probabilistic_policy = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    critic_layers = []
    if hasattr(state_layer, "width"):
        critic_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            add_progress=cfg.network.critic.add_progress,
            _recursive_=False,
            width=cfg.graph.config.n,
        )
    else:
        critic_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            add_progress=cfg.network.critic.add_progress,
            _recursive_=False,
        )

    _td_critic_state = td_nn.TensorDictModule(
        critic_state_module,
        in_keys=["observation"],
        out_keys=state_output_keys,
    )
    output_dim = critic_state_module.output_dim
    critic_layers.append(_td_critic_state)

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
        output_dim=1,
        _recursive_=False,
    )

    _td_critic_output = td_nn.TensorDictModule(
        critic_output_module,
        in_keys=["embed"],
        out_keys=["state_value"],
    )
    critic_layers.append(_td_critic_output)

    critic_module = td_nn.TensorDictSequential(*critic_layers, inplace=True)
    value_operator = critic_module

    return ActorCriticModule(probabilistic_policy, value_operator), lstm_mod

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
        print(
            f"Loaded policy weights from {ckpt_path} into "
            f"{target_module.__class__.__name__} (missing={len(missing)}, unexpected={len(unexpected)})."
        )
        return True
    except Exception as e:
        print(f"Failed to load policy weights into target module: {e}")
        return False