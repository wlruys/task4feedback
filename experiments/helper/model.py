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


def MultiHeadCategorical(**kwargs):
    return torch.distributions.Categorical(**kwargs).to_event(1)   # equivalent to Independent(base, 1)

def MultiHeadCategorical(**kwargs):
    # kwargs will contain 'logits' read from the TensorDict by Probabilistic* module
    base = torch.distributions.Categorical(**kwargs)          # batch_shape: [..., 64], event_shape: ()
    return torch.distributions.Independent(base, 1)           # reinterpret the last batch dim as event -> joint log_prob

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
) -> tuple[nn.Module, nn.Module, LSTMModule | None]:
        
    graph_config = instantiate(cfg.graph.config)

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
            length=get_length_from_config(graph_config),
            feature_config=feature_cfg,
            _recursive_=False,
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
    print(f"Policy state output dim: {output_dim}")

    print(layers)

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
        out_keys=["action"],
        #distribution_class=torch.distributions.Categorical,
        distribution_class=MultiHeadCategorical,
        return_log_prob=True,
    )

    critic_layers = []
    reference_layers = []

    if hasattr(state_layer, "width") and hasattr(state_layer, "length"):
        critic_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            add_progress=cfg.network.critic.add_progress,
            _recursive_=False,
            width=graph_config.n,
            length=get_length_from_config(graph_config)
        )
        reference_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            add_progress=cfg.network.critic.add_progress,
            _recursive_=False,
            width=graph_config.n,
            length=get_length_from_config(graph_config)
        )
    else:
        critic_state_module = instantiate(
            state_layer,
            feature_config=feature_cfg,
            add_progress=cfg.network.critic.add_progress,
            _recursive_=False,
        )
        reference_state_module  = instantiate(
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
        output_dim=1,
        _recursive_=False,
        
    )
    reference_output_module = instantiate(
        critic_layer,
        input_dim=output_dim,
        output_dim=8,
        _recursive_=False,
    )

    if "input_keys" in cfg.network.critic:
        critic_input_keys = cfg.network.critic.input_keys
        print(f"Critic input keys: {critic_input_keys}")
    else:
        critic_input_keys = state_output_keys

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
        print(
            f"Loaded policy weights from {ckpt_path} into "
            f"{target_module.__class__.__name__} (missing={len(missing)}, unexpected={len(unexpected)})."
        )
        return True
    except Exception as e:
        print(f"Failed to load policy weights into target module: {e}")
        return False