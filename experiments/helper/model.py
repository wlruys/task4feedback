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

    # remove "add_progress" key from critic_layer config
    critic_layer
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

    policy_state_module = instantiate(
        state_layer,
        feature_config=feature_cfg,
        _recursive_=False,
    )

    _td_policy_state = td_nn.TensorDictModule(
        policy_state_module,
        in_keys=["observation"],
        out_keys=["embed"],
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

    policy_output_module = instantiate(
        actor_layer,
        input_dim=output_dim,
        output_dim=cfg.system.n_devices - 1,
        _recursive_=False,
    )

    _td_policy_output = td_nn.TensorDictModule(
        policy_output_module,
        in_keys=["embed"],
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

    critic_state_module = instantiate(
        state_layer,
        feature_config=feature_cfg,
        add_progress=cfg.network.critic.add_progress,
        _recursive_=False,
    )

    _td_critic_state = td_nn.TensorDictModule(
        critic_state_module,
        in_keys=["observation"],
        out_keys=["embed"],
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
