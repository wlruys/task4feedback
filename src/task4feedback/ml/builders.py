from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchrl.modules import ProbabilisticActor, ValueOperator

from task4feedback.exp_utils.model import _timing_options
from task4feedback.ml.base import UnifiedRLModule
from task4feedback.ml.models.distributions import MultiHeadCategoricalMasked
from task4feedback.ml.wrappers import InferenceTimingProbabilisticActor, LogitInferenceTimingWrapper

from .tdgraph import NamespacingConf, TDGraph, ValidationConf, to_key


def _get_distribution_config(base_rt: dict, component_rt: dict | None = None) -> dict:
    """Extract distribution configuration from runtime parameters.

    Returns a dict with:
        - distribution_class: Class to use for distribution (default: MultiHeadCategoricalMasked)
        - distribution_kwargs: Kwargs for distribution (default: {"inactive_action": 0})
        - mask_key_name: Name of mask key in in_keys (default: "head_mask")
        - mask_key_value: Tuple/str value for mask key (default: ("observation", "aux", "candidate_mask"))
        - return_log_prob: Whether to return log probabilities (default: True)
    """
    rt = dict(base_rt)
    if component_rt:
        rt.update(component_rt)

    dist_cfg = rt.get("distribution", {})

    return {
        "distribution_class": dist_cfg.get("class", MultiHeadCategoricalMasked),
        "distribution_kwargs": dist_cfg.get("kwargs", {"inactive_action": 0}),
        "mask_key_name": dist_cfg.get("mask_key_name", "head_mask"),
        "mask_key_value": dist_cfg.get("mask_key_value", ("observation", "aux", "candidate_mask")),
        "return_log_prob": dist_cfg.get("return_log_prob", True),
    }


def _wrap_policy_with_distribution(
    policy_graph: TDGraph,
    base_rt: dict,
    component_rt: dict | None = None,
) -> nn.Module:
    """Wrap a policy TDGraph with timing wrappers and ProbabilisticActor.

    Args:
        policy_graph: TDGraph for policy network
        base_rt: Base runtime parameters
        component_rt: Component-specific runtime parameters

    Returns:
        Policy module wrapped with timing and distribution (or returned
        directly when actor_kind is set to deterministic).
    """
    timing_cfg = _timing_options(
        base_rt.get("cfg") if "cfg" in base_rt else OmegaConf.create({})
    )

    actor_kind = (component_rt or {}).get("actor_kind", base_rt.get("actor_kind", "probabilistic"))

    # Apply logit timing wrapper if configured
    policy_module: nn.Module = policy_graph
    if timing_cfg.get("logits"):
        policy_module = LogitInferenceTimingWrapper(
            policy_module,
            timing_key=timing_cfg.get("logit_key", "logit_inference_time_s"),
            store_in_tensordict=timing_cfg.get("store_in_td", True),
            sync_cuda=timing_cfg.get("sync_cuda", False),
            log_timing=timing_cfg.get("log", False),
            conversion_timing_key=timing_cfg.get("conversion_key", "data_conversion_time_s"),
            subtract_conversion_time=timing_cfg.get("subtract_conversion", True),
        )

    if actor_kind == "deterministic":
        return policy_module

    # Get distribution configuration
    dist_cfg = _get_distribution_config(base_rt, component_rt)

    # Choose actor class based on timing configuration
    actor_cls = InferenceTimingProbabilisticActor if timing_cfg.get("actions") else ProbabilisticActor

    # Build timing kwargs if needed
    timing_kwargs = {}
    if timing_cfg.get("actions"):
        timing_kwargs = dict(
            timing_key=timing_cfg.get("action_key"),
            store_in_tensordict=timing_cfg.get("store_in_td", True),
            sync_cuda=timing_cfg.get("sync_cuda", False),
            log_timing=timing_cfg.get("log", False),
        )

    # Wrap in ProbabilisticActor
    probabilistic_policy = actor_cls(
        module=policy_module,
        in_keys={"logits": "logits", dist_cfg["mask_key_name"]: dist_cfg["mask_key_value"]},
        out_keys=["action"],
        distribution_class=dist_cfg["distribution_class"],
        distribution_kwargs=dist_cfg["distribution_kwargs"],
        return_log_prob=dist_cfg["return_log_prob"],
        **timing_kwargs,
    )

    return probabilistic_policy


def _check_public_key_overlap(
    graphs: list[tuple[str, TDGraph]],
    ns: NamespacingConf,
) -> None:
    """Check for public key overlaps between multiple TDGraphs.

    Args:
        graphs: List of (name, graph) tuples to check
        ns: Namespacing configuration

    Raises:
        ValueError: If any public keys overlap between graphs
    """
    def public_provides(g: TDGraph) -> set:
        keys = set()
        for node in g.nodes.values():
            for k in getattr(node, "provides", []):
                k = to_key(k)
                if ns.is_public(k):
                    keys.add(k)
        return keys

    # Check all pairs for overlaps
    for i, (name1, graph1) in enumerate(graphs):
        keys1 = public_provides(graph1)
        for name2, graph2 in graphs[i+1:]:
            keys2 = public_provides(graph2)
            overlap = keys1 & keys2
            if overlap:
                raise ValueError(
                    f"{name1}/{name2} public key overlap: {sorted(overlap)}"
                )


def build_unified_model(
    *,
    policy: DictConfig | None = None,
    value: DictConfig | None = None,
    qvalue: DictConfig | None = None,
    namespacing: DictConfig | Mapping[str, Any] | None = None,
    validation: DictConfig | Mapping[str, Any] | None = None,
    runtime: dict | None = None,
    policy_runtime: dict | None = None,
    value_runtime: dict | None = None,
    qvalue_runtime: dict | None = None,
    **_ignored,
) -> Tuple[nn.Module, None, None]:
    """Unified builder for all RL model architectures.

    Inspects the provided DictConfig and builds any combination of policy,
    value, and qvalue networks. Returns UnifiedRLModule containing the
    built components.

    This replaces build_actor_critic, build_off_policy_model, and build_model
    with a single unified pathway that reads the Hydra config and builds
    whatever components are defined.

    Args:
        policy: Policy network configuration (if building actor/policy)
        value: Value network configuration (if building critic/value)
        qvalue: Q-value network configuration (if building Q-network)
        namespacing: TDGraph namespacing configuration
        validation: TDGraph validation configuration
        runtime: Base runtime parameters for all networks (includes distribution config)
        policy_runtime: Additional runtime parameters for policy network
        value_runtime: Additional runtime parameters for value network
        qvalue_runtime: Additional runtime parameters for qvalue network
        **_ignored: Ignored parameters for backward compatibility

    Returns:
        Tuple of (UnifiedRLModule, None, None)

    Example runtime distribution config:
        runtime = {
            "distribution": {
                "class": MultiHeadCategoricalMasked,  # or MaskedCategorical
                "kwargs": {"inactive_action": 0},
                "mask_key_name": "head_mask",  # or "mask"
                "mask_key_value": ("observation", "aux", "candidate_mask"),
                "return_log_prob": True,  # or False
            }
        }
    """
    ns = NamespacingConf.from_cfg(namespacing)
    val = ValidationConf.from_cfg(validation)
    base_rt = runtime or {}

    # Build policy network if defined
    policy_module = None
    policy_graph = None
    if policy is not None:
        p_rt = dict(base_rt)
        if policy_runtime:
            p_rt.update(policy_runtime)

        policy_graph = TDGraph.from_config(policy, namespacing=ns, validation=val, runtime=p_rt)
        policy_module = _wrap_policy_with_distribution(policy_graph, base_rt, policy_runtime)

    # Build value network if defined
    value_module = None
    value_graph = None
    if value is not None:
        v_rt = dict(base_rt)
        if value_runtime:
            v_rt.update(value_runtime)
        value_graph = TDGraph.from_config(value, namespacing=ns, validation=val, runtime=v_rt)
        value_module = value_graph

    # Build qvalue network if defined
    qvalue_module = None
    qvalue_graph = None
    if qvalue is not None:
        q_rt = dict(base_rt)
        if qvalue_runtime:
            q_rt.update(qvalue_runtime)
        qvalue_graph = TDGraph.from_config(qvalue, namespacing=ns, validation=val, runtime=q_rt)
        qvalue_module = qvalue_graph

    # Check for public key overlaps between all built graphs
    graphs = []
    if policy_graph is not None:
        graphs.append(("policy", policy_graph))
    if value_graph is not None:
        graphs.append(("value", value_graph))
    if qvalue_graph is not None:
        graphs.append(("qvalue", qvalue_graph))

    if len(graphs) > 1:
        _check_public_key_overlap(graphs, ns)

    return UnifiedRLModule(policy=policy_module, value=value_module, qvalue=qvalue_module), None, None


# Backward compatibility aliases - these now delegate to the unified builder
def build_actor_critic(
    *,
    actor: DictConfig,
    critic: DictConfig,
    namespacing: DictConfig | Mapping[str, Any] | None = None,
    validation: DictConfig | Mapping[str, Any] | None = None,
    runtime: dict | None = None,
    actor_runtime: dict | None = None,
    critic_runtime: dict | None = None,
    **_ignored,
) -> Tuple[nn.Module, None, None]:
    """Legacy builder for actor/critic models - delegates to build_unified_model.

    Maintained for backward compatibility with existing configs that use
    _target_: build_actor_critic.
    """
    # Create a copy to avoid modifying OmegaConf-wrapped dicts
    # Don't include class objects that OmegaConf can't wrap
    base_rt = dict(runtime) if runtime else {}
    if "distribution" not in base_rt:
        base_rt["distribution"] = {
            "kwargs": {"inactive_action": 0},
            "mask_key_name": "head_mask",
            "mask_key_value": ("observation", "aux", "candidate_mask"),
            "return_log_prob": True,
        }

    return build_unified_model(
        policy=actor,
        value=critic,
        qvalue=None,
        namespacing=namespacing,
        validation=validation,
        runtime=base_rt,
        policy_runtime=actor_runtime,
        value_runtime=critic_runtime,
        **_ignored,
    )


def build_off_policy_model(
    *,
    actor: DictConfig | None = None,
    qvalue: DictConfig | None = None,
    value: DictConfig | None = None,
    namespacing: DictConfig | Mapping[str, Any] | None = None,
    validation: DictConfig | Mapping[str, Any] | None = None,
    runtime: dict | None = None,
    actor_runtime: dict | None = None,
    qvalue_runtime: dict | None = None,
    value_runtime: dict | None = None,
    **_ignored,
) -> Tuple[nn.Module, None, None]:
    """Legacy builder for off-policy models - delegates to build_unified_model.

    Maintained for backward compatibility with existing configs that use
    _target_: build_off_policy_model.
    """
    # Create a copy to avoid modifying OmegaConf-wrapped dicts
    # Don't include class objects that OmegaConf can't wrap
    base_rt = dict(runtime) if runtime else {}
    if actor is not None and "distribution" not in base_rt:
        base_rt["distribution"] = {
            "kwargs": {"inactive_action": 0},
            "mask_key_name": "head_mask",
            "mask_key_value": ("observation", "aux", "candidate_mask"),
            "return_log_prob": True,  # SAC/off-policy needs sample_log_prob for temperature/actor losses
        }

    return build_unified_model(
        policy=actor,
        value=value,
        qvalue=qvalue,
        namespacing=namespacing,
        validation=validation,
        runtime=base_rt,
        policy_runtime=actor_runtime,
        value_runtime=value_runtime,
        qvalue_runtime=qvalue_runtime,
        **_ignored,
    )


def build_model(
    *,
    policy: DictConfig | None = None,
    value: DictConfig | None = None,
    qvalue: DictConfig | None = None,
    namespacing: DictConfig | Mapping[str, Any] | None = None,
    validation: DictConfig | Mapping[str, Any] | None = None,
    runtime: dict | None = None,
    actor_runtime: dict | None = None,
    policy_runtime: dict | None = None,
    value_runtime: dict | None = None,
    qvalue_runtime: dict | None = None,
    **_ignored,
) -> Tuple[nn.Module, None, None]:
    """Legacy generic builder - delegates to build_unified_model.

    Maintained for backward compatibility with existing configs that use
    _target_: build_model.
    """
    # Don't modify the incoming runtime dict (it may be from OmegaConf)
    # Create a copy and avoid storing class objects that OmegaConf can't wrap
    base_rt = dict(runtime) if runtime else {}
    if policy is not None and "distribution" not in base_rt:
        # Don't include "class" key here - _get_distribution_config will provide the default
        base_rt["distribution"] = {
            "kwargs": {"inactive_action": 0, "reinterpreted_batch_ndims": 0},
            "mask_key_name": "head_mask",
            "mask_key_value": ("observation", "aux", "candidate_mask"),
            "return_log_prob": True,  # Changed to True - needed for PPO and on-policy algorithms
        }

    # Support legacy actor_runtime by mapping it to policy_runtime.
    # If both are provided, policy_runtime takes precedence.
    merged_policy_runtime = None
    if actor_runtime or policy_runtime:
        merged_policy_runtime = {}
        if actor_runtime:
            merged_policy_runtime.update(actor_runtime)
        if policy_runtime:
            merged_policy_runtime.update(policy_runtime)

    return build_unified_model(
        policy=policy,
        value=value,
        qvalue=qvalue,
        namespacing=namespacing,
        validation=validation,
        runtime=base_rt,
        policy_runtime=merged_policy_runtime,
        value_runtime=value_runtime,
        qvalue_runtime=qvalue_runtime,
        **_ignored,
    )
