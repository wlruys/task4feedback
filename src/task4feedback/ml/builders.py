from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchrl.modules import ProbabilisticActor, ValueOperator

from task4feedback.exp_utils.model import MultiHeadCategoricalMasked, _timing_options
from task4feedback.ml.base import ActorCriticModule
from task4feedback.ml.wrappers import InferenceTimingProbabilisticActor, LogitInferenceTimingWrapper

from .tdgraph import NamespacingConf, TDGraph, ValidationConf, to_key


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
) -> Tuple[nn.Module, nn.Module, Optional[Any]]:
    """Hydra entrypoint to build independent actor/critic TDGraphs."""

    ns = NamespacingConf.from_cfg(namespacing)
    val = ValidationConf.from_cfg(validation)

    base_rt = runtime or {}
    actor_rt = dict(base_rt)
    if actor_runtime:
        actor_rt.update(actor_runtime)
    critic_rt = dict(base_rt)
    if critic_runtime:
        critic_rt.update(critic_runtime)

    actor_graph = TDGraph.from_config(actor, namespacing=ns, validation=val, runtime=actor_rt)
    critic_graph = TDGraph.from_config(critic, namespacing=ns, validation=val, runtime=critic_rt)

    # Public key overlap check.
    def public_provides(g: TDGraph) -> set:
        keys = set()
        for node in g.nodes.values():
            for k in getattr(node, "provides", []):
                k = to_key(k)
                if ns.is_public(k):
                    keys.add(k)
        return keys

    overlap = public_provides(actor_graph) & public_provides(critic_graph)
    if overlap:
        raise ValueError(f"Actor/Critic public key overlap: {sorted(overlap)}")

    timing_cfg = _timing_options(
        base_rt.get("cfg") if "cfg" in base_rt else OmegaConf.create({})
    )
    actor_cls = InferenceTimingProbabilisticActor if timing_cfg.get("actions") else ProbabilisticActor
    actor_module: nn.Module = actor_graph
    if timing_cfg.get("logits"):
        actor_module = LogitInferenceTimingWrapper(
            actor_module,
            timing_key=timing_cfg.get("logit_key", "logit_inference_time_s"),
            store_in_tensordict=timing_cfg.get("store_in_td", True),
            sync_cuda=timing_cfg.get("sync_cuda", False),
            log_timing=timing_cfg.get("log", False),
            conversion_timing_key=timing_cfg.get("conversion_key", "data_conversion_time_s"),
            subtract_conversion_time=timing_cfg.get("subtract_conversion", True),
        )

    probabilistic_policy = actor_cls(
        module=actor_module,
        in_keys={"logits": "logits", "head_mask": ("observation", "aux", "candidate_mask")},
        out_keys=["action"],
        distribution_class=MultiHeadCategoricalMasked,
        distribution_kwargs={"inactive_action": 0},
        return_log_prob=True,
        **(
            dict(
                timing_key=timing_cfg.get("action_key"),
                store_in_tensordict=timing_cfg.get("store_in_td", True),
                sync_cuda=timing_cfg.get("sync_cuda", False),
                log_timing=timing_cfg.get("log", False),
            )
            if timing_cfg.get("actions")
            else {}
        ),
    )

    value_operator: nn.Module | ValueOperator = critic_graph

    return ActorCriticModule(probabilistic_policy, value_operator), None, None
