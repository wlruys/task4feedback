from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from tensordict import TensorDict
from torchrl.envs.transforms import Transform


TensorKey = tuple[str, ...]


def _as_key_tuple(key: Sequence[str] | str) -> TensorKey:
    if isinstance(key, str):
        return (key,)
    return tuple(key)


def _as_key_tuple_list(
    keys: Sequence[Sequence[str] | str] | Sequence[str] | str,
) -> tuple[TensorKey, ...]:
    if isinstance(keys, str):
        return (_as_key_tuple(keys),)
    if len(keys) == 0:
        return ()
    first = keys[0]
    if isinstance(first, str):
        return (_as_key_tuple(keys),)
    return tuple(_as_key_tuple(key) for key in keys)


def _lookup_first(td: TensorDict, keys: tuple[TensorKey, ...]):
    for key in keys:
        value = td.get(key, default=None)
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class NodeSpec:
    name: str
    attr_key: TensorKey | Sequence[str] | str
    count_keys: tuple[TensorKey, ...] | Sequence[Sequence[str] | str] | Sequence[str] | str = ()
    required: bool = True

    def __post_init__(self):
        object.__setattr__(self, "attr_key", _as_key_tuple(self.attr_key))
        object.__setattr__(self, "count_keys", _as_key_tuple_list(self.count_keys))


@dataclass(frozen=True)
class EdgeSpec:
    name: str
    row0_node: str
    row1_node: str
    idx_keys: tuple[TensorKey, ...] | Sequence[Sequence[str] | str] | Sequence[str] | str
    count_keys: tuple[TensorKey, ...] | Sequence[Sequence[str] | str] | Sequence[str] | str = ()
    required: bool = True

    def __post_init__(self):
        object.__setattr__(self, "idx_keys", _as_key_tuple_list(self.idx_keys))
        object.__setattr__(self, "count_keys", _as_key_tuple_list(self.count_keys))


@dataclass(frozen=True)
class HeteroGraphSchema:
    nodes: tuple[NodeSpec, ...] | Sequence[NodeSpec]
    edges: tuple[EdgeSpec, ...] | Sequence[EdgeSpec]
    graph_batch_key: TensorKey | Sequence[str] | str = ("graph_batch",)

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "graph_batch_key", _as_key_tuple(self.graph_batch_key))


def _reshape_by_batch(x: torch.Tensor, batch_shape: torch.Size, B_flat: int) -> torch.Tensor:
    if len(batch_shape) == 0:
        return x.unsqueeze(0)
    return x.reshape(B_flat, *x.shape[len(batch_shape) :])


def _reshape_count_tensor(count: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    if len(batch_shape) == 0:
        return count.reshape(())
    return count.reshape(*batch_shape)


def _reshape_edge_index_tensor(edge_index: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    if len(batch_shape) == 0:
        return edge_index.reshape(2, edge_index.shape[-1])
    return edge_index.reshape(*batch_shape, 2, edge_index.shape[-1])


def _compact_invalid_edges(
    row0: torch.Tensor,
    row1: torch.Tensor,
    edge_mask: torch.Tensor,
    edge_count: torch.Tensor,
    compact_graph_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, E_max = edge_mask.shape
    bad_graphs = compact_graph_mask.nonzero(as_tuple=False).reshape(-1).tolist()
    if len(bad_graphs) == 0:
        return row0, row1, edge_count

    for graph_id in bad_graphs:
        valid_slots = edge_mask[graph_id].nonzero(as_tuple=False).reshape(-1)
        valid_count = int(valid_slots.numel())
        if valid_count > 0:
            row0[graph_id, :valid_count] = row0[graph_id, valid_slots]
            row1[graph_id, :valid_count] = row1[graph_id, valid_slots]
        if valid_count < E_max:
            row0[graph_id, valid_count:] = 0
            row1[graph_id, valid_count:] = 0
        edge_count[graph_id] = valid_count

    return row0, row1, edge_count


def build_graph_batch_from_schema(
    observation: TensorDict,
    schema: HeteroGraphSchema,
    *,
    strict: bool = False,
) -> TensorDict:
    batch_shape = observation.batch_size
    B_flat = int(math.prod(batch_shape)) if len(batch_shape) > 0 else 1

    node_specs = {node.name: node for node in schema.nodes}
    if len(node_specs) != len(schema.nodes):
        raise ValueError("Node names in HeteroGraphSchema must be unique.")

    node_counts: dict[str, torch.Tensor] = {}
    node_offsets: dict[str, torch.Tensor] = {}
    node_max_sizes: dict[str, int] = {}
    node_out = TensorDict({}, batch_size=batch_shape)

    for node in schema.nodes:
        attr = observation.get(node.attr_key, default=None)
        if attr is None:
            if node.required:
                raise KeyError(f"Missing required node attr key: {node.attr_key}")
            continue

        attr_batched = _reshape_by_batch(attr, batch_shape, B_flat)
        if attr_batched.dim() < 2:
            raise RuntimeError(
                f"Node attr tensor for {node.name} must have at least rank 2 after batching, "
                f"got shape {tuple(attr_batched.shape)}."
            )

        max_nodes = int(attr_batched.shape[1])
        count_raw = _lookup_first(observation, node.count_keys) if len(node.count_keys) > 0 else None
        if count_raw is None:
            count = torch.full((B_flat,), max_nodes, dtype=torch.long, device=attr_batched.device)
        else:
            count = count_raw.reshape(B_flat, -1)[:, 0].to(device=attr_batched.device, dtype=torch.long)

        invalid_count = (count < 0) | (count > max_nodes)
        if bool(invalid_count.any()):
            if strict:
                bad_graphs = invalid_count.nonzero(as_tuple=False).reshape(-1).tolist()
                raise RuntimeError(
                    f"Invalid node count for {node.name}. Expected [0, {max_nodes}], "
                    f"invalid entries at graph ids {bad_graphs}."
                )
            count = count.clamp(min=0, max=max_nodes)

        node_counts[node.name] = count
        node_offsets[node.name] = torch.cumsum(count, dim=0) - count
        node_max_sizes[node.name] = max_nodes

        node_out.set(
            node.name,
            TensorDict(
                {"count": _reshape_count_tensor(count, batch_shape)},
                batch_size=batch_shape,
                device=count.device,
            ),
        )

    edge_out = TensorDict({}, batch_size=batch_shape)
    for edge in schema.edges:
        if edge.row0_node not in node_specs or edge.row1_node not in node_specs:
            raise KeyError(
                f"Edge {edge.name} references undefined node types "
                f"({edge.row0_node}, {edge.row1_node})."
            )
        if edge.row0_node not in node_counts or edge.row1_node not in node_counts:
            if edge.required:
                raise KeyError(
                    f"Edge {edge.name} requires node counts for "
                    f"{edge.row0_node} and {edge.row1_node}, but one is missing."
                )
            continue

        idx_raw = _lookup_first(observation, edge.idx_keys)
        if idx_raw is None:
            if edge.required:
                raise KeyError(f"Missing required edge index keys for {edge.name}: {edge.idx_keys}")
            continue

        idx_local = _reshape_by_batch(idx_raw, batch_shape, B_flat).to(dtype=torch.long)
        if idx_local.dim() != 3 or idx_local.shape[1] != 2:
            raise RuntimeError(
                f"Edge index tensor for {edge.name} must reshape to [B,2,E], got {tuple(idx_local.shape)}."
            )

        E_max = int(idx_local.shape[-1])
        count_raw = _lookup_first(observation, edge.count_keys) if len(edge.count_keys) > 0 else None
        if count_raw is None:
            edge_count = torch.full((B_flat,), E_max, dtype=torch.long, device=idx_local.device)
        else:
            edge_count = count_raw.reshape(B_flat, -1)[:, 0].to(device=idx_local.device, dtype=torch.long)

        invalid_edge_count = (edge_count < 0) | (edge_count > E_max)
        if bool(invalid_edge_count.any()):
            if strict:
                bad_graphs = invalid_edge_count.nonzero(as_tuple=False).reshape(-1).tolist()
                raise RuntimeError(
                    f"Invalid edge count for {edge.name}. Expected [0, {E_max}], "
                    f"invalid entries at graph ids {bad_graphs}."
                )
            edge_count = edge_count.clamp(min=0, max=E_max)

        arange_e = torch.arange(E_max, device=idx_local.device).unsqueeze(0)
        edge_mask = arange_e < edge_count.unsqueeze(1)
        row0 = idx_local[:, 0, :].clone()
        row1 = idx_local[:, 1, :].clone()

        row0_bound = node_counts[edge.row0_node].to(idx_local.device).unsqueeze(1)
        row1_bound = node_counts[edge.row1_node].to(idx_local.device).unsqueeze(1)
        invalid_row0 = edge_mask & ((row0 < 0) | (row0 >= row0_bound))
        invalid_row1 = edge_mask & ((row1 < 0) | (row1 >= row1_bound))
        invalid_edge = invalid_row0 | invalid_row1
        if bool(invalid_edge.any()):
            if strict:
                bad_locs = invalid_edge.nonzero(as_tuple=False)
                preview = bad_locs[:8].tolist()
                raise RuntimeError(
                    f"Edge index for {edge.name} contains invalid local node ids. "
                    f"First invalid (graph_id, edge_slot): {preview}."
                )
            edge_mask = edge_mask & (~invalid_edge)
            row0, row1, edge_count = _compact_invalid_edges(
                row0,
                row1,
                edge_mask,
                edge_count,
                invalid_edge.any(dim=1),
            )

        row0_offset = node_offsets[edge.row0_node].to(idx_local.device).unsqueeze(1)
        row1_offset = node_offsets[edge.row1_node].to(idx_local.device).unsqueeze(1)
        idx_global = torch.empty_like(idx_local)
        idx_global[:, 0, :] = row0 + row0_offset
        idx_global[:, 1, :] = row1 + row1_offset

        edge_out.set(
            edge.name,
            TensorDict(
                {
                    "count": _reshape_count_tensor(edge_count, batch_shape),
                    "idx_global": _reshape_edge_index_tensor(idx_global, batch_shape),
                },
                batch_size=batch_shape,
                device=idx_global.device,
            ),
        )

    return TensorDict(
        {
            "nodes": node_out,
            "edges": edge_out,
        },
        batch_size=batch_shape,
    )


class HeteroGraphBatchTransform(Transform):
    """
    Build schema-driven graph batch metadata at replay buffer sample time.
    """

    def __init__(
        self,
        schema: HeteroGraphSchema,
        observation_keys: Sequence[Sequence[str] | str] = (("observation",),),
        *,
        strict: bool = False,
    ):
        super().__init__(in_keys=[], out_keys=[])
        self.schema = schema
        self.observation_keys = _as_key_tuple_list(observation_keys)
        self.strict = bool(strict)

    def _schema_is_present(self, observation: TensorDict) -> bool:
        for node in self.schema.nodes:
            if node.required and observation.get(node.attr_key, default=None) is None:
                return False
        for edge in self.schema.edges:
            if not edge.required:
                continue
            if _lookup_first(observation, edge.idx_keys) is None:
                return False
        return True

    def forward(self, tensordict: TensorDict) -> TensorDict:
        for obs_key in self.observation_keys:
            observation = tensordict.get(obs_key, default=None)
            if observation is None or not isinstance(observation, TensorDict):
                continue
            if not self._schema_is_present(observation):
                continue

            graph_batch = build_graph_batch_from_schema(
                observation,
                self.schema,
                strict=self.strict,
            )
            observation.set(self.schema.graph_batch_key, graph_batch)
        return tensordict


def candidate_task_graph_schema() -> HeteroGraphSchema:
    return HeteroGraphSchema(
        nodes=(
            NodeSpec(
                name="tasks",
                attr_key=("nodes", "tasks", "attr"),
                count_keys=(
                    ("nodes", "tasks", "count"),
                    ("aux", "candidates", "count"),
                ),
            ),
        ),
        edges=(
            EdgeSpec(
                name="tasks_tasks",
                row0_node="tasks",
                row1_node="tasks",
                idx_keys=(
                    ("edges", "tasks_tasks", "idx"),
                    ("graph", "edge_index_local"),
                ),
                count_keys=(
                    ("edges", "tasks_tasks", "count"),
                    ("graph", "edge_count"),
                ),
            ),
        ),
        graph_batch_key=("graph_batch",),
    )


def task_data_hetero_graph_schema() -> HeteroGraphSchema:
    return HeteroGraphSchema(
        nodes=(
            NodeSpec(
                name="tasks",
                attr_key=("nodes", "tasks", "attr"),
                count_keys=(
                    ("nodes", "tasks", "count"),
                    ("aux", "candidates", "count"),
                ),
            ),
            NodeSpec(
                name="data",
                attr_key=("nodes", "data", "attr"),
                count_keys=(("nodes", "data", "count"),),
            ),
            NodeSpec(
                name="devices",
                attr_key=("nodes", "devices", "attr"),
                count_keys=(("nodes", "devices", "count"),),
                required=False,
            ),
        ),
        edges=(
            EdgeSpec(
                name="tasks_tasks",
                row0_node="tasks",
                row1_node="tasks",
                idx_keys=(("edges", "tasks_tasks", "idx"),),
                count_keys=(("edges", "tasks_tasks", "count"),),
            ),
            EdgeSpec(
                name="tasks_read_data",
                row0_node="tasks",
                row1_node="data",
                idx_keys=(("edges", "tasks_read_data", "idx"),),
                count_keys=(("edges", "tasks_read_data", "count"),),
            ),
            EdgeSpec(
                name="tasks_write_data",
                row0_node="tasks",
                row1_node="data",
                idx_keys=(("edges", "tasks_write_data", "idx"),),
                count_keys=(("edges", "tasks_write_data", "count"),),
                required=False,
            ),
            EdgeSpec(
                name="tasks_devices",
                row0_node="tasks",
                row1_node="devices",
                idx_keys=(("edges", "tasks_devices", "idx"),),
                count_keys=(("edges", "tasks_devices", "count"),),
                required=False,
            ),
            EdgeSpec(
                name="data_devices",
                row0_node="data",
                row1_node="devices",
                idx_keys=(("edges", "data_devices", "idx"),),
                count_keys=(("edges", "data_devices", "count"),),
                required=False,
            ),
        ),
        graph_batch_key=("graph_batch",),
    )


class ReplayBufferGraphBatchTransform(HeteroGraphBatchTransform):
    """
    Backward-compatible name for candidate-task graph batching transform.
    """

    def __init__(
        self,
        observation_keys: Sequence[Sequence[str] | str] = (("observation",),),
        *,
        strict: bool = False,
        schema: HeteroGraphSchema | None = None,
    ):
        super().__init__(
            schema=candidate_task_graph_schema() if schema is None else schema,
            observation_keys=observation_keys,
            strict=strict,
        )
