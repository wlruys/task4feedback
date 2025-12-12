from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Type, TYPE_CHECKING
import cxxfilt
import torch
from tensordict.tensordict import TensorDict
from torch_geometric.data import HeteroData
import warnings

import task4feedback.trip as trip
from .utils import (
    _make_node_tensor,
    _make_edge_tensor,
    _make_index_tensor,
    HashHolder,
)

if TYPE_CHECKING:
    from .simulator import SimulatorDriver

class AccessType(IntEnum):
    READ_WRITE: int = 0
    READ: int = 1
    WRITE: int = 2
    READ_MAPPED: int = 3
    RETIRE: int = 4


class NeighborhoodType(IntEnum):
    DEPENDENCIES: int = 0
    DEPENDENTS: int = 1
    BIDIRECTIONAL: int = 2
    ITERATIVE: int = 3

_EDGE_KEY_CACHE = {}

def observation_to_heterodata(observation: TensorDict, idx: int = 0, device="cpu", actions=None, truncate: bool = False) -> HeteroData:
    hetero_data = HeteroData()

    hetero_data["time"].x = observation["aux", "time"].unsqueeze(0)
    hetero_data["progress"].x = observation["aux", "progress"].unsqueeze(0)
    hetero_data["device_load"].x = observation["aux", "device_load"].unsqueeze(0)
    hetero_data["device_memory"].x = observation["aux", "device_memory"].unsqueeze(0)
    hetero_data["z_ch"].x = observation["aux", "z_ch"].unsqueeze(0)
    hetero_data["z_spa"].x = observation["aux", "z_spa"].unsqueeze(0)
    hetero_data["baseline"].x = observation["aux", "baseline"].unsqueeze(0)

    if actions is not None:
        hetero_data["actions"].x = actions

    for node_type, node_data in observation["nodes"].items():
        if truncate:
            count = int(node_data["count"][0].item())
            hetero_data[f"{node_type}"].x = node_data["attr"][:count]
        else:
            count = node_data["count"]
            hetero_data[f"{node_type}"].x = node_data["attr"]
            hetero_data[f"{node_type}_count"].x = count

    for edge_key, edge_data in observation["edges"].items():
        if edge_key not in _EDGE_KEY_CACHE:
            splits = edge_key.split("_")
            if len(splits) == 2:
                _EDGE_KEY_CACHE[edge_key] = (splits[0], None, splits[1])
            elif len(splits) == 3:
                _EDGE_KEY_CACHE[edge_key] = (splits[0], splits[1], splits[2])
            else:
                raise ValueError(f"Invalid edge key format: {edge_key}")
        
        target, usage, source = _EDGE_KEY_CACHE[edge_key]

        if truncate:
            count = int(edge_data["count"][0].item())
            idx_slice = edge_data["idx"][:, :count]
            attr_slice = edge_data["attr"][:count] if "attr" in edge_data else None
        else:
            count = edge_data["count"]
            idx_slice = edge_data["idx"]
            attr_slice = edge_data["attr"] if "attr" in edge_data else None

        if usage is None:
            hetero_data[target, "to", source].edge_index = idx_slice
            if attr_slice is not None:
                hetero_data[target, "to", source].edge_attr = attr_slice

            if source != target:
                hetero_data[source, "to", target].edge_index = hetero_data[target, "to", source].edge_index.flip(0)
                if attr_slice is not None:
                    hetero_data[source, "to", target].edge_attr = hetero_data[target, "to", source].edge_attr

            if source == target:
                hetero_data[source, "from", target].edge_index = hetero_data[target, "to", source].edge_index.flip(0)
                if attr_slice is not None:
                    hetero_data[source, "from", target].edge_attr = hetero_data[target, "to", source].edge_attr
        else:
            hetero_data[target, usage, source].edge_index = idx_slice
            if attr_slice is not None:
                hetero_data[target, usage, source].edge_attr = attr_slice

            if source != target:
                hetero_data[source, usage, target].edge_index = hetero_data[target, usage, source].edge_index.flip(0)
                if attr_slice is not None:
                    hetero_data[source, usage, target].edge_attr = hetero_data[target, usage, source].edge_attr

    return hetero_data.to(device)

def observation_to_heterodata_truncate(observation: TensorDict, idx: int = 0, device="cpu", actions=None) -> HeteroData:
    """Deprecated: Use observation_to_heterodata(..., truncate=True) instead."""
    return observation_to_heterodata(observation, idx, device, actions, truncate=True)


class FeatureExtractorFactory:
    def __init__(
        self,
        feature_list: Optional[list] = None,
        options: Optional[dict[Type, tuple]] = None,
    ):
        if feature_list is None:
            feature_list = []
        if options is None:
            options = {}

        self.options = options
        self.feature_list = feature_list

    def create(self, state: trip.SchedulerState):
        feature_extractor = trip.RuntimeFeatureExtractor()
        for feature_t in self.feature_list:
            args = self.options.get(feature_t, tuple())
            feature_extractor.add_feature(feature_t.create(state, *args))
        return feature_extractor

    def add(self, feature_t: Type, *args):
        self.feature_list.append(feature_t)

        if args:
            self.options[feature_t] = args


class EdgeFeatureExtractorFactory:
    def __init__(
        self,
        feature_list: Optional[list] = None,
        options: Optional[dict[Type, tuple]] = None,
    ):
        if feature_list is None:
            feature_list = []
        if options is None:
            options = {}

        self.options = options
        self.feature_list = feature_list

    def create(self, state: trip.SchedulerState):
        feature_extractor = trip.RuntimeEdgeFeatureExtractor()
        for feature_t in self.feature_list:
            args = self.options.get(feature_t, tuple())
            feature_extractor.add_feature(feature_t.create(state, *args))
        return feature_extractor

    def add(self, feature_t: Type, *args):
        self.feature_list.append(feature_t)

        if args:
            self.options[feature_t] = args


@dataclass
class ExternalObserver:
    simulator: "SimulatorDriver"
    graph_spec: trip.GraphSpec
    graph_extractor: Optional[trip.GraphExtractor] = None 
    task_features: Optional[trip.RuntimeFeatureExtractor] = None 
    data_features: Optional[trip.RuntimeFeatureExtractor] = None 
    device_features: Optional[trip.RuntimeFeatureExtractor] = None 
    task_task_features: Optional[trip.RuntimeEdgeFeatureExtractor] = None
    task_read_data_features: Optional[trip.RuntimeEdgeFeatureExtractor] = None
    task_write_data_features: Optional[trip.RuntimeEdgeFeatureExtractor] = None
    task_data_features: Optional[trip.RuntimeEdgeFeatureExtractor] = None
    task_device_features: Optional[trip.RuntimeEdgeFeatureExtractor] = None
    data_device_features: Optional[trip.RuntimeEdgeFeatureExtractor] = None
    device_device_features: Optional[trip.RuntimeEdgeFeatureExtractor] = None
    data_data_features: Optional[trip.RuntimeEdgeFeatureExtractor] = None
    truncate: bool = True
    cache: bool = False

    def __post_init__(self):
        # Fix: Initialize HashHolder as instance variable
        self.cache_holder = HashHolder()
        self.observation_buffer = self.new_observation_buffer(self.graph_spec)

    def store_feature_types(self):
        """
        Store feature type information in the provided config dictionary.
        Includes both the feature extractor class names and the specific feature types.
        """
        config_dictionary = {}
        for field_name, field_value in self.__dict__.items():
            if field_name.endswith("features") and field_value is not None:
                # Store the class name of the feature extractor
                config_dictionary[field_name] = field_value.__class__.__name__

                # Store the specific feature type names
                if hasattr(field_value, "feature_type_names"):
                    feature_types = field_value.feature_type_names
                    feature_types = [cxxfilt.demangle(t) for t in feature_types]
                    # Format feature type names for better readability
                    formatted_types = [t.split("::")[-1] if "::" in t else t for t in feature_types]

                    print(f"Feature types for {field_name}: {formatted_types}")
                    config_dictionary[f"{field_name}_types"] = formatted_types

        return config_dictionary

    @property
    def task_feature_dim(self):
        if self.task_features is None:
            return 0
        return self.task_features.feature_dim

    @property
    def data_feature_dim(self):
        if self.data_features is None:
            return 0
        return self.data_features.feature_dim

    @property
    def device_feature_dim(self):
        if self.device_features is None:
            return 0
        return self.device_features.feature_dim

    @property
    def task_data_edge_dim(self):
        if self.task_data_features is None:
            return 0
        return self.task_data_features.feature_dim

    @property
    def task_read_data_edge_dim(self):
        if self.task_read_data_features is None:
            return 0
        return self.task_read_data_features.feature_dim

    @property
    def task_write_data_edge_dim(self):
        if self.task_write_data_features is None:
            return 0
        return self.task_write_data_features.feature_dim

    @property
    def task_device_edge_dim(self):
        if self.task_device_features is None:
            return 0
        return self.task_device_features.feature_dim

    @property
    def data_device_edge_dim(self):
        if self.data_device_features is None:
            return 0
        return self.data_device_features.feature_dim

    @property
    def task_task_edge_dim(self):
        if self.task_task_features is None:
            return 0
        return self.task_task_features.feature_dim

    def get_task_features(self, task_ids, workspace):
        length = self.task_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_data_features(self, data_ids, workspace):
        length = self.data_features.get_features_batch(data_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_device_features(self, device_ids, workspace):
        length = self.device_features.get_features_batch(device_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_task_features(self, task_ids, workspace):
        length = self.task_task_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_data_features(self, task_ids, workspace):
        length = self.task_data_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_read_data_features(self, task_ids, workspace):
        length = self.task_read_data_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_write_data_features(self, task_ids, workspace):
        length = self.task_write_data_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_device_features(self, task_ids, workspace):
        length = self.task_device_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_data_device_features(self, data_ids, workspace):
        length = self.data_device_features.get_features_batch(data_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_k_hop_neighborhood(self, task_ids, workspace, depth: int = 1):
        length = self.graph_extractor.get_k_hop_neighborhood(task_ids, depth, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_k_hop_bidirectional(self, task_ids, workspace, depth: int = 1):
        length = self.graph_extractor.get_k_hop_bidirectional(task_ids, depth, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_k_hop_dependencies(self, task_ids, workspace, depth: int = 1):
        length = self.graph_extractor.get_k_hop_dependencies(task_ids, depth, workspace)
        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_k_hop_dependents(self, task_ids, workspace, depth: int = 1):
        length = self.graph_extractor.get_k_hop_dependents(task_ids, depth, workspace)
        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_unique_data(self, task_ids, workspace):
        length = self.graph_extractor.get_unique_data(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_used_filtered_data(self, task_ids, workspace):
        """
        Only return data whose most recent writer has been mapped
        """
        length = self.graph_extractor.get_unique_filtered_data(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_task_edges(self, task_ids, workspace, global_workspace):
        length = self.graph_extractor.get_task_task_edges(task_ids, workspace, global_workspace)

        if self.truncate:
            workspace = workspace[:, :length]
        return workspace, length

    def get_task_data_edges(
        self,
        task_ids,
        data_ids,
        workspace,
        global_workspace,
        access_type: AccessType = AccessType.READ_WRITE,
    ):
        if access_type == AccessType.READ_WRITE:
            length = self.graph_extractor.get_task_data_edges_all(task_ids, data_ids, workspace, global_workspace)
        elif access_type == AccessType.READ:
            length = self.graph_extractor.get_task_data_edges_read(task_ids, data_ids, workspace, global_workspace)
        elif access_type == AccessType.WRITE:
            length = self.graph_extractor.get_task_data_edges_write(task_ids, data_ids, workspace, global_workspace)
        elif access_type == AccessType.READ_MAPPED:
            length = self.graph_extractor.get_task_data_edges_read_mapped(task_ids, data_ids, workspace, global_workspace)
        else:
            raise ValueError(f"Invalid access type operation for get_task_data_edges: {access_type}")

        if self.truncate:
            workspace = workspace[:, :length]
        return workspace, length

    def get_task_task_edges_reverse(self, task_ids, workspace, global_workspace):
        length = self.graph_extractor.get_task_task_edges_reverse(task_ids, workspace, global_workspace)

        if self.truncate:
            workspace = workspace[:, :length]
        return workspace, length

    def get_task_device_edges(self, task_ids, workspace, global_workspace):
        length = self.graph_extractor.get_task_device_edges(task_ids, workspace, global_workspace)

        if self.truncate:
            workspace = workspace[:, :length]
        return workspace, length

    def get_data_device_edges(self, data_ids, workspace, global_workspace):
        length = self.graph_extractor.get_data_device_edges(data_ids, workspace, global_workspace)

        if self.truncate:
            workspace = workspace[:, :length]

        return workspace, length

    def _local_to_global(self, global_ids, local_ids, workspace=None):
        if workspace is not None:
            workspace[: len(local_ids)] = global_ids[local_ids]
            return workspace
        else:
            return global_ids[local_ids]

    def get_device_memory(self, output: TensorDict):
        self.graph_extractor.get_device_memory(output["aux"]["device_memory"])

    def get_device_load(self, output: TensorDict):
        self.graph_extractor.get_device_load(output["aux"]["device_load"])

    def _local_to_global2D(self, g1, g2, l, workspace=None):
        if workspace is not None:
            size = len(l[0, :])
            workspace[0, :size] = g1[l[0, :]][:size]
            workspace[1, :size] = g2[l[1, :]][:size]
            return workspace
        else:
            id1 = g1[l[0, :]]
            id2 = g2[l[1, :]]
            return torch.stack((id1, id2), dim=0)

    def _local_to_global2D_same(self, g1, l, workspace=None):
        if workspace is not None:
            size = len(l[0, :])
            workspace[:, :size] = g1[l][:size]
            return workspace
        else:
            return g1[l]

    def new_observation_buffer(self, spec: Optional[trip.GraphSpec] = None):
        if spec is None:
            spec = self.graph_spec

        node_dict = {}

        if self.task_features is not None:
            node_dict["tasks"] = _make_node_tensor(spec.max_tasks, self.task_features.feature_dim)
        
        if self.data_features is not None:
            node_dict["data"] = _make_node_tensor(spec.max_data, self.data_features.feature_dim)
        
        if self.device_features is not None:
            node_dict["devices"] = _make_node_tensor(spec.max_devices, self.device_features.feature_dim)

        node_tensor = TensorDict(node_dict)


        edge_dict = {}

        if self.task_task_features is not None:
            edge_dict["tasks_tasks"] = _make_edge_tensor(spec.max_edges_tasks_tasks, self.task_task_features.feature_dim)

        if self.task_read_data_features is not None:
            edge_dict["tasks_read_data"] = _make_edge_tensor(spec.max_edges_tasks_data, self.task_read_data_features.feature_dim)

        if self.task_write_data_features is not None:
            edge_dict["tasks_write_data"] = _make_edge_tensor(spec.max_edges_tasks_data, self.task_write_data_features.feature_dim)

        edge_tensor = TensorDict(edge_dict)

        aux_tensor = TensorDict(
            {
                "candidates": _make_index_tensor(spec.max_candidates),
                "candidate_mask": torch.zeros((spec.max_candidates), dtype=torch.bool),
                "time": torch.zeros((1), dtype=torch.int64),
                "improvement": torch.zeros((1), dtype=torch.float32),
                "progress": torch.zeros((1), dtype=torch.float32),
                "baseline": torch.zeros((1), dtype=torch.float32),
                "last_action": torch.zeros(
                    (spec.max_candidates, spec.max_devices),
                    dtype=torch.float32,
                ),
                "device_memory": torch.zeros(1 * (spec.max_devices), dtype=torch.float32),
                "device_load": torch.zeros(2 * (spec.max_devices), dtype=torch.float32),
                "z_ch": torch.zeros((8), dtype=torch.float32),
                "z_spa": torch.zeros((8), dtype=torch.float32),
            }
        )

        obs_tensor = TensorDict(
            {
                "nodes": node_tensor,
                "edges": edge_tensor,
                "aux": aux_tensor,
            }
        )

        return obs_tensor

    def task_observation(
        self,
        output: TensorDict,
        task_ids: Optional[torch.Tensor] = None,
        k: int = 1,
        neighborhood_type: NeighborhoodType = NeighborhoodType.ITERATIVE,
    ):
        if self.task_features is None:
            return
        
        if task_ids is None:
            n_candidates = output["aux", "candidates", "count"][0]
            task_ids = output["aux", "candidates", "idx"][:n_candidates]

        t = None
        if self.cache:
            key = (task_ids,)
            t = self.cache_holder.get(key, ("nodes", "tasks", "glb"))
            if t is not None:
                output["nodes", "tasks", "glb"] = t
                output["nodes", "tasks", "count"] = self.cache_holder.get(key, ("nodes", "tasks", "count"))
                count = output["nodes", "tasks", "count"][0]

        if t is None:
            if neighborhood_type == NeighborhoodType.BIDIRECTIONAL:
                _, count = self.get_k_hop_bidirectional(task_ids, output["nodes", "tasks", "glb"], k)
            elif neighborhood_type == NeighborhoodType.DEPENDENCIES:
                _, count = self.get_k_hop_dependencies(task_ids, output["nodes", "tasks", "glb"])

            elif neighborhood_type == NeighborhoodType.DEPENDENTS:
                _, count = self.get_k_hop_dependents(task_ids, output["nodes", "tasks", "glb"])

            elif neighborhood_type == NeighborhoodType.ITERATIVE:
                _, count = self.get_k_hop_neighborhood(task_ids, output["nodes", "tasks", "glb"], k)
            else:
                raise ValueError(f"Invalid neighborhood type operation for task observation: {neighborhood_type}")

            output.set_at_(("nodes", "tasks", "count"), count, 0)

            if self.cache:
                self.cache_holder.add(key, ("nodes", "tasks", "glb"), output["nodes", "tasks", "glb"].detach().clone())
                self.cache_holder.add(key, ("nodes", "tasks", "count"), output["nodes", "tasks", "count"].detach().clone())

        self.get_task_features(output["nodes", "tasks", "glb"][:count], output["nodes", "tasks", "attr"])

    def data_observation(self, output: TensorDict):
        if self.data_features is None:
            return

        n_candidates = output["aux", "candidates", "count"][0]
        candidate_ids = output["aux", "candidates", "idx"][:n_candidates]
        t = None
        if self.cache:
            key = (candidate_ids,)
            t = self.cache_holder.get(key, ("nodes", "data", "glb"))
            if t is not None:
                output["nodes", "data", "glb"] = t
                output["nodes", "data", "count"] = self.cache_holder.get(key, ("nodes", "data", "count"))
                count = output["nodes", "data", "count"][0]

        if t is None:
            ntasks = output["nodes", "tasks", "count"][0]
            _, count = self.get_unique_data(output["nodes", "tasks", "glb"][:ntasks], output["nodes", "data", "glb"])
            output.set_at_(("nodes", "data", "count"), count, 0)

            if self.cache:
                self.cache_holder.add(key, ("nodes", "data", "glb"), output["nodes", "data", "glb"].detach().clone())
                self.cache_holder.add(key, ("nodes", "data", "count"), output["nodes", "data", "count"].detach().clone())

        self.get_data_features(output["nodes", "data", "glb"][:count], output["nodes", "data", "attr"])

    def read_data_observation(self, output: TensorDict):
        # print("Read Data observation")
        ntasks = output["nodes", "tasks", "count"][0]
        _, count = self.get_used_filtered_data(
            output["nodes", "tasks", "glb"][:ntasks], output["nodes", "read_data", "glb"]
        )
        output.set_at_(("nodes", "read_data", "count"), count, 0)
        self.get_data_features(
            output["nodes", "read_data", "glb"][:count], output["nodes", "read_data", "attr"]
        )

    def write_data_observation(self, output: TensorDict):
        # print("Write Data observation")
        ntasks = output["nodes", "tasks", "count"][0]
        _, count = self.get_used_filtered_data(
            output["nodes", "tasks", "glb"][:ntasks], output["nodes", "write_data", "glb"]
        )
        output.set_at_(("nodes", "write_data", "count"), count, 0)
        self.get_data_features(
            output["nodes", "write_data", "glb"][:count], output["nodes", "write_data", "attr"]
        )

    def device_observation(self, output: TensorDict):
        if self.device_features is None:
            return 
        count = output["nodes", "devices", "glb"].shape[0]
        output.set_at_(("nodes", "devices", "count"), count, 0)
        output["nodes", "devices", "glb"][:count] = torch.arange(count, dtype=torch.int64)
        self.get_device_features(
            output["nodes", "devices", "glb"][:count],
            output["nodes", "devices", "attr"],
        )

    def task_task_observation(self, output: TensorDict):
        if self.task_task_features is None:
            return
        
        ntasks = output["nodes", "tasks", "count"][0]

        n_candidates = output["aux", "candidates", "count"][0]
        candidate_ids = output["aux", "candidates", "idx"][:n_candidates]
        t = None
        if self.cache:
            key = (candidate_ids,)
            t = self.cache_holder.get(key, ("edges", "tasks_tasks", "glb"))
            if t is not None:
                output["edges", "tasks_tasks", "glb"] = t
                output["edges", "tasks_tasks", "idx"] = self.cache_holder.get(key, ("edges", "tasks_tasks", "idx"))
                output["edges", "tasks_tasks", "count"] = self.cache_holder.get(key, ("edges", "tasks_tasks", "count"))
                count = output["edges", "tasks_tasks", "count"][0]

        if t is None:
            _, count = self.get_task_task_edges(
                output["nodes", "tasks", "glb"][:ntasks],
                output["edges", "tasks_tasks", "idx"],
                output["edges", "tasks_tasks", "glb"],
            )

            output.set_at_(("edges", "tasks_tasks", "count"), count, 0)

            if self.cache:
                self.cache_holder.add(key, ("edges", "tasks_tasks", "glb"), output["edges", "tasks_tasks", "glb"].detach().clone())
                self.cache_holder.add(key, ("edges", "tasks_tasks", "idx"), output["edges", "tasks_tasks", "idx"].detach().clone())
                self.cache_holder.add(key, ("edges", "tasks_tasks", "count"), output["edges", "tasks_tasks", "count"].detach().clone())

        if "attr" in output["edges", "tasks_tasks"]:
            self.get_task_task_features(
                output["edges", "tasks_tasks", "glb"][:, :count],
                output["edges", "tasks_tasks", "attr"],
            )

    def task_data_observation(self, output: TensorDict):
        if self.task_read_data_features is None:
            return
        
        ntasks = output["nodes", "tasks", "count"][0]
        ndata = output["nodes", "data", "count"][0]

        n_candidates = output["aux", "candidates", "count"][0]
        candidate_ids = output["aux", "candidates", "idx"][:n_candidates]
        t = None
        if self.cache:
            key = (candidate_ids,)
            t = self.cache_holder.get(key, ("edges", "tasks_read_data", "glb"))
            if t is not None:
                output["edges", "tasks_read_data", "glb"] = t
                output["edges", "tasks_read_data", "idx"] = self.cache_holder.get(key, ("edges", "tasks_read_data", "idx"))
                output["edges", "tasks_read_data", "count"] = self.cache_holder.get(key, ("edges", "tasks_read_data", "count"))
                read_count = output["edges", "tasks_read_data", "count"][0]

        if t is None:
            _, read_count = self.get_task_data_edges(
                output["nodes", "tasks", "glb"][:ntasks],
                output["nodes", "data", "glb"][:ndata],
                output["edges", "tasks_read_data", "idx"],
                output["edges", "tasks_read_data", "glb"],
                AccessType.READ_MAPPED,
            )
            output.set_at_(("edges", "tasks_read_data", "count"), read_count, 0)

            if self.cache:
                self.cache_holder.add(key, ("edges", "tasks_read_data", "glb"), output["edges", "tasks_read_data", "glb"].detach().clone())
                self.cache_holder.add(key, ("edges", "tasks_read_data", "idx"), output["edges", "tasks_read_data", "idx"].detach().clone())
                self.cache_holder.add(key, ("edges", "tasks_read_data", "count"), output["edges", "tasks_read_data", "count"].detach().clone())

        if "attr" in output["edges", "tasks_read_data"]:
            self.get_task_read_data_features(
                output["edges", "tasks_read_data", "glb"][:, :read_count],
                output["edges", "tasks_read_data", "attr"],
            )

    def task_device_observation(self, output: TensorDict, use_all_tasks=False):
        if self.task_device_features is None:
            return
        
        if not use_all_tasks:
            ncandidates = output["aux", "candidates", "count"][0]
            task_ids = output["aux", "candidates", "idx"][:ncandidates]
        else:
            ntasks = output["nodes", "tasks", "count"][0]
            task_ids = output["nodes", "tasks", "glb"][:ntasks]

        ndevices = output["nodes", "devices", "count"][0]

        _, count = self.get_task_device_edges(
            task_ids,
            output["edges", "tasks_devices", "idx"],
            output["edges", "tasks_devices", "glb"],
        )

        output.set_at_(("edges", "tasks_devices", "count"), count, 0)

        self.get_task_device_features(
            output["edges", "tasks_devices", "glb"][:, :count],
            output["edges", "tasks_devices", "attr"],
        )

    def candidate_observation(self, output: TensorDict):
        count = self.simulator.simulator.get_mappable_candidates(output["aux", "candidates", "idx"])
        output.set_at_(("aux", "candidates", "count"), count, 0)
        output["aux", "candidate_mask"][:count] = True 


    def get_candidate_to_action(self, candidate_idx: int, task_id: int):
        return candidate_idx

    def get_action_to_candidate(self, action_idx: int):
        raise NotImplementedError("get_action_to_candidate is not implemented in ExternalObserver.")


    def get_observation(self, output: Optional[TensorDict] = None):
        if output is None:
            output = self.observation_buffer

        # Get mappable candidates
        self.candidate_observation(output)

        # Node observations (all nodes must be processed before edges)
        self.task_observation(output, k=1)
        self.data_observation(output)

        # Edge observations (edges depend on ids collected during node observation)
        self.task_task_observation(output)
        self.task_data_observation(output)

        # Auxiliary observations
        output.set_at_(("aux", "progress"), -2.0, 0)
        output.set_at_(("aux", "time"), self.simulator.time, 0)
        output.set_at_(("aux", "improvement"), -100.0, 0)

        self.get_device_load(output)
        self.get_device_memory(output)

        return output

    def reset(self):
        """
        Reset the observer state.
        This method can be overridden by subclasses to implement specific reset logic.
        """
        self.cache_holder.clear()


@dataclass
class ExternalObserverFactory:
    graph_spec: trip.GraphSpec
    graph_extractor_t: Type[trip.GraphExtractor]
    task_feature_factory: Optional[FeatureExtractorFactory] = None
    data_feature_factory: Optional[FeatureExtractorFactory] = None
    device_feature_factory: Optional[FeatureExtractorFactory] = None
    task_task_feature_factory: Optional[EdgeFeatureExtractorFactory] = None
    task_data_feature_factory: Optional[EdgeFeatureExtractorFactory] = None
    task_device_feature_factory: Optional[EdgeFeatureExtractorFactory] = None
    data_device_feature_factory: Optional[EdgeFeatureExtractorFactory] = None
    task_read_data_feature_factory: Optional[EdgeFeatureExtractorFactory] = None
    task_write_data_feature_factory: Optional[EdgeFeatureExtractorFactory] = None
    observer_t: Type[ExternalObserver] = ExternalObserver

    def set_graph_spec(self, spec: trip.GraphSpec):
        self.graph_spec = spec

    def create(self, simulator: "SimulatorDriver"):
        state = simulator.get_state()
        graph_spec = self.graph_spec
        graph_extractor = self.graph_extractor_t(state) if self.graph_extractor_t is not None else None
        task_feature_extractor = self.task_feature_factory.create(state) if self.task_feature_factory is not None else None
        data_feature_extractor = self.data_feature_factory.create(state) if self.data_feature_factory is not None else None
        device_feature_extractor = self.device_feature_factory.create(state) if self.device_feature_factory is not None else None
        task_task_feature_extractor = self.task_task_feature_factory.create(state) if self.task_task_feature_factory is not None else None
        task_data_feature_extractor = self.task_data_feature_factory.create(state) if self.task_data_feature_factory is not None else None

        task_device_feature_extractor = self.task_device_feature_factory.create(state) if self.task_device_feature_factory is not None else None
        data_device_feature_extractor = self.data_device_feature_factory.create(state) if self.data_device_feature_factory is not None else None
        task_read_data_feature_extractor = self.task_read_data_feature_factory.create(state) if self.task_read_data_feature_factory is not None else None
        task_write_data_feature_extractor = self.task_write_data_feature_factory.create(state) if self.task_write_data_feature_factory is not None else None

        return self.observer_t(
            simulator,
            graph_spec,
            graph_extractor,
            task_features=task_feature_extractor,
            data_features=data_feature_extractor,
            device_features=device_feature_extractor,
            task_task_features=task_task_feature_extractor,
            task_data_features=task_data_feature_extractor,
            task_device_features=task_device_feature_extractor,
            data_device_features=data_device_feature_extractor,
            task_read_data_features=task_read_data_feature_extractor,
            task_write_data_features=task_write_data_feature_extractor,
        )


class DefaultObserverFactory(ExternalObserverFactory):
    def __init__(self, spec: trip.GraphSpec):
        graph_extractor_t = trip.GraphExtractor
        task_feature_factory = FeatureExtractorFactory()
        task_feature_factory.add(trip.InDegreeTaskFeature)
        task_feature_factory.add(trip.OutDegreeTaskFeature)
        task_feature_factory.add(trip.OneHotMappedDeviceTaskFeature)
        task_feature_factory.add(trip.EmptyTaskFeature, 1)

        data_feature_factory = FeatureExtractorFactory()
        data_feature_factory.add(trip.DataSizeFeature)
        data_feature_factory.add(trip.DataMappedLocationsFeature)

        device_feature_factory = FeatureExtractorFactory()
        device_feature_factory.add(trip.DeviceArchitectureFeature)
        device_feature_factory.add(trip.DeviceIDFeature)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(trip.TaskTaskSharedDataFeature)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        task_data_feature_factory.add(trip.TaskDataRelativeSizeFeature)

        task_device_feature_factory = EdgeFeatureExtractorFactory()
        task_device_feature_factory.add(trip.TaskDeviceDefaultEdgeFeature)

        data_device_feature_factory = None

        super().__init__(
            spec,
            graph_extractor_t,
            task_feature_factory,
            data_feature_factory,
            device_feature_factory,
            task_task_feature_factory,
            task_data_feature_factory,
            task_device_feature_factory,
            data_device_feature_factory,
        )


class CandidateTaskObserver(ExternalObserver):
    """
    Observer that only collects candidate information.
    """

    def new_observation_buffer(self, spec: Optional[trip.GraphSpec] = None):
        if spec is None:
            spec = self.graph_spec

        node_tensor = TensorDict({"tasks": _make_node_tensor(spec.max_candidates, self.task_features.feature_dim)})

        aux_tensor = TensorDict(
            {
                "candidates": _make_index_tensor(spec.max_candidates),
                "candidate_mask": torch.zeros((spec.max_candidates), dtype=torch.bool),
                "time": torch.zeros((1), dtype=torch.int64),
                "improvement": torch.zeros((1), dtype=torch.float32),
                "progress": torch.zeros((1), dtype=torch.float32),
                "baseline": torch.ones((1), dtype=torch.float32),
                "last_action": torch.zeros(
                    (spec.max_candidates, spec.max_devices),
                    dtype=torch.float32,
                ),
                "device_memory": torch.zeros(1 * (spec.max_devices), dtype=torch.float32),
                "device_load": torch.zeros(2 * (spec.max_devices), dtype=torch.float32),
                "z_ch": torch.zeros((8), dtype=torch.float32),
                "z_spa": torch.zeros((8), dtype=torch.float32),
            }
        )

        obs_tensor = TensorDict(
            {
                "nodes": node_tensor,
                "aux": aux_tensor,
            }
        )

        return obs_tensor

    def get_observation(self, output: Optional[TensorDict] = None):
        if output is None:
            output = self.new_observation_buffer(self.graph_spec)
            # Fix: Use warnings.warn instead of raise Warning
            warnings.warn("Allocating new observation buffer, this is not efficient!")

        # Get mappable candidates
        self.candidate_observation(output)

        output.set_(("nodes", "tasks", "glb"), output["aux", "candidates", "idx"])
        output.set_at_(("nodes", "tasks", "count"), output["aux", "candidates", "count"][0], 0)

        self.get_task_features(output["nodes", "tasks", "glb"], output["nodes", "tasks", "attr"])

        # Auxiliary observations
        output.set_at_(("aux", "progress"), -2.0, 0)
        output.set_at_(("aux", "time"), self.simulator.time, 0)
        output.set_at_(("aux", "improvement"), -100.0, 0)

        self.get_device_load(output)
        self.get_device_memory(output)

        return output


class GridTaskObserver(ExternalObserver):
    """
    Observer that collects 2d flattened grid of task features.
    """

    task_ids = None

    def new_observation_buffer(self, spec: Optional[trip.GraphSpec] = None):
        if spec is None:
            spec = self.graph_spec
        graph = self.simulator.input.graph

        aux_tensor = TensorDict(
            {
                "candidates": _make_index_tensor(spec.max_candidates),
                "candidate_mask": torch.zeros((spec.max_candidates), dtype=torch.bool),
                "time": torch.zeros((1), dtype=torch.int64),
                "improvement": torch.zeros((1), dtype=torch.float32),
                "progress": torch.zeros((1), dtype=torch.float32),
                "baseline": torch.ones((1), dtype=torch.float32),
                "last_action": torch.zeros(
                    (spec.max_candidates, spec.max_devices),
                    dtype=torch.float32,
                ),
                "device_memory": torch.zeros(1 * (spec.max_devices), dtype=torch.float32),
                "device_load": torch.zeros(2 * (spec.max_devices), dtype=torch.float32),
                "z_ch": torch.zeros((8), dtype=torch.float32),
                "z_spa": torch.zeros((8), dtype=torch.float32),
            }
        )

        obs_tensor = TensorDict(
            {
                "nodes": TensorDict(
                    {
                        "tasks": TensorDict(
                            {
                                "attr": torch.zeros(
                                    (graph.nx * graph.ny, self.task_features.feature_dim),
                                    dtype=torch.float32,
                                )
                            }
                        )
                    }
                ),
                "aux": aux_tensor,
            }
        )

        return obs_tensor
    
    def get_candidate_to_action(self, candidate_idx: int, task_id: int):
        action_idx = self.simulator.input.graph.xy_from_id(task_id)
        return action_idx

    def get_action_to_candidate(self, action_idx: int):
        raise NotImplementedError("GridTaskObserver does not support action to candidate mapping.")

    def get_observation(self, output: Optional[TensorDict] = None):
        graph = self.simulator.input.graph
        if output is None:
            output = self.new_observation_buffer(self.graph_spec)
            warnings.warn("Allocating new observation buffer, this is not efficient!")
        if self.task_ids is None:
            self.task_ids = torch.Tensor([-1 for _ in range(graph.nx * graph.ny)])

        # Get mappable candidates
        self.candidate_observation(output)

        assert output["aux", "candidates", "count"][0] == graph.nx * graph.ny or output["aux", "candidates", "count"][0] == 0, "GridTaskObserver expects {} candidates but got {}.".format(
            graph.nx * graph.ny, output["aux", "candidates", "count"][0].item()
        )
        for task_id in output["aux", "candidates", "idx"]:
            idx = graph.xy_from_id(task_id.item())
            self.task_ids[idx] = task_id.item()

        self.get_task_features(self.task_ids, output["nodes", "tasks", "attr"])
        self.get_device_load(output)
        self.get_device_memory(output)

        # Auxiliary observations
        output.set_at_(("aux", "progress"), -2.0, 0)
        output.set_at_(("aux", "time"), self.simulator.time, 0)
        output.set_at_(("aux", "improvement"), -100.0, 0)

        return output
