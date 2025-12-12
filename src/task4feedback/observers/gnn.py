from dataclasses import dataclass
from omegaconf import DictConfig
from task4feedback import trip as trip
from task4feedback.interface.wrappers import (
    ExternalObserverFactory,
    ExternalObserver,
    FeatureExtractorFactory,
    EdgeFeatureExtractorFactory,
    SimulatorDriver
)
from .base import ConfigurableObserverFactory

@dataclass(kw_only=True)
class GNNExternalObserverFactory(ExternalObserverFactory):
    def create(self, simulator: SimulatorDriver):
        state = simulator.get_state()
        graph_spec = self.graph_spec
        graph_extractor = self.graph_extractor_t(state)
        task_feature_extractor = self.task_feature_factory.create(state)
        data_feature_extractor = self.data_feature_factory.create(state)
        device_feature_extractor = self.device_feature_factory.create(state)
        task_task_feature_extractor = self.task_task_feature_factory.create(state)
        task_read_data_feature_extractor = self.task_read_data_feature_factory.create(state)
        task_write_data_feature_extractor = self.task_write_data_feature_factory.create(state)

        return ExternalObserver(
            simulator,
            graph_spec,
            graph_extractor,
            task_features=task_feature_extractor,
            data_features=data_feature_extractor,
            device_features=device_feature_extractor,
            task_task_features=task_task_feature_extractor,
            task_read_data_features=task_read_data_feature_extractor,
            task_write_data_features=task_write_data_feature_extractor,
            cache=True,
        )

class GNNObserverFactory(GNNExternalObserverFactory, ConfigurableObserverFactory):
    def __init__(self, spec: trip.GraphSpec, features: DictConfig, add_degree: bool = False, **_ignored):
        if features is None:
            raise ValueError(
                "The 'features' parameter is required. "
                "Please specify a feature set via Hydra config (e.g., feature.defaults: [sets/gnn/io_data_mapped])."
            )

        graph_extractor_t = trip.GraphExtractor
        task_feature_factory = FeatureExtractorFactory()
        data_feature_factory = FeatureExtractorFactory()
        device_feature_factory = FeatureExtractorFactory()
        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_read_data_feature_factory = EdgeFeatureExtractorFactory()
        task_write_data_feature_factory = EdgeFeatureExtractorFactory()

        if add_degree:
            task_feature_factory.add(trip.InDegreeTaskFeature)
            task_feature_factory.add(trip.OutDegreeTaskFeature)
            task_feature_factory.add(trip.ReadDegreeTaskFeature)

        # Add features from config
        self._add_features(task_feature_factory, features.get("task", []))
        self._add_features(data_feature_factory, features.get("data", []))
        self._add_features(device_feature_factory, features.get("device", []))
        self._add_features(task_task_feature_factory, features.get("task_task_edge", []))
        self._add_features(task_read_data_feature_factory, features.get("task_data_edge", []))
        self._add_features(task_write_data_feature_factory, features.get("data_device_edge", []))

        # Add default features if not specified in config
        if not features.get("device"):
            device_feature_factory.add(trip.EmptyDeviceFeature, 1)
        if not features.get("task_task_edge"):
            task_task_feature_factory.add(trip.TaskTaskDefaultEdgeFeature)
        if not features.get("task_data_edge"):
            task_read_data_feature_factory.add(trip.TaskDataMappedFeature)
        if not features.get("data_device_edge"):
            task_write_data_feature_factory.add(trip.TaskDeviceDefaultEdgeFeature)

        super().__init__(
            spec,
            graph_extractor_t,
            task_feature_factory=task_feature_factory,
            data_feature_factory=data_feature_factory,
            device_feature_factory=device_feature_factory,
            task_task_feature_factory=task_task_feature_factory,
            task_read_data_feature_factory=task_read_data_feature_factory,
            task_write_data_feature_factory=task_write_data_feature_factory,
        )
