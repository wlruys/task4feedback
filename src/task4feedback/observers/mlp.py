from omegaconf import DictConfig
from task4feedback import trip as trip
from task4feedback.interface.observer import CandidateTaskObserver
from task4feedback.interface.wrappers import ExternalObserverFactory
from .base import ConfigurableObserverFactory

class MLPTaskObserverFactory(ExternalObserverFactory, ConfigurableObserverFactory):
    def __init__(self, spec: trip.GraphSpec, features: DictConfig, **_ignored):
        if features is None:
            raise ValueError(
                "The 'features' parameter is required. "
                "Please specify a feature set via Hydra config (e.g., feature.defaults: [sets/mlp/io_basic])."
            )

        graph_extractor_t = trip.GraphExtractor
        self.observer_t = CandidateTaskObserver

        context = {}
        ret = self._add_features(features, context)

        super().__init__(
            spec,
            graph_extractor_t,
            **ret,
            observer_t=self.observer_t,
        )
