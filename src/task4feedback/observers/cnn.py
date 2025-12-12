from dataclasses import dataclass
from omegaconf import DictConfig
from task4feedback import trip as trip
from task4feedback.interface.wrappers import (
    ExternalObserverFactory,
    GridTaskObserver,
)
from .base import ConfigurableObserverFactory

@dataclass(kw_only=True)
class GridObserverFactory(ExternalObserverFactory, ConfigurableObserverFactory):
    def __init__(
        self,
        spec: trip.GraphSpec,
        width: int,
        length: int,
        prev_frames: int = 1,
        features: DictConfig = None,
        **_ignored,
    ):
        if features is None:
            raise ValueError(
                "The 'features' parameter is required. "
                "Please specify a feature set via Hydra config (e.g., feature.defaults: [sets/cnn/read_basic])."
            )

        self.observer_t = GridTaskObserver
        self.batched = True

        graph_extractor_t = trip.GraphExtractor

        if not (spec.max_candidates == width * length):
            raise ValueError(f"Grid width * length ({width * length}) must match spec.max_candidates ({spec.max_candidates})")

        context = {
            "width": width,
            "length": length,
            "prev_frames": prev_frames
        }
        ret = self._add_features(features, context)

        super().__init__(
            spec,
            graph_extractor_t,
            **ret,
            observer_t=self.observer_t,
        )
