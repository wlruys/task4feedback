from typing import Optional, List, Dict, Union, Any
from omegaconf import DictConfig, ListConfig
from task4feedback import trip as trip
import logging

from task4feedback.interface.observer import EdgeFeatureExtractorFactory, FeatureExtractorFactory
logger = logging.getLogger(__name__)

class ConfigurableObserverFactory:
    def _resolve_arg(self, arg: Any, context: Optional[Dict[str, Any]]) -> Any:
        if isinstance(arg, str) and arg.startswith("$") and context:
            key = arg[1:]
            if key in context:
                return context[key]
        return arg
    
    def _add_features(self, features: DictConfig, context: Optional[Dict[str, Any]] = None):
        if features is None:
            logger.debug("No features to add: features is None.")
            return None
        
        ret = {}
        
        if task_features := features.get("task"):
            task_feature_factory = FeatureExtractorFactory()
            self._add_features_impl(task_feature_factory, task_features, context)
            ret["task_feature_factory"] = task_feature_factory
        
        if data_features := features.get("data"):
            data_feature_factory = FeatureExtractorFactory()
            self._add_features_impl(data_feature_factory, data_features, context)
            ret["data_feature_factory"] = data_feature_factory

        if device_features := features.get("device"):
            device_feature_factory = FeatureExtractorFactory()
            self._add_features_impl(device_feature_factory, device_features, context)
            ret["device_feature_factory"] = device_feature_factory

        if task_task_features := features.get("task_task"):
            task_task_feature_factory = EdgeFeatureExtractorFactory()
            self._add_features_impl(task_task_feature_factory, task_task_features, context)
            ret["task_task_feature_factory"] = task_task_feature_factory

        if task_data_features := features.get("task_data"):
            task_data_feature_factory = EdgeFeatureExtractorFactory()
            self._add_features_impl(task_data_feature_factory, task_data_features, context)
            ret["task_data_feature_factory"] = task_data_feature_factory

        if task_device_features := features.get("task_device"):
            task_device_feature_factory = EdgeFeatureExtractorFactory()
            self._add_features_impl(task_device_feature_factory, task_device_features, context)
            ret["task_device_feature_factory"] = task_device_feature_factory

        if data_device_features := features.get("data_device"):
            data_device_feature_factory = EdgeFeatureExtractorFactory()
            self._add_features_impl(data_device_feature_factory, data_device_features, context)
            ret["data_device_feature_factory"] = data_device_feature_factory

        if device_device_features := features.get("device_device"):
            device_device_feature_factory = EdgeFeatureExtractorFactory()
            self._add_features_impl(device_device_feature_factory, device_device_features, context)
            ret["device_device_feature_factory"] = device_device_feature_factory

        return ret

    def _add_features_impl(self, factory, feature_list, context: Optional[Dict[str, Any]] = None):
        if feature_list is None or len(feature_list) == 0:
            logger.debug("No features to add: feature_list is empty or None.")
            return None
        for feat in feature_list:
            if isinstance(feat, str):
                logger.debug(f"Adding feature '{feat}' from trip module.")
                factory.add(getattr(trip, feat))
            elif isinstance(feat, (dict, DictConfig)):
                name = feat.get("name")
                args = feat.get("args", [])
                if not isinstance(args, (list, ListConfig)):
                    args = [args]
                resolved_args = [self._resolve_arg(arg, context) for arg in args]
                logger.debug(f"Adding feature '{name}' from trip module with args {resolved_args}.")
                factory.add(getattr(trip, name), *resolved_args)
            else:
                logger.warning(f"Unknown feature type: {type(feat)}. Feature: {feat}")
