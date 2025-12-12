import logging
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.profiler import record_function
from torch import Tensor

from tensordict import TensorDict, TensorDictBase
from torch_geometric.data import Batch, HeteroData
from torchrl.modules import ProbabilisticActor

from task4feedback.interface.wrappers import observation_to_heterodata_truncate


class InferenceTimingProbabilisticActor(ProbabilisticActor):
    """
    ProbabilisticActor wrapper that records wall-clock latency for action selection.
    """

    def __init__(self, *args, timing_key: str = "action_inference_time_s", store_in_tensordict: bool = True, sync_cuda: bool = False, log_timing: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_key = timing_key
        self.store_in_tensordict = store_in_tensordict
        self.sync_cuda = sync_cuda
        self.last_inference_time_s: Optional[float] = None
        self.log_timing = log_timing
        self._logger = logging.getLogger(__name__)

    def forward(self, tensordict: TensorDictBase):
        sync = self.sync_cuda and torch.cuda.is_available()
        if sync:
            torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        with record_function("ProbabilisticActor.action_selection"):
            out = super().forward(tensordict)
        if sync:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start_ns) / 1e9
        self.last_inference_time_s = elapsed
        if self.log_timing:
            self._logger.info("Action selection latency: %.3f ms", elapsed * 1e3)

        if self.store_in_tensordict and isinstance(out, TensorDictBase):
            device = getattr(out, "device", None)
            batch_shape = tuple(out.batch_size) if out.batch_size else ()
            timing_tensor = torch.full(batch_shape, elapsed, device=device) if batch_shape else torch.tensor(elapsed, device=device)
            out.set(self.timing_key, timing_tensor)
        return out


class LogitInferenceTimingWrapper(nn.Module):
    """
    Wraps a module that produces logits to measure forward-pass latency before sampling.
    """

    def __init__(self, module: nn.Module, timing_key: str = "logit_inference_time_s", store_in_tensordict: bool = True, sync_cuda: bool = False, log_timing: bool = False, conversion_timing_key: Optional[str] = "data_conversion_time_s", subtract_conversion_time: bool = True):
        super().__init__()
        self.module = module
        self.timing_key = timing_key
        self.store_in_tensordict = store_in_tensordict
        self.sync_cuda = sync_cuda
        self.last_logit_inference_time_s: Optional[float] = None
        self.last_conversion_time_s: Optional[float] = None
        self.conversion_timing_key = conversion_timing_key
        self.subtract_conversion_time = subtract_conversion_time
        self.log_timing = log_timing
        self._logger = logging.getLogger(__name__)
        self._in_keys = getattr(module, "in_keys", None)
        self._out_keys = getattr(module, "out_keys", None)

    @property
    def in_keys(self):
        return getattr(self.module, "in_keys", self._in_keys)

    @in_keys.setter
    def in_keys(self, value):
        self._in_keys = value
        if hasattr(self.module, "in_keys"):
            self.module.in_keys = value

    @property
    def out_keys(self):
        return getattr(self.module, "out_keys", self._out_keys)

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys = value
        if hasattr(self.module, "out_keys"):
            self.module.out_keys = value

    def forward(self, tensordict: TensorDictBase):
        sync = self.sync_cuda and torch.cuda.is_available()
        if sync:
            torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        with record_function("Actor.logit_inference"):
            out = self.module(tensordict)
        if sync:
            torch.cuda.synchronize()
        elapsed_total = (time.perf_counter_ns() - start_ns) / 1e9
        conversion_time = self._collect_conversion_time()
        inference_elapsed = max(0.0, elapsed_total - conversion_time) if (self.subtract_conversion_time and conversion_time is not None) else elapsed_total

        self.last_conversion_time_s = conversion_time
        self.last_logit_inference_time_s = inference_elapsed
        if self.log_timing:
            self._logger.info("Logit inference latency (excl. conversion): %.3f ms", inference_elapsed * 1e3)
            if conversion_time is not None and conversion_time > 0:
                self._logger.info("Data conversion latency: %.3f ms", conversion_time * 1e3)

        if self.store_in_tensordict and isinstance(out, TensorDictBase):
            device = getattr(out, "device", None)
            batch_shape = tuple(out.batch_size) if out.batch_size else ()
            timing_tensor = torch.full(batch_shape, inference_elapsed, device=device) if batch_shape else torch.tensor(inference_elapsed, device=device)
            out.set(self.timing_key, timing_tensor)
            if self.conversion_timing_key is not None and conversion_time is not None:
                conv_tensor = torch.full(batch_shape, conversion_time, device=device) if batch_shape else torch.tensor(conversion_time, device=device)
                out.set(self.conversion_timing_key, conv_tensor)
        return out

    def _collect_conversion_time(self) -> Optional[float]:
        seen = set()
        total = 0.0
        found = False
        for m in self.module.modules():
            if hasattr(m, "last_conversion_time_s"):
                if id(m) in seen:
                    continue
                seen.add(id(m))
                ct = getattr(m, "last_conversion_time_s", None)
                if ct is not None:
                    total += float(ct)
                    found = True
        return total if found else None


class BatchWrapper(nn.Module):
    def __init__(self, network: nn.Module, device: Optional[str] = "cpu"):
        super().__init__()
        self.network = network
        self.register_parameter("dummy_param_0", nn.Parameter(torch.randn(1)))

    def _is_batch(self, obs: TensorDict) -> bool:
        if not obs.batch_size:
            return False
        return True

    def _convert_to_heterodata(self, obs: TensorDict, is_batch: bool = False) -> HeteroData | Batch:
        if not is_batch:
            return obs["hetero_data"]

        hetero_data_list = obs["hetero_data"]
        batches = []
        for hlist in hetero_data_list:
            batches.append(Batch.from_data_list(hlist))
        return batches

    def forward(self, obs: TensorDict):
        is_batch = self._is_batch(obs)
        with torch.no_grad():
            data = self._convert_to_heterodata(obs, is_batch)
        out = self.network(data)
        return out


class HeteroDataWrapper(nn.Module):
    def __init__(self, device: Optional[str] = "cpu"):
        super().__init__()
        self.register_parameter("dummy_param_0", nn.Parameter(torch.randn(1)))
        self.last_conversion_time_s: Optional[float] = None

    def _is_batch(self, obs: TensorDict) -> bool:
        if not obs.batch_size:
            return False
        return True

    def _convert_to_heterodata(self, obs: TensorDict, is_batch: bool = False, actions: Optional[TensorDict] = None) -> HeteroData:
        is_cuda = any(p.is_cuda for p in self.parameters())

        if not is_batch:
            if actions is not None:
                _obs = observation_to_heterodata_truncate(obs, actions=actions)
            else:
                _obs = observation_to_heterodata_truncate(obs)

            if is_cuda:
                _obs = _obs.to("cuda", non_blocking=True)

            return _obs

        self.batch_size = obs.batch_size
        obs = obs.reshape(-1)
        _h_data = []
        for i in range(obs.batch_size[0]):
            if actions is not None:
                _obs = observation_to_heterodata_truncate(obs[i], actions=actions[i])
            else:
                _obs = observation_to_heterodata_truncate(obs[i])
            _h_data.append(_obs)

        batch_obs = Batch.from_data_list(_h_data)

        if isinstance(batch_obs, tuple):
            batch_obs = batch_obs[0]

        if is_cuda:
            batch_obs = batch_obs.to("cuda", non_blocking=True)

        return batch_obs

    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)

        with torch.no_grad():
            self.last_conversion_time_s = None
            start_ns = time.perf_counter_ns()
            data = self._convert_to_heterodata(obs, is_batch, actions=actions)
            self.last_conversion_time_s = (time.perf_counter_ns() - start_ns) / 1e9

        return data
