from ..legacy_types import Device, DataID
from typing import Dict, Sequence
from dataclasses import dataclass
from .eviction.base import DataPool
from .eviction.base import EvictionPool
from .device import SimulatedDevice
from copy import deepcopy


@dataclass(slots=True)
class DeviceDataPools:
    devices: Sequence[SimulatedDevice] = None
    device_datapool: Dict[Device, DataPool] = None
    device_evictionpool: Dict[Device, EvictionPool] = None
    init: bool = True

    def __post_init__(self):
        if self.init:
            self.device_datapool = {}
            self.device_evictionpool = {}

            for device in self.devices:
                self.device_datapool[device.name] = DataPool()
                self.device_evictionpool[device.name] = device.eviction_pool_type()
            self.init = False

    def add(self, device: Device, data: DataID):
        self.device_datapool[device].add(data)

    def remove(self, device: Device, data: DataID):
        self.device_datapool[device].remove(data)

    def add_evictable(self, device: Device, data: DataID, size: int):
        self.device_evictionpool[device].add(data, size)

    def remove_evictable(self, device: Device, data: DataID, size: int) -> bool:
        return self.device_evictionpool[device].remove(data, size)

    @property
    def pool(self) -> Dict[Device, DataPool]:
        return self.device_datapool

    @property
    def evictable(self) -> Dict[Device, EvictionPool]:
        return self.device_evictionpool

    def __deepcopy__(self, memo):
        device_datapool = deepcopy(self.device_datapool)
        device_evictionpool = deepcopy(self.device_evictionpool)
        return DeviceDataPools(
            devices=self.devices,
            device_datapool=device_datapool,
            device_evictionpool=device_evictionpool,
        )
