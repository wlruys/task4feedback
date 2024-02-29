from ..types import Device, DataID
from typing import Dict, Set, Sequence
from dataclasses import dataclass, field, InitVar

from .eviction.base import DataPool
from .eviction.base import EvictionPool
from .device import SimulatedDevice
from .topology import SimulatedTopology


@dataclass(slots=True)
class DeviceDataPools:
    devices: InitVar[Sequence[SimulatedDevice]]
    device_datapool: Dict[Device, DataPool] = field(init=False)
    device_evictionpool: Dict[Device, EvictionPool] = field(init=False)

    def __post_init__(self, devices: Sequence[SimulatedDevice]):
        self.device_datapool = {}
        self.device_evictionpool = {}

        for device in devices:
            self.device_datapool[device.name] = DataPool()
            self.device_evictionpool[device.name] = device.eviction_pool_type()

    def add(self, device: Device, data: DataID):
        self.device_datapool[device].add(data)

    def remove(self, device: Device, data: DataID):
        self.device_datapool[device].remove(data)

    def add_evictable(self, device: Device, data: DataID, size: int):
        self.device_evictionpool[device].add(data, size)

    def remove_evictable(self, device: Device, data: DataID, size: int):
        self.device_evictionpool[device].remove(data, size)

    @property
    def pool(self) -> Dict[Device, DataPool]:
        return self.device_datapool

    @property
    def evictable(self) -> Dict[Device, EvictionPool]:
        return self.device_evictionpool
