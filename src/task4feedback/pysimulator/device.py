from ..legacy_types import Architecture, Device, TaskID, TaskState, ResourceType, Time
from dataclasses import dataclass, field
from .queue import *
from .datapool import *
from enum import IntEnum
from typing import List, Dict, Set, Tuple, Optional, Type
from fractions import Fraction
from decimal import Decimal
from collections import defaultdict as DefaultDict

from .eviction.base import EvictionPool
from .eviction.lru import LRUEvictionPool
from .resourceset import FasterResourceSet, Numeric
from copy import deepcopy


@dataclass(slots=True)
class DeviceStats:
    active_movement: int = 0
    active_compute: int = 0

    last_active_compute: Time = field(default_factory=Time)
    last_active_movement: Time = field(default_factory=Time)

    idle_time_compute: Time = field(default_factory=Time)
    idle_time_movement: Time = field(default_factory=Time)
    idle_time: Time = field(default_factory=Time)

    outgoing_transfers: int = 0
    incoming_transfers: int = 0

    next_free_compute: Dict[TaskState, Time] = field(
        default_factory=lambda: DefaultDict(Time)
    )

    next_free: Dict[TaskState, Time] = field(default_factory=lambda: DefaultDict(Time))


@dataclass(slots=True)
class SimulatedDevice:
    name: Device
    resources: FasterResourceSet
    stats: DeviceStats = field(default_factory=DeviceStats)
    eviction_pool_type: Type[EvictionPool] = LRUEvictionPool
    eviction_targets: List[Device] = field(default_factory=list)
    memory_space: Device = field(init=False)
    init: bool = True

    def __post_init__(self):
        if self.init:
            self.eviction_targets = [Device(Architecture.CPU, 0)]
            self.memory_space = self.name
            self.init = False

    def __str__(self) -> str:
        return f"Device({self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, SimulatedDevice):
            return self.name == other.name
        elif isinstance(other, Device):
            return self.name == other

    def __lt__(self, other):
        if isinstance(other, SimulatedDevice):
            return self.name < other.name
        elif isinstance(other, Device):
            return self.name < other

    def __getitem__(self, key: ResourceType) -> Numeric:  # type: ignore
        if key == ResourceType.VCU:
            return self.resources.vcus
        elif key == ResourceType.MEMORY:
            return self.resources.memory
        elif key == ResourceType.COPY:
            return self.resources.copy

    def __deepcopy__(self, memo):
        return SimulatedDevice(
            name=self.name,
            resources=deepcopy(self.resources),
            stats=deepcopy(self.stats),
            eviction_pool_type=self.eviction_pool_type,
            eviction_targets=deepcopy(self.eviction_targets),
        )
