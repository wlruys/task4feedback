from ..types import Architecture, Device, TaskID, TaskState, ResourceType, Time
from dataclasses import dataclass, field
from .queue import *
from .datapool import *
from enum import IntEnum
from typing import List, Dict, Set, Tuple, Optional, Self, Type
from fractions import Fraction
from decimal import Decimal
from collections import defaultdict as DefaultDict

from .eviction.base import EvictionPool
from .eviction.lru import LRUEvictionPool


Numeric = int | float | Fraction | Decimal

resource_names = {
    ResourceType.VCU: "vcu",
    ResourceType.MEMORY: "memory",
    ResourceType.COPY: "copy",
}


@dataclass(slots=True, init=False)
class FasterResourceSet:
    vcus: Fraction = Fraction(0)
    memory: int = 0
    copy: int = 0

    def __init__(self, vcus: Numeric, memory: int, copy: int):
        self.vcus = Fraction(vcus)
        self.memory = memory
        self.copy = copy

    def __add__(self, other: Self):
        return FasterResourceSet(
            self.vcus + other.vcus, self.memory + other.memory, self.copy + other.copy
        )

    def __sub__(self, other: Self):
        return FasterResourceSet(
            self.vcus - other.vcus, self.memory - other.memory, self.copy - other.copy
        )

    def __iadd__(self, other: Self):
        self.vcus += other.vcus
        self.memory += other.memory
        self.copy += other.copy
        return self

    def __isub__(self, other: Self):
        self.vcus -= other.vcus
        self.memory -= other.memory
        self.copy -= other.copy
        return self

    def __lt__(self, other: Self) -> bool:
        status = (
            self.vcus < other.vcus
            and self.memory < other.memory
            and self.copy < other.copy
        )
        return status

    def __le__(self, other: Self) -> bool:
        status = (
            self.vcus <= other.vcus
            and self.memory <= other.memory
            and self.copy <= other.copy
        )
        return status

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Self):
            return (
                self.vcus == other.vcus
                and self.memory == other.memory
                and self.copy == other.copy
            )
        else:
            return False

    def __str__(self) -> str:
        return f"ResourceSet(vcus={self.vcus}, memory={self.memory}, copy={self.copy})"

    def verify(self):
        if self.vcus < 0 or self.memory < 0 or self.copy < 0:
            raise ValueError(f"ResourceSet {self} contains negative value.")


@dataclass(slots=True, init=False)
class ResourceSet:
    store: DefaultDict[ResourceType, Numeric] = field(
        default_factory=lambda: DefaultDict(int)
    )

    def __init__(self, vcus: Numeric, memory: int, copy: int):
        self.store = DefaultDict(int)

        self.store[ResourceType.VCU] = Fraction(vcus)
        self.store[ResourceType.MEMORY] = memory
        self.store[ResourceType.COPY] = copy

    def __getitem__(self, key: ResourceType) -> Numeric:
        return self.store[key]

    def __setitem__(self, key: ResourceType, value: Numeric):
        self.store[key] = value

    def __iter__(self):  # For unpack operator
        return iter(self.store)

    def add_types(self, other: Self, resource_types: List[ResourceType]) -> Self:
        for key in resource_types:
            if key in other.store and key in self.store:
                self.store[key] += other.store[key]
        return self

    def add_all(self, other: Self) -> Self:
        for key in self.store:
            if key in other.store and key in self.store:
                self.store[key] += other.store[key]
        return self

    def subtract_types(self, other: Self, resource_types: List[ResourceType]) -> Self:
        for key in resource_types:
            if key in other.store and key in self.store:
                self.store[key] -= other.store[key]
        return self

    def subtract_all(self, other: Self) -> Self:
        for key in self.store:
            if key in other.store and key in self.store:
                self.store[key] -= other.store[key]
        return self

    def verify(self, max_resources: Optional[Self] = None):
        for key in self.store:
            if self.store[key] < 0:
                raise ValueError(
                    f"ResourceSet {self} contains negative value for {key}"
                )

        if max_resources is not None:
            for key in self.store:
                if self.store[key] > max_resources.store[key]:
                    raise ValueError(
                        f"ResourceSet {self} exceeds maximum resources {max_resources}"
                    )

    def __str__(self) -> str:
        return f"ResourceSet({self.store})"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: Self) -> bool:
        for key in other.store:
            if self.store[key] >= other.store[key]:
                return False
        return True

    def __le__(self, other: Self) -> bool:
        for key in other.store:
            if self.store[key] > other.store[key]:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Self):
            for key in other.store:
                if self.store[key] != other.store[key]:
                    return False
            return True
        else:
            return False

    def __str__(self):
        string = f"ResourceSet("
        for key in self.store:
            string += f"{resource_names[key]}={self.store[key]} "
        string += ")"
        return string

    def __add__(self, other: Self):
        return ResourceSet(0, 0, 0).add_all(self).add_all(other)

    def __sub__(self, other: Self):
        return ResourceSet(0, 0, 0).add_all(self).subtract_all(other)

    def __iadd__(self, other: Self) -> Self:
        return self.add_all(other)

    def __isub__(self, other: Self) -> Self:
        return self.subtract_all(other)


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
    datapool: DataPool = field(default_factory=DataPool)
    eviction_pool_type: Type[EvictionPool] = LRUEvictionPool
    eviction_pool: EvictionPool = field(init=False)
    eviction_targets: List[Device] = field(default_factory=list)
    memory_space: Device = field(init=False)

    def __post_init__(self):
        self.eviction_pool = self.eviction_pool_type()
        self.eviction_targets = [Device(Architecture.CPU, 0)]
        self.memory_space = self.name

    def __str__(self) -> str:
        return f"Device({self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __getitem__(self, key: ResourceType) -> Numeric:
        if key == ResourceType.VCU:
            return self.resources.vcus
        elif key == ResourceType.MEMORY:
            return self.resources.memory
        elif key == ResourceType.COPY:
            return self.resources.copy

    def add_data(self, data: SimulatedData):
        self.datapool.add(data)

    def remove_data(self, data: SimulatedData):
        self.datapool.remove(data)

    def add_evictable(self, data: SimulatedData):
        self.eviction_pool.add(data)

    def remove_evictable(self, data: SimulatedData):
        self.eviction_pool.remove(data)
