from ..types import Architecture, Device, TaskID, TaskState, ResourceType, Time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Set, Tuple, Optional, Self, Type
from fractions import Fraction
from decimal import Decimal
from collections import defaultdict as DefaultDict

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

    def __len__(self):
        return (self.vcus > 0) + (self.memory > 0) + (self.copy > 0)


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
