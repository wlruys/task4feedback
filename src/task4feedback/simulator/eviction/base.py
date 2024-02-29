from __future__ import annotations
from ...types import *
from dataclasses import dataclass, field


@dataclass(slots=True)
class DataPool:
    pool: Set[DataID] = field(default_factory=set)

    def add(self, data: DataID):
        self.pool.add(data)

    def remove(self, data: DataID):
        self.pool.remove(data)

    def __contains__(self, data: DataID):
        return data in self.pool

    def __len__(self):
        return len(self.pool)


@dataclass(slots=True)
class EvictionPool:
    evictable_size: int = 0

    def add(self, data: DataID, size: int):
        raise NotImplementedError

    def remove(self, data: DataID, size: int):
        raise NotImplementedError

    def peek(self) -> DataID:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __contains__(self, data: DataID):
        raise NotImplementedError

    def empty(self) -> bool:
        raise NotImplementedError
