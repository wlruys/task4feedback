from __future__ import annotations
from ...types import DataID
from ..data import *
from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Set, Tuple, Union, Self


@dataclass(slots=True)
class EvictionPool:
    evictable_size: int = 0

    def add(self, data: SimulatedData):
        raise NotImplementedError

    def remove(self, data: SimulatedData):
        raise NotImplementedError

    def peek(self) -> DataID:
        raise NotImplementedError

    def get(self) -> DataID:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __contains__(self, data: SimulatedData | DataID):
        raise NotImplementedError

    def empty(self) -> bool:
        raise NotImplementedError
