from __future__ import annotations
from ..types import DataID
from .data import *
from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Set, Tuple, Union, Self

@dataclass(slots=True)
class DataPool:
    datalist: Set[DataID] = field(default_factory=set)

    def add(self, data: SimulatedData | DataID):
        if isinstance(data, SimulatedData):
            data = data.name
        self.datalist.add(data)

    def remove(self, data: SimulatedData | DataID):
        if isinstance(data, SimulatedData):
            data = data.name
        self.datalist.remove(data)

    def __contains__(self, data: SimulatedData | DataID):
        if isinstance(data, SimulatedData):
            data = data.name
        return data in self.datalist

    def __len__(self):
        return len(self.datalist)
