from __future__ import annotations
from ...types import DataID
from ..data import *
from dataclasses import dataclass, field
from typing import Dict

from .base import EvictionPool


@dataclass(slots=True)
class DataNode:
    data: Optional[DataInfo] = None
    next: Optional[DataNode] = None
    prev: Optional[DataNode] = None


@dataclass(slots=True)
class DataNodeList:
    head: DataNode = field(default_factory=DataNode)
    tail: DataNode = field(default_factory=DataNode)
    size: int = 0
    map: Dict[DataID, DataNode] = field(default_factory=dict)

    def __post_init__(self):
        self.head = DataNode()
        self.tail = DataNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def append(self, data: DataInfo) -> bool:
        if data.id in self.map:
            return False

        node = DataNode(data)
        self.map[data.id] = node

        node.next = self.tail
        node.prev = self.tail.prev

        assert self.tail.prev is not None

        self.tail.prev.next = node
        self.tail.prev = node
        self.size += 1

        return True

    def remove(self, data: DataInfo) -> bool:
        if data.id not in self.map:
            return False

        node = self.map[data.id]
        del self.map[data.id]

        assert node.prev is not None
        assert node.next is not None

        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

        return True

    def __iter__(self):
        node = self.head.next
        assert node is not None
        assert node.data is not None

        while node.data is not None:
            yield node.data
            node = node.next
            assert node is not None

    def __len__(self):
        return self.size

    def __str__(self):
        return f"DataNodeList({self.size}, {self.map.keys()})"

    def __repr__(self):
        return self.__str__()


@dataclass(slots=True)
class LRUEvictionPool(EvictionPool):
    datalist: DataNodeList = field(default_factory=DataNodeList)

    def add(self, data: SimulatedData):
        if self.datalist.append(data.info):
            # print("Adding {data.info} to eviction pool: {data.size}, {self.evictable_size}")
            self.evictable_size += data.size

    def remove(self, data: SimulatedData):
        if self.datalist.remove(data.info):
            # print(f"Removing {data.info} from eviction pool: {data.size}")
            self.evictable_size -= data.size
            assert self.evictable_size >= 0

    def peek(self) -> DataID:
        assert self.datalist.head.next is not None
        data = self.datalist.head.next.data
        assert data is not None
        return data.id

    def get(self) -> DataID:
        assert self.datalist.head.next is not None
        data = self.datalist.head.next.data
        assert data is not None
        self.datalist.remove(data)
        self.evictable_size -= data.size
        assert self.evictable_size >= 0
        return data.id

    def __len__(self):
        return len(self.datalist)

    def __contains__(self, data: SimulatedData | DataID):
        if isinstance(data, SimulatedData):
            return data.info.id in self.datalist.map
        elif isinstance(data, DataID):
            return data in self.datalist.map
        else:
            raise TypeError(f"Expected SimulatedData or DataID, got {type(data)}")

    def __str__(self):
        return f"LRUEvictionPool({self.evictable_size}, {self.datalist})"

    def __repr__(self):
        return self.__str__()

    def empty(self) -> bool:
        return len(self.datalist) == 0
