from dataclasses import dataclass
from typing import Optional, Type, Self
from .types import DeviceTuple, TaskTuple, DataBlockTuple, VariantTuple, ConnectionTuple
from task4feedback.fastsim2 import DeviceType


class VariantBuilder:
    @staticmethod
    def build_variant(arch: DeviceType, task: TaskTuple) -> Optional[VariantTuple]:
        memory_usage = 0
        vcu_usage = 1
        expected_time = 1000
        return VariantTuple(arch, memory_usage, vcu_usage, expected_time)


class TaskLabeler:
    @staticmethod
    def label(task: TaskTuple) -> int:
        return 0


class DataBlockTransformer:
    @staticmethod
    def transform(block: DataBlockTuple) -> Optional[DataBlockTuple]:
        return block
