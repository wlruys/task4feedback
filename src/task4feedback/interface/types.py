from dataclasses import dataclass
from enum import Enum
from ..fastsim2 import DeviceType, BYTES_IN_POWER


@dataclass
class DeviceTuple:
    name: str
    global_id: int = -1
    local_id: int = -1
    arch: DeviceType = DeviceType.NONE
    memory: int = 0
    vcu: int = 0


@dataclass
class TaskTuple:
    id: int
    name: str
    tag: int
    dependencies: list
    read: list
    write: list
    type: int


@dataclass
class DataBlockTuple:
    id: int
    name: str
    size: int
    location: int
    tag: int = 0
    type: int = 0


@dataclass
class VariantTuple:
    arch: DeviceType
    memory_usage: int
    vcu_usage: int
    expected_time: int


@dataclass
class ConnectionTuple:
    source: DeviceTuple
    destination: DeviceTuple
    bandwidth: int
    latency: int
    max_connections: int


def _bytes_to_readable(size):
    base = BYTES_IN_POWER

    if size < base:
        return f"{size} B"
    if size < base**2:
        return f"{size / base} KB"
    if size < base**3:
        return f"{size / base**2} MB"
    if size < base**4:
        return f"{size / base**3} GB"
    return f"{size / base**4} TB"
