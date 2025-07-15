from dataclasses import dataclass, field
from enum import Enum
from ..fastsim2 import DeviceType, BYTES_IN_POWER


@dataclass
class DeviceTuple:
    name: str
    global_id: int = -1
    local_id: int = -1
    arch: DeviceType = DeviceType.CPU
    memory: int = 0
    vcu: int = 0


@dataclass
class TaskTuple:
    id: int = 0
    name: str = ""
    tag: int = 0
    dependencies: list = field(default_factory=list)
    read: list = field(default_factory=list)
    write: list = field(default_factory=list)
    retire: list = field(default_factory=list)
    type: int = 0


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
