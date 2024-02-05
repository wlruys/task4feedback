from ..types import (
    Architecture,
    Device,
    TaskID,
    DataID,
    DataInfo,
    TaskState,
    ResourceType,
)
from typing import List, Dict, Set, Tuple, Optional, Sequence
from enum import IntEnum
from .device import SimulatedDevice, ResourceSet, FasterResourceSet
from dataclasses import dataclass, InitVar, field

import numpy as np

NamedDevice = Device | SimulatedDevice


class ResourceGroup(IntEnum):
    PERSISTENT = 0
    NONPERSISTENT = 1
    ALL = 2


@dataclass(slots=True)
class FasterResourcePool:
    devices: InitVar[Sequence[SimulatedDevice]]
    devicemap: Dict[Device, SimulatedDevice] = field(init=False)
    pool: Dict[Device, Dict[TaskState, FasterResourceSet]] = field(init=False)

    def __post_init__(self, devices: Sequence[SimulatedDevice]):
        self.pool = {}
        self.devicemap = {}
        for device in devices:
            self.pool[device.name] = {
                TaskState.MAPPED: FasterResourceSet(vcus=0, memory=0, copy=0),
                TaskState.RESERVED: FasterResourceSet(vcus=0, memory=0, copy=0),
                TaskState.LAUNCHED: FasterResourceSet(vcus=0, memory=0, copy=0),
            }
            self.devicemap[device.name] = device

    def _build_set(
        self, type: ResourceGroup, resources: FasterResourceSet
    ) -> FasterResourceSet:
        if type == ResourceGroup.PERSISTENT:
            resources = FasterResourceSet(vcus=0, memory=resources.memory, copy=0)
        elif type == ResourceGroup.NONPERSISTENT:
            resources = FasterResourceSet(
                vcus=resources.vcus, memory=0, copy=resources.copy
            )
        else:
            resources = resources

        return resources

    def add_device_resource(
        self,
        device: Device,
        pool_state: TaskState,
        type: ResourceGroup,
        resources: FasterResourceSet,
    ):
        resource_set = self.pool[device][pool_state]
        resources = self._build_set(type, resources)

        resource_set += resources
        resource_set.verify()

    def remove_device_resources(
        self,
        device: Device,
        pool_state: TaskState,
        type: ResourceGroup,
        resources: FasterResourceSet,
    ):
        resource_set = self.pool[device][pool_state]
        resources = self._build_set(type, resources)

        resource_set -= resources
        resource_set.verify()

    def add_resources(
        self,
        devices: Sequence[Device],
        state: TaskState,
        type: ResourceGroup,
        resources: Sequence[FasterResourceSet],
    ):
        for device, resource in zip(devices, resources):
            self.add_device_resource(device, state, type, resource)

    def remove_resources(
        self,
        devices: Sequence[Device],
        state: TaskState,
        type: ResourceGroup,
        resources: Sequence[FasterResourceSet],
    ):
        for device, resource in zip(devices, resources):
            self.remove_device_resources(device, state, type, resource)

    def check_device_resources(
        self,
        device: Device,
        state: TaskState,
        type: ResourceGroup,
        proposed_resources: FasterResourceSet,
    ) -> bool:
        if device not in self.pool:
            return False

        if state not in self.pool[device]:
            raise ValueError(
                f"Invalid state {state} for Device Resource Request. Valid states are {self.pool[device].keys()}"
            )
        max_resources = self.devicemap[device].resources
        current_resources = self.pool[device][state]
        proposed_resources = self._build_set(type, proposed_resources)
        # max_resources = self._build_set(type, max_resources)

        if max_resources >= (current_resources + proposed_resources):
            return True
        return False

    def check_resources(
        self,
        devices: Sequence[Device],
        state: TaskState,
        type: ResourceGroup,
        resources: Sequence[FasterResourceSet],
    ) -> bool:
        for device, resource in zip(devices, resources):
            if not self.check_device_resources(device, state, type, resource):
                return False
        return True

    def __str__(self) -> str:
        return f"FasterResourcePool({self.pool})"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(
        self, device: Device, state: Optional[TaskState] = None
    ) -> Dict[TaskState, FasterResourceSet] | FasterResourceSet:
        if state is None:
            return self.pool[device]
        else:
            return self.pool[device][state]

    def __setitem__(self, device: Device, state: TaskState, value: FasterResourceSet):
        self.pool[device][state] = value

    def __contains__(self, device: Device, state: TaskState) -> bool:
        return device in self.pool and state in self.pool[device]

    def print_device_status(self, device: Device):
        print(f"Device {device} has resources: {self.pool[device]}")


@dataclass(slots=True)
class ResourcePool:
    devices: InitVar[Sequence[SimulatedDevice]]
    devicemap: Dict[Device, SimulatedDevice] = field(init=False)
    pool: Dict[Device, Dict[TaskState, ResourceSet]] = field(init=False)

    def __post_init__(self, devices: Sequence[SimulatedDevice]):
        self.pool = {}
        self.devicemap = {}
        for device in devices:
            self.pool[device.name] = {
                TaskState.MAPPED: ResourceSet(vcus=0, memory=0, copy=0),
                TaskState.RESERVED: ResourceSet(vcus=0, memory=0, copy=0),
                TaskState.LAUNCHED: ResourceSet(vcus=0, memory=0, copy=0),
            }
            self.devicemap[device.name] = device

    def add_device_resource(
        self,
        device: Device,
        pool_state: TaskState,
        types: List[ResourceType],
        resources: ResourceSet,
    ):
        resource_set = self.pool[device][pool_state]
        resource_set.add_types(resources, types)
        resource_set.verify()

    def remove_device_resources(
        self,
        device: Device,
        pool_state: TaskState,
        types: List[ResourceType],
        resources: ResourceSet,
    ):
        resource_set = self.pool[device][pool_state]
        resource_set.subtract_types(resources, types)
        resource_set.verify()

    def add_resources(
        self,
        devices: Sequence[Device],
        state: TaskState,
        types: List[ResourceType],
        resources: Sequence[ResourceSet],
    ):
        for device, resource in zip(devices, resources):
            self.add_device_resource(device, state, types, resource)

    def remove_resources(
        self,
        devices: Sequence[Device],
        state: TaskState,
        types: List[ResourceType],
        resources: Sequence[ResourceSet],
    ):
        for device, resource in zip(devices, resources):
            self.remove_device_resources(device, state, types, resource)

    def check_device_resources(
        self,
        device: Device,
        state: TaskState,
        types: List[ResourceType],
        proposed_resources: ResourceSet,
    ) -> bool:
        if device not in self.pool:
            return False

        if state not in self.pool[device]:
            raise ValueError(
                f"Invalid state {state} for Device Resource Request. Valid states are {self.pool[device].keys()}"
            )
        max_resources = self.devicemap[device].resources
        current_resources = self.pool[device][state]

        for resourcekey in types:
            if (
                current_resources[resourcekey] + proposed_resources[resourcekey]
                > max_resources[resourcekey]
            ):
                return False
        return True

    def check_resources(
        self,
        devices: Sequence[Device],
        state: TaskState,
        types: List[ResourceType],
        resources: Sequence[ResourceSet],
    ) -> bool:
        for device, resource in zip(devices, resources):
            if not self.check_device_resources(device, state, types, resource):
                return False
        return True

    def __str__(self) -> str:
        return f"ResourcePool({self.pool})"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(
        self, device: Device, state: Optional[TaskState] = None
    ) -> Dict[TaskState, ResourceSet] | ResourceSet:
        if state is None:
            return self.pool[device]
        else:
            return self.pool[device][state]

    def __setitem__(self, device: Device, state: TaskState, value: ResourceSet):
        self.pool[device][state] = value

    def __contains__(self, device: Device, state: TaskState) -> bool:
        return device in self.pool and state in self.pool[device]

    def print_device_status(self, device: Device):
        print(f"Device {device} has resources: {self.pool[device]}")
