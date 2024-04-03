from ..types import (
    Architecture,
    Device,
    TaskID,
    DataID,
    DataInfo,
    TaskState,
    TaskStatus,
    AccessType,
)
from typing import List, Dict, Set, Tuple, Optional, Sequence
from dataclasses import dataclass, field, InitVar
from collections import defaultdict as DefaultDict
from enum import IntEnum
from ..logging import logger
from ..types import Time
from .datapool import *


class DataMovementFlags(IntEnum):
    """
    Status of data movement to a device
    """

    FIRST_MOVE = 0
    """ Data is being moved to the device for the first time """
    ALREADY_MOVING = 1
    """ Data is already being moved to the device """
    ALREADY_THERE = 2
    """ Data is already on the device due to a previous move """


class DataState(IntEnum):
    """
    Internal state of data on a device (not all states are used)
    Note: Not to be confused with TaskState
    """

    NONE = -1
    """ Data is not present on the device """
    PLANNED = 0
    """ Data is planned to be present on the device (not on the device yet, but requested by a task)"""
    MOVING = 1
    """ Data is in transit to the device """
    VALID = 2
    """ Data is valid on the device """
    EVICTABLE = 3
    """ Data is evictable from the device """
    STALE = 3
    """ Data is stale on the device """


class DataUses(IntEnum):
    """
    State representing how data is being used by a task
    """

    MAPPED = 1
    """ A mapped compute task is using the data """
    RESERVED = 2
    """ A reserved compute task is using the data """
    MOVING_TO = 3
    """ A data task is currently moving the data to the device """
    MOVING_FROM = 4
    """ A data task is currently moving the data from the device """
    USED = 5
    """ A launched compute task is using the data """
    CHECKING = 6
    """ A task is locking the data during a check """
    EVICTING = 7
    """ A task is evicting the data from the device """


TaskStateToUse = {}
TaskStateToUse[TaskState.MAPPED] = DataUses.MAPPED
TaskStateToUse[TaskState.RESERVED] = DataUses.RESERVED
TaskStateToUse[TaskState.LAUNCHED] = DataUses.USED
TaskStateToUse[TaskState.COMPLETED] = DataUses.USED

NonEvictableUses = [
    DataUses.EVICTING,
    DataUses.RESERVED,
    DataUses.MOVING_TO,
    DataUses.MOVING_FROM,
    DataUses.USED,
]

from copy import deepcopy


@dataclass(slots=True)
class DataStats:
    read_count: int = 0
    write_count: int = 0
    move_count: int = 0
    move_time: Time = field(default_factory=Time)

    def __deepcopy__(self, memo):
        return DataStats(
            read_count=self.read_count,
            write_count=self.write_count,
            move_count=self.move_count,
            move_time=self.move_time,
        )


@dataclass(slots=True)
class DataUse:
    name: DataID
    devices_uses_tasks: DefaultDict[Device, DefaultDict[DataUses, Set[TaskID]]]
    tasks_uses_devices: DefaultDict[TaskID, DefaultDict[DataUses, Set[Device]]]
    nonevictable_usage_count: DefaultDict[Device, int]
    eviction_tasks: Set[TaskID]
    size: int = 0
    init: bool = True

    def __post_init__(self):
        if self.init:
            self.nonevictable_usage_count = DefaultDict(lambda: 0)
            self.devices_uses_tasks = DefaultDict(lambda: DefaultDict(lambda: set()))  # type: ignore
            self.tasks_uses_devices = DefaultDict(lambda: DefaultDict(lambda: set()))  # type: ignore
            self.eviction_tasks = set()
            self.init = False

    def __deepcopy__(self, memo):
        devices_uses_tasks = deepcopy(self.device_uses_tasks)
        tasks_uses_devices = deepcopy(self.tasks_uses_devices)
        nonevictable_usage_count = deepcopy(self.nonevictable_usage_count)
        eviction_tasks = deepcopy(self.eviction_tasks)
        return DataUse(
            name=self.name,
            size=self.size,
            devices_uses_tasks=devices_uses_tasks,
            tasks_uses_devices=tasks_uses_devices,
            nonevictable_usage_count=nonevictable_usage_count,
            eviction_tasks=eviction_tasks,
            init=self.init,
        )

    def add_task_use_to_device(
        self,
        task: TaskID,
        use: DataUses,
        device: Device,
        pools: DeviceDataPools,
    ):
        print(f"Adding {task} to {use.name} of {self.name} on {device}.")
        self.devices_uses_tasks[device][use].add(task)
        self.tasks_uses_devices[task][use].add(device)

        if use in NonEvictableUses:
            self.nonevictable_usage_count[device] += 1
            print(
                f"Removing {self.name} from eviction pool on device {device}. Last usage was {task} with use {str(use.name)}."
            )
            pools.remove_evictable(device, self.name, self.size)

    def remove_task_use_from_device(
        self, task: TaskID, use: DataUses, device: Device, pools: DeviceDataPools
    ):
        print(f"Removing {task} from {use.name} of {self.name} on {device}.")
        self.devices_uses_tasks[device][use].remove(task)
        self.tasks_uses_devices[task][use].remove(device)

        if use in NonEvictableUses:
            self.nonevictable_usage_count[device] -= 1
            assert self.nonevictable_usage_count[device] >= 0, f"Negative usage count."

        if self.is_evictable(device) and not use == DataUses.EVICTING:
            print(
                f"Adding {self.name} to eviction pool on device {device}. Last usage was {task} with use {str(use.name)}."
            )
            pools.add_evictable(device, self.name, self.size)

    def get_devices_from_task_use(self, task: TaskID, use: DataUses) -> Set[Device]:
        return self.tasks_uses_devices[task][use]

    def get_tasks_from_device_use(self, device: Device, use: DataUses) -> Set[TaskID]:
        return self.devices_uses_tasks[device][use]

    def add_task_use_to_devices(
        self,
        task: TaskID,
        use: DataUses,
        devices: Sequence[Device],
        pools: DeviceDataPools,
    ):
        for device in devices:
            self.add_task_use_to_device(task, use, device, pools)

    def remove_task_use(self, task: TaskID, use: DataUses, pools: DeviceDataPools):
        devices = list(self.get_devices_from_task_use(task, use))
        for device in devices:
            self.remove_task_use_from_device(task, use, device, pools)

    def add_eviction_task(self, task: TaskID, device: Device, pools: DeviceDataPools):
        self.eviction_tasks.add(task)
        self.add_task_use_to_device(task, DataUses.EVICTING, device, pools)

    def remove_eviction_task(
        self, task: TaskID, device: Device, pools: DeviceDataPools
    ):
        self.eviction_tasks.remove(task)
        self.remove_task_use(task, DataUses.EVICTING, pools)

    def is_evictable(self, device: Device) -> bool:
        return self.nonevictable_usage_count[device] == 0

    def __rich_repr__(self):
        yield "tasks", self.devices_uses_tasks

    def is_used_on_device(self, use: DataUses, device: Device) -> bool:
        return len(self.devices_uses_tasks[device][use]) > 0

    def get_use_count_on_device(self, use: DataUses, device: Device) -> int:
        return len(self.devices_uses_tasks[device][use])


@dataclass(slots=True)
class DataStatus:
    id: DataID
    size: int
    devices: Sequence[Device]
    device2state: Dict[TaskState, Dict[Device, DataState]]
    state2device: Dict[TaskState, Dict[DataState, Set[Device]]]
    uses: DataUse
    init: bool = True

    def __post_init__(self):
        if self.init:
            self.device2state = {}
            self.state2device = {}

            for state in [
                TaskState.SPAWNED,
                TaskState.MAPPED,
                TaskState.RESERVED,
                TaskState.LAUNCHED,
            ]:
                self.device2state[state] = {}
                self.state2device[state] = {}

                for device in self.devices:
                    self.device2state[state][device] = DataState.NONE

                for data_state in DataState:
                    self.state2device[state][data_state] = set()
            self.init = False

        self.uses = DataUse(
            name=self.id,
            size=self.size,
            devices_uses_tasks=None,
            tasks_uses_devices=None,
            nonevictable_usage_count=None,
            eviction_tasks=None,
        )

    def __deepcopy__(self, memo):
        device2state = {
            k: {d: v for d, v in v2.items()} for k, v2 in self.device2state.items()
        }
        state2device = {
            k: {d: {v for v in v2} for d, v2 in v3.items()}
            for k, v3 in self.state2device.items()
        }
        uses = deepcopy(uses)

        return DataStatus(
            id=self.id,
            size=size,
            devices=self.devices,
            device2state=device2state,
            state2device=state2device,
            uses=uses,
            init=self.init,
        )

    def set_data_state(
        self, device: Device, state: TaskState, data_state: DataState, initial=False
    ) -> Optional[DataState]:
        prior_state = None

        if not initial:
            prior_state = self.device2state[state][device]

            if prior_state == data_state:
                return prior_state

            if logger.ENABLE_LOGGING:
                logger.data.debug(
                    f"Setting data state of {self.id} on device {device} from {prior_state.name} to {data_state.name} in phase {state.name}",
                    extra=dict(
                        data=self.id,
                        device=device,
                        state=state,
                        data_state=data_state,
                        prior_state=prior_state,
                    ),
                )

            self.state2device[state][prior_state].remove(device)

        self.device2state[state][device] = data_state
        self.state2device[state][data_state].add(device)

        return prior_state

    def check_data_state(
        self, device: Device, state: TaskState, data_state: DataState
    ) -> bool:
        return self.device2state[state][device] == data_state

    def get_data_state(self, device: Device, state: TaskState) -> DataState:
        return self.device2state[state][device]

    def get_devices_from_states(
        self, states: Sequence[TaskState], data_states: Sequence[DataState]
    ) -> List[Device]:
        devices = []

        for task_state in states:
            for data_state in data_states:
                devices.extend(self.state2device[task_state][data_state])

        return devices

    def get_device_set_from_state(
        self, state: TaskState, data_state: DataState
    ) -> Set[Device]:
        return self.state2device[state][data_state]

    def add_task(
        self, device: Device, task: TaskID, use: DataUses, pools: DeviceDataPools
    ):
        self.uses.add_task_use_to_device(task, use, device, pools)

    def add_eviction_task(self, task: TaskID, device: Device, pools: DeviceDataPools):
        self.uses.add_eviction_task(task, device, pools)

    def remove_eviction_task(
        self, task: TaskID, device: Device, pools: DeviceDataPools
    ):
        self.uses.remove_eviction_task(task, device, pools)

    def remove_task_from_device(
        self, device: Device, task: TaskID, use: DataUses, pools: DeviceDataPools
    ):
        self.uses.remove_task_use_from_device(task, use, device, pools)

    def remove_task(self, task: TaskID, use: DataUses, pools: DeviceDataPools):
        self.uses.remove_task_use(task, use, pools)

    def get_tasks_from_usage(self, device: Device, use: DataUses) -> Set[TaskID]:
        return self.uses.get_tasks_from_device_use(device, use)

    def is_evictable(self, device: Device) -> bool:
        return self.uses.is_evictable(device)

    def is_used(self, device: Device, use: DataUses) -> bool:
        return self.uses.is_used_on_device(use, device)

    def get_use_count(self, device: Device, use: DataUses) -> int:
        return self.uses.get_use_count_on_device(use, device)

    def advance_state(
        self,
        device: Device,
        state: TaskState,
        new_data_state: DataState,
        force: bool = False,
    ) -> bool:
        if force:
            self.set_data_state(device, state, new_data_state)
            return True  # Change
        else:
            prior_data_state = self.get_data_state(device, state)
            if new_data_state > prior_data_state:
                self.set_data_state(device, state, new_data_state)
                return True  # Change
        return False  # No change

    def verify_write(self, device: Device, state: TaskState, check_use: bool = True):
        status = self.device2state[state]

        # Ensure no device is moving the data
        for device in status.keys():
            if status[device] == DataState.MOVING:
                raise RuntimeError(
                    f"Cannot write while device {device} is moving data {self.id}. Status: {status}"
                )

        # Ensure no device is using the data if check_use is True
        if check_use:
            for device in status.keys():
                if self.is_used(device=device, use=DataUses.USED):
                    raise RuntimeError(
                        f"Cannot write while a device {device} that is using that data {self.id}. Status: {status}"
                    )

    def get_eviction_target(
        self,
        source_device: Device,
        potential_targets: Sequence[Device],
        state: TaskState,
    ) -> Device:
        print(
            f"Getting eviction target for {self.id} from {source_device} to {potential_targets}"
        )
        valid_copies = self.get_device_set_from_state(state, DataState.VALID)
        target_device = source_device
        current_state = self.get_data_state(source_device, state)
        print("Current state of eviction target on source device", current_state.name)
        assert (
            current_state == DataState.VALID
        ), f"Data {self.id} must be VALID to be evicted, but is {current_state} on {source_device} in phase {state}."

        if len(valid_copies) == 1:
            target_device = potential_targets[0]

        # print("Target Device", target_device)
        # print("Source Device", source_device)

        return target_device

    def initialize_eviction(
        self,
        task: TaskID,
    ) -> None:
        self.uses.add_eviction_task(task)

    def start_eviction(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        state: TaskState,
        pools: DeviceDataPools,
        verify: bool = False,
        verbose: bool = False,
    ) -> Tuple[DataState, List[Device]]:
        assert state == TaskState.LAUNCHED

        if logger.ENABLE_LOGGING:
            logger.data.info(
                f"Start eviction of {self.id} from device {source_device} to {target_device}.",
                extra=dict(
                    task=task,
                    data=self.id,
                    source=source_device,
                    target=target_device,
                    state=state,
                ),
            )

        current_state = self.get_data_state(source_device, state)
        assert (
            current_state == DataState.VALID
        ), f"Data {self.id} must be VALID to be evicted, but is {current_state} on {source_device}."

        if source_device != target_device:
            self.start_move(task, source_device, target_device, pools, verbose=verbose)

        return current_state, [source_device]

    def finish_eviction(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        state: TaskState,
        pools: DeviceDataPools,
        verify: bool = False,
        verbose: bool = False,
    ):
        assert state == TaskState.LAUNCHED

        if logger.ENABLE_LOGGING:
            logger.data.info(
                f"Finish eviction of {self.id} from device {source_device} to {target_device}.",
                extra=dict(
                    task=task,
                    data=self.id,
                    source=source_device,
                    target=target_device,
                    state=state,
                ),
            )

        current_state = self.get_data_state(source_device, state)

        if source_device != target_device:
            self.finish_move(task, source_device, target_device, pools, verbose=verbose)

        self.set_data_state(source_device, state, DataState.NONE)
        self.remove_eviction_task(task, source_device, pools)

        return current_state, [source_device]

    def evict(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        state: TaskState,
        pools: DeviceDataPools,
        verify: bool = False,
        verbose: bool = False,
    ) -> Tuple[DataState, List[Device]]:
        if logger.ENABLE_LOGGING:
            logger.data.info(
                f"Evicting data {self.id} from device {source_device} to device {target_device} for task {task} in phase {state}",
                extra=dict(
                    task=task,
                    data=self.id,
                    source=source_device,
                    target=target_device,
                    state=state,
                ),
            )

        if state == TaskState.LAUNCHED:
            raise ValueError(
                f"Incorrect usage. Use start_eviction and finish_eviction for {state} phase."
            )

        current_state = self.get_data_state(source_device, state)
        assert (
            current_state == DataState.VALID
        ), f"Data {self.id} must be VALID to be evicted, but is {current_state} on {source_device}."

        if source_device != target_device:
            self.set_data_state(target_device, state, DataState.VALID)

        self.set_data_state(source_device, state, DataState.NONE)

        return current_state, [source_device]

    def write(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        pools: DeviceDataPools,
        verify: bool = False,
        update: bool = False,
        initial: bool = False,
        verbose: bool = False,
    ) -> Tuple[Optional[DataState], List[Device]]:
        if verify and not initial:
            # Assumes that this happens before the task is added to the device uses list
            self.verify_write(target_device, state)

        if logger.ENABLE_LOGGING:
            logger.data.info(
                f"Performing write of data {self.id} on device {target_device} for task {task} in phase {state}",
                extra=dict(
                    data=self.id,
                    task=task,
                    device=target_device,
                    state=state,
                    update=update,
                    initial=initial,
                ),
            )

        evicted_locations = []
        old_state = self.get_data_state(target_device, state)

        # Invalidate all other devices and check that the target device is valid
        status = self.device2state[state]
        for device in status.keys():
            if device == target_device:
                if update:
                    self.set_data_state(
                        device=device, state=state, data_state=DataState.VALID
                    )
                else:
                    if not self.check_data_state(device, state, DataState.VALID):
                        raise RuntimeError(
                            f"Task {task} cannot write to data {self.id} that is not valid on device {device}. Status: {status}"
                        )
            else:
                assert (
                    self.get_data_state(device, state) != DataState.MOVING
                ), f"Task {task} cannot invalidate data that is moving. Status: {status}"
                prev_state = self.set_data_state(device, state, DataState.NONE)

                if prev_state == DataState.VALID:
                    evicted_locations.append(device)

        from rich import print

        print(f"Write on {self.id} to {target_device}")
        print(self.state2device)

        return old_state, evicted_locations

    def read(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        pools: DeviceDataPools,
        update: bool = False,
        initial: bool = False,
        verbose: bool = False,
    ) -> Optional[DataState]:
        # Ensure that the target device is valid
        status = self.device2state[state]

        if logger.ENABLE_LOGGING:
            logger.data.info(
                f"Performing read of data {self.id} on device {target_device} for task {task} in phase {state}",
                extra=dict(
                    data=self.id,
                    task=task,
                    device=target_device,
                    state=state,
                    update=update,
                    initial=initial,
                ),
            )

        if update:
            prior_state = self.set_data_state(target_device, state, DataState.VALID)
        else:
            if not self.check_data_state(target_device, state, DataState.VALID):
                raise RuntimeError(
                    f"Task {task} cannot read from data {self.id} that is not valid on device {target_device}. Status: {status}"
                )
                prior_state = None
            prior_state = self.get_data_state(target_device, state)

        return prior_state

    def start_use(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        pools: DeviceDataPools,
        operation: AccessType,
        update: bool = False,
        initial: bool = False,
        verbose: bool = False,
    ) -> Tuple[Optional[DataState], List[Device]]:
        if logger.ENABLE_LOGGING:
            logger.data.debug(
                f"Using data on device {target_device} from task {task} in phase {state} with operation {operation.name}",
                extra=dict(task=task, data=self.id, operation=operation, state=state),
            )

        if operation == AccessType.READ:
            evicted_locations = []
            old_state = self.read(
                task=task,
                target_device=target_device,
                state=state,
                pools=pools,
                update=update,
                initial=False,
                verbose=verbose,
            )
        elif operation == AccessType.READ_WRITE:
            # This is a write WITH a read
            # Assume that this means data usage
            # for compute thats this means that the data must be valid
            # before this is called
            old_state, evicted_locations = self.write(
                task=task,
                target_device=target_device,
                state=state,
                pools=pools,
                update=update,
                initial=False,
                verbose=verbose,
            )
        elif operation == AccessType.WRITE:
            # This is a write WITHOUT a read
            # Assume that this means data creation
            # The write is always valid and evicts all other data

            old_state, evicted_locations = self.write(
                task=task,
                target_device=target_device,
                state=state,
                pools=pools,
                update=True,
                initial=True,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Invalid data operation {operation}")

        print(
            f"Start use on {self.id} to {target_device}, {TaskStateToUse[state].name}"
        )
        self.add_task(target_device, task, TaskStateToUse[state], pools)

        for device in evicted_locations:
            self.add_task(device=device, task=task, use=DataUses.EVICTING, pools=pools)

        return old_state, evicted_locations

    def finish_use(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        pools: DeviceDataPools,
        operation: AccessType,
        update: bool = False,
        verbose: bool = False,
    ):
        if logger.ENABLE_LOGGING:
            logger.data.debug(
                f"Finished using data {self.id} on device {target_device} from task {task} in phase {state} with operation {operation.name}",
                extra=dict(task=task, data=self.id, operation=operation, state=state),
            )
        self.remove_task(task=task, use=TaskStateToUse[state], pools=pools)

    def start_move(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        pools: DeviceDataPools,
        verbose: bool = False,
    ) -> DataState:
        if logger.ENABLE_LOGGING:
            logger.data.info(
                f"Starting move of data {self.id} from device {source_device} to device {target_device} for task {task}",
                extra=dict(
                    task=task, data=self.id, source=source_device, target=target_device
                ),
            )

        if not self.check_data_state(
            source_device, TaskState.LAUNCHED, DataState.VALID
        ):
            raise RuntimeError(
                f"Task {task} cannot move data {self.id} from a device that is not valid."
            )

        prior_target_state = self.get_data_state(target_device, TaskState.LAUNCHED)

        if prior_target_state != DataState.VALID:
            self.set_data_state(target_device, TaskState.LAUNCHED, DataState.MOVING)

        if source_device != target_device:
            self.add_task(source_device, task, DataUses.MOVING_FROM, pools=pools)
            self.add_task(target_device, task, DataUses.MOVING_TO, pools=pools)

        return prior_target_state

    def finish_move(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        pools: DeviceDataPools,
        verbose: bool = False,
    ) -> DataState:
        # from rich import print

        if logger.ENABLE_LOGGING:
            logger.data.info(
                f"Finishing move of data {self.id} from device {source_device} to device {target_device} for task {task}",
                extra=dict(
                    task=task, data=self.id, source=source_device, target=target_device
                ),
            )

        if not self.check_data_state(
            source_device, TaskState.LAUNCHED, DataState.VALID
        ):
            raise RuntimeError(
                f"Task {task} cannot move data {self.id} from a device that is not valid."
            )

        prior_target_state = self.get_data_state(target_device, TaskState.LAUNCHED)

        if prior_target_state == DataState.VALID:
            pass  # Do nothing
        elif prior_target_state == DataState.MOVING:
            self.set_data_state(target_device, TaskState.LAUNCHED, DataState.VALID)
        else:
            raise RuntimeError(
                f"Task {task} cannot finish moving data {self.id} to a device that is not valid or moving."
            )

        if source_device != target_device:
            self.remove_task(task=task, use=DataUses.MOVING_TO, pools=pools)
            self.remove_task(task=task, use=DataUses.MOVING_FROM, pools=pools)

        return prior_target_state

    def __rich_repr__(self):
        yield "MAPPED", self.device2state[TaskState.MAPPED]
        yield "RESERVED", self.device2state[TaskState.RESERVED]
        yield "LAUNCHED", self.device2state[TaskState.LAUNCHED]


@dataclass(slots=True)
class SimulatedData:
    system_devices: Sequence[Device] = None
    info: DataInfo = None
    status: DataStatus = None
    init: bool = True
    eviction_count: int = 0

    def __deepcopy__(self, memo):
        return SimulatedData(
            system_devices=self.system_devices,
            info=self.info,
            status=deepcopy(self.status),
            init=self.init,
            eviction_count=self.eviction_count,
        )

    def __post_init__(self):
        system_devices = self.system_devices
        if self.init:
            self.status = DataStatus(
                id=self.info.id,
                devices=system_devices,
                size=self.info.size,
                device2state=None,
                state2device=None,
                uses=None,
            )

            starting_devices = self.info.location
            assert starting_devices is not None

            if not isinstance(starting_devices, tuple):
                starting_devices = (starting_devices,)

            for device in system_devices:
                for state in [
                    TaskState.SPAWNED,
                    TaskState.MAPPED,
                    TaskState.RESERVED,
                    TaskState.LAUNCHED,
                ]:
                    if device in starting_devices:
                        self.status.set_data_state(
                            device, state, DataState.VALID, initial=True
                        )
                    else:
                        self.status.set_data_state(
                            device, state, DataState.NONE, initial=True
                        )
                self.init = False

    @property
    def name(self) -> DataID:
        return self.info.id

    @property
    def size(self) -> int:
        return self.info.size

    def get_state(self, device: Device, state: TaskState) -> DataState:
        return self.status.get_data_state(device, state)

    def start_use(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        pools: DeviceDataPools,
        operation: AccessType,
        update: bool = False,
        verbose: bool = False,
    ) -> Tuple[Optional[DataState], List[Device]]:
        old_state, evicted_locations = self.status.start_use(
            task=task,
            target_device=target_device,
            state=state,
            pools=pools,
            operation=operation,
            update=update,
            verbose=verbose,
        )
        return old_state, evicted_locations

    def finish_use(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        pools: DeviceDataPools,
        operation: AccessType,
        update: bool = False,
        verbose: bool = False,
    ):
        self.status.finish_use(
            task=task,
            target_device=target_device,
            state=state,
            pools=pools,
            operation=operation,
            update=update,
            verbose=verbose,
        )

    def start_move(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        pools: DeviceDataPools,
        verbose: bool = False,
    ) -> DataState:
        return self.status.start_move(
            task=task,
            source_device=source_device,
            target_device=target_device,
            pools=pools,
            verbose=verbose,
        )

    def finish_move(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        pools: DeviceDataPools,
        verbose: bool = False,
    ) -> DataState:
        return self.status.finish_move(
            task=task,
            source_device=source_device,
            target_device=target_device,
            pools=pools,
            verbose=verbose,
        )

    def __str__(self):
        return f"Data({self.name}) | Status: {self.status}"

    def __repr__(self):
        return self.__str__()

    def __rich_repr__(self):
        yield "info", self.info
        yield "status", self.status
        yield "used_by", self.status.uses

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def get_devices_from_states(
        self, states: Sequence[TaskState], data_states: Sequence[DataState]
    ) -> List[Device]:
        return self.status.get_devices_from_states(states, data_states)

    def get_device_set_from_states(
        self, state: TaskState, data_state: DataState
    ) -> Set[Device]:
        return self.status.get_device_set_from_state(state, data_state)

    def get_tasks_from_usage(self, device: Device, use: DataUses) -> Set[TaskID]:
        return self.status.get_tasks_from_usage(device, use)

    def is_valid(self, device: Device, state: TaskState) -> bool:
        return self.status.check_data_state(device, state, DataState.VALID)

    def is_valid_or_moving(self, device: Device, state: TaskState) -> bool:
        return self.status.check_data_state(
            device, state, DataState.VALID
        ) or self.status.check_data_state(device, state, DataState.MOVING)

    def is_evictable(self, device: Device) -> bool:
        return self.status.is_evictable(device)

    def set_evictable(self, device: Device, pools: DeviceDataPools):
        pools.add_evictable(device, self.name, self.size)

    def remove_evictable(self, device: Device, pools: DeviceDataPools) -> bool:
        return pools.remove_evictable(device, self.name, self.size)

    def get_eviction_target(
        self,
        source_device: Device,
        potential_targets: Sequence[Device],
        state: TaskState,
    ) -> Device:
        return self.status.get_eviction_target(source_device, potential_targets, state)


SimulatedDataMap = Dict[DataID, SimulatedData]
