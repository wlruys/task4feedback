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
    USED = 4
    """ A launched compute task is using the data """


TaskStateToUse = {}
TaskStateToUse[TaskState.MAPPED] = DataUses.MAPPED
TaskStateToUse[TaskState.RESERVED] = DataUses.RESERVED
TaskStateToUse[TaskState.LAUNCHED] = DataUses.USED
TaskStateToUse[TaskState.COMPLETED] = DataUses.USED

NonEvictableUses = [
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
    tasks: Dict[DataUses, Set[TaskID]] = field(default_factory=dict)
    counters: Dict[DataUses, int] = field(default_factory=dict)
    init: bool = True

    def __deepcopy__(self, memo):
        # print("Deepcopying DataUse")
        tasks = {k: {t for t in v} for k, v in self.tasks.items()}
        counters = {k: v for k, v in self.counters.items()}

        return DataUse(tasks=tasks, counters=counters, init=self.init)

    def __deepcopy__(self, memo):
        return DataUse(
            tasks={k: {l for l in v} for k, v in self.tasks.items()},
            counters={k: v for k, v in self.counters.items()},
        )

    def __post_init__(self):
        if self.init:
            for use in DataUses:
                self.tasks[use] = set()
                self.counters[use] = 0
            self.init = False

    def is_evictable(self):
        for use in NonEvictableUses:
            if self.counters[use] > 0:
                return False
        return True

    def is_used(self, use: DataUses) -> bool:
        return self.counters[use] > 0

    def get_use_count(self, use: DataUses) -> int:
        return self.counters[use]

    def add_task(self, task: TaskID, use: DataUses):
        # print(f"Adding task {task} to {use}")
        self.tasks[use].add(task)
        self.counters[use] += 1
        # print(f"Tasks: {self.tasks}")

    def remove_task(self, task: TaskID, use: DataUses):
        # print(f"Removing task {task} from {use}", self)
        self.tasks[use].remove(task)
        self.counters[use] -= 1
        # print(f"Tasks: {self.tasks}")

    def __rich_repr__(self):
        yield "tasks", self.tasks


@dataclass(slots=True)
class DataStatus:
    id: DataID
    devices: Sequence[Device]
    device2state: Dict[TaskState, Dict[Device, DataState]] = field(default_factory=dict)
    state2device: Dict[TaskState, Dict[DataState, Set[Device]]] = field(
        default_factory=dict
    )
    device2uses: Dict[Device, DataUse] = field(default_factory=dict)
    eviction_tasks: Set[TaskID] = field(default_factory=set)
    init: bool = True

    def __deepcopy__(self, memo):
        device2state = {
            k: {d: v for d, v in v2.items()} for k, v2 in self.device2state.items()
        }
        state2device = {
            k: {d: {v for v in v2} for d, v2 in v3.items()}
            for k, v3 in self.state2device.items()
        }
        device2uses = {k: deepcopy(v) for k, v in self.device2uses.items()}

        return DataStatus(
            id=self.id,
            devices=self.devices,
            device2state=device2state,
            state2device=state2device,
            device2uses=device2uses,
            eviction_tasks={t for t in self.eviction_tasks},
            init=self.init,
        )

    def __post_init__(self):
        devices = self.devices
        if self.init:
            for state in TaskState:
                self.device2state[state] = {}
                self.state2device[state] = {}

                for device in devices:
                    self.device2state[state][device] = DataState.NONE

                for data_state in DataState:
                    self.state2device[state][data_state] = set()

            for device in devices:
                self.device2uses[device] = DataUse()
            self.init = False

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
                    f"Setting data state of {self.id} on device {device} from {prior_state} to {data_state}",
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

    def get_data_use(self, device: Device) -> DataUse:
        return self.device2uses[device]

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

    def add_task(self, device: Device, task: TaskID, use: DataUses):
        self.device2uses[device].add_task(task, use)

    def add_eviction_task(self, task: TaskID):
        # print(f"Adding eviction task {task} to {self.id}")
        self.eviction_tasks.add(task)

    def remove_eviction_task(self, task: TaskID):
        # print(f"Removing eviction task {task} from {self.id}")
        self.eviction_tasks.remove(task)

    def remove_task(self, device: Device, task: TaskID, use: DataUses):
        self.device2uses[device].remove_task(task, use)

    def get_tasks_from_usage(self, device: Device, use: DataUses) -> List[TaskID]:
        return list(self.device2uses[device].tasks[use])

    def is_evictable(self, device: Device) -> bool:
        return self.device2uses[device].is_evictable()

    def is_used(self, device: Device, use: DataUses) -> bool:
        return self.device2uses[device].is_used(use)

    def get_use_count(self, device: Device, use: DataUses) -> int:
        return self.device2uses[device].get_use_count(use)

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
        # print(
        #     f"Getting eviction target for {self.id} from {source_device} to {potential_targets}"
        # )
        valid_copies = self.get_device_set_from_state(state, DataState.VALID)
        target_device = source_device

        # print("Valid Copies", valid_copies)

        current_state = self.get_data_state(source_device, state)
        assert (
            current_state == DataState.VALID
        ), f"Data {self.id} must be VALID to be evicted, but is {current_state} on {source_device}."

        if len(valid_copies) == 1:
            target_device = potential_targets[0]

        # print("Target Device", target_device)
        # print("Source Device", source_device)

        return target_device

    def initialize_eviction(
        self,
        task: TaskID,
    ) -> None:
        self.eviction_tasks.add(task)

    def start_eviction(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        state: TaskState,
        verify: bool = False,
        verbose: bool = False,
    ) -> Tuple[DataState, List[Device]]:
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
            self.start_move(task, source_device, target_device, verbose=verbose)

        return current_state, [source_device]

    def finish_eviction(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        state: TaskState,
        verify: bool = False,
        verbose: bool = False,
    ):
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
            self.finish_move(task, source_device, target_device, verbose=verbose)

        self.set_data_state(source_device, state, DataState.NONE)
        self.remove_eviction_task(task)

        return current_state, [source_device]

    def evict(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        state: TaskState,
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

        return current_state, [source_device]

    def write(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
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

        return old_state, evicted_locations

    def read(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
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
        operation: AccessType,
        update: bool = False,
        initial: bool = False,
        verbose: bool = False,
    ) -> Tuple[Optional[DataState], List[Device]]:
        if logger.ENABLE_LOGGING:
            logger.data.debug(
                f"Using data on device {target_device} from task {task}",
                extra=dict(task=task, data=self.id, operation=operation, state=state),
            )

        if operation == AccessType.READ:
            evicted_locations = []
            old_state = self.read(
                task=task,
                target_device=target_device,
                state=state,
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
                update=True,
                initial=True,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Invalid data operation {operation}")

        self.add_task(target_device, task, TaskStateToUse[state])

        return old_state, evicted_locations

    def finish_use(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        operation: AccessType,
        update: bool = False,
        verbose: bool = False,
    ):
        if logger.ENABLE_LOGGING:
            logger.data.debug(
                f"Finished using data {self.id} on device {target_device} from task {task}",
                extra=dict(task=task, data=self.id, operation=operation, state=state),
            )
        self.remove_task(target_device, task, TaskStateToUse[state])

    def start_move(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
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
            self.add_task(source_device, task, DataUses.MOVING_FROM)
            self.add_task(target_device, task, DataUses.MOVING_TO)

        return prior_target_state

    def finish_move(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
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
            self.remove_task(target_device, task, DataUses.MOVING_TO)
            self.remove_task(source_device, task, DataUses.MOVING_FROM)

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
    # stats: DataStats = field(default_factory=DataStats)

    def __deepcopy__(self, memo):
        return SimulatedData(
            system_devices=self.system_devices,
            info=self.info,
            status=deepcopy(self.status),
            init=self.init,
        )

    def __post_init__(self):
        system_devices = self.system_devices
        if self.init:
            self.status = DataStatus(id=self.info.id, devices=system_devices)

            starting_devices = self.info.location
            assert starting_devices is not None

            if isinstance(starting_devices, Device):
                starting_devices = (starting_devices,)

            for device in system_devices:
                for state in TaskState:
                    if device not in starting_devices:
                        self.status.set_data_state(
                            device, state, DataState.NONE, initial=True
                        )
                    else:
                        self.status.set_data_state(
                            device, state, DataState.VALID, initial=True
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
        operation: AccessType,
        update: bool = False,
        verbose: bool = False,
    ) -> Tuple[Optional[DataState], List[Device]]:
        old_state, evicted_locations = self.status.start_use(
            task, target_device, state, operation, update=update, verbose=verbose
        )
        return old_state, evicted_locations

    def finish_use(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        operation: AccessType,
        update: bool = False,
        verbose: bool = False,
    ):
        self.status.finish_use(
            task, target_device, state, operation, update=update, verbose=verbose
        )

    def start_move(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        verbose: bool = False,
    ) -> DataState:
        return self.status.start_move(
            task, source_device, target_device, verbose=verbose
        )

    def finish_move(
        self,
        task: TaskID,
        source_device: Device,
        target_device: Device,
        verbose: bool = False,
    ) -> DataState:
        return self.status.finish_move(
            task, source_device, target_device, verbose=verbose
        )

    def __str__(self):
        return f"Data({self.name}) | Status: {self.status}"

    def __repr__(self):
        return self.__str__()

    def __rich_repr__(self):
        yield "info", self.info
        yield "status", self.status
        yield "used_by", self.status.device2uses

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

    def get_tasks_from_usage(self, device: Device, use: DataUses) -> Sequence[TaskID]:
        return self.status.get_tasks_from_usage(device, use)

    def is_valid(self, device: Device, state: TaskState) -> bool:
        return self.status.check_data_state(device, state, DataState.VALID)

    def is_valid_or_moving(self, device: Device, state: TaskState) -> bool:
        return self.status.check_data_state(
            device, state, DataState.VALID
        ) or self.status.check_data_state(device, state, DataState.MOVING)

    def is_evictable(self, device: Device) -> bool:
        return self.status.is_evictable(device)

    def get_eviction_target(
        self,
        source_device: Device,
        potential_targets: Sequence[Device],
        state: TaskState,
    ) -> Device:
        return self.status.get_eviction_target(source_device, potential_targets, state)


SimulatedDataMap = Dict[DataID, SimulatedData]
