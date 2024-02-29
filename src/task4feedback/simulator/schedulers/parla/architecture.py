from ...task import SimulatedTask, SimulatedDataTask, SimulatedComputeTask
from ...data import *
from ...device import *
from ...queue import *
from ...events import *
from ...resources import *
from ...task import *
from ...topology import *

from ....types import Architecture, Device, TaskID, TaskState, TaskType, Time
from ....types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap

from typing import List, Dict, Set, Tuple, Optional, Callable, Sequence
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict

from ..state import SystemState
from .state import ParlaState, StatesToResources, AllResources
from ..architecture import SchedulerArchitecture, SchedulerOptions

from ...eviction.usage import *

from rich import print


def chose_random_placement(task: SimulatedTask) -> Tuple[Device, ...]:

    # This contains any
    devices = task.info.runtime.locations

    # # random.shuffle(devices)
    device = devices[0]

    if not isinstance(device, Tuple):
        device = (device,)

    return device
    # return (Device(Architecture.GPU, 3),)


def map_task(
    task: SimulatedTask, scheduler_state: ParlaState, verbose: bool = False
) -> Optional[Tuple[Device, ...]]:
    phase = TaskState.MAPPED
    objects = scheduler_state.objects
    assert objects is not None

    current_time = scheduler_state.time
    assert current_time is not None

    # Check if task is mappable
    if check_status := scheduler_state.check_task_status(task, TaskStatus.MAPPABLE):

        if (
            check_limiter := scheduler_state.num_mapped_tasks
            < scheduler_state.threshold
        ):
            # task.assigned_devices = apply_mapping_function(task, scheduler_state)
            chosen_devices = chose_random_placement(task)
            task.assigned_devices = chosen_devices
            scheduler_state.acquire_resources(phase, task, verbose=verbose)
            scheduler_state.use_data(phase, task, verbose=verbose)
            scheduler_state.num_mapped_tasks += 1
            print(f"Mapped task {task.name} to device {task.assigned_devices}.")

            return chosen_devices

    if logger.ENABLE_LOGGING:
        logger.runtime.debug(
            f"Task {task.name} cannot be mapped: Invalid status.",
            extra=dict(
                task=task.name, phase=phase, counters=task.counters, status=task.status
            ),
        )
    return None


def run_device_eviction(
    parent_task: TaskID,
    device_id: Device,
    requested_resources: FasterResourceSet,
    scheduler_state: ParlaState,
    verbose: bool = False,
) -> List[Tuple[Device, TaskID]]:

    objects = scheduler_state.objects
    device = objects.get_device(device_id)
    assert device is not None

    eviction_pool = scheduler_state.data_pool.evictable[device_id]
    quota = requested_resources.memory

    new_eviction_tasks = []

    while quota > 0 and not eviction_pool.empty():
        # print(f"Quota: {quota}.")
        data_id = eviction_pool.peek()
        data = objects.get_data(data_id)
        # print("Data ID: ", data_id, data.info.size)
        # print("Data Pool: ", eviction_pool)
        assert data is not None

        eviction_task = eviction_init(parent_task, scheduler_state, device, data)
        assert eviction_task.assigned_devices is not None

        objects.add_task(eviction_task)
        new_eviction_tasks.append(
            (eviction_task.assigned_devices[0], eviction_task.name)
        )

        if logger.ENABLE_LOGGING:
            logger.runtime.info(
                f"Created eviction task {eviction_task.name} for data {data.name} on device {device.name} with size {data.info.size}. Target: {eviction_task.assigned_devices} Source: {eviction_task.source}",
                extra=dict(
                    task=eviction_task.name,
                    data=data.name,
                    device=device.name,
                    size=data.info.size,
                ),
            )
        # print(
        #    f"Created eviction task {eviction_task.name} for data {data.name} on device {device.name} with size {data.info.size}."
        # )
        size = data.info.size
        quota -= size
        popped = eviction_pool.remove(data_id, size)

    if quota > 0:
        raise RuntimeError(
            f"Eviction quota not met for device {device.name}. Remaining: {quota}."
        )

    return new_eviction_tasks


def run_eviction(
    scheduler_state: ParlaState, event: Eviction, verbose: bool = False
) -> List[Tuple[Device, TaskID]]:

    # print(f"Running eviction requested by {event.parent_task}.")

    eviction_tasks = []
    for device, requested_resources in event.requested_resources.items():
        tasks = run_device_eviction(
            event.parent_task,
            device,
            requested_resources,
            scheduler_state,
            verbose=verbose,
        )
        eviction_tasks.extend(tasks)

    return eviction_tasks


def reserve_task(
    task: SimulatedTask, scheduler_state: ParlaState, verbose: bool = False
) -> Tuple[bool, Optional[Eviction]]:
    phase = TaskState.RESERVED
    objects = scheduler_state.objects
    assert objects is not None

    current_time = scheduler_state.time
    assert current_time is not None

    if check_status := scheduler_state.check_task_status(task, TaskStatus.RESERVABLE):
        if can_fit := scheduler_state.check_resources(phase, task, verbose=verbose):
            scheduler_state.acquire_resources(phase, task, verbose=verbose)
            scheduler_state.use_data(phase, task, verbose=verbose)
            return True, None

        else:
            if logger.ENABLE_LOGGING:
                logger.runtime.debug(
                    f"Task {task.name} cannot be reserved: Insufficient resources.",
                    extra=dict(task=task.name, phase=phase),
                )
            # print(f"Task {task.name} cannot be reserved: Insufficient resources.")

            return False, scheduler_state.check_eviction(task)
    else:
        if logger.ENABLE_LOGGING:
            logger.runtime.debug(
                f"Task {task.name} cannot be reserved: Invalid status.",
                extra=dict(
                    task=task.name,
                    phase=phase,
                    counters=task.counters,
                    status=task.status,
                ),
            )
            # print(f"Task {task.name} cannot be reserved: Invalid status.")
        return False, None


def launch_task(
    task: SimulatedTask, scheduler_state: ParlaState, verbose: bool = False
) -> bool:
    phase = TaskState.LAUNCHED

    # print(f"Trying to launch task {task.name}.")
    # if isinstance(task, SimulatedEvictionTask):
    #     print(
    #         f"Trying to launch eviction task {task.name}. Source Device: {task.source}. Target Device: {task.assigned_devices}."
    #     )

    # print(f"Trying to launch task {task.name}.")
    # print(f"Task status: {task.status}.")
    # print(f"Task counters: {task.counters}.")

    if check_status := scheduler_state.check_task_status(task, TaskStatus.LAUNCHABLE):
        # if isinstance(task, SimulatedEvictionTask):
        #     print(
        #         f"Task {task.name} is launchable. Source Device: {task.source}. Target Device: {task.assigned_devices}."
        #     )
        if can_fit := scheduler_state.check_resources(phase, task):
            # if isinstance(task, SimulatedEvictionTask):
            #     print(
            #         f"Task {task.name} can fit. Source Device: {task.source}. Target Device: {task.assigned_devices}."
            #     )
            if logger.ENABLE_LOGGING:
                logger.runtime.critical(
                    f"Launching task {task.name} on devices {task.assigned_devices}",
                    extra=dict(task=task.name, devices=task.assigned_devices),
                )

            # if isinstance(task, SimulatedEvictionTask):
            #     print(f"Task {task.name} before acquire_resources: {task.source}.")
            scheduler_state.acquire_resources(phase, task)
            # if isinstance(task, SimulatedEvictionTask):
            #     print(f"Task {task.name} before duration: {task.source}.")
            duration, completion_time = scheduler_state.get_task_duration(
                task, task.assigned_devices, verbose=verbose
            )
            # if isinstance(task, SimulatedEvictionTask):
            #     print(f"Task {task.name} before use_data: {task.source}.")
            scheduler_state.use_data(phase, task, verbose=verbose)
            # if isinstance(task, SimulatedEvictionTask):
            #     print(f"Task {task.name} after use_data: {task.source}.")
            task.duration = duration
            task.completion_time = completion_time
            scheduler_state.launch_stats(task)
            return True
        else:
            if logger.ENABLE_LOGGING:
                # print(f"Task {task.name} cannot be launched: Insufficient resources.")
                logger.runtime.debug(
                    f"Task {task.name} cannot be launched: Insufficient resources.",
                    extra=dict(task=task.name, phase=phase),
                )
    else:
        if logger.ENABLE_LOGGING:
            # print(f"Task {task.name} cannot be launched: Invalid status.")
            logger.runtime.debug(
                f"Task {task.name} cannot be launched: Invalid status.",
                extra=dict(
                    task=task.name,
                    phase=phase,
                    counters=task.counters,
                    status=task.status,
                ),
            )
    return False


def complete_task(
    task: SimulatedTask, scheduler_state: ParlaState, verbose: bool = False
) -> bool:
    phase = TaskState.COMPLETED
    scheduler_state.release_data(phase, task, verbose=verbose)
    scheduler_state.release_resources(phase, task, verbose=verbose)
    scheduler_state.completion_stats(task)
    scheduler_state.num_mapped_tasks -= 1

    return True


@SchedulerOptions.register_architecture("parla")
@dataclass(slots=True)
class ParlaArchitecture(SchedulerArchitecture):
    topology: InitVar[SimulatedTopology]
    spawned_tasks: TaskQueue = TaskQueue()
    # Mapping Phase
    mappable_tasks: TaskQueue = TaskQueue()
    # Reserving Phase
    reservable_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)
    # Launching Phase
    launchable_tasks: Dict[Device, Dict[TaskType, TaskQueue]] = field(
        default_factory=dict
    )
    launched_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)
    success_count: int = 0
    active_scheduler: int = 0
    eviction_occured: bool = False

    def __post_init__(self, topology: SimulatedTopology):
        assert topology is not None

        for device in topology.devices:
            self.reservable_tasks[device.name] = TaskQueue()

            self.launchable_tasks[device.name] = dict()
            self.launchable_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.launchable_tasks[device.name][TaskType.COMPUTE] = TaskQueue()
            self.launchable_tasks[device.name][TaskType.EVICTION] = TaskQueue()

            self.launched_tasks[device.name] = TaskQueue()

    def initialize(
        self, tasks: List[TaskID], scheduler_state: SystemState
    ) -> List[EventPair]:
        objects = scheduler_state.objects

        task_objects = [objects.get_task(task) for task in tasks]

        # Initialize the set of visible tasks
        self.add_initial_tasks(task_objects, scheduler_state)

        # Initialize memory for starting data blocks
        for data in objects.datamap.values():
            devices = data.info.location
            if devices is not None:
                if isinstance(devices, Device):
                    devices = (devices,)

                for device in devices:
                    for pool in [
                        TaskState.MAPPED,
                        TaskState.RESERVED,
                        TaskState.LAUNCHED,
                    ]:
                        resource_set = FasterResourceSet(
                            memory=data.info.size, vcus=0, copy=0
                        )

                        if logger.ENABLE_LOGGING:
                            logger.data.info(
                                f"Initializing data resource {data.name} on device {device} in pool {pool} with resources: {resource_set}",
                                extra=dict(
                                    data=data.name,
                                    device=device,
                                    pool=pool,
                                    resources=resource_set,
                                ),
                            )

                        scheduler_state.resource_pool.add_device_resource(
                            device,
                            pool,
                            ResourceGroup.PERSISTENT,
                            resource_set,
                        )

                        # Default state is evictable
                        data.set_evictable(device, scheduler_state.data_pool)

        # Initialize the event queue
        next_event = Mapper()
        next_time = Time(0)
        self.active_scheduler += 1
        return [(next_time, next_event)]

    def add_initial_tasks(
        self, tasks: List[SimulatedTask], scheduler_state: SystemState
    ):
        """
        Append an initial task who does not have any dependency to
        a spawned task queue.
        """
        for task in tasks:
            self.spawned_tasks.put(task)

    def mapper(self, scheduler_state: SystemState, event: Mapper) -> List[EventPair]:
        self.success_count = 0
        next_tasks = TaskIterator(self.spawned_tasks)

        current_time = scheduler_state.time
        objects = scheduler_state.objects

        if logger.ENABLE_LOGGING:
            logger.runtime.info(
                "Mapping tasks", extra=dict(time=current_time, phase=TaskState.MAPPED)
            )

        for priority, taskid in next_tasks:
            task = objects.get_task(taskid)
            assert task is not None

            if devices := map_task(task, scheduler_state):
                for device in devices:
                    self.reservable_tasks[device].put_id(
                        task_id=taskid, priority=priority
                    )
                if logger.ENABLE_LOGGING:
                    logger.runtime.info(
                        f"Task {task.name}, mapped successfully.",
                        extra=dict(task=taskid, device=devices),
                    )
                event.tasks.add(taskid)
                task.notify_state(TaskState.MAPPED, objects.taskmap, current_time)
                next_tasks.success()
                self.success_count += 1
            else:
                next_tasks.fail()
                continue

        reserver_pair = (current_time + Time(10), Reserver())
        return [reserver_pair]

    def _add_eviction_dependencies_from_access(
        self,
        task: SimulatedTask,
        accesses: List[DataAccess],
        scheduler_state: SystemState,
    ):
        objects = scheduler_state.objects
        for data_access in accesses:
            data = objects.get_data(data_access.id)

            active_eviction_tasks = data.status.uses.eviction_tasks
            # print("Active eviction tasks: ", active_eviction_tasks)
            for eviction_task_id in active_eviction_tasks:
                eviction_task = objects.get_task(eviction_task_id)
                assert eviction_task is not None
                task.add_eviction_dependency(eviction_task)

    def _add_eviction_dependencies(
        self, task: SimulatedTask, scheduler_state: SystemState
    ):
        self._add_eviction_dependencies_from_access(
            task, task.read_accesses, scheduler_state
        )
        self._add_eviction_dependencies_from_access(
            task, task.write_accesses, scheduler_state
        )
        self._add_eviction_dependencies_from_access(
            task, task.read_write_accesses, scheduler_state
        )

    def _enqueue_data_tasks(self, task: SimulatedTask, scheduler_state: SystemState):
        objects = scheduler_state.objects
        assert objects is not None
        assert task.assigned_devices is not None

        if task.data_tasks is not None:
            for data_task_id in task.data_tasks:
                data_task: SimulatedDataTask = objects.get_task(data_task_id)  # type: ignore
                assert data_task is not None

                if logger.ENABLE_LOGGING:
                    logger.runtime.debug(
                        f"Enqueuing data task {data_task_id}",
                        extra=dict(
                            task=data_task_id,
                            device=task.assigned_devices,
                            phase=TaskState.RESERVED,
                        ),
                    )

                device = task.assigned_devices[data_task.local_index]
                data_task.assigned_devices = (device,)
                data_task.set_state(
                    TaskState.RESERVED, scheduler_state.time, verify=False
                )
                data_task.set_status(
                    TaskStatus.RESERVABLE, scheduler_state.time, verify=False
                )
                self._add_eviction_dependencies(data_task, scheduler_state)

                self.launchable_tasks[device][TaskType.DATA].put_id(
                    task_id=data_task_id, priority=data_task.priority
                )

    def reserver(
        self, scheduler_state: SystemState, event: Reserver
    ) -> List[EventPair]:
        objects = scheduler_state.objects
        current_time = scheduler_state.time

        if logger.ENABLE_LOGGING:
            logger.runtime.info(
                "Reserving tasks.",
                extra=dict(time=current_time, phase=TaskState.RESERVED),
            )

        # print(f"Reserving tasks")

        next_tasks = MultiTaskIterator(self.reservable_tasks)
        for priority, taskid in next_tasks:
            task = objects.get_task(taskid)
            assert task is not None

            # print(f"Atempting to reserve task {taskid}.")

            reserve_success, eviction_event = reserve_task(task, scheduler_state)

            if reserve_success is True:
                devices = task.assigned_devices
                assert devices is not None
                device = devices[0]

                self._enqueue_data_tasks(task, scheduler_state)
                self._add_eviction_dependencies(task, scheduler_state)
                # print(f"Task {taskid} reserved successfully.")
                # print(task.dependencies)
                # print(task.counters)

                self.launchable_tasks[device][TaskType.COMPUTE].put_id(
                    task_id=taskid, priority=priority
                )
                if logger.ENABLE_LOGGING:
                    logger.runtime.info(
                        f"Task {taskid} reserved successfully.",
                        extra=dict(task=taskid),
                    )
                event.tasks.add(taskid)
                task.notify_state(TaskState.RESERVED, objects.taskmap, current_time)
                next_tasks.success()
                self.success_count += 1
            elif isinstance(eviction_event, Eviction):
                next_tasks.fail()
                return [(current_time + Time(10), eviction_event)]
            else:
                next_tasks.fail()
                continue

        launcher_pair = (current_time + Time(10), Launcher())
        return [launcher_pair]

    def eviction(
        self, scheduler_state: SystemState, event: Eviction
    ) -> List[EventPair]:

        self.eviction_occured = True

        objects = scheduler_state.objects
        current_time = scheduler_state.time

        if logger.ENABLE_LOGGING:
            logger.runtime.info(
                "Evicting data.",
                extra=dict(time=current_time),
            )
        # print("Evicting data.")
        eviction_tasks = run_eviction(scheduler_state, event)
        for device, eviction_task in eviction_tasks:
            # print(device, eviction_task)
            task_i = objects.get_task(eviction_task)

            self.launchable_tasks[device][TaskType.EVICTION].put_id(
                task_id=eviction_task, priority=0
            )

        reserver_pair = (current_time + Time(10), Launcher())
        return [reserver_pair]

    def launcher(
        self, scheduler_state: SystemState, event: Launcher
    ) -> List[EventPair]:
        objects = scheduler_state.objects
        current_time = scheduler_state.time

        if logger.ENABLE_LOGGING:
            logger.runtime.info(
                "Launching tasks.",
                extra=dict(time=current_time, phase=TaskState.LAUNCHED),
            )

        next_events: List[EventPair] = []

        next_tasks = MultiTaskIterator(self.launchable_tasks)
        for priority, taskid in next_tasks:
            task = objects.get_task(taskid)
            assert task is not None

            # Process LAUNCHABLE state
            if launch_success := launch_task(task, scheduler_state):
                task.notify_state(TaskState.LAUNCHED, objects.taskmap, current_time)
                completion_time = task.completion_time

                device = task.assigned_devices[0]  # type: ignore
                self.launched_tasks[device].put_id(taskid, completion_time)

                if logger.ENABLE_LOGGING:
                    logger.runtime.info(
                        f"Task {taskid} launched successfully on {device}.",
                        extra=dict(
                            task=taskid,
                            device=device,
                            time=current_time,
                            completion_time=completion_time,
                        ),
                    )
                    # if isinstance(task, SimulatedEvictionTask):
                    #     print("Source Device: ", task.source)

                event.tasks.add(taskid)

                # Create completion event
                completion_event = TaskCompleted(task=taskid)
                next_events.append((completion_time, completion_event))
                next_tasks.success()
                self.success_count += 1
            else:
                next_tasks.fail()
                continue

        self.active_scheduler -= 1

        if (
            remaining_tasks := length(self.launchable_tasks) and self.success_count
        ) or self.eviction_occured:
            mapping_pair = (current_time + Time(10), Mapper())
            next_events.append(mapping_pair)
            self.active_scheduler += 1
            self.eviction_occured = False

        return next_events

    def _verify_correct_task_completed(
        self, task: SimulatedTask, scheduler_state: SystemState
    ):
        taskid = task.name
        # Remove task from launched queues
        devices = task.assigned_devices
        if devices is None:
            raise ValueError(f"Task {task.name} has no assigned devices.")
        device = devices[0]

        if expected := self.launched_tasks[device].peek():
            expected_time, expected_task = expected
            if expected_task != taskid:
                raise ValueError(
                    f"Invalid state: Task {task.name} is not at the head of the launched queue. Expected: {expected_task}, Actual: {taskid}"
                )
            if expected_time != scheduler_state.time:
                raise ValueError(
                    f"Invalid state: Task {task.name} is not expected to complete at this time. Expected: {expected_time}, Actual: {scheduler_state.time}"
                )
            # Remove task from launched queue (it has completed)
            self.launched_tasks[device].get()
            self.completed_tasks.append(taskid)
        else:
            raise ValueError(
                f"Invalid state: Launch queue for device {device} is empty."
            )

    def complete_task(
        self, scheduler_state: SystemState, event: TaskCompleted
    ) -> List[EventPair]:
        objects = scheduler_state.objects
        task = objects.get_task(event.task)
        current_time = scheduler_state.time

        if logger.ENABLE_LOGGING:
            # print(f"Completing task {event.task}")
            logger.runtime.critical(
                f"Completing task {event.task}",
                extra=dict(
                    task=event.task, time=current_time, phase=TaskState.COMPLETED
                ),
            )

        next_events: List[EventPair] = []

        self._verify_correct_task_completed(task, scheduler_state)
        complete_task(task, scheduler_state)

        # Update status of dependencies
        task.notify_state(TaskState.COMPLETED, objects.taskmap, scheduler_state.time)

        self.success_count += 1
        if self.active_scheduler == 0:
            mapping_pair = (current_time, Mapper())
            next_events.append(mapping_pair)
            self.active_scheduler += 1

        return next_events

    def complete(self, scheduler_state: SystemState) -> bool:
        complete_flag = self.spawned_tasks.empty()
        for device in self.reservable_tasks:
            complete_flag = complete_flag and self.reservable_tasks[device].empty()
        for device in self.launchable_tasks:
            for task_type in self.launchable_tasks[device]:
                complete_flag = (
                    complete_flag and self.launchable_tasks[device][task_type].empty()
                )

        return complete_flag
