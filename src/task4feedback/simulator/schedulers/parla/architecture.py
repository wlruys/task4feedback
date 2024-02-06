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

from rich import print


def chose_random_placement(task: SimulatedTask) -> Tuple[Device, ...]:
    devices = task.info.runtime.locations
    # random.shuffle(devices)
    device = devices[0]

    if not isinstance(device, Tuple):
        device = (device,)

    return device


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
        chosen_devices = chose_random_placement(task)
        task.assigned_devices = chosen_devices
        scheduler_state.acquire_resources(phase, task, verbose=verbose)
        scheduler_state.use_data(phase, task, verbose=verbose)

        return chosen_devices

    if logger.ENABLE_LOGGING:
        logger.runtime.debug(
            f"Task {task.name} cannot be mapped: Invalid status.",
            extra=dict(
                task=task.name, phase=phase, counters=task.counters, status=task.status
            ),
        )
    return None


def reserve_task(
    task: SimulatedTask, scheduler_state: ParlaState, verbose: bool = False
) -> bool:
    phase = TaskState.RESERVED
    objects = scheduler_state.objects
    assert objects is not None

    current_time = scheduler_state.time
    assert current_time is not None

    if check_status := scheduler_state.check_task_status(task, TaskStatus.RESERVABLE):
        if can_fit := scheduler_state.check_resources(phase, task, verbose=verbose):
            scheduler_state.acquire_resources(phase, task, verbose=verbose)
            scheduler_state.use_data(phase, task, verbose=verbose)
            return True
        if logger.ENABLE_LOGGING:
            logger.runtime.debug(
                f"Task {task.name} cannot be reserved: Insufficient resources.",
                extra=dict(task=task.name, phase=phase),
            )
    if logger.ENABLE_LOGGING:
        logger.runtime.debug(
            f"Task {task.name} cannot be reserved: Invalid status.",
            extra=dict(
                task=task.name, phase=phase, counters=task.counters, status=task.status
            ),
        )
    return False


def launch_task(
    task: SimulatedTask, scheduler_state: ParlaState, verbose: bool = False
) -> bool:
    phase = TaskState.LAUNCHED

    if check_status := scheduler_state.check_task_status(task, TaskStatus.LAUNCHABLE):
        if can_fit := scheduler_state.check_resources(phase, task):
            if logger.ENABLE_LOGGING:
                logger.runtime.critical(
                    f"Launching task {task.name} on devices {task.assigned_devices}",
                    extra=dict(task=task.name, devices=task.assigned_devices),
                )
            scheduler_state.acquire_resources(phase, task)
            duration, completion_time = scheduler_state.get_task_duration(
                task, task.assigned_devices, verbose=verbose
            )
            scheduler_state.use_data(phase, task, verbose=verbose)
            task.duration = duration
            task.completion_time = completion_time
            scheduler_state.launch_stats(task)
            return True
        if logger.ENABLE_LOGGING:
            logger.runtime.debug(
                f"Task {task.name} cannot be launched: Insufficient resources.",
                extra=dict(task=task.name, phase=phase),
            )
    if logger.ENABLE_LOGGING:
        logger.runtime.debug(
            f"Task {task.name} cannot be launched: Invalid status.",
            extra=dict(
                task=task.name, phase=phase, counters=task.counters, status=task.status
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

    def __post_init__(self, topology: SimulatedTopology):
        assert topology is not None

        for device in topology.devices:
            self.reservable_tasks[device.name] = TaskQueue()

            self.launchable_tasks[device.name] = dict()
            self.launchable_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.launchable_tasks[device.name][TaskType.COMPUTE] = TaskQueue()

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

        reserver_pair = (current_time, Reserver())
        return [reserver_pair]

    def _enqueue_data_tasks(self, task: SimulatedTask, scheduler_state: SystemState):
        objects = scheduler_state.objects
        assert objects is not None
        assert task.assigned_devices is not None

        if task.data_tasks is not None:
            for data_task_id in task.data_tasks:
                data_task: SimulatedDataTask = objects.get_task(data_task_id)
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

        next_tasks = MultiTaskIterator(self.reservable_tasks)
        for priority, taskid in next_tasks:
            task = objects.get_task(taskid)
            assert task is not None

            if reserve_success := reserve_task(task, scheduler_state):
                devices = task.assigned_devices
                assert devices is not None
                device = devices[0]

                self._enqueue_data_tasks(task, scheduler_state)

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
            else:
                next_tasks.fail()
                continue

        launcher_pair = (current_time, Launcher())
        return [launcher_pair]

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
                        f"Task {taskid} launched successfully.",
                        extra=dict(
                            task=taskid,
                            device=device,
                            time=current_time,
                            completion_time=completion_time,
                        ),
                    )
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

        if remaining_tasks := length(self.launchable_tasks) and self.success_count:
            mapping_pair = (current_time + Time(10), Mapper())
            next_events.append(mapping_pair)
            self.active_scheduler += 1

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
