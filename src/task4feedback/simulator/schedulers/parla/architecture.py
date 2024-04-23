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
from .state import RLState, ParlaState, StatesToResources, AllResources
from ..architecture import SchedulerArchitecture, SchedulerOptions

from ...eviction.usage import *

# from rich import print


def chose_random_placement(task: SimulatedTask) -> Tuple[Device, ...]:
    devices = task.info.runtime.locations
    # random.shuffle(devices)
    device = devices[0]

    if not isinstance(device, Tuple):
        device = (device,)

    return device


def map_task(
    simulator,
    task: SimulatedTask,
    verbose: bool = False,
) -> Optional[Tuple[Device, ...]]:
    phase = TaskState.MAPPED

    scheduler_state: ParlaState = simulator.state
    scheduler_arch: SchedulerArchitecture = simulator.mechanisms
    mapper = simulator.mapper

    objects = scheduler_state.objects
    assert objects is not None

    current_time = scheduler_state.time
    assert current_time is not None

    # Check if task is mappable
    if check_status := scheduler_state.check_task_status(task, TaskStatus.MAPPABLE):
        chosen_devices = mapper.map_task(task, simulator)
        if chosen_devices is not None:
            task.assigned_devices = chosen_devices
            task.wait_time = current_time
            task_workload = float(scheduler_state.get_task_duration(
                                      task, task.info.runtime.locations[0]).scale_to("ms"))
            scheduler_state.acquire_resources(phase, task, verbose=verbose)
            scheduler_state.use_data(phase, task, verbose=verbose)
            scheduler_state.mapped_active_tasks += 1
            scheduler_state.total_num_mapped_tasks += 1
            scheduler_state.perdev_active_workload[chosen_devices[0]] += task_workload
            scheduler_state.total_active_workload += task_workload

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

    eviction_pool = device.eviction_pool
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
        objects.add_task(eviction_task)
        new_eviction_tasks.append(
            (eviction_task.assigned_devices[0], eviction_task.name)
        )
        # print(
        #    f"Created eviction task {eviction_task.name} for data {data.name} on device {device.name} with size {data.info.size}."
        # )
        quota -= data.info.size
        popped = eviction_pool.get()

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
            scheduler_state.reserved_active_tasks += 1
            return True, None

        else:
            if logger.ENABLE_LOGGING:
                logger.runtime.debug(
                    f"Task {task.name} cannot be reserved: Insufficient resources.",
                    extra=dict(task=task.name, phase=phase),
                )
            # print(f"Task {task.name} cannot be reserved: Insufficient resources.")

            eviction_event = scheduler_state.check_eviction(task)

            return False, eviction_event
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
    # print(f"Task status: {task.status}.")
    # print(f"Task counters: {task.counters}.")

    if check_status := scheduler_state.check_task_status(task, TaskStatus.LAUNCHABLE):
        if can_fit := scheduler_state.check_resources(phase, task):
            if logger.ENABLE_LOGGING:
                logger.runtime.critical(
                    f"Launching task {task.name} on devices {task.assigned_devices}",
                    extra=dict(task=task.name, devices=task.assigned_devices),
                )
            scheduler_state.acquire_resources(phase, task)
            duration, completion_time = scheduler_state.get_task_duration_completion(
                task, task.assigned_devices, verbose=verbose
            )
            scheduler_state.use_data(phase, task, verbose=verbose)
            task.duration = duration
            task.completion_time = completion_time
            scheduler_state.num_tasks += 1
            scheduler_state.launched_active_tasks += 1

            devices = task.assigned_devices
            assert devices is not None

            if isinstance(task, SimulatedComputeTask):
                scheduler_state.total_num_mapped_tasks -= 1
                task.wait_time = scheduler_state.time - task.wait_time
                scheduler_state.wait_time_accum += task.wait_time


            for device_id in devices:
                device = scheduler_state.objects.get_device(device_id)
                assert device is not None

                if device.stats.active_compute == 0 and device.stats.active_movement == 0:
                    last_active = max(
                        device.stats.last_active_compute, device.stats.last_active_movement
                    )
                    duration = scheduler_state.time - last_active
                    device.stats.idle_time += duration

                    if logger.ENABLE_LOGGING:
                        logger.stats.debug(
                            f"Device {device_id} was idle. Duration: {duration}, Last Active: {last_active}",
                            extra=dict(device=device_id),
                        )

                if isinstance(task, SimulatedComputeTask):
                    if device.stats.active_compute == 0:
                        duration = scheduler_state.time - device.stats.last_active_compute
                        device.stats.idle_time_compute += duration
                        device.stats.last_idle_compute = scheduler_state.time 

                        if logger.ENABLE_LOGGING:
                            logger.stats.debug(
                                f"Device {device_id} was compute idle. Duration: {duration}, Last Active: {device.stats.last_active_compute}",
                                extra=dict(device=device_id),
                            )

                elif isinstance(task, SimulatedDataTask) and task.real:
                    if device.stats.active_movement == 0:
                        duration = scheduler_state.time - device.stats.last_active_movement
                        device.stats.idle_time_movement += duration
                        device.stats.last_idle_movement = scheduler_state.time 

                        if logger.ENABLE_LOGGING:
                            logger.stats.debug(
                                f"Device {device_id} was movement idle. Duration: {duration}, Last Active: {device.stats.last_active_movement}",
                                extra=dict(device=device_id),
                            )

                if isinstance(task, SimulatedComputeTask):
                    device.stats.last_active_compute = task.completion_time
                    device.stats.active_compute += 1
                elif isinstance(task, SimulatedDataTask) and task.real:
                    device.stats.last_active_movement = task.completion_time
                    device.stats.active_movement += 1

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
    scheduler_state.mapped_active_tasks -= 1
    scheduler_state.reserved_active_tasks -= 1
    scheduler_state.launched_active_tasks -= 1
    logger.runtime.debug(f"Task completion: {task.name}\n")
    if isinstance(task, SimulatedComputeTask):
        scheduler_state.total_num_completed_tasks += 1
        logger.runtime.debug(f"Total num completed tasks: {scheduler_state.total_num_completed_tasks}\n")
        task_workload = float(
            scheduler_state.get_task_duration(
                task, task.info.runtime.locations[0]).scale_to("ms"))
        scheduler_state.total_active_workload -= task_workload
    for device_id in task.assigned_devices:
        device = scheduler_state.objects.get_device(device_id)
        assert device is not None

        if isinstance(task, SimulatedComputeTask):
            device.stats.active_compute -= 1
            scheduler_state.perdev_active_workload[device_id] -= task_workload

            if device.stats.active_compute == 0:
                device.stats.active_time_compute += (scheduler_state.time - device.stats.last_idle_compute)

        elif isinstance(task, SimulatedDataTask) and task.real:
            device.stats.active_movement -= 1
            data_id = task.read_accesses[0].id
            data = scheduler_state.objects.get_data(data_id)
            data.stats.move_time += task.duration
            data.stats.move_count += 1

            if device.stats.active_movement == 0:
                device.stats.active_time_movement += (scheduler_state.time - device.stats.last_idle_movement)

    return True


@SchedulerOptions.register_architecture("parla")
@dataclass(slots=True)
class ParlaArchitecture(SchedulerArchitecture):
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
    is_nothing_launched: Dict[Device, bool] = field(default_factory=dict)
    success_count: int = 0
    active_scheduler: int = 0
    eviction_occured: bool = False

    def __deepcopy__(self, memo):
        spawned_tasks = deepcopy(self.spawned_tasks)
        mappable_tasks = deepcopy(self.mappable_tasks)
        reservable_tasks = deepcopy(self.reservable_tasks)
        launchable_tasks = deepcopy(self.launchable_tasks)
        launched_tasks = deepcopy(self.launched_tasks)
        is_nothing_launched = deepcopy(self.is_nothing_launched)
        completed_tasks = [t for t in self.completed_tasks]

        # print("Mappable Tasks: ", launchable_tasks, self.launchable_tasks)

        return ParlaArchitecture(
            topology=None,
            spawned_tasks=spawned_tasks,
            mappable_tasks=mappable_tasks,
            reservable_tasks=reservable_tasks,
            launchable_tasks=launchable_tasks,
            launched_tasks=launched_tasks,
            is_nothing_launched=is_nothing_launched,
            success_count=self.success_count,
            active_scheduler=self.active_scheduler,
            eviction_occured=self.eviction_occured,
            completed_tasks=completed_tasks,
        )

    def __post_init__(self):
        topology = self.topology

        if topology is not None:
            for device in topology.devices:
                self.reservable_tasks[device.name] = TaskQueue()

                self.launchable_tasks[device.name] = dict()
                self.launchable_tasks[device.name][TaskType.DATA] = TaskQueue()
                self.launchable_tasks[device.name][TaskType.COMPUTE] = TaskQueue()
                self.launchable_tasks[device.name][TaskType.EVICTION] = TaskQueue()

                self.launched_tasks[device.name] = TaskQueue()
                self.is_nothing_launched[device.name] = True

    def initialize(
        self, tasks: List[TaskID], scheduler_state: SystemState, **kwargs
    ) -> List[EventPair]:

        objects = scheduler_state.objects

        task_objects = [objects.get_task(task) for task in tasks]

        # Initialize a scheduler state
        scheduler_state.initialize(tasks, task_objects)
        # Initialize the set of visible tasks
        self.add_initial_tasks(task_objects, scheduler_state)
         
        simulator = kwargs["simulator"]
        mapper_type = kwargs["mapper_type"]
        simulator.mapper.initialize(task_objects, scheduler_state, mapper_type)

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
                        scheduler_state.objects.get_device(device).add_evictable(data)

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
        print(f"Number of tasks: {len(tasks)}")
        for task in tasks:
            print(">> task:", task.name)
            self.spawned_tasks.put(task)

    def mapper(
        self, scheduler_state: SystemState, event: Mapper, **kwargs
    ) -> List[EventPair]:
        self.success_count = 0
        next_tasks = TaskIterator(self.spawned_tasks)

        task_mapper = kwargs["mapper"]
        simulator = kwargs["simulator"]

        current_time = scheduler_state.time
        objects = scheduler_state.objects

        if logger.ENABLE_LOGGING:
            logger.runtime.info(
                "Mapping tasks", extra=dict(time=current_time, phase=TaskState.MAPPED)
            )

        for priority, taskid in next_tasks:
            task = objects.get_task(taskid)
            assert task is not None

            if not task_mapper.check_allowed(task):
                # print(f"Task {task.name} not allowed. Draining task.")
                task.notify_state(TaskState.MAPPED, objects.taskmap, current_time)
                next_tasks.success()
                self.success_count += 1
            elif devices := map_task(simulator, task):
            # if scheduler_state.total_num_mapped_tasks >= scheduler_state.mapper_threshold:
            #     break
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
                # 
                task.notify_state(TaskState.MAPPED, objects.taskmap, current_time)
                next_tasks.success()
                self.success_count += 1

            else:
                next_tasks.fail()
                continue

        reserver_pair = (current_time, Reserver())
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

            active_eviction_tasks = data.status.eviction_tasks
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
                # print("Spawning data:", data_task_id)
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
                data_task.info.order = task.info.order
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

                scheduler_state.reserved_active_tasks += 1
                scheduler_state.mapped_active_tasks += 1

    def reserver(
        self, scheduler_state: SystemState, event: Reserver, **kwargs
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

            # print(f"Atempting to reserve task {taskid}.")

            reserve_success, eviction_event = reserve_task(task, scheduler_state)

            if reserve_success is True:
                # print(task, " reserved [done]")
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

        launcher_pair = (current_time, Launcher())
        return [launcher_pair]

    def eviction(
        self, scheduler_state: SystemState, event: Eviction, **kwargs
    ) -> List[EventPair]:

        """
        self.eviction_occured = True

        objects = scheduler_state.objects

        if logger.ENABLE_LOGGING:
            logger.runtime.info(
                "Evicting data.",
                extra=dict(time=current_time),
            )
        print("Evicting data.")
        eviction_tasks = run_eviction(scheduler_state, event)
        for device, eviction_task in eviction_tasks:
            self.launchable_tasks[device][TaskType.EVICTION].put_id(
                task_id=eviction_task, priority=0
            )

        """
        current_time = scheduler_state.time
        reserver_pair = (current_time, Launcher())
        return [reserver_pair]

    def launcher(
        self, scheduler_state: SystemState, event: Launcher, **kwargs
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

            # print(task, " launched")
            # Process LAUNCHABLE state
            if launch_success := launch_task(task, scheduler_state):
                task.notify_state(TaskState.LAUNCHED, objects.taskmap, current_time)
                completion_time = task.completion_time

                # print(task, " launched [done]")
                device = task.assigned_devices[0]  # type: ignore
                self.launched_tasks[device].put_id(taskid, completion_time)

                if self.is_nothing_launched[device]:
                    self.is_nothing_launched[device] = False

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

        if (
            remaining_tasks := length(self.launchable_tasks) and self.success_count
        ) or self.eviction_occured:
            mapping_pair = (current_time + Time(10), Mapper())
            next_events.append(mapping_pair)
            self.active_scheduler += 1
            self.eviction_occured = False

        return next_events

    def _verify_correct_task_completed(
        self, task: SimulatedTask, scheduler_state: SystemState, **kwargs
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
        self, scheduler_state: SystemState, event: TaskCompleted, **kwargs
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

    def complete(self, scheduler_state: SystemState, **kwargs) -> bool:
        complete_flag = self.spawned_tasks.empty()
        # print("spawned tasks:", self.spawned_tasks)
        for device in self.reservable_tasks:
            complete_flag = complete_flag and self.reservable_tasks[device].empty()
            # print("reservable tasks on ", device, ", ", self.reservable_tasks[device])
        for device in self.launchable_tasks:
            for task_type in self.launchable_tasks[device]:
                complete_flag = (
                    complete_flag and self.launchable_tasks[device][task_type].empty()
                )
        for device in self.launched_tasks:
            complete_flag = complete_flag and self.launched_tasks[device].empty()

        scheduler_state.complete()
        return complete_flag
