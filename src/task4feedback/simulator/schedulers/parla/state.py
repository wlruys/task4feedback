from task4feedback.simulator.data import TaskID, TaskState
from task4feedback.simulator.device import TaskID, TaskState
from task4feedback.simulator.events import TaskID, TaskState
from task4feedback.simulator.queue import TaskID
from task4feedback.simulator.resources import TaskID, TaskState
from task4feedback.simulator.task import TaskID, TaskState
from task4feedback.simulator.topology import TaskID
from task4feedback.types import TaskID, TaskState
from ....types import *
from ...data import *
from ...device import *

from ..state import *
from ..architecture import *

from dataclasses import dataclass, field, InitVar

StatesToResources: Dict[TaskState, list[ResourceType]] = {}
StatesToResources[TaskState.MAPPED] = [
    ResourceType.VCU,
    ResourceType.MEMORY,
    ResourceType.COPY,
]
StatesToResources[TaskState.LAUNCHED] = [ResourceType.VCU, ResourceType.COPY]
StatesToResources[TaskState.RESERVED] = [ResourceType.MEMORY]
StatesToResources[TaskState.COMPLETED] = []
AllResources = [ResourceType.VCU, ResourceType.MEMORY, ResourceType.COPY]


def get_required_memory_for_data(
    phase: TaskState, device: Device, data_id: DataID, objects: ObjectRegistry
) -> int:
    data = objects.get_data(data_id)
    if is_valid := data.is_valid(device, phase):
        return 0
    else:
        return data.size


def get_required_memory(
    memory: List[int],
    phase: TaskState,
    devices: Tuple[Device, ...],
    data_accesses: List[DataAccess],
    objects: ObjectRegistry,
) -> None:
    for data_access in data_accesses:
        idx = data_access.device
        device = devices[idx]
        memory[idx] += get_required_memory_for_data(
            phase, device, data_access.id, objects
        )


def get_required_resources(
    phase: TaskState,
    task: SimulatedTask,
    devices: Devices,
    objects: ObjectRegistry,
    count_data: bool = True,
    verbose: bool = False,
) -> List[ResourceSet]:
    if isinstance(devices, Device):
        devices = (devices,)

    task.set_resources(devices)

    memory: List[int] = [s[ResourceType.MEMORY] for s in task.resources]

    if count_data:
        get_required_memory(memory, phase, devices, task.read_accesses, objects)

    resources = []
    for i in range(len(devices)):
        t_req = task.resources[i]
        vcus: Numeric = t_req[ResourceType.VCU]
        mem: int = memory[i]
        copy: int = t_req[ResourceType.COPY]
        resources.append(ResourceSet(vcus=vcus, memory=mem, copy=copy))

    return resources


def _check_nearest_source(
    state: SystemState,
    task: SimulatedDataTask,
    verbose: bool = False,
) -> Optional[Device | SimulatedDevice]:
    assert isinstance(task, SimulatedDataTask)
    devices = task.assigned_devices
    assert devices is not None
    assert isinstance(devices, Device) or len(devices) == 1

    data_id = task.read_accesses[0].id
    data = state.objects.get_data(data_id)
    assert data is not None

    device = devices[0] if isinstance(devices, tuple) else devices
    device = state.objects.get_device(device)
    assert device is not None

    valid_sources = data.get_devices_from_states(
        [TaskState.LAUNCHED], [DataState.VALID]
    )
    valid_sources = [state.objects.get_device(d) for d in valid_sources]
    print(f"Valid sources: {valid_sources}")
    print(f"Target device: {device}")

    source_device = state.topology.nearest_valid_connection(
        device, valid_sources, require_copy_engines=True, require_symmetric=True
    )

    return source_device


def _acquire_resources_mapped(
    state: SystemState, task: SimulatedTask, verbose: bool = False
):
    devices = task.assigned_devices

    if isinstance(task, SimulatedDataTask):
        raise RuntimeError(
            f"Data tasks should never hit the Mapper. Invalid task: {task}"
        )

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources = get_required_resources(
        TaskState.MAPPED, task, devices, state.objects, count_data=True
    )
    state.resource_pool.add_resources(
        devices, TaskState.MAPPED, AllResources, resources
    )


def _check_resources_reserved(
    state: SystemState, task: SimulatedTask, verbose: bool = False
) -> bool:
    devices = task.assigned_devices

    if isinstance(task, SimulatedDataTask):
        raise RuntimeError(
            f"Data tasks should never hit the Reserver. Invalid task: {task}"
        )

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources_types = StatesToResources[TaskState.RESERVED]

    resources = get_required_resources(
        TaskState.RESERVED, task, devices, state.objects, count_data=True
    )

    can_fit = state.resource_pool.check_resources(
        devices=devices,
        state=TaskState.RESERVED,
        types=resources_types,
        resources=resources,
    )

    return can_fit


def _acquire_resources_reserved(
    state: SystemState, task: SimulatedTask, verbose: bool = False
):
    devices = task.assigned_devices

    if isinstance(task, SimulatedDataTask):
        raise RuntimeError(
            f"Data tasks should never hit the Reserver. Invalid task: {task}"
        )

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resource_types = StatesToResources[TaskState.RESERVED]

    resources = get_required_resources(
        TaskState.RESERVED, task, devices, state.objects, count_data=True
    )
    state.resource_pool.add_resources(
        devices, TaskState.RESERVED, resource_types, resources
    )


def _check_resources_launched(
    state: SystemState, task: SimulatedTask, verbose: bool = False
) -> bool:
    devices = task.assigned_devices

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources_types = StatesToResources[TaskState.LAUNCHED]

    resources = get_required_resources(
        TaskState.RESERVED, task, devices, state.objects, count_data=False
    )

    can_fit = state.resource_pool.check_resources(
        devices=devices,
        state=TaskState.RESERVED,
        types=resources_types,
        resources=resources,
    )

    print(f"Can fit: {can_fit}")

    if isinstance(task, SimulatedDataTask):
        source_device = _check_nearest_source(state, task)
        print(f"Nearest source: {source_device}")
        if source_device is None:
            return False

        if isinstance(source_device, SimulatedDevice):
            source_device = source_device.name

        task.source = source_device

    return can_fit


def _acquire_resources_launched(
    state: SystemState, task: SimulatedTask, verbose: bool = False
):
    devices = task.assigned_devices

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources = get_required_resources(
        TaskState.LAUNCHED, task, devices, state.objects, count_data=False
    )

    resource_types = StatesToResources[TaskState.LAUNCHED]

    print(
        f"Acquiring resources for task {task} on devices {devices} with resources {resources}."
    )

    state.resource_pool.add_resources(
        devices, TaskState.RESERVED, resource_types, resources
    )

    state.resource_pool.add_resources(
        devices, TaskState.LAUNCHED, AllResources, resources
    )

    if isinstance(task, SimulatedDataTask):
        assert len(devices) == 1
        target_device = devices[0]
        source_device = task.source
        assert source_device is not None

        state.topology.acquire_connection(source_device, target_device)

    print("Resources after acquiring:")
    for device in devices:
        print(
            f"Device {device}: {state.resource_pool.pool[device][TaskState.LAUNCHED]}"
        )


def _release_resources_completed(
    state: SystemState, task: SimulatedTask, verbose: bool = False
):
    devices = task.assigned_devices

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources = get_required_resources(
        TaskState.COMPLETED, task, devices, state.objects, count_data=False
    )

    print(
        f"Releasing resources for task {task} on devices {devices} with resources {resources}."
    )

    # Free resources from all pools
    if isinstance(task, SimulatedComputeTask):
        state.resource_pool.remove_resources(
            devices=devices,
            state=TaskState.MAPPED,
            types=AllResources,
            resources=task.resources,
        )
        state.resource_pool.remove_resources(
            devices=devices,
            state=TaskState.RESERVED,
            types=AllResources,
            resources=task.resources,
        )
    elif isinstance(task, SimulatedDataTask):
        assert len(devices) == 1
        target_device = devices[0]
        source_device = task.source
        assert source_device is not None

        state.topology.release_connection(source_device, target_device)

    state.resource_pool.remove_resources(
        devices=devices,
        state=TaskState.LAUNCHED,
        types=AllResources,
        resources=task.resources,
    )


def _use_data(
    state: SystemState,
    phase: TaskState,
    data_accesses: List[DataAccess],
    task: SimulatedTask,
    access_type: AccessType,
    verbose: bool = False,
):
    devices = task.assigned_devices
    assert devices is not None

    for data_access in data_accesses:
        data_id = data_access.id
        device_idx = data_access.device
        device = devices[device_idx]

        data = state.objects.get_data(data_id)
        assert data is not None

        update_state = True
        initial_state = False

        if phase == TaskState.MAPPED:
            initial_state = True
        elif phase == TaskState.RESERVED:
            data.finish_use(task.name, device, TaskState.MAPPED, operation=access_type)
        elif phase == TaskState.LAUNCHED:
            data.finish_use(
                task.name, device, TaskState.RESERVED, operation=access_type
            )
            # State updates at runtime are managed by data movement tasks
            # Compute tasks only verify and evict
            update_state = False

        data.start_use(
            task.name,
            device,
            phase,
            operation=access_type,
            update=update_state,
            verbose=verbose,
        )


def _release_data(
    state: SystemState,
    phase: TaskState,
    data_accesses: List[DataAccess],
    task: SimulatedTask,
    access_type: AccessType,
    verbose: bool = False,
):
    assert phase == TaskState.COMPLETED

    devices = task.assigned_devices
    assert devices is not None

    for data_access in data_accesses:
        data_id = data_access.id
        device_idx = data_access.device
        device = devices[device_idx]

        data = state.objects.get_data(data_id)
        assert data is not None

        data.finish_use(task.name, device, phase, operation=access_type)


def _move_data(
    state: SystemState,
    data_accesses: List[DataAccess],
    task: SimulatedTask,
    verbose: bool = False,
):
    devices = task.assigned_devices
    assert devices is not None

    # Move data is called from a data movement task at launch time
    assert isinstance(task, SimulatedDataTask)

    # move data is called from a data movement task at launch time
    # each data movement task moves one data item onto a single target device
    assert len(data_accesses) == 1
    assert len(devices) == 1
    target_device = devices[0]
    assert target_device is not None

    data = state.objects.get_data(data_accesses[0].id)
    assert data is not None

    # Assumes source device is set by the prior check_resources_launched call
    source_device = task.source
    assert source_device is not None

    # Mark data as moving onto target device
    prior_state = data.start_move(task.name, source_device, target_device)


def _finish_move(
    state: SystemState, data_accesses: List[DataAccess], task: SimulatedTask
):
    devices = task.assigned_devices
    assert devices is not None

    # Move data is called from a data movement task at launch time
    assert isinstance(task, SimulatedDataTask)

    # move data is called from a data movement task at launch time
    # each data movement task moves one data item onto a single target device
    assert len(data_accesses) == 1
    assert len(devices) == 1
    target_device = devices[0]
    assert target_device is not None

    data = state.objects.get_data(data_accesses[0].id)
    assert data is not None

    # Assumes source device is set by the prior check_resources_launched call
    source_device = task.source
    assert source_device is not None

    # Mark data as valid on target device
    prior_state = data.finish_move(task.name, source_device, target_device)


def _compute_task_duration(
    state: SystemState,
    task: SimulatedComputeTask,
    devices: Devices,
    verbose: bool = False,
) -> Time:
    assert isinstance(task, SimulatedComputeTask)
    assert devices is not None

    runtime_infos = task.get_runtime_info(devices)
    max_time = max([runtime_info.task_time for runtime_info in runtime_infos])
    return Time(max_time)


def _data_task_duration(
    state: SystemState,
    task: SimulatedDataTask,
    devices: Devices,
    verbose: bool = False,
) -> Time:
    return Time(10)


@SchedulerOptions.register_state("parla")
@dataclass(slots=True)
class ParlaState(SystemState):
    def check_resources(
        self, phase: TaskState, task: SimulatedTask, verbose: bool = False
    ) -> bool:
        if phase == TaskState.MAPPED:
            return True
        elif phase == TaskState.RESERVED:
            return _check_resources_reserved(self, task)
        elif phase == TaskState.LAUNCHED:
            return _check_resources_launched(self, task)
        else:
            raise RuntimeError(
                f"Invalid phase {phase} in check_resource for task {task}"
            )

    def acquire_resources(
        self, phase: TaskState, task: SimulatedTask, verbose: bool = False
    ):
        if phase == TaskState.MAPPED:
            _acquire_resources_mapped(self, task)
        elif phase == TaskState.RESERVED:
            _acquire_resources_reserved(self, task)
        elif phase == TaskState.LAUNCHED:
            _acquire_resources_launched(self, task)
        else:
            raise RuntimeError(
                f"Invalid phase {phase} in acquire_resource for task {task}"
            )

    def release_resources(
        self, phase: TaskState, task: SimulatedTask, verbose: bool = False
    ):
        if phase == TaskState.COMPLETED:
            _release_resources_completed(self, task)
        else:
            raise RuntimeError(
                f"Invalid phase {phase} in release_resource for task {task}"
            )

    def use_data(self, phase: TaskState, task: SimulatedTask, verbose: bool = False):
        if isinstance(task, SimulatedComputeTask):
            _use_data(self, phase, task.read_accesses, task, AccessType.READ)
            _use_data(self, phase, task.write_accesses, task, AccessType.WRITE)
        elif isinstance(task, SimulatedDataTask):
            # Data movement tasks only exist at launch time
            assert phase == TaskState.LAUNCHED

            # All data movement tasks are single data item tasks
            # They read from a single source device onto a single target device
            _move_data(self, task.read_accesses, task)

    def release_data(
        self,
        phase: TaskState,
        task: SimulatedTask,
        verbose: bool = False,
    ):
        assert phase == TaskState.COMPLETED

        if isinstance(task, SimulatedComputeTask):
            _release_data(self, phase, task.read_accesses, task, AccessType.READ)
            _release_data(self, phase, task.write_accesses, task, AccessType.WRITE)
        elif isinstance(task, SimulatedDataTask):
            _finish_move(self, task.read_accesses, task)

    def get_task_duration(
        self, task: SimulatedTask, devices: Devices, verbose: bool = False
    ) -> Time:
        if isinstance(task, SimulatedComputeTask):
            return _compute_task_duration(self, task, devices, verbose=verbose)
        elif isinstance(task, SimulatedDataTask):
            return _data_task_duration(self, task, devices, verbose=verbose)
        else:
            raise RuntimeError(f"Invalid task type for {task} of type {type(task)}")

    def check_task_status(
        self, task: SimulatedTask, status: TaskStatus, verbose: bool = False
    ) -> bool:
        return task.check_status(status, self.objects.taskmap, self.time)
