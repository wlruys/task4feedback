from .base import *
from ..schedulers import *


def _create_eviction_task(
    parent_task: TaskID,
    data: SimulatedData,
    source_device: Device,
    target_device: Device,
) -> SimulatedEvictionTask:

    # Assume each parent task only requests one eviction task for each of its data dependencies

    eviction_space = str(parent_task) + "_eviction"
    eviction_id = TaskID(taskspace=eviction_space, task_idx=data.name.idx)
    eviction_dependencies = list(data.status.eviction_tasks)
    data_dependencies = TaskDataInfo(read=[DataAccess(id=data.name, device=0)])
    eviction_runtime = TaskPlacementInfo()
    eviction_runtime.add(Device(Architecture.ANY, -1), TaskRuntimeInfo())

    info = TaskInfo(
        id=eviction_id,
        dependencies=eviction_dependencies,
        data_dependencies=data_dependencies,
        runtime=eviction_runtime,
    )

    eviction_task = SimulatedEvictionTask(
        name=eviction_id,
        info=info,
        source=source_device,
    )

    eviction_task.assigned_devices = (target_device,)

    return eviction_task


def _check_target_resources(
    state: SystemState,
    source_device: SimulatedDevice,
    target_device: SimulatedDevice,
    data: SimulatedData,
) -> bool:

    if source_device == target_device:
        # Backup copy already exists, no need to make a new one
        return True

    requested_resources = FasterResourceSet(vcus=0, memory=data.size, copy=0)

    can_fit = state.resource_pool.check_device_resources(
        target_device.name, TaskState.RESERVED, ResourceGroup.ALL, requested_resources
    )

    if not can_fit:
        raise RuntimeError(
            f"Eviction target does not have enough memory: {target_device}. Requested: {requested_resources}"
        )

    return can_fit


def _update_target_resources(
    state: SystemState,
    source_device: SimulatedDevice,
    target_device: SimulatedDevice,
    data: SimulatedData,
) -> None:
    if source_device == target_device:
        # Backup copy already exists, no need to make a new one
        return None

    requested_resources = FasterResourceSet(vcus=0, memory=data.size, copy=1)

    state.resource_pool.add_device_resource(
        target_device.name,
        TaskState.MAPPED,
        ResourceGroup.ALL,
        requested_resources,
    )

    state.resource_pool.add_device_resource(
        target_device.name,
        TaskState.RESERVED,
        ResourceGroup.PERSISTENT,
        requested_resources,
    )

    return None


def _update_data_state(
    task: SimulatedEvictionTask,
    state: SystemState,
    source_device: SimulatedDevice,
    target_device: SimulatedDevice,
    data: SimulatedData,
) -> None:
    data.status.evict(
        task.name,
        source_device=source_device.name,
        target_device=target_device.name,
        state=TaskState.MAPPED,
    )
    data.status.evict(
        task.name,
        source_device=source_device.name,
        target_device=target_device.name,
        state=TaskState.RESERVED,
    )


def eviction_init(
    parent_task: TaskID,
    state: SystemState,
    device: SimulatedDevice,
    data: SimulatedData,
) -> SimulatedEvictionTask:

    target_device_id = data.get_eviction_target(
        device.name, device.eviction_targets, TaskState.RESERVED
    )

    target_device = state.objects.get_device(target_device_id)
    can_fit = _check_target_resources(state, device, target_device, data)
    assert can_fit

    eviction_task = _create_eviction_task(
        parent_task=parent_task,
        data=data,
        source_device=device.name,
        target_device=target_device.name,
    )

    _update_target_resources(
        state=state, source_device=device, target_device=target_device, data=data
    )
    _update_data_state(
        task=eviction_task,
        state=state,
        source_device=device,
        target_device=target_device,
        data=data,
    )

    data.status.add_eviction_task(eviction_task.name)

    return eviction_task
