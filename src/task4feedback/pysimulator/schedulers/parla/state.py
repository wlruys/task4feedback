from task4feedback.pysimulator import task
from ....legacy_types import *
from ...data import *
from ...device import *
from ...utility import *
from ...randomizer import *

from ..state import *
from ..architecture import *

from ....logging import logger

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
    phase: TaskState,
    device: Device,
    data_id: DataID,
    objects: ObjectRegistry,
    access_type: AccessType,
    always_count: bool = False,
) -> int:
    data = objects.get_data(data_id)

    if not always_count and (
        is_valid := data.is_valid_or_moving(device, phase)
        and (access_type == AccessType.READ or access_type == AccessType.READ_WRITE)
    ):
        return 0
    else:
        return data.size


def get_required_memory(
    memory: List[int],
    phase: TaskState,
    devices: Tuple[Device, ...],
    data_accesses: List[DataAccess],
    objects: ObjectRegistry,
    access_type: AccessType = AccessType.READ,
    always_count: bool = False,
) -> None:
    for data_access in data_accesses:
        idx = data_access.device
        device = devices[idx]
        memory[idx] += get_required_memory_for_data(
            phase, device, data_access.id, objects, access_type, always_count
        )


def get_required_resources(
    phase: TaskState,
    task: SimulatedTask,
    devices: Devices,
    objects: ObjectRegistry,
    count_data: bool = True,
    verbose: bool = False,
    always_count_data: bool = False,
) -> List[FasterResourceSet]:
    if isinstance(devices, Device):
        devices = (devices,)

    if phase == TaskState.MAPPED or (
        phase == TaskState.RESERVED and isinstance(task, SimulatedDataTask)
    ):
        resources = task.get_resources(devices)
        task.resources = resources
    else:
        resources = task.resources

    memory: List[int] = [s.memory for s in resources]

    if count_data:
        get_required_memory(
            memory, phase, devices, task.read_accesses, objects, always_count_data
        )
        get_required_memory(
            memory, phase, devices, task.read_write_accesses, objects, always_count_data
        )
        get_required_memory(
            memory, phase, devices, task.write_accesses, objects, always_count_data
        )

    resources = []
    for i in range(len(devices)):
        t_req = task.resources[i]
        mem = memory[i]
        vcus = t_req.vcus
        copy = t_req.copy
        resources.append(FasterResourceSet(vcus=vcus, memory=mem, copy=copy))

    if logger.ENABLE_LOGGING:
        logger.resource.debug(
            f"Required resources {resources} for {task.name} in {phase} phase",
            extra=dict(task=task.name, phase=phase, resources=resources),
        )

    return resources


def _check_eviction(
    state: SystemState,
    task: SimulatedTask,
    verbose: bool = False,
) -> Optional[Eviction]:
    from rich import print

    # print("Checking eviction for task: ", task.name)

    # Only generate an eviction event if it would free enough resources to run the next task
    resource_differences = _get_difference_reserved(state, task, verbose=verbose)

    # verbose = True
    # print("Time: ", state.time)
    if verbose:
        print(
            f"Cannot allocate memory for task {task.name}. Requires {resource_differences} more memory."
        )

    # Remove the task's own data from the eviction pool
    # for data_access in task.info.data_dependencies.all_accesses():
    #     data = state.objects.get_data(data_access.id)
    #     assert data is not None
    #     for device in state.topology.devices:
    #         data.status.add_task(
    #             device.name, task.name, DataUses.CHECKING, state.data_pool
    #         )

    for device, resources in resource_differences.items():
        evictable_memory = state.data_pool.evictable[device].evictable_size
        unresolved_requests = task.requested_eviction_bytes[device]

        # print(f"Evictable memory on {device}: {evictable_memory}")
        # print(f"Incomplete eviction requests on {device}: {unresolved_requests}")

        resources.memory -= unresolved_requests

        # print(f"Missing Memory on {device}: {resources.memory}")

        # print(f"Data on {device}:)")
        # for data in state.objects.datamap.values():
        #     if data.is_valid(device, TaskState.RESERVED):
        #         print(f"{data.name} is valid on {device}")
        #         print(data.status.uses)

        # print(f"Eviction pool on Device {device}")
        # print(state.data_pool.evictable[device])

        if verbose:
            print(f"Evictable memory on {device}: {evictable_memory}")
            print(f"Missing Memory on {device}: {resources.memory}")

        if resources.memory > evictable_memory or resources.memory < 0:
            # if : request is too big and not satisfiable by eviction, do not run eviction
            # if : request is already satisfied by previously enqueued (but not complete) eviction, do not run eviction
            if state.reserved_active_tasks == 0:
                devices = task.assigned_devices
                if _check_eviction_status(state, task, check_complete=True):
                    task_data_print(state, task, devices, TaskState.RESERVED)
                    resource_error_print(
                        state, task, devices, [resources], resource_types=["memory"]
                    )
                    raise RuntimeError(
                        f"Failure to acquire resources for task {task.name}."
                    )

            # Return tasks data to the eviction pool
            # for data_access in task.info.data_dependencies.all_accesses():
            #     data = state.objects.get_data(data_access.id)
            #     assert data is not None
            #     data.status.uses.remove_task_use(
            #         task.name, DataUses.CHECKING, state.data_pool
            #     )

            return None

    return Eviction(parent_task=task.name, requested_resources=resource_differences)


def _check_nearest_source(
    state: SystemState,
    task: SimulatedDataTask,
    verbose: bool = False,
) -> Optional[Device | SimulatedDevice]:
    from rich import print

    assert isinstance(task, SimulatedDataTask)
    devices = task.assigned_devices
    assert devices is not None
    assert isinstance(devices, Device) or len(devices) == 1

    data_id = task.read_accesses[0].id
    data = state.objects.get_data(data_id)
    assert data is not None

    device_id = devices[0] if isinstance(devices, tuple) else devices
    device = state.objects.get_device(device_id)
    assert device is not None

    valid_sources_ids = data.get_device_set_from_states(
        TaskState.LAUNCHED, DataState.VALID
    )
    valid_sources = [state.objects.get_device(d) for d in valid_sources_ids]

    source_device = state.topology.nearest_valid_connection(
        device, valid_sources, require_copy_engines=True, require_symmetric=True
    )

    # print(f"Source set for {data.name}: {valid_sources_ids}")
    # print("Data Status")
    # print(data.status)

    # print("Data Uses")
    # print(data.status.uses)

    for eviction_task in data.status.uses.eviction_tasks:
        eviction_task: SimulatedEvictionTask = state.objects.get_task(eviction_task)
        # print(
        #     f"Eviction task: {eviction_task} uses {data.name}: evicting at {eviction_task.source}, moving to {eviction_task.assigned_devices}"
        # )
        assert isinstance(eviction_task, SimulatedEvictionTask)
        if eviction_task.source == source_device:
            if eviction_task.in_ready_queue:
                # print(
                #     f"Eviction task {eviction_task} is already evicting from {source_device}"
                # )
                return None

            if eviction_task not in task.dependents:
                # print(f"Adding dependency on eviction task {eviction_task} for {task.name}")
                eviction_task.add_dependency(
                    task.name, states=[TaskState.LAUNCHED, TaskState.COMPLETED]
                )
                task.add_dependent(eviction_task.name)

    if logger.ENABLE_LOGGING:
        logger.data.debug(
            f"Finding nearest data source for {data.name} on {device_id}: {valid_sources_ids} -> {source_device}",
            extra=dict(
                data=data.name,
                target=device_id,
                valid_sources=valid_sources_ids,
                source=source_device,
                time=state.time,
            ),
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
        devices, TaskState.MAPPED, ResourceGroup.ALL, resources
    )

    if logger.ENABLE_LOGGING:
        for device in devices:
            remaining = state.resource_pool.pool[device][TaskState.MAPPED]

            logger.resource.debug(
                f"Resources after acquiring for mapped task {task.name} on {device}: {remaining}",
                extra=dict(
                    task=task.name,
                    device=device,
                    resources=remaining,
                    phase=TaskState.MAPPED,
                    pool=TaskState.MAPPED,
                    time=state.time,
                ),
            )


def _get_difference_reserved(
    state: SystemState, task: SimulatedTask, verbose: bool = False
) -> Dict[Device, FasterResourceSet]:
    devices = task.assigned_devices

    if isinstance(task, SimulatedDataTask):
        raise RuntimeError(
            f"Data tasks should never hit the Reserver. Invalid task: {task}"
        )

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources = get_required_resources(
        TaskState.RESERVED, task, devices, state.objects, count_data=True
    )

    # print(f"Resources at time of eviction request")
    # task_data_print(state, task, devices, TaskState.RESERVED)
    # resource_error_print(state, task, task.assigned_devices, resources)

    # print(f"Resources required for task {task.name} in RESERVED state: {resources}")
    # print(
    #     f"Resources in use in RESERVED state: {state.resource_pool.pool[devices[0]][TaskState.RESERVED]}"
    # )

    missing_resources = state.resource_pool.get_difference(
        devices=devices,
        state=TaskState.RESERVED,
        type=ResourceGroup.PERSISTENT,
        resources=resources,
    )

    return missing_resources


def task_data_print(
    state: SystemState, task: SimulatedTask, devices: Devices, phase: TaskState
):
    from rich import print
    from rich.console import Console
    from rich.table import Table

    if isinstance(devices, Device):
        devices = (devices,)

    table_task = Table(
        title=f"Task {task.name} on {devices[0]}",
        show_header=True,
        header_style="bold",
    )

    table_task.add_column("Device", justify="right", style="cyan", no_wrap=True)
    table_task.add_column("Resource Type", style="magenta")
    table_task.add_column("Requested", justify="right", style="yellow")

    resources = task.get_resources(devices)

    for resource in resources:
        table_task.add_row(str(devices[0]), "VCUs", str(resource.vcus))
        table_task.add_row(str(devices[0]), "Memory", str(resource.memory))
        table_task.add_row(str(devices[0]), "Copy Engines", str(resource.copy))

    table_data = Table(
        title=f"Data Accesses for Task {task.name}",
        show_header=True,
        header_style="bold",
    )

    table_data.add_column("Device", justify="right", style="cyan", no_wrap=True)
    table_data.add_column("Data ID", style="magenta")
    table_data.add_column("Access Type", style="yellow")
    table_data.add_column("Data Size", justify="right", style="green")
    table_data.add_column("Local?", justify="right", style="blue")

    for data_access in task.read_accesses:
        data_id = data_access.id
        device_idx = data_access.device
        device_id = devices[device_idx]

        data = state.objects.get_data(data_id)
        assert data is not None

        is_local = data.is_valid(device_id, phase)

        is_local_text = Text("Yes" if is_local else "No")
        if is_local:
            is_local_text.stylize("bold green", 0, 6)
        else:
            is_local_text.stylize("bold red", 0, 6)

        table_data.add_row(
            str(device_id),
            str(data.name),
            "READ",
            str(data.size),
            is_local_text,
        )

    for data_access in task.read_write_accesses:
        data_id = data_access.id
        device_idx = data_access.device
        device_id = devices[device_idx]

        data = state.objects.get_data(data_id)
        assert data is not None

        is_local = data.is_valid(device_id, phase)

        is_local_text = Text("Yes" if is_local else "No")
        if is_local:
            is_local_text.stylize("bold green", 0, 6)
        else:
            is_local_text.stylize("bold red", 0, 6)

        table_data.add_row(
            str(device_id), str(data.name), "READ_WRITE", str(data.size), is_local_text
        )

    for data_access in task.write_accesses:
        data_id = data_access.id
        device_idx = data_access.device
        device_id = devices[device_idx]

        data = state.objects.get_data(data_id)
        assert data is not None

        is_local = data.is_valid(device_id, phase)

        is_local_text = Text("Yes" if is_local else "No")
        if is_local:
            is_local_text.stylize("bold green", 0, 6)
        else:
            is_local_text.stylize("bold red", 0, 6)

        table_data.add_row(
            str(device_id), str(data.name), "WRITE", str(data.size), is_local_text
        )

    console = Console()
    console.print(table_task)
    console.print(table_data)


def resource_error_print(
    state: SystemState,
    task: SimulatedTask,
    devices: Devices,
    requested_resources: List[FasterResourceSet],
    resource_types: List[ResourceType] = ["vcus", "memory", "copy"],
):
    from rich import print
    from rich.console import Console
    from rich.table import Table

    if isinstance(devices, Device):
        devices = (devices,)

    table_resources = Table(
        title=f"Resources Requested for Task {task.name}",
        show_header=True,
        header_style="bold",
    )

    table_resources.add_column("Device", justify="right", style="cyan", no_wrap=True)
    table_resources.add_column("Resource", style="magenta")
    table_resources.add_column("Maximum", justify="right", style="yellow")
    table_resources.add_column("Requested", justify="right", style="green")
    table_resources.add_column("Available", justify="right", style="blue")
    table_resources.add_column("Difference", justify="right", style="red")

    for device, resources in zip(devices, requested_resources):
        used_resources = state.resource_pool.pool[device][TaskState.RESERVED]
        max_resources = state.objects.devicemap[device].resources
        available_resources = max_resources - used_resources

        for resource in resource_types:
            used_of_resource_type = getattr(used_resources, resource)
            max_of_resource_type = getattr(max_resources, resource)
            requested_of_resource_type = getattr(resources, resource)
            available_of_resource_type = getattr(available_resources, resource)
            difference_of_resource_type = (
                available_of_resource_type - requested_of_resource_type
            )

            difference_text = Text(str(difference_of_resource_type))

            if difference_of_resource_type < 0:
                difference_text.stylize("bold red", 0, len(difference_text))
            else:
                difference_text.stylize("bold green", 0, len(difference_text))

            table_resources.add_row(
                str(device),
                str(resource),
                str(max_of_resource_type),
                str(requested_of_resource_type),
                str(available_of_resource_type),
                difference_text,
            )
    console = Console()
    console.print(table_resources)


def _check_eviction_status(state, task, check_complete=True):
    if not state.use_eviction:
        return True

    if check_complete:
        # if: there are any outstanding eviction requests, don't throw an error yet
        outstanding_bytes = list(task.requested_eviction_bytes.values())
        print(f">> outstanding bytes for task {task}", outstanding_bytes)
        if any([d > 0 for d in outstanding_bytes]):
            return False
        return True
    else:
        return False


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

    resources = get_required_resources(
        TaskState.RESERVED, task, devices, state.objects, count_data=True
    )

    can_fit = state.resource_pool.check_resources(
        devices=devices,
        state=TaskState.RESERVED,
        type=ResourceGroup.PERSISTENT,
        resources=resources,
    )

    # print(f"Can fit resources: {can_fit}: ", resources)
    # print(f"Reserved active tasks: {state.reserved_active_tasks}")

    if not can_fit and state.reserved_active_tasks == 0:
        # check if task could NEVER fit
        max_resources = get_required_resources(
            TaskState.RESERVED,
            task,
            devices,
            state.objects,
            count_data=True,
            always_count_data=True,
        )
        can_fit_max = state.resource_pool.check_max_resources(
            devices=devices, type=ResourceGroup.PERSISTENT, resources=max_resources
        )
        if not can_fit_max:
            task_data_print(state, task, devices, TaskState.RESERVED)
            resource_error_print(
                state, task, devices, max_resources, resource_types=["memory"]
            )
            raise RuntimeError(
                f"Task {task.name} requires more memory than is satisifiable on its chosen devices (at max capacity)."
            )

        # print("Checking eviction status")
        if _check_eviction_status(state, task, check_complete=False):
            task_data_print(state, task, devices, TaskState.RESERVED)
            resource_error_print(
                state, task, devices, resources, resource_types=["memory"]
            )
            raise RuntimeError(f"Failure to acquire resources for task {task.name}.")

    # print(f"Resources required for task {task.name} in RESERVED state: {resources}")
    # print(
    #    f"Current resources in use for task {task.name} in RESERVED state: {state.resource_pool.pool[devices[0]][TaskState.RESERVED]}"
    # )
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

    # resource_types = StatesToResources[TaskState.RESERVED]

    resources = get_required_resources(
        TaskState.RESERVED, task, devices, state.objects, count_data=True
    )

    state.resource_pool.add_resources(
        devices, TaskState.RESERVED, ResourceGroup.PERSISTENT, resources
    )
    # print(
    #     f"Resources acquired for task {task.name} in RESERVED state on device {devices[0]}"
    # )

    if logger.ENABLE_LOGGING:
        for device in devices:
            remaining = state.resource_pool.pool[device][TaskState.RESERVED]
            logger.resource.debug(
                f"Resources after acquiring for reserved task {task.name} on {device}: {remaining}",
                extra=dict(
                    task=task.name,
                    device=device,
                    resources=remaining,
                    phase=TaskState.RESERVED,
                    pool=TaskState.RESERVED,
                    time=state.time,
                ),
            )


def _check_resources_launched(
    state: SystemState, task: SimulatedTask, verbose: bool = False
) -> bool:
    devices = task.assigned_devices

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    # resources_types = StatesToResources[TaskState.LAUNCHED]

    resources = get_required_resources(
        TaskState.RESERVED, task, devices, state.objects, count_data=False
    )

    can_fit = state.resource_pool.check_resources(
        devices=devices,
        state=TaskState.RESERVED,
        type=ResourceGroup.NONPERSISTENT,
        resources=resources,
    )

    if not can_fit and state.launched_active_tasks == 0:
        task_data_print(state, task, devices, TaskState.LAUNCHED)
        resource_error_print(
            state, task, devices, resources, resource_types=["vcus", "copy"]
        )
        raise RuntimeError(f"Failure to acquire resources for task {task.name}.")

    if isinstance(task, SimulatedDataTask) and not isinstance(
        task, SimulatedEvictionTask
    ):
        source_device = _check_nearest_source(state, task)

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
        TaskState.LAUNCHED, task, devices, state.objects, count_data=True
    )
    # print(f"Task {task.name} launching with resources: {resources}")

    state.resource_pool.add_resources(
        devices, TaskState.RESERVED, ResourceGroup.NONPERSISTENT, resources
    )

    state.resource_pool.add_resources(
        devices, TaskState.LAUNCHED, ResourceGroup.ALL, resources
    )

    if isinstance(task, SimulatedDataTask) or isinstance(task, SimulatedEvictionTask):
        assert len(devices) == 1
        target_device = devices[0]
        source_device = task.source
        assert source_device is not None

        state.topology.acquire_connection(source_device, target_device)
        if logger.ENABLE_LOGGING:
            logger.resource.info(
                "Acquired connection",
                extra=dict(
                    task=task.name,
                    source=source_device,
                    target=target_device,
                    time=state.time,
                ),
            )

    if logger.ENABLE_LOGGING:
        for device in devices:
            remaining_reserved = state.resource_pool.pool[device][TaskState.RESERVED]
            remaining_launched = state.resource_pool.pool[device][TaskState.LAUNCHED]

            logger.resource.debug(
                f"Reserved Resources after acquiring launched task {task.name} on {device}: {remaining_reserved}",
                extra=dict(
                    task=task.name,
                    device=device,
                    resources=remaining_reserved,
                    phase=TaskState.LAUNCHED,
                    pool=TaskState.RESERVED,
                    time=state.time,
                ),
            )

            logger.resource.debug(
                f"Launched Resources after acquiring launched task {task.name} on {device}: {remaining_launched}",
                extra=dict(
                    task=task.name,
                    device=device,
                    resources=remaining_launched,
                    phase=TaskState.LAUNCHED,
                    pool=TaskState.LAUNCHED,
                    time=state.time,
                ),
            )


def _release_resources_completed(
    state: SystemState, task: SimulatedTask, verbose: bool = False
):
    devices = task.assigned_devices

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources = get_required_resources(
        TaskState.LAUNCHED, task, devices, state.objects, count_data=False
    )

    # Free resources from all pools
    if isinstance(task, SimulatedComputeTask):
        state.resource_pool.remove_resources(
            devices=devices,
            state=TaskState.MAPPED,
            type=ResourceGroup.ALL,
            resources=task.resources,
        )
    elif isinstance(task, SimulatedDataTask) or isinstance(task, SimulatedEvictionTask):
        assert len(devices) == 1
        target_device = devices[0]
        source_device = task.source
        assert source_device is not None

        state.topology.release_connection(source_device, target_device)
        if logger.ENABLE_LOGGING:
            logger.resource.info(
                "Released connection",
                extra=dict(
                    task=task.name,
                    source=source_device,
                    target=target_device,
                    time=state.time,
                ),
            )

        if isinstance(task, SimulatedEvictionTask):
            # print(f"I am an eviction task: {task.name} my parent is {task.parent}")
            parent_task = state.objects.get_task(task.parent)
            data = task.read_accesses[0].id
            data = state.objects.get_data(data)
            parent_task.requested_eviction_bytes[source_device] -= data.size

    state.resource_pool.remove_resources(
        devices=devices,
        state=TaskState.RESERVED,
        type=ResourceGroup.ALL,
        resources=task.resources,
    )

    state.resource_pool.remove_resources(
        devices=devices,
        state=TaskState.LAUNCHED,
        type=ResourceGroup.ALL,
        resources=task.resources,
    )

    if logger.ENABLE_LOGGING:
        for device in devices:
            remaining_reserved = state.resource_pool.pool[device][TaskState.RESERVED]
            remaining_mapped = state.resource_pool.pool[device][TaskState.MAPPED]
            remaining_launched = state.resource_pool.pool[device][TaskState.LAUNCHED]

            logger.resource.debug(
                f"Mapped resources after releasing task {task} on {device}: {resources}",
                extra=dict(
                    task=task.name,
                    device=device,
                    resources=remaining_mapped,
                    phase=TaskState.COMPLETED,
                    pool=TaskState.MAPPED,
                    time=state.time,
                ),
            )

            logger.resource.debug(
                f"Reserved Resources after releasing task {task} on {device}: {remaining_reserved}",
                extra=dict(
                    task=task.name,
                    device=device,
                    resources=remaining_reserved,
                    phase=TaskState.COMPLETED,
                    pool=TaskState.RESERVED,
                    time=state.time,
                ),
            )

            logger.resource.debug(
                f"Launched Resources after releasing task {task} on {device}: {remaining_launched}",
                extra=dict(
                    task=task.name,
                    device=device,
                    resources=remaining_launched,
                    phase=TaskState.COMPLETED,
                    pool=TaskState.LAUNCHED,
                ),
            )

            # print(
            #     f"Removed resources after completing task {task} on {device}, time: {state.time}"
            # )


def _remove_memory(
    data: SimulatedData,
    locations: Sequence[Device],
    phases: Sequence[TaskState],
    state: SystemState,
):
    for device in locations:
        for phase in phases:
            state.resource_pool.remove_device_resources(
                device,
                phase,
                ResourceGroup.PERSISTENT,
                FasterResourceSet(memory=data.size, vcus=0, copy=0),
            )


def _use_data(
    state: SystemState,
    phase: TaskState,
    data_accesses: List[DataAccess],
    task: SimulatedTask,
    access_type: AccessType,
    verbose: bool = False,
):
    if len(data_accesses) == 0:
        return

    devices = task.assigned_devices
    assert devices is not None

    for data_access in data_accesses:
        data_id = data_access.id
        device_idx = data_access.device
        device_id = devices[device_idx]

        data = state.objects.get_data(data_id)
        assert data is not None

        device = state.objects.get_device(device_id)
        assert device is not None

        update_state = True
        initial_state = False

        if phase == TaskState.MAPPED:
            initial_state = True
        elif phase == TaskState.RESERVED:
            data.finish_use(
                task=task.name,
                target_device=device_id,
                state=TaskState.MAPPED,
                operation=access_type,
                pools=state.data_pool,
            )

        elif phase == TaskState.LAUNCHED:
            data.finish_use(
                task=task.name,
                target_device=device_id,
                state=TaskState.RESERVED,
                operation=access_type,
                pools=state.data_pool,
            )
            # State updates at runtime are managed by data movement tasks
            # Compute tasks only verify and evict
            update_state = False

        old_state, evicted_locations = data.start_use(
            task=task.name,
            target_device=device_id,
            state=phase,
            operation=access_type,
            pools=state.data_pool,
            update=update_state,
            verbose=verbose,
        )

        if phase == TaskState.LAUNCHED:
            _remove_memory(
                data,
                evicted_locations,
                [TaskState.MAPPED, TaskState.RESERVED, TaskState.LAUNCHED],
                state,
            )
            data.status.remove_task(task.name, DataUses.EVICTING, pools=state.data_pool)


def _release_data(
    state: SystemState,
    phase: TaskState,
    data_accesses: List[DataAccess],
    task: SimulatedTask,
    access_type: AccessType,
    verbose: bool = False,
):
    # from rich import print

    if len(data_accesses) == 0:
        return

    assert phase == TaskState.COMPLETED

    devices = task.assigned_devices
    assert devices is not None

    for data_access in data_accesses:
        data_id = data_access.id
        device_idx = data_access.device
        device_id = devices[device_idx]

        device = state.objects.get_device(device_id)
        assert device is not None

        data = state.objects.get_data(data_id)
        assert data is not None

        data.finish_use(
            task=task.name,
            target_device=device_id,
            state=TaskState.LAUNCHED,
            operation=access_type,
            pools=state.data_pool,
        )


def _move_data(
    state: SystemState,
    data_accesses: List[DataAccess],
    task: SimulatedTask,
    verbose: bool = False,
):
    if len(data_accesses) == 0:
        return

    devices = task.assigned_devices
    assert devices is not None

    # Move data is called from a data movement task at launch time
    assert isinstance(task, SimulatedDataTask)

    # move data is called from a data movement task at launch time
    # each data movement task moves one data item onto a single target device
    assert len(data_accesses) == 1, (
        f"Data Task {task.name} should only move one data item: {data_accesses}"
    )
    assert len(devices) == 1, (
        f"Data Task {task.name} should only move to one device: {devices}"
    )
    target_device = devices[0]
    assert target_device is not None

    data = state.objects.get_data(data_accesses[0].id)
    assert data is not None

    # Assumes source device is set by the prior check_resources_launched call
    source_device = task.source
    assert source_device is not None

    # Mark data as moving onto target device
    prior_state = data.start_move(
        task=task.name,
        source_device=source_device,
        target_device=target_device,
        pools=state.data_pool,
    )


def _finish_move(
    state: SystemState, data_accesses: List[DataAccess], task: SimulatedTask
):
    if len(data_accesses) == 0:
        return

    devices = task.assigned_devices
    assert devices is not None

    # Move data is called from a data movement task at launch time
    assert isinstance(task, SimulatedDataTask)

    # move data is called from a data movement task at launch time
    # each data movement task moves one data item onto a single target device
    assert len(data_accesses) == 1, (
        f"Data Task {task.name} should only move one data item: {data_accesses}"
    )
    assert len(devices) == 1, (
        f"Data Task {task.name} should only move to one device: {devices}"
    )
    target_device = devices[0]
    assert target_device is not None

    data = state.objects.get_data(data_accesses[0].id)
    assert data is not None

    # Assumes source device is set by the prior check_resources_launched call
    source_device = task.source
    assert source_device is not None

    # Mark data as valid on target device
    prior_state = data.finish_move(
        task=task.name,
        source_device=source_device,
        target_device=target_device,
        pools=state.data_pool,
    )


def _start_evict(
    state: SystemState,
    data_accesses: List[DataAccess],
    task: SimulatedEvictionTask,
    access_type: AccessType,
    verbose: bool = False,
):
    if len(data_accesses) == 0:
        return

    devices = task.assigned_devices
    assert devices is not None

    assert len(devices) == 1
    assert len(data_accesses) == 1

    assert task.source is not None

    # print(
    #     f"Starting eviction for task {task.name} on {devices[0]}, source {task.source}"
    # )

    for data_access in data_accesses:
        data_id = data_access.id
        device_idx = data_access.device
        device_id = devices[device_idx]

        data = state.objects.get_data(data_id)
        assert data is not None

        device = state.objects.get_device(device_id)
        assert device is not None

        data.status.start_eviction(
            task=task.name,
            target_device=device_id,
            source_device=task.source,
            state=TaskState.LAUNCHED,
            pools=state.data_pool,
        )


def _finish_evict(
    state: SystemState,
    data_accesses: List[DataAccess],
    task: SimulatedEvictionTask,
    access_type: AccessType,
    verbose: bool = False,
):
    if len(data_accesses) == 0:
        return

    devices = task.assigned_devices
    assert devices is not None

    assert len(devices) == 1
    assert len(data_accesses) == 1

    assert task.source is not None

    for data_access in data_accesses:
        data_id = data_access.id
        device_idx = data_access.device
        device_id = devices[device_idx]

        data = state.objects.get_data(data_id)
        assert data is not None

        device = state.objects.get_device(device_id)
        assert device is not None

        old_state, evicted_locations = data.status.finish_eviction(
            task=task.name,
            target_device=device_id,
            source_device=task.source,
            state=TaskState.LAUNCHED,
            pools=state.data_pool,
        )

        _remove_memory(
            data, evicted_locations, [TaskState.RESERVED, TaskState.LAUNCHED], state
        )


def _compute_task_duration(
    state: SystemState,
    task: SimulatedComputeTask,
    devices: Devices,
    verbose: bool = False,
) -> Tuple[Time, Time]:
    assert isinstance(task, SimulatedComputeTask)
    assert devices is not None

    runtime_infos = task.get_runtime_info(devices)

    max_time = max([runtime_info.task_time for runtime_info in runtime_infos])
    duration = Time(max_time)

    completion_time = state.time + duration
    return duration, completion_time


def _data_task_duration(
    state: SystemState,
    task: SimulatedDataTask,
    target_devices: Devices,
    verbose: bool = False,
) -> Tuple[Time, Time]:
    assert target_devices is not None
    assert task.source is not None

    if isinstance(target_devices, Tuple):
        target = target_devices[0]
    else:
        target = target_devices

    data = state.objects.get_data(task.read_accesses[0].id)
    assert data is not None

    other_moving_tasks = data.get_tasks_from_usage(target, DataUses.MOVING_TO)
    if len(other_moving_tasks) > 0:
        task.real = False
        duration = Time(0)
        other_task = list(other_moving_tasks)[0]
        assert other_task != task.name, (
            f"Current task {task} should not be in the list of moving tasks {other_moving_tasks} during duration calculation."
        )

        other_task = state.objects.get_task(other_task)
        completion_time = other_task.completion_time
    else:
        if task.source == target:
            task.real = False
        duration = state.topology.get_transfer_time(task.source, target, data.size)
        completion_time = state.time + duration

    # print(f"Data task {task.name} duration: {duration}, size: {data.size}")
    return duration, completion_time


def _eviction_task_duration(
    state: SystemState,
    task: SimulatedEvictionTask,
    devices: Devices,
    verbose: bool = False,
) -> Tuple[Time, Time]:
    assert isinstance(task, SimulatedEvictionTask)
    duration, completion_time = _data_task_duration(
        state, task, devices, verbose=verbose
    )
    duration = Time(1000)
    completion_time = state.time + duration

    return duration, completion_time


@SchedulerOptions.register_state("parla")
@dataclass(slots=True)
class ParlaState(SystemState):
    mapped_active_tasks: int = 0
    reserved_active_tasks: int = 0
    launched_active_tasks: int = 0
    # Total workload planned across devices
    total_active_workload: float = 0
    # Workload per device
    perdev_active_workload: Dict[Device, int] = field(default_factory=dict)
    # Earliest device available time estimation
    # It considers task dependencies and hence, is different from active workload.
    # e.g., taskA -> taskB, taskA -> taskC, each task takes 3ms, there are 2 GPUS,
    # taskA was mapped to GPU0, and others were mapped to GPU1.
    # Then,
    # perdev_active_workload:
    # GPU0: 3, GPU1: 6
    # But, perdev_earliest_avail_time:
    # GPU0: 3, GPU1: 9 since taskB and taskC cannot start until taskA is completed.
    #
    # NOTE that this is only utilized for EFT mapping policies.
    # This information provides more accurate estimation for the earliest ready time
    # of a task.
    perdev_earliest_avail_time: Dict[Device, int] = field(default_factory=dict)
    # # of completed tasks
    total_num_completed_tasks: int = 0
    # Threshold of the number of tasks that can be mapped per each mapper event
    mapper_num_tasks_threshold: int = -1
    # # of tasks in (mapped~launchable) states
    total_num_mapped_tasks: int = 0

    def initialize(
        self,
        task_ids: List[TaskID],
        task_objects: List[SimulatedTask],
        mapper_type: str,
        consider_initial_placement: bool,
    ):
        self.mapper_num_tasks_threshold = len(self.topology.devices) * 4
        for device in self.objects.devicemap:
            self.perdev_active_workload[device] = 0
            self.perdev_earliest_avail_time[device] = 0

        if self.task_order_mode == TaskOrderType.RANDOM:
            # Calculation of HEFT is required when using HEFT mapper
            if mapper_type == "heft":
                _ = calculate_heft(
                    task_objects,
                    self.objects.taskmap,
                    len(self.objects.devicemap) - 1,
                    self,
                    False,
                    consider_initial_placement,
                )
            if self.load_task_order:
                print("REPLAY RANDOM SORT")
                task_objects[:] = load_task_order(task_objects)
            else:
                print("RANDOM SORT")
                # Deep copy
                task_objects[:] = self.randomizer.task_order(
                    task_objects, self.objects.taskmap
                )
                if self.save_task_order:
                    print("save task order..")
                    save_task_order(task_objects)
        elif self.task_order_mode == TaskOrderType.HEFT:
            print("HEFT SORT")
            # Tasks are sorted in-place
            _ = calculate_heft(
                task_objects,
                self.objects.taskmap,
                len(self.objects.devicemap) - 1,
                self,
                True,
                consider_initial_placement,
            )
        elif self.task_order_mode == TaskOrderType.OPTIMAL:
            print("OPTIMAL SORT")

            # Calculation of HEFT is required when using HEFT mapper
            if mapper_type == "heft":
                _ = calculate_heft(
                    task_objects,
                    self.objects.taskmap,
                    len(self.objects.devicemap) - 1,
                    self,
                    False,
                    consider_initial_placement,
                )
            task_objects = sorted(task_objects, key=lambda x: x.info.z3_order)
            # print("OPTIMAL ORDER")
            for task in task_objects:
                # print(task.info.z3_order, task.name)
                task.info.order = task.info.z3_order

        task_ids[:] = [t.name for t in task_objects]

    def __deepcopy__(self, memo):
        s = clock()
        topology = deepcopy(self.topology)
        # print(f"Time to deepcopy topology: {clock() - s}")

        s = clock()
        data_pool = deepcopy(self.data_pool)
        # print(f"Time to deepcopy data_pool: {clock() - s}")

        s = clock()
        resource_pool = deepcopy(self.resource_pool)
        # print(f"Time to deepcopy resource_pool: {clock() - s}")

        s = clock()
        objects = deepcopy(self.objects)
        # print(f"Time to deepcopy objects: {clock() - s}")

        s = clock()
        time = deepcopy(self.time)
        # print(f"Time to deepcopy time: {clock() - s}")

        s = clock()
        perdev_active_workload = deepcopy(self.perdev_active_workload)

        s = clock()
        perdev_earliest_avail_time = deepcopy(self.perdev_earliest_avail_time)

        return ParlaState(
            topology=topology,
            data_pool=data_pool,
            resource_pool=resource_pool,
            objects=objects,
            time=time,
            init=self.init,
            use_eviction=self.use_eviction,
            mapped_active_tasks=self.mapped_active_tasks,
            reserved_active_tasks=self.reserved_active_tasks,
            launched_active_tasks=self.launched_active_tasks,
            total_active_workload=self.total_active_workload,
            perdev_active_workload=perdev_active_workload,
            perdev_earliest_avail_time=perdev_earliest_avail_time,
            total_num_completed_tasks=self.total_num_completed_tasks,
            mapper_num_tasks_threshold=self.mapper_num_tasks_threshold,
            total_num_mapped_tasks=self.total_num_mapped_tasks,
            randomizer=self.randomizer,
            task_order_mode=self.task_order_mode,
            save_task_order=self.save_task_order,
            load_task_order=self.load_task_order,
            save_task_noise=self.save_task_noise,
            load_task_noise=self.load_task_noise,
        )

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

    def get_resource_difference(
        self, phase: TaskState, task: SimulatedTask, verbose: bool = False
    ) -> Dict[Device, FasterResourceSet]:
        if phase == TaskState.RESERVED:
            return _get_difference_reserved(self, task)
        else:
            raise NotImplementedError(
                f"Invalid phase {phase} in get_resource_difference for task {task}"
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
            _use_data(
                self, phase, task.read_write_accesses, task, AccessType.READ_WRITE
            )
            _use_data(self, phase, task.write_accesses, task, AccessType.WRITE)
        elif isinstance(task, SimulatedEvictionTask):
            # All eviction tasks store data in "read access"
            _start_evict(self, task.read_accesses, task, AccessType.EVICT)
        elif isinstance(task, SimulatedDataTask):
            # Data movement tasks only exist at launch time
            assert phase == TaskState.LAUNCHED

            # All data movement tasks are single data item tasks
            # They read from a single source device onto a single target device
            _move_data(self, task.read_accesses, task)
            _move_data(self, task.read_write_accesses, task)

    def release_data(
        self,
        phase: TaskState,
        task: SimulatedTask,
        verbose: bool = False,
    ):
        assert phase == TaskState.COMPLETED
        # print("RELEASING DATA")
        # print("Task is type: ", type(task))
        if isinstance(task, SimulatedComputeTask):
            # print("COMPLETED COMPUTE")
            _release_data(self, phase, task.read_accesses, task, AccessType.READ)
            _release_data(
                self, phase, task.read_write_accesses, task, AccessType.READ_WRITE
            )
            _release_data(self, phase, task.write_accesses, task, AccessType.WRITE)
        elif isinstance(task, SimulatedEvictionTask):
            # print("COMPLETED EVICTION")
            _finish_evict(self, task.read_accesses, task, AccessType.EVICT)
        elif isinstance(task, SimulatedDataTask):
            # print("COMPLETED DATA")
            _finish_move(self, task.read_accesses, task)
            _finish_move(self, task.read_write_accesses, task)

    def get_task_duration(
        self, task: SimulatedTask, devices: Devices, verbose: bool = False
    ) -> Tuple[Time, Time]:
        if isinstance(task, SimulatedComputeTask):
            duration, completion_time = _compute_task_duration(
                self, task, devices, verbose=verbose
            )
        elif isinstance(task, SimulatedDataTask):
            duration, completion_time = _data_task_duration(
                self, task, devices, verbose=verbose
            )
        else:
            raise RuntimeError(f"Invalid task type for {task} of type {type(task)}")

        return duration

    def get_task_duration_completion(
        self, task: SimulatedTask, devices: Devices, verbose: bool = False
    ) -> Tuple[Time, Time]:
        if isinstance(task, SimulatedComputeTask):
            duration, completion_time = _compute_task_duration(
                self, task, devices, verbose=verbose
            )
        elif isinstance(task, SimulatedDataTask):
            duration, completion_time = _data_task_duration(
                self, task, devices, verbose=verbose
            )
        else:
            raise RuntimeError(f"Invalid task type for {task} of type {type(task)}")

        assert (
            self.use_duration_noise == False and self.load_task_noise == False
        ) or self.use_duration_noise != self.load_task_noise
        noise = Time()

        if "eviction" not in str(task.name) and "data" not in str(task.name):
            # if True:
            if self.use_duration_noise:
                # noise = Time(abs(gaussian_noise(duration.duration, self.noise_scale)))
                noise = log_normal_noise(duration.duration, self.noise_scale)
            elif self.load_task_noise:
                noise = int(self.loaded_task_noises[str(task.name)])

            if self.save_task_noise:
                save_task_noise(task, noise)

        return duration, completion_time + noise

    def check_task_status(
        self, task: SimulatedTask, status: TaskStatus, verbose: bool = False
    ) -> bool:
        return task.check_status(status, self.objects.taskmap, self.time)

    def check_eviction(
        self, task: SimulatedTask, verbose: bool = False
    ) -> Optional[Eviction]:
        if self.use_eviction:
            return _check_eviction(self, task, verbose=verbose)
        else:
            return None

    def complete(self):
        pass


@SchedulerOptions.register_state("rl")
@dataclass(slots=True)
class RLState(ParlaState):
    target_exec_time: float = 0

    def initialize(
        self,
        task_ids: List[TaskID],
        task_objects: List[SimulatedTask],
        mapper_type: str,
        consider_initial_placement: bool,
    ):
        self.mapper_num_tasks_threshold = len(self.topology.devices) * 4
        for device in self.objects.devicemap:
            self.perdev_active_workload[device] = 0

        # RL state always requires HEFT calculation for rewarding
        # Tasks are sorted in-place
        self.target_exec_time = calculate_heft(
            task_objects,
            self.objects.taskmap,
            len(self.objects.devicemap) - 1,
            self,
            self.task_order_mode == TaskOrderType.HEFT,
            consider_initial_placement,
        )

        if self.task_order_mode == TaskOrderType.RANDOM:
            if self.load_task_order:
                print("REPLAY RANDOM SORT")
                task_objects[:] = load_task_order(task_objects)
            else:
                print("RANDOM SORT")
                # Deep copy
                task_objects[:] = self.randomizer.task_order(
                    task_objects, self.objects.taskmap
                )
                if self.save_task_order:
                    print("save task order..")
                    save_task_order(task_objects)
        elif self.task_order_mode == TaskOrderType.HEFT:
            print("HEFT SORT")

        task_ids[:] = [t.name for t in task_objects]

    def complete(self):
        total_exec_time = convert_to_float(self.time.scale_to("ms"))
        if self.rl_mapper.is_training_mode():
            reward = (
                0
                if total_exec_time == 0
                else (self.target_exec_time - total_exec_time) / self.target_exec_time
            )
            # reward = -(1-reward) if reward < 0.8 else reward
            self.rl_mapper.optimize_model(reward, self)
        self.rl_mapper.complete_episode(total_exec_time)
