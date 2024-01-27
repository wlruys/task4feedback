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


def _acquire_resources_mapped(state: SystemState, taskid: TaskID):
    task = state.objects.get_task(taskid)
    devices = task.assigned_devices

    if isinstance(task, SimulatedDataTask):
        raise RuntimeError("Data tasks should never hit the Mapper.", taskid)

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources = get_required_resources(
        TaskState.MAPPED, task, devices, state.objects, count_data=True
    )
    state.resource_pool.add_resources(
        devices, TaskState.MAPPED, AllResources, resources
    )


def _check_resources_reserved(state: SystemState, taskid: TaskID) -> bool:
    task = state.objects.get_task(taskid)
    devices = task.assigned_devices

    if isinstance(task, SimulatedDataTask):
        raise RuntimeError("Data tasks should never hit the Reserver.", taskid)

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


def _acquire_resources_reserved(state: SystemState, taskid: TaskID):
    task = state.objects.get_task(taskid)
    devices = task.assigned_devices

    if isinstance(task, SimulatedDataTask):
        raise RuntimeError("Data tasks should never hit the Reserver.", taskid)

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


def _check_resources_launched(state: SystemState, taskid: TaskID) -> bool:
    task = state.objects.get_task(taskid)
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

    return can_fit


def _acquire_resources_launched(state: SystemState, taskid: TaskID):
    task = state.objects.get_task(taskid)
    devices = task.assigned_devices

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources = get_required_resources(
        TaskState.LAUNCHED, task, devices, state.objects, count_data=False
    )

    resource_types = StatesToResources[TaskState.LAUNCHED]

    print(
        f"Acquiring resources for task {taskid} on devices {devices} with resources {resources}."
    )

    state.resource_pool.add_resources(
        devices, TaskState.RESERVED, resource_types, resources
    )

    state.resource_pool.add_resources(
        devices, TaskState.LAUNCHED, AllResources, resources
    )
    print(f"Resources after acquiring:")
    for device in devices:
        print(
            f"Device {device}: {state.resource_pool.pool[device][TaskState.LAUNCHED]}"
        )


def _release_resources_completed(state: SystemState, taskid: TaskID):
    task = state.objects.get_task(taskid)
    devices = task.assigned_devices

    assert devices is not None
    if isinstance(devices, Device):
        devices = (devices,)

    resources = get_required_resources(
        TaskState.COMPLETED, task, devices, state.objects, count_data=False
    )

    print(
        f"Releasing resources for task {taskid} on devices {devices} with resources {resources}."
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

    state.resource_pool.remove_resources(
        devices=devices,
        state=TaskState.LAUNCHED,
        types=AllResources,
        resources=task.resources,
    )


@SchedulerOptions.register_state("parla")
@dataclass(slots=True)
class ParlaState(SystemState):
    def check_resources(self, phase: TaskState, taskid: TaskID) -> bool:
        if phase == TaskState.MAPPED:
            return True
        elif phase == TaskState.RESERVED:
            return _check_resources_reserved(self, taskid)
        elif phase == TaskState.LAUNCHED:
            return _check_resources_launched(self, taskid)
        else:
            raise RuntimeError(
                f"Invalid phase {phase} in check_resource for task {taskid}"
            )

    def acquire_resources(self, phase: TaskState, taskid: TaskID):
        if phase == TaskState.MAPPED:
            _acquire_resources_mapped(self, taskid)
        elif phase == TaskState.RESERVED:
            _acquire_resources_reserved(self, taskid)
        elif phase == TaskState.LAUNCHED:
            _acquire_resources_launched(self, taskid)
        else:
            raise RuntimeError(
                f"Invalid phase {phase} in acquire_resource for task {taskid}"
            )

    def release_resources(self, phase: TaskState, taskid: TaskID):
        if phase == TaskState.COMPLETED:
            _release_resources_completed(self, taskid)
        else:
            raise RuntimeError(
                f"Invalid phase {phase} in release_resource for task {taskid}"
            )

    def use_data(self, phase: TaskState, taskid: TaskID):
        task = self.objects.get_task(taskid)

        if isinstance(task, SimulatedComputeTask):
            print(f"Task {taskid} is a compute task.")
        elif isinstance(task, SimulatedDataTask):
            print(f"Task {taskid} is a data task.")

    def release_data(
        self, phase: TaskState, taskid: TaskID, dataid: DataID, access: AccessType
    ):
        pass
