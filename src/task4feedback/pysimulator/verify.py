from .task import *
from ..legacy_types import *
from .topology import *


def verify_task_order(task: SimulatedTask, taskmap: SimulatedTaskMap):
    times = task.times

    # Verify state times are ascending
    times.state_times
    for state_1 in TaskState:
        for state_2 in TaskState:
            if state_2 >= state_1:
                assert times.state_times[state_1] <= times.state_times[state_2], (
                    f"Task {task.name} has state {state_1} at time {times.state_times[state_1]}. Task {task.name} has state {state_2} at time {times.state_times[state_2]}."
                )

    # Verify status times are ascending
    for status_1 in TaskStatus:
        for status_2 in TaskStatus:
            if status_2 >= status_1:
                assert times.status_times[status_1] <= times.status_times[status_2], (
                    f"Task {task.name} has status {status_1} at time {times.status_times[status_1]}. Task {task.name} has status {status_2} at time {times.status_times[status_2]}."
                )

    # Verify state times are after all dependencies
    for state in TaskState:
        for dependency in task.dependencies:
            dependency = taskmap[dependency]
            assert times.state_times[state] >= dependency.times.state_times[state], (
                f"Task {task.name} has state {state} at time {times.state_times[state]}. Dependency {dependency.name} has state {state} at time {dependency.times.state_times[state]}."
            )

    # Verify status times are after all dependencies
    for status in TaskStatus:
        if isinstance(task, SimulatedDataTask) and status < TaskStatus.RESERVABLE:
            continue

        for dependency in task.dependencies:
            dependency = taskmap[dependency]
            if (
                isinstance(dependency, SimulatedDataTask)
                and status <= TaskStatus.RESERVABLE
            ):
                continue
            assert (
                times.status_times[status] >= dependency.times.status_times[status]
            ), (
                f"Task {task.name} has status {status} at time {times.status_times[status]}. Dependency {dependency.name} has status {status} at time {dependency.times.status_times[status]}."
            )


def verify_order(tasks: SimulatedTaskMap):
    for task in tasks.values():
        verify_task_order(task, tasks)


def verify_resources_at_snapshot(
    running_tasks: Set[TaskID],
    tasks: SimulatedTaskMap,
    devicemap: Mapping[Device, SimulatedDevice],
):
    current_usage = {}
    for device in devicemap.keys():
        current_usage[device] = FasterResourceSet(vcus=0, memory=0, copy=0)

    for task in running_tasks:
        task = tasks[task]
        devices = task.assigned_devices
        assert devices is not None, (
            f"Task {task.name} is running but has no assigned devices."
        )
        if not isinstance(devices, tuple):
            devices = (devices,)

        resources = task.resources

        for i, device in enumerate(devices):
            current_usage[device] += resources[i]

    for device, usage in current_usage.items():
        assert usage <= devicemap[device].resources, (
            f"Device {device} has resources {devicemap[device].resources} but is using {usage}."
        )


def verify_runtime_resources(
    tasks: SimulatedTaskMap, devices: Mapping[Device, SimulatedDevice]
):
    task_events = []
    for task in tasks.values():
        launched_time = task.times.state_times[TaskState.LAUNCHED]
        completed_time = task.times.state_times[TaskState.COMPLETED]

        task_events.append((launched_time, TaskState.LAUNCHED, task))
        task_events.append((completed_time, TaskState.COMPLETED, task))

    sorted_events = sorted(task_events, key=lambda x: x[0])
    running_tasks = set()

    for event in sorted_events:
        task: SimulatedTask = event[2]
        state: TaskState = event[1]
        time: Time = event[0]

        if state == TaskState.LAUNCHED:
            running_tasks.add(task.name)
        elif state == TaskState.COMPLETED:
            running_tasks.remove(task.name)

        verify_resources_at_snapshot(running_tasks, tasks, devices)
