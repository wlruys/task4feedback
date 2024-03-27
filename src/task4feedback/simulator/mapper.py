from .schedulers.state import SystemState
from .schedulers.architecture import SchedulerArchitecture
from ..types import *
from .task import SimulatedTask
from dataclasses import dataclass
from copy import deepcopy
from .watcher import *
from functools import partial

MappingFunction = Callable[
    [SimulatedTask, "SimulatedScheduler"],
    Optional[Devices],
]


def default_mapping_policy(task: SimulatedTask, simulator) -> Optional[Devices]:

    scheduler_state: SystemState = simulator.state
    scheduler_arch: SchedulerArchitecture = simulator.mechanisms

    potential_devices = task.info.runtime.locations
    potential_device = potential_devices[0]
    if isinstance(potential_device, Tuple):
        potential_device = potential_device[0]

    if potential_device.architecture == Architecture.ANY:
        potential_device = Device(Architecture.GPU, 0)

    if potential_device.device_id == -1:
        potential_device = Device(potential_device.architecture, 0)

    return (potential_device,)


def earliest_finish_time(task: SimulatedTask, simulator) -> Optional[Devices]:

    min_time = Time(9999999999999999999)
    chosen_device = None

    for devices in task.info.runtime.locations:
        simulator_copy = deepcopy(simulator)
        mapper = TaskMapper(
            mapping_function=default_mapping_policy,
            restrict_tasks=True,
            allowed_set={task.name},
            assignments={task.name: devices},
        )
        simulator_copy.set_mapper(mapper)
        task_completion_check = partial(check_for_task_completion, tasks={task.name})
        simulator_copy.add_stop_condition(task_completion_check)

        simulator_copy.run()
        print(f"EFT of {task.name} on device {devices} is {simulator_copy.time}")

        if simulator_copy.time < min_time:
            min_time = simulator_copy.time
            chosen_device = devices

    print("----")
    # print(simulator_copy.current_event)

    return chosen_device  # default_mapping_policy(task, simulator)


def latest_finish_time(task: SimulatedTask, simulator) -> Optional[Devices]:

    max_time = Time(0)
    chosen_device = None

    for devices in task.info.runtime.locations:
        simulator_copy = deepcopy(simulator)
        mapper = TaskMapper(
            mapping_function=default_mapping_policy,
            restrict_tasks=True,
            allowed_set={task.name},
            assignments={task.name: devices},
        )
        simulator_copy.set_mapper(mapper)
        task_completion_check = partial(check_for_task_completion, tasks={task.name})
        simulator_copy.add_stop_condition(task_completion_check)

        simulator_copy.run()
        print(f"EFT of {task.name} on device {devices} is {simulator_copy.time}")

        if simulator_copy.time > max_time:
            max_time = simulator_copy.time
            chosen_device = devices

    print("----")
    # print(simulator_copy.current_event)

    return chosen_device  # default_mapping_policy(task, simulator)


@dataclass(slots=True)
class TaskMapper:
    restrict_tasks: bool = False
    allowed_set: Set[TaskID] = field(default_factory=set)
    assignments: Dict[TaskID, Devices] = field(default_factory=dict)
    mapping_function: MappingFunction = latest_finish_time

    def map_task(self, task: SimulatedTask, simulator) -> Optional[Devices]:

        if self.check_allowed(task) is False:
            return None

        if task.name in self.assignments:
            # print(f"Task {task.name} mapped to {self.assignments[task.name]}.")
            return self.assignments[task.name]

        return self.mapping_function(task, simulator)

    def set_assignments(self, assignments: Dict[TaskID, Devices]):
        self.assignments = assignments

    def set_allowed(self, task_ids: Set[TaskID]):
        self.allowed_set = task_ids
        self.restrict_tasks = True

    def check_allowed(self, task: SimulatedTask) -> bool:
        return (not self.restrict_tasks) or (task.name in self.allowed_set)

    def __deepcopy__(self, memo):
        return TaskMapper(
            mapping_function=self.mapping_function,
        )
