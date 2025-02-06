from ..types import *
from .schedulers import SystemState, SchedulerArchitecture
from .events import *
from .task import *


@dataclass(slots=True)
class Watcher:
    conditions: List[Callable[[SystemState, SchedulerArchitecture, Event], bool]] = (
        field(default_factory=list)
    )

    def add_condition(
        self, condition: Callable[[SystemState, SchedulerArchitecture, Event], bool]
    ):
        self.conditions.append(condition)

    def check_conditions(
        self, state: SystemState, arch: SchedulerArchitecture, event: Event
    ):
        return all([c(state, arch, event) for c in self.conditions])


def check_for_task_completion(
    state: SystemState, arch: SchedulerArchitecture, event: Event, tasks: Set[TaskID]
):
    # print("Checking for task completion", event, tasks)
    if isinstance(event, TaskCompleted):
        if event.task in tasks:
            return False
    return True


def check_for_mapper(state: SystemState, arch: SchedulerArchitecture, event: Event):
    if isinstance(event, Mapper):
        return True
    return False


def check_for_data_read(
    state: SystemState,
    arch: SchedulerArchitecture,
    event: Event,
    data: Set[DataID],
):
    if isinstance(event, Launcher):
        for tasks in event.tasks:
            task = state.objects.taskmap[tasks]
            if isinstance(task, SimulatedDataTask):
                continue
            dset = set([d.id for d in task.read_accesses])

            if dset.intersection(data):
                return False
    return True


def check_for_time(
    state: SystemState, arch: SchedulerArchitecture, event: Event, time: Time
):
    if state.time < time:
        return False
    return True


def check_for_task_launch(
    state: SystemState, arch: SchedulerArchitecture, event: Event, tasks: Set[TaskID]
):
    if isinstance(event, Launcher):
        for task in tasks:
            if task in event.tasks:
                return True
    return False
