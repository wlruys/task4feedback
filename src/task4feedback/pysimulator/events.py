from ..legacy_types import TaskID, TaskInfo, TaskState, Optional, Time, Device, Devices
from dataclasses import dataclass, field
from .resourceset import FasterResourceSet
from typing import Dict, Set


@dataclass(slots=True)
class Event:
    func: str
    time: Optional[Time] = None
    verbose: bool = False

    def __eq__(self, other):
        return self.func == other.func and self.time == other.time

    def __lt__(self, other):
        return self.time < other.time

    def __hash__(self):
        return hash((self.func, self.time))


@dataclass(slots=True)
class PhaseEvent(Event):
    max_tasks: int | None = None
    tasks: set[TaskID] = field(default_factory=set)


@dataclass(slots=True)
class TaskEvent(Event):
    task: TaskID = field(default_factory=TaskID)


@dataclass(slots=True)
class Eviction(Event):
    func: str = "eviction"
    parent_task: TaskID = field(default_factory=TaskID)
    requested_resources: Dict[Device, FasterResourceSet] = field(default_factory=list)


@dataclass(slots=True)
class Mapper(PhaseEvent):
    func: str = "mapper"


@dataclass(slots=True)
class Reserver(PhaseEvent):
    func: str = "reserver"


@dataclass(slots=True)
class Launcher(PhaseEvent):
    func: str = "launcher"


@dataclass(slots=True)
class TaskCompleted(TaskEvent):
    func: str = "complete_task"
