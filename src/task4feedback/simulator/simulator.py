from .task import SimulatedTask, SimulatedDataTask, SimulatedComputeTask
from .data import *
from .device import *
from .queue import *
from .events import *
from .resources import *
from .task import *
from .topology import *

from ..types import Architecture, Device, TaskID, TaskState, TaskType, Time
from ..types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap

from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict

from .schedulers import *

from rich import print

from .analysis.recorder import RecorderList


@dataclass(slots=True)
class SimulatedScheduler:
    topology: InitVar[SimulatedTopology]
    scheduler_type: InitVar[str] = "parla"
    tasks: List[TaskID] = field(default_factory=list)
    name: str = "SimulatedScheduler"
    mechanisms: SchedulerArchitecture = field(init=False)
    state: SystemState = field(init=False)
    log_level: int = 0
    recorders: RecorderList = field(default_factory=RecorderList)

    events: EventQueue = EventQueue()

    def __post_init__(self, topology: SimulatedTopology, scheduler_type: str = "parla"):
        scheduler_arch = SchedulerOptions.get_architecture(scheduler_type)
        scheduler_state = SchedulerOptions.get_state(scheduler_type)

        print(f"Scheduler Architecture: {scheduler_arch}")
        print(f"Scheduler State: {scheduler_state}")

        self.state = scheduler_state(topology=topology)
        self.mechanisms = scheduler_arch(topology=topology)

    def __str__(self):
        return f"Scheduler {self.name} | Current Time: {self.time}"

    @property
    def time(self):
        return self.state.time

    @time.setter
    def time(self, time):
        self.state.time = time

    def register_taskmap(self, taskmap: SimulatedTaskMap):
        self.state.register_tasks(taskmap)

    def register_datamap(self, datamap: SimulatedDataMap):
        self.state.register_data(datamap)

    def add_initial_tasks(self, tasks: List[TaskID]):
        self.tasks.extend(tasks)

    def __repr__(self):
        return self.__str__()

    def __rich_repr__(self):
        yield "name", self.name
        yield "time", self.time
        yield "architecture", self.mechanisms
        yield "events", self.events

    def record(self, event: Event, new_events: Sequence[EventPair]):
        self.recorders.save(self.time, self.mechanisms, self.state, event, new_events)

    def process_event(self, event: Event) -> List[EventPair]:
        # New events are created from the current event.
        new_event_pairs = self.mechanisms[event](self.state)

        # Append new events and their completion times to the event queue
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

        return new_event_pairs

    def run(self):
        new_event_pairs = self.mechanisms.initialize(self.tasks, self.state)
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

        from rich import print

        event_count = 0

        next_events = EventIterator(self.events, peek=False)
        for event_pair in next_events:
            if event_pair:
                event_count += 1
                completion_time, event = event_pair

                # Advance time
                self.time = max(self.time, completion_time)

                # Process Event
                new_events = self.process_event(event)

                # Update Log
                self.record(event, new_events)

        self.state.finalize_stats()

        print(f"Event Count: {event_count}")
        if not self.mechanisms.complete(self.state):
            raise RuntimeError("Scheduler terminated without completing all tasks.")
