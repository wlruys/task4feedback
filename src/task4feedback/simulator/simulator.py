from .task import SimulatedTask, SimulatedDataTask, SimulatedComputeTask
from .data import *
from .device import *
from .resourceset import *
from .queue import *
from .events import *
from .resources import *
from .task import *
from .topology import *

from ..types import DataMap, Architecture, Device, TaskID, TaskState, TaskType, Time
from ..types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap

from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict

from .schedulers import *

# from rich import print

from .analysis.recorder import RecorderList, Recorder
from .randomizer import Randomizer
from .watcher import Watcher
from .mapper import *

from enum import Enum


@dataclass(slots=True)
class SimulatedScheduler:
    topology: SimulatedTopology | None = None
    scheduler_type: str = "parla"
    tasks: List[TaskID] = field(default_factory=list)
    name: str = "SimulatedScheduler"
    mechanisms: SchedulerArchitecture | None = None
    state: SystemState | None = None
    log_level: int = 0
    recorders: RecorderList = field(default_factory=RecorderList)
    randomizer: Randomizer = field(default_factory=Randomizer)
    watcher: Watcher = field(default_factory=Watcher)
    mapper: TaskMapper = field(default_factory=TaskMapper)
    current_event: Event | None = None

    events: EventQueue = EventQueue()
    event_count: int = 0
    init: bool = True
    use_eviction: bool = True

    def __post_init__(
        self,
    ):
        if self.state is None:
            scheduler_state = SchedulerOptions.get_state(self.scheduler_type)
            self.state = scheduler_state(
                topology=self.topology, use_eviction=self.use_eviction
            )
        if self.mechanisms is None:
            scheduler_arch = SchedulerOptions.get_architecture(self.scheduler_type)
            self.mechanisms = scheduler_arch(topology=self.topology)

    def __deepcopy__(self, memo):
        tasks = [t for t in self.tasks]
        start_t = clock()
        state = deepcopy(self.state)
        end_t = clock()
        # print(f"Time to deepcopy state: {end_t - start_t}")

        start_t = clock()
        mechanisms = deepcopy(self.mechanisms)
        end_t = clock()
        # print(f"Time to deepcopy mechanisms: {end_t - start_t}")

        start_t = clock()
        events = deepcopy(self.events)
        end_t = clock()
        # print(f"Time to deepcopy events: {end_t - start_t}")

        return SimulatedScheduler(
            topology=self.topology,
            scheduler_type=self.scheduler_type,
            tasks=tasks,
            name=self.name,
            mechanisms=mechanisms,
            state=state,
            log_level=self.log_level,
            events=events,
            event_count=self.event_count,
            init=self.init,
            randomizer=deepcopy(self.randomizer),
            current_event=deepcopy(self.current_event),
            use_eviction=self.use_eviction,
        )

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

    def set_mapper(self, mapper: TaskMapper):
        self.mapper = mapper

    def set_randomizer(self, randomizer: Randomizer):
        self.randomizer = randomizer

    @property
    def taskmap(self):
        return self.state.objects.taskmap

    @property
    def datamap(self):
        return self.state.objects.datamap

    @property
    def devicemap(self):
        return self.state.objects.devicemap

    def add_stop_condition(
        self, condition: Callable[[SystemState, SchedulerArchitecture, Event], bool]
    ):
        self.watcher.add_condition(condition)

    def add_initial_tasks(self, tasks: List[TaskID], apply_sort: bool = True):
        if apply_sort:
            tasks = self.randomizer.task_order(tasks, self.taskmap)
            # print(f"Initial Task Order: {tasks}")
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
        assert self.mechanisms is not None

        self.current_event = event

        # New events are created from the current event.
        new_event_pairs = self.mechanisms[event](
            self.state, simulator=self, mapper=self.mapper
        )

        # Append new events and their completion times to the event queue
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

        return new_event_pairs

    def run(self) -> bool:
        watcher_status = True

        if self.init:
            new_event_pairs = self.mechanisms.initialize(self.tasks, self.state)
            for completion_time, new_event in new_event_pairs:
                self.events.put(new_event, completion_time)
            self.init = False

        # from rich import print
        from copy import deepcopy
        from time import perf_counter as clock

        next_events = EventIterator(self.events, peek=True)
        for event_pair in next_events:
            if event_pair:
                self.event_count += 1
                completion_time, event = event_pair

                # Advance time
                self.time = max(self.time, completion_time)

                # Process Event
                new_events = self.process_event(event)

                # Update Log
                self.record(event, new_events)

                # Check Watcher Conditions
                watcher_status = self.watcher.check_conditions(
                    self.state, self.mechanisms, event
                )

                if not watcher_status:
                    break

                next_events.success()

        self.recorders.finalize(self.time, self.mechanisms, self.state)

        # print(f"Event Count: {self.event_count}")

        is_complete = self.mechanisms.complete(self.state)
        events_empty = self.events.empty()

        if not is_complete and watcher_status:
            raise RuntimeError("Scheduler terminated without completing all tasks.")

        return is_complete and events_empty
