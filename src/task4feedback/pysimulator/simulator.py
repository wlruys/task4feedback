from .task import SimulatedTask, SimulatedDataTask, SimulatedComputeTask
from .data import *
from .device import *
from .resourceset import *
from .queue import *
from .events import *
from .resources import *
from .task import *
from .topology import *

from .rl.models.model import *
from .rl.models.env import *

from ..legacy_types import (
    DataMap,
    Architecture,
    Device,
    TaskID,
    TaskState,
    TaskType,
    Time,
)
from ..legacy_types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap, ExecutionMode
from ..legacy_types import TaskOrderType

from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict as DefaultDict

from .schedulers.parla.state import RLState

from .schedulers import *

# from rich import print

from .analysis.recorder import RecorderList, Recorder
from .randomizer import Randomizer
from .watcher import Watcher
from .mapper import *

from enum import Enum


def get_scheduler_state(mapper_type: str):
    """
    This function returns a scheduler state type based on the specified
    mapper type.
    Different mappers may need different information due to their policies.
    This information is usually tracked by the state.
    NOTE that if you add new policy, you also should add the corresponding
    state on here.
    """
    if (
        mapper_type == "random"
        or mapper_type == "parla"
        or mapper_type == "loadbalance"
        or mapper_type == "heft"
        or mapper_type == "eft_without_data"
        or mapper_type == "eft_with_data"
        or mapper_type == "opt"
    ):
        return "parla"
    elif mapper_type == "rl":
        return "rl"
    else:
        print(f"Unsupported mapper type: {mapper_type}")
        return None


@dataclass(slots=True)
class SimulatedScheduler:
    topology: SimulatedTopology | None = None
    scheduler_type: str = "parla"
    mapper_type: str = "parla"
    # Consider data's initial placement for HEFT
    consider_initial_placement: bool = True
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
    task_order_mode: TaskOrderType = TaskOrderType.DEFAULT
    use_duration_noise: bool = False
    noise_scale: float = 0
    save_task_order: bool = False
    load_task_order: bool = False
    save_task_noise: bool = False
    load_task_noise: bool = False

    ###########################
    # RL related fields
    ###########################

    rl_env: RLBaseEnvironment = None
    rl_mapper: RLModel = None

    def __post_init__(self):
        if self.state is None:
            scheduler_state_type = get_scheduler_state(self.mapper_type)
            scheduler_state = SchedulerOptions.get_state(scheduler_state_type)
            self.state = scheduler_state(
                topology=self.topology,
                rl_env=self.rl_env,
                rl_mapper=self.rl_mapper,
                task_order_mode=self.task_order_mode,
                use_duration_noise=self.use_duration_noise,
                noise_scale=self.noise_scale,
                save_task_order=self.save_task_order,
                load_task_order=self.load_task_order,
                save_task_noise=self.save_task_noise,
                load_task_noise=self.load_task_noise,
                randomizer=self.randomizer,
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

        start_t = clock()
        mapper = deepcopy(self.mapper)
        end_t = clock()

        start_t = clock()
        rl_env = deepcopy(self.rl_env)
        end_t = clock()

        start_t = clock()
        rl_mapper = deepcopy(self.rl_mapper)
        end_t

        return SimulatedScheduler(
            topology=self.topology,
            mapper_type=self.mapper_type,
            consider_initial_placement=self.consider_initial_placement,
            scheduler_type=self.scheduler_type,
            tasks=tasks,
            name=self.name,
            mechanisms=mechanisms,
            state=state,
            log_level=self.log_level,
            events=events,
            event_count=self.event_count,
            mapper=mapper,
            init=self.init,
            randomizer=deepcopy(self.randomizer),
            use_duration_noise=self.use_duration_noise,
            save_task_order=self.save_task_order,
            load_task_order=self.load_task_order,
            save_task_noise=self.save_task_noise,
            load_task_noise=self.load_task_noise,
            noise_scale=self.noise_scale,
            current_event=deepcopy(self.current_event),
            use_eviction=self.use_eviction,
            task_order_mode=self.task_order_mode,
            rl_env=rl_env,
            rl_mapper=rl_mapper,
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

    def add_initial_tasks(self, tasks: List[TaskID]):
        # if apply_sort:
        #     tasks = self.randomizer.task_order(tasks, self.taskmap)
        self.tasks.extend(tasks)

    # def add_initial_tasks(self, tasks: List[TaskID]):
    #     self.tasks.extend(tasks)

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
            new_event_pairs = self.mechanisms.initialize(
                tasks=self.tasks,
                scheduler_state=self.state,
                simulator=self,
                mapper_type=self.mapper_type,
                consider_initial_placement=self.consider_initial_placement,
            )
            for completion_time, new_event in new_event_pairs:
                self.events.put(new_event, completion_time)
            self.init = False

        from rich import print
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

                # print("Event: ", event)
                # print(self.mechanisms)

                # Check Watcher Conditions
                watcher_status = self.watcher.check_conditions(
                    self.state, self.mechanisms, event
                )

                if not watcher_status:
                    break

                next_events.success()

        self.recorders.finalize(self.time, self.mechanisms, self.state)

        for device in self.topology.devices:
            last_active = max(
                device.stats.last_active_compute, device.stats.last_active_movement
            )

            device.stats.idle_time += self.time - last_active

            device.stats.idle_time_compute += (
                self.time - device.stats.last_active_compute
            )
            device.stats.idle_time_movement += (
                self.time - device.stats.last_active_movement
            )

            print(f"{device.name},idle,{device.stats.idle_time}")

        print(f"{self.mapper_type},simtime,{float(self.time.scale_to('s'))}")
        print(
            f"{self.mapper_type},wait_time,{float(self.state.wait_time_accum.scale_to('s')) / self.state.num_tasks}"
        )

        # print(f"Event Count: {self.event_count}")

        is_complete = self.mechanisms.complete(self.state)
        events_empty = self.events.empty()

        if not is_complete and watcher_status:
            print("Event Queue", self.events)
            print("Arch", self.mechanisms)
            raise RuntimeError("Scheduler terminated without completing all tasks.")

        return self.time, self.tasks, (is_complete and events_empty)
