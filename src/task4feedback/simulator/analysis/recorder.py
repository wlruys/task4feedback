from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Optional, Sequence, Set, Tuple, Self

from ..events import Event
from ..data import SimulatedData
from ..device import SimulatedDevice
from ..schedulers import SchedulerArchitecture, SystemState
from ..task import SimulatedTask
from ...types import TaskID, TaskState, Time, Device, Architecture, TaskType

from copy import copy, deepcopy


@dataclass(slots=True)
class Recorder:
    def __getitem__(
        self, time: Time
    ) -> Tuple[List[SchedulerArchitecture], List[SystemState]]:
        raise NotImplementedError()

    def save(
        self, time: Time, arch_state: SchedulerArchitecture, system_state: SystemState
    ):
        raise NotImplementedError()


@dataclass(slots=True)
class Snapshots(Recorder):
    arch_states: Dict[Time, List[SchedulerArchitecture]] = field(default_factory=dict)
    system_states: Dict[Time, List[SystemState]] = field(default_factory=dict)

    def __getitem__(
        self, time: Time
    ) -> Tuple[List[SchedulerArchitecture], List[SystemState]]:
        return self.arch_states[time], self.system_states[time]

    def save(
        self, time: Time, arch_state: SchedulerArchitecture, system_state: SystemState
    ):
        if time not in self.arch_states:
            self.arch_states[time] = []
        if time not in self.system_states:
            self.system_states[time] = []

        # self.arch_states[time].append((arch_state))
        # self.system_states[time].append((system_state))
