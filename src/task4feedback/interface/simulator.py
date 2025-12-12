from dataclasses import dataclass
from typing import Optional, Type, Self, TYPE_CHECKING
import torch
import task4feedback.trip as trip
from task4feedback.trip import (
    TaskNoise,
    LognormalTaskNoise,
    SchedulerInput,
    ExecutionState,
)
from .graph import TaskGraph, DataBlocks
from .system import System
from .observer import ExternalObserverFactory, ExternalObserver

if TYPE_CHECKING:
    from .observer import ExternalObserver

class ExternalMapper:
    def __init__(self, mapper: Optional[Self] = None):
        pass

    def map_tasks(self, simulator: "SimulatorDriver") -> list[trip.Action]:
        raise NotImplementedError("ExternalMapper.map_tasks must be implemented by subclass")


class StaticExternalMapper:
    def __init__(self, mapper: Optional[Self] = None, mapping_dict: Optional[dict] = None):
        if mapper is not None:
            self.mapping_dict = mapper.mapping_dict

        elif mapping_dict is not None:
            self.mapping_dict = mapping_dict
        else:
            self.mapping_dict = {}

    def set_mapping_dict(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def map_tasks(self, simulator: "SimulatorDriver") -> list[trip.Action]:
        candidates = torch.zeros((1), dtype=torch.int64)
        count = simulator.simulator.get_mappable_candidates(candidates)
        
        if count == 0:
            return []

        global_task_id = candidates[0].item()
        local_id = 0
        
        if global_task_id not in self.mapping_dict:
            # Fallback or error?
            # For now, let's assume device 0 if not found to avoid crash, or raise error
            raise KeyError(f"Task {global_task_id} not found in static mapping dict")
            
        device = self.mapping_dict[global_task_id]
        state = simulator.simulator.get_state()
        mapping_priority = state.get_mapping_priority(global_task_id)
        return [trip.Action(local_id, device, mapping_priority, mapping_priority)]


@dataclass
class NoiseConfig:
    task_noise: TaskNoise

    def __init__(
        self,
        graph: TaskGraph,
        duration_seed: int = 0,
        priority_seed: int = 0,
    ):
        self.task_noise = TaskNoise(graph.static_graph, duration_seed, priority_seed)


@dataclass
class LognormalNoiseConfig(NoiseConfig):

    def __init__(
        self,
        graph: TaskGraph,
        duration_seed: int = 0,
        priority_seed: int = 0,
    ):
        super().__init__(graph, duration_seed, priority_seed)
        self.task_noise = LognormalTaskNoise(graph.static_graph, duration_seed, priority_seed)


@dataclass
class SimulatorInput:
    graph: TaskGraph
    data: DataBlocks
    system: System
    task_noise: TaskNoise
    transition_conditions: trip.TransitionConditions
    top_k_candidates: int = 1

    def __init__(
        self,
        graph: TaskGraph,
        data: DataBlocks,
        system: System,
        task_noise: Optional[TaskNoise] = None,
        transition_conditions: Optional[trip.TransitionConditions] = None,
        top_k_candidates: int = 1,
    ):
        if transition_conditions is None:
            transition_conditions = trip.RangeTransitionConditions(5, 5, 16)

        if task_noise is None:
            task_noise = TaskNoise(graph.static_graph)

        self.task_noise = task_noise
        self.graph = graph
        self.data = data
        self.system = system
        self.transition_conditions = transition_conditions
        self.top_k_candidates = top_k_candidates

    def to_input(self):
        return SchedulerInput(
            self.graph.graph,
            self.graph.static_graph,
            self.data.data,
            self.system.devices,
            self.system.topology,
            self.task_noise,
            self.transition_conditions,
            self.top_k_candidates,
        )

@dataclass
class SimulatorDriver:
    input: SimulatorInput
    internal_mapper: trip.Mapper
    external_mapper: ExternalMapper
    simulator: trip.Simulator
    observer_factory: Optional[ExternalObserverFactory]
    observer: Optional[ExternalObserver]
    use_external_mapper: bool = False

    def __init__(
        self,
        input: SimulatorInput,
        internal_mapper: trip.Mapper | Type[trip.Mapper] = trip.DequeueEFTMapper,
        external_mapper: ExternalMapper | Type[ExternalMapper] = ExternalMapper,
        observer_factory: Optional[ExternalObserverFactory] = None,
        simulator: Optional[trip.Simulator] = None,
    ):
        self.input = input
        if isinstance(internal_mapper, type):
            internal_mapper = internal_mapper()

        if isinstance(external_mapper, type):
            external_mapper = external_mapper()

        self.internal_mapper = internal_mapper
        self.external_mapper = external_mapper

        if simulator is None:
            self.simulator = trip.Simulator(input.to_input(), self.internal_mapper)
        else:
            self.simulator = simulator
            self.simulator.set_mapper(self.internal_mapper)

        if observer_factory is not None:
            self.observer_factory = observer_factory
            self.observer = observer_factory.create(self)

    def get_state(self):
        return self.simulator.get_state()

    @property
    def state(self):
        return self.simulator.get_state()

    @property
    def processed_events(self):
        return self.simulator.processed_events

    @property
    def status(self):
        return self.simulator.last_execution_state

    def get_mappable_candidates(self, candidates: torch.Tensor):
        """
        Get the mappable candidates from the simulator.
        """
        return self.simulator.get_mappable_candidates(candidates)

    def get_mapping_priority(self, task_id: int):
        """
        Get the mapping priority for a task.
        """
        return self.state.get_mapping_priority(task_id)

    def initialize(self):
        """
        Initialize the simulator (creates workspaces for current tasks, state, etc).
        The GRAPH input SHOULD NOT be modified after this is called.
        The NOISE input SHOULD NOT be modified after this is called.
        THE SYSTEM input SHOULD NOT be modified after this is called.
        """
        self.simulator.initialize()

    def initialize_data(self):
        """
        Initialize the simulator data manager.
        This finalizes the starting locations of all data blocks and their initial memory usage.
        The DATA input SHOULD NOT be modified after this is called.
        """
        self.simulator.initialize_data()

    @property
    def mapper(self):
        if self.use_external_mapper:
            return self.external_mapper
        return self.internal_mapper

    def enable_external_mapper(self, external_mapper: Optional[ExternalMapper | Type[ExternalMapper]] = None):
        """
        Use external mapper for mapping tasks (run Python callback).
        """
        if external_mapper is not None:
            if isinstance(external_mapper, type):
                external_mapper = external_mapper()

            self.external_mapper = external_mapper

        self.use_external_mapper = True
        self.simulator.enable_python_mapper()

    def disable_external_mapper(self):
        """
        Use internal mapper for mapping tasks (do not run Python callback).
        """
        self.use_external_mapper = False
        self.simulator.disable_python_mapper()

        if self.simulator.last_execution_state == ExecutionState.EXTERNAL_MAPPING:
            self.simulator.skip_external_mapping()

    def fresh_copy(self) -> "SimulatorDriver":
        """
        Initialize a fresh (uninitialized) copy of the simulator driver with the same initial input and configuration.
        """
        internal_mapper_t = type(self.internal_mapper)
        external_mapper_t = type(self.external_mapper)

        internal_mapper_copy = internal_mapper_t()

        try:
            external_mapper_copy = external_mapper_t()
        except ValueError:
            external_mapper_copy = external_mapper_t(geometry=self.external_mapper.geometry)

        observer_factory = self.observer_factory

        return SimulatorDriver(
            input=self.input,
            internal_mapper=internal_mapper_copy,
            external_mapper=external_mapper_copy,
            observer_factory=observer_factory,
            simulator=None,
        )

    def set_steps(self, steps: int):
        """
        Set the number of mapping steps to run the simulator.
        Will return in a breakpoint state.
        """
        self.simulator.set_steps(steps)

    def start_drain(self):
        self.simulator.start_drain()

    def stop_drain(self):
        self.simulator.stop_drain()

    def reset(self):
        """
        Return a fresh copy of the simulator driver with the same initial input and configuration.
        (This is equivalent to calling fresh_copy()).
        """
        return self.fresh_copy()

    @property
    def time(self) -> int:
        """
        Returns the current time (in microseconds) of the simulator state.
        """
        return self.simulator.get_current_time()

    @property
    def max_mem_usage(self) -> int:
        """
        Returns the maximum memory usage (in bytes) of the simulator state.
        """
        return self.simulator.get_max_memory_usage()

    def total_data_movement(self):
        return self.simulator.get_total_data_movement()

    def total_eviction_movement(self):
        return self.simulator.get_eviction_data_movement()

    def set_task_breakpoint(self, event: trip.EventType, task_id: int) -> int:
        self.simulator.add_task_breakpoint(event, task_id)

    def clear_breakpoints(self):
        self.simulator.clear_breakpoints()

    def copy(self) -> "SimulatorDriver":
        """
        Initialize a copy of the simulator driver at the current state (may be initialized if the source simulator is).
        Mappers and their internal state (if any) are copied as well.
        """
        internal_mapper_t = type(self.internal_mapper)
        external_mapper_t = type(self.external_mapper)

        internal_mapper_copy = internal_mapper_t(self.internal_mapper)
        external_mapper_copy = external_mapper_t(self.external_mapper)

        observer_factory = self.observer_factory

        simulator_copy = trip.Simulator(self.simulator)

        new_sim_driver = SimulatorDriver(
            input=self.input,
            internal_mapper=internal_mapper_copy,
            external_mapper=external_mapper_copy,
            observer_factory=observer_factory,
            simulator=simulator_copy,
        )
        return new_sim_driver

    def run_until_external_mapping(self) -> ExecutionState:
        """
        Run the simulator until a breakpoint, error, completion, or external mapping is reached.
        Will return the current state of the simulator at the exitpoint.
        """
        sim_state = self.simulator.run()
        return sim_state

    def run(self) -> ExecutionState:
        """
        Run the simulator until a breakpoint, error, or completion is reached.
        This DOES NOT STOP for external mapping. Use run_until_external_mapping() for that.
        External mapping will be called, if enabled, inside this function.
        Will return the current state of the simulator at the exitpoint.
        """
        sim_state = ExecutionState.RUNNING
        while sim_state == ExecutionState.RUNNING:
            sim_state = self.simulator.run()

            if sim_state == ExecutionState.BREAKPOINT:
                return sim_state

            if sim_state == ExecutionState.ERROR:
                return sim_state

            if sim_state == ExecutionState.EXTERNAL_MAPPING:
                actions = self.external_mapper.map_tasks(self)
                self.simulator.map_tasks(actions)
                sim_state = ExecutionState.RUNNING
        return sim_state


def create_graph_spec(
    max_tasks: int = 100,
    max_data: int = 100,
    max_devices: int = 5,
    max_edges_tasks_tasks: int = 200,
    max_edges_tasks_data: int = 200,
    max_edges_data_devices: int = 200,
    max_edges_tasks_devices: int = 200,
    max_candidates: int = 1,
):
    """
    Create a graph spec with the specified limits for tasks, data, devices, edges, and candidates.
    """
    spec = trip.GraphSpec()
    spec.max_tasks = max_tasks
    spec.max_data = max_data
    spec.max_devices = max_devices
    spec.max_edges_tasks_tasks = max_edges_tasks_tasks
    spec.max_edges_tasks_data = max_edges_tasks_data
    spec.max_edges_data_devices = max_edges_data_devices
    spec.max_edges_tasks_devices = max_edges_tasks_devices

    spec.max_candidates = max_candidates
    return spec


class SimulatorFactory:
    def __init__(
        self,
        input: SimulatorInput,
        graph_spec: trip.GraphSpec,
        observer_factory: ExternalObserverFactory | Type[ExternalObserverFactory],
        internal_mapper: trip.Mapper | Type[trip.Mapper] = trip.DequeueEFTMapper,
        external_mapper: ExternalMapper | Type[ExternalMapper] = ExternalMapper,
        seed: int = 0,
        priority_seed: int = 0,
        comm_seed: int = 0,
    ):
        self.input = input
        self.graph_spec = graph_spec
        self.internal_mapper = internal_mapper
        self.external_mapper = external_mapper

        self.seed = seed
        self.pseed = priority_seed
        self.cseed = comm_seed

        if isinstance(observer_factory, type):
            observer_factory = observer_factory(graph_spec)
        self.observer_factory = observer_factory

    def create(
        self,
        duration_seed: Optional[int] = None,
        priority_seed: Optional[int] = None,
        comm_seed: Optional[int] = None,
        use_external_mapper: bool = True,
    ) -> SimulatorDriver:
        if duration_seed is None:
            duration_seed = self.seed

        if priority_seed is None:
            priority_seed = self.pseed

        if comm_seed is None:
            comm_seed = self.cseed

        self.input.task_noise.set_seed(duration_seed)
        self.input.task_noise.set_pseed(priority_seed)

        simulator = SimulatorDriver(
            self.input,
            observer_factory=self.observer_factory,
            internal_mapper=self.internal_mapper,
            external_mapper=self.external_mapper,
        )
        self.input.task_noise.randomize_duration(self.input.graph.static_graph)
        self.input.task_noise.randomize_priority(self.input.graph.static_graph)

        simulator.initialize()
        simulator.initialize_data()

        if use_external_mapper:
            simulator.enable_external_mapper()
        else:
            simulator.disable_external_mapper()

        return simulator

    def set_seed(
        self,
        seed: Optional[int] = None,
        priority_seed: Optional[int] = None,
    ):
        """
        Set the seed for the simulator.
        """
        if seed is not None:
            self.seed = seed
        if priority_seed is not None:
            self.pseed = priority_seed
