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


def random_mapping_policy(task: SimulatedTask, simulator) -> Optional[Devices]:
    scheduler_state: SystemState = simulator.state
    scheduler_arch: SchedulerArchitecture = simulator.mechanisms

    np.random.seed(None)
    potential_devices = task.info.runtime.locations
    index = np.random.randint(0, len(potential_devices))

    potential_device = potential_devices[index]

    if isinstance(potential_device, Tuple):
        potential_device = potential_device[0]

    return (potential_device,)


def heft_mapping_policy(task: SimulatedTask, simulator) -> Optional[Devices]:
    potential_device = Device(Architecture.GPU, task.info.heft_allocation)
    # print("task:", task.name, " heft device:", potential_device.device_id)

    return (potential_device,)


def optimal_mapping_policy(task: SimulatedTask, simulator) -> Optional[Devices]:
    if task.info.z3_allocation == -1:
        raise ValueError(
            "Optimal mapping policy requires a valid allocation. Currently only supported in random graph."
        )
    potential_device = Device(Architecture.GPU, task.info.z3_allocation)
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
        # print(f"EFT of {task.name} on device {devices} is {simulator_copy.time}")

        if simulator_copy.time < min_time:
            min_time = simulator_copy.time
            chosen_device = devices

    # print("----")
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
        # print(f"EFT of {task.name} on device {devices} is {simulator_copy.time}")

        if simulator_copy.time > max_time:
            max_time = simulator_copy.time
            chosen_device = devices

    # print("----")
    # print(simulator_copy.current_event)

    return chosen_device  # default_mapping_policy(task, simulator)


def load_balancing(task: SimulatedTask, simulator) -> Optional[Devices]:
    scheduler_state: SystemState = simulator.state
    scheduler_arch: SchedulerArchitecture = simulator.mechanisms

    lowest_workload = 9999999999999999999
    potential_device = None
    potential_devices = task.info.runtime.locations
    # potential_devices = scheduler_state.topology.devices

    for device in potential_devices:
        if isinstance(device, Tuple):
            device = device[0]

        # if device.name.architecture == Architecture.CPU:
        #     continue
        workload = scheduler_state.perdev_active_workload[device]
        if potential_device is None or workload < lowest_workload:
            lowest_workload = workload
            potential_device = device

    if isinstance(potential_device, Tuple):
        potential_device = potential_device[0]

    # print("load balancing device:", potential_device)
    return (potential_device,)


def eft_without_data(task: SimulatedTask, simulator) -> Optional[Devices]:
    scheduler_state: SystemState = simulator.state
    scheduler_arch: SchedulerArchitecture = simulator.mechanisms
    perdev_earliest_avail_time = scheduler_state.perdev_earliest_avail_time
    taskmap = scheduler_state.objects.taskmap

    est_ready_time: float = 0
    for dependency in task.dependencies:
        est_ready_time = max(taskmap[dependency].est_completion_time, est_ready_time)

    lowest_workload = 9999999999999999999
    potential_device = None
    potential_devices = task.info.runtime.locations
    # potential_devices = scheduler_state.topology.devices
    for device in potential_devices:
        if isinstance(device, Tuple):
            device = device[0]

        # if device.name.architecture == Architecture.CPU:
        #     continue
        workload = max(perdev_earliest_avail_time[device], est_ready_time)
        if potential_device is None or workload < lowest_workload:
            lowest_workload = workload
            potential_device = device

    task.est_completion_time = lowest_workload + float(
        scheduler_state.get_task_duration(
            task, task.info.runtime.locations[0]
        ).scale_to("ms")
    )
    perdev_earliest_avail_time[potential_device] = max(
        task.est_completion_time, perdev_earliest_avail_time[potential_device]
    )

    if logger.ENABLE_LOGGING:
        logger.mapping.debug(
            (
                f"Task {task.name}, start:{lowest_workload}, end:{task.est_completion_time}, device:{potential_device}"
            )
        )

    # print("task ", task.name ," start:", lowest_workload, " complete:", task.est_completion_time, " potential device:", potential_device)
    # print("load balancing device:", potential_device)
    return (potential_device,)


def eft_with_data(task: SimulatedTask, simulator) -> Optional[Devices]:
    scheduler_state: SystemState = simulator.state
    scheduler_arch: SchedulerArchitecture = simulator.mechanisms
    perdev_earliest_avail_time = scheduler_state.perdev_earliest_avail_time
    taskmap = scheduler_state.objects.taskmap
    datamap = scheduler_state.objects.datamap

    est_ready_time: float = 0
    for dependency in task.dependencies:
        est_ready_time = max(taskmap[dependency].est_completion_time, est_ready_time)

    lowest_workload = 9999999999999999999
    potential_device = None
    potential_devices = task.info.runtime.locations
    # potential_devices = scheduler_state.topology.devices
    for device in potential_devices:
        # if device.name.architecture == Architecture.CPU:
        #     continue

        if isinstance(device, Tuple):
            device = device[0]

        workload = max(perdev_earliest_avail_time[device], est_ready_time)
        if task.data_tasks is not None:
            nonlocal_data = 0
            for dtask_id in task.data_tasks:
                dtask = taskmap[dtask_id]
                for data_id in dtask.info.data_dependencies.all_ids():
                    data = datamap[data_id]
                    is_valid = data.is_valid(device, TaskState.MAPPED)
                    nonlocal_data += data.size if not is_valid else 0
            # NOTE This assumes that GPU connection bandwidths are all equal
            # NOTE This also assumes that # of GPUs >= 2
            bandwidth = scheduler_state.topology.connection_pool.bandwidth[1, 2]
            # print(
            #     f"\t task: {task.name} tests device: {device} data size: {nonlocal_data} new workload: {nonlocal_data / bandwidth * 1000 * 1000} bw: {bandwidth}"
            # )
            # Convert to ms
            workload += (nonlocal_data / bandwidth) * 1000

        if potential_device is None or workload < lowest_workload:
            lowest_workload = workload
            potential_device = device

    task.est_completion_time = lowest_workload + float(
        scheduler_state.get_task_duration(
            task, task.info.runtime.locations[0]
        ).scale_to("ms")
    )
    perdev_earliest_avail_time[potential_device] = max(
        task.est_completion_time, perdev_earliest_avail_time[potential_device]
    )

    if logger.ENABLE_LOGGING:
        logger.mapping.debug(
            (
                f"Task {task.name}, start:{lowest_workload}, end:{task.est_completion_time}, device:{potential_device}"
            )
        )

    # print("task ", task.name ," start:", lowest_workload, " complete:", task.est_completion_time, " potential device:", potential_device)
    # print("load balancing device:", potential_device)
    return (potential_device,)


def parla_mapping_policy(task: SimulatedTask, simulator) -> Optional[Devices]:
    scheduler_state: SystemState = simulator.state
    scheduler_arch: SchedulerArchitecture = simulator.mechanisms

    taskmap = scheduler_state.objects.taskmap
    datamap = scheduler_state.objects.datamap

    highest_workload = -1
    potential_device = None
    potential_devices = task.info.runtime.locations
    # potential_devices = scheduler_state.topology.devices

    total_workload = 0
    for device in potential_devices:
        if isinstance(device, Tuple):
            device = device[0]

        # if device.name.architecture == Architecture.CPU:
        #     continue

        total_workload += scheduler_state.perdev_active_workload[device]

    for device in potential_devices:
        if isinstance(device, Tuple):
            device = device[0]

        # if device.name.architecture == Architecture.CPU:
        #     continue

        workload = scheduler_state.perdev_active_workload[device]
        norm_workload = workload / total_workload if total_workload != 0 else workload
        local_data = 0
        nonlocal_data = 0
        total_data = 0
        if task.data_tasks is not None:
            for dtask_id in task.data_tasks:
                dtask = taskmap[dtask_id]
                for data_id in dtask.info.data_dependencies.all_ids():
                    data = datamap[data_id]
                    is_valid = data.is_valid(device, TaskState.MAPPED)
                    local_data += data.size if is_valid else 0
                    nonlocal_data += data.size if not is_valid else 0
                    total_data += data.size
            local_data = local_data / total_data if total_data > 0 else local_data
            nonlocal_data = (
                nonlocal_data / total_data if total_data > 0 else nonlocal_data
            )
        score = 50 + (30 * local_data - 30 * nonlocal_data - 10 * norm_workload)
        if score > highest_workload:
            highest_workload = score
            potential_device = device

    if isinstance(potential_device, Tuple):
        potential_device = potential_device[0]

    # print("potential device:", potential_device)
    return (potential_device,)


@dataclass(slots=True)
class TaskMapper:
    restrict_tasks: bool = False
    allowed_set: Set[TaskID] = field(default_factory=set)
    assignments: Dict[TaskID, Devices] = field(default_factory=dict)
    mapping_function: MappingFunction = random_mapping_policy

    def initialize(
        self, tasks: List[SimulatedTask], scheduler_state: SystemState, mapper_type: str
    ):
        if mapper_type == "random":
            print("RANDOM mapper is enabled")
            self.mapping_function = random_mapping_policy
        elif mapper_type == "parla":
            print("PARLA mapper is enabled")
            self.mapping_function = parla_mapping_policy
        elif mapper_type == "loadbalance":
            print("Load balancing mapper is enabled")
            self.mapping_function = load_balancing
        elif mapper_type == "eft_with_data":
            print("EFT with data is enabled")
            self.mapping_function = eft_with_data
        elif mapper_type == "eft_without_data":
            print("EFT without data is enabled")
            self.mapping_function = eft_without_data
        elif mapper_type == "heft":
            self.mapping_function = heft_mapping_policy
        elif mapper_type == "opt":
            self.mapping_function = optimal_mapping_policy

    def map_task(self, task: SimulatedTask, simulator) -> Optional[Devices]:
        sched_state: SystemState = simulator.state

        if self.check_allowed(task) is False:
            return None

        if sched_state.total_num_mapped_tasks >= sched_state.mapper_num_tasks_threshold:
            return None

        if task.name in self.assignments:
            assigned_device = self.assignments[task.name]
        else:
            assigned_device = self.mapping_function(task, simulator)

        # print(f"Task {task.name} mapped to {assigned_device}.")
        return assigned_device

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


@dataclass(slots=True)
class RLTaskMapper(TaskMapper):
    # ***********************************
    # Our RL model needs to collect task graph information for its state features
    # ***********************************
    # Max out/indegree
    max_outdegree: int = 0
    max_indegree: int = 0
    # Max expected execution time
    max_duration: float = 0
    # Depth from a root task
    max_depth: int = 0
    # # of total tasks
    total_num_tasks: int = 0
    # degree information
    outdegree: Dict[TaskID, int] = field(default_factory=dict)
    indegree: Dict[TaskID, int] = field(default_factory=dict)

    def initialize(
        self,
        tasks: List[SimulatedTask],
        scheduler_state: SystemState,
        scheduler_state_type: str,
    ):
        self.collect_task_graph_info(tasks, scheduler_state)

    def map_task(self, task: SimulatedTask, simulator) -> Optional[Devices]:
        sched_state: SystemState = simulator.state

        if self.check_allowed(task) is False:
            return None

        if sched_state.total_num_mapped_tasks >= sched_state.mapper_num_tasks_threshold:
            return None

        curr_state = sched_state.rl_env.create_state(task, simulator, self)
        assigned_devices = (
            Device(
                Architecture.GPU,
                sched_state.rl_mapper.select_device(task, curr_state, sched_state),
            ),
        )

        # print(f"Task {task.name} mapped to {assigned_devices}.")
        return assigned_devices

    def __deepcopy__(self, memo):
        return TaskMapper(
            mapping_function=self.mapping_function,
        )

    def collect_task_graph_info(
        self, tasks: List[SimulatedTask], scheduler_state: SystemState
    ):
        """
        Collect a task graph information before simulation starts.
        """
        taskmap = scheduler_state.objects.taskmap
        for task in tasks:
            num_dependents = 0
            num_dependencies = 0
            for dependent in task.dependents:
                if isinstance(taskmap[dependent], SimulatedComputeTask):
                    num_dependents += 1
            for dependency in task.dependencies:
                if isinstance(taskmap[dependency], SimulatedComputeTask):
                    num_dependencies += 1
            self.outdegree[task.name] = num_dependents
            self.indegree[task.name] = num_dependencies
            self.max_outdegree = max(self.max_outdegree, num_dependents)
            self.max_indegree = max(self.max_indegree, num_dependencies)
            task_duration_float = float(
                scheduler_state.get_task_duration(
                    task, task.info.runtime.locations[0]
                ).scale_to("us")
            )
            self.max_duration = max(self.max_duration, task_duration_float)

            # Propagate depth to its successors
            for dep in task.dependencies:
                task.info.depth = max(task.info.depth, taskmap[dep].info.depth + 1)

            if task.info.depth == -1:
                # If its depth is not initialized
                task.info.depth = 0

            self.max_depth = max(self.max_depth, task.info.depth)
        self.total_num_tasks = len(tasks)

        print(
            f"max degree: {self.max_outdegree}, "
            f"in-degree: {self.max_indegree}"
            f" total tasks: {self.total_num_tasks}, "
            f"max depth: {self.max_depth} \n"
            f"max execution time: {self.max_duration}"
        )
        if logger.ENABLE_LOGGING:
            logger.runtime.info(
                f"Total tasks: {self.total_num_tasks}\n"
                f"Max out-degree: {self.max_outdegree} "
                f"and in-degree: {self.max_indegree}."
            )

    def get_indegree(self, task: SimulatedTask):
        return self.indegree[task.name]

    def get_outdegree(self, task: SimulatedTask):
        return self.outdegree[task.name]
