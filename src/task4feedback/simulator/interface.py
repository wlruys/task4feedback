from .simulator import *
from ..types import *
from .data import *
from .device import *
from .preprocess import *

from typing import List, Dict, Set, Tuple, Optional, Callable, Type, Sequence
from .randomizer import Randomizer

from .rl.models.model import *
from .rl.models.env import *


@dataclass(slots=True)
class SimulatorConfig:
    topology: SimulatedTopology
    tasks: TaskMap
    data: DataMap
    task_order_log: List[TaskID] | None = None
    scheduler_type: str = "parla"
    scheduler_state_type: str = "parla"
    name: str = "Simulator"
    recorders: List[Type[Recorder]] = field(default_factory=list)
    simulated_tasks: SimulatedTaskMap = field(init=False)
    simulated_data: SimulatedDataMap = field(init=False)
    use_data: bool = True
    randomizer: Randomizer = field(default_factory=Randomizer)
    watcher: Watcher = field(default_factory=Watcher)
    mapper: TaskMapper = field(default_factory=TaskMapper)
    use_eviction: bool = True
    task_order_mode: TaskOrderType = TaskOrderType.DEFAULT
    use_duration_noise: bool = False
    noise_scale: float = 0
    save_task_order: bool = False
    save_task_noise: bool = False
    load_task_noise: bool = False

    ###########################
    # RL related fields
    ###########################
    rl_env: RLBaseEnvironment = None
    rl_mapper: RLModel = None


def create_simulator(config: SimulatorConfig):
    simulated_data = create_data_objects(config.data, topology=config.topology)
    recorders = RecorderList(recorder_types=config.recorders)

    # print("state config:", config.scheduler_state_type)

    scheduler = SimulatedScheduler(
        topology=config.topology,
        scheduler_type=config.scheduler_type,
        scheduler_state_type=config.scheduler_state_type,
        recorders=recorders,
        randomizer=config.randomizer,
        rl_env=config.rl_env,
        rl_mapper=config.rl_mapper,
        watcher=config.watcher,
        mapper=config.mapper,
        use_eviction=config.use_eviction,
        task_order_mode=config.task_order_mode,
        use_duration_noise=config.use_duration_noise,
        noise_scale=config.noise_scale,
        save_task_order=config.save_task_order,
        save_task_noise=config.save_task_noise,
        load_task_noise=config.load_task_noise
    )

    tasklist, simulated_tasks = create_sim_graph(
        config.tasks, config.data, use_data=config.use_data,
        task_order_mode=config.task_order_mode,
        task_order_log=config.task_order_log
    )

    # for task in simulated_tasks.values():
    #     print(task.name, task.dependencies)

    config.simulated_tasks = simulated_tasks
    config.simulated_data = simulated_data

    scheduler.register_datamap(simulated_data)
    scheduler.register_taskmap(simulated_tasks)

    scheduler.add_initial_tasks(tasklist)

    # print("-----")
    # for task in scheduler.state.objects.taskmap.values():
    #     print(task.name, task.dependencies)

    return scheduler
