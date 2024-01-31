from task4feedback.graphs import *
from task4feedback.load import *

from rich import print

# from utility.execute import run
from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *

from time import perf_counter as clock


def test_serial():
    cpu = Device(Architecture.CPU, 0)
    gpu0 = Device(Architecture.GPU, 0)
    gpu1 = Device(Architecture.GPU, 1)

    def initial_data_placement(data_id: DataID) -> Devices:
        return Device(Architecture.CPU, 0)

    def sizes(data_id: DataID) -> int:
        return 1

    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        if task_id.task_idx[0] % 2 == 0:
            device_tuple = (gpu0,)
        else:
            device_tuple = (gpu1,)

        runtime_info = TaskRuntimeInfo(task_time=10000, device_fraction=1, memory=0)
        placement_info = TaskPlacementInfo()
        placement_info.add(device_tuple, runtime_info)

        return placement_info

    data_config = ChainDataGraphConfig()
    data_config.initial_placement = initial_data_placement
    data_config.initial_sizes = sizes

    config = ChainConfig(steps=10, chains=1, task_config=task_placement)

    tasks, data = make_graph(config, data_config=data_config)

    tasklist, task_map = create_sim_graph(tasks, data, use_data=True)

    # print(task_map)

    topology = TopologyManager().get_generator("frontera")(None)
    data_map = create_data_objects(data, topology=topology)
    # print(data_map)

    scheduler = SimulatedScheduler(topology=topology, scheduler_type="parla")
    scheduler.register_taskmap(task_map)
    scheduler.register_datamap(data_map)

    topological_sort(tasklist, task_map)
    print(tasklist)

    scheduler.add_initial_tasks(tasklist)

    # print(task_map)

    start_t = clock()
    scheduler.run()
    end_t = clock()
    print(f"Time: {end_t - start_t}")

    print(scheduler.time)


test_serial()
