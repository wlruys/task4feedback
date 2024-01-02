from task4feedback.graphs import *
from task4feedback.load import *

from rich import print

# from utility.execute import run
from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *


def run():
    cpu = Device(Architecture.CPU, 0)
    gpu = Device(Architecture.GPU, -1)

    gpu1 = Device(Architecture.GPU, 1)
    gpu2 = Device(Architecture.GPU, 2)

    task_configs = TaskPlacementInfo()

    runtime_info = TaskRuntimeInfo(task_time=100000, device_fraction=1)
    task_configs.add((gpu), runtime_info)

    data_config = NoDataGraphConfig()

    def custom_tasks(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()

        import random

        time = random.randint(10000, 20000)
        device_tuple = (Device(Architecture.GPU, -1),)
        runtime_info = TaskRuntimeInfo(task_time=time)

        placement_info.add(device_tuple, runtime_info)

        return placement_info

    config = CholeskyConfig(blocks=4, data_config=data_config, task_config=custom_tasks)
    # config = SerialConfig(
    #     steps=2, chains=1, task_config=custom_tasks, data_config=data_config
    # )

    # tasks, data = make_serial_graph(config)
    tasks, data = make_cholesky_graph(config)

    write_tasks_to_yaml(tasks, "graph")
    write_data_to_yaml(data, "graph")

    tasklist, taskmap, datamap = read_graph("graph")

    topology = TopologyManager.get_generator("frontera")(None)

    scheduler = SimulatedScheduler(topology=topology, scheduler_type="parla")
    scheduler.register_taskmap(taskmap)
    scheduler.add_initial_tasks(tasklist)

    start_t = time.perf_counter()
    scheduler.run()
    end_t = time.perf_counter()
    print("Simulated Time: ", scheduler.time)


run()
