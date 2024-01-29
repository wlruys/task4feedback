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

    def custom_tasks(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()

        import random

        time = random.randint(10000, 20000)
        device_tuple = (Device(Architecture.GPU, -1),)
        runtime_info = TaskRuntimeInfo(task_time=time)

        placement_info.add(device_tuple, runtime_info)

        time = random.randint(50000, 70000)
        device_tuple = (Device(Architecture.CPU, -1),)
        runtime_info = TaskRuntimeInfo(task_time=time)
        placement_info.add(device_tuple, runtime_info)

        return placement_info

    data_config = SweepDataGraphConfig()
    config = SweepConfig(width=4, steps=4, dimensions=1, task_config=custom_tasks)
    # config = SerialConfig(chains=2, steps=10, data_config=data_config)
    # config = MapReduceConfig(data_config=data_config)
    # config = ScatterReductionConfig(levels=3, branch_factor=3, data_config=data_config)

    # tasks, data = make_sweep_graph(config)
    tasks, data = make_graph(config, data_config=data_config)

    write_tasks_to_yaml(tasks, "graph")
    write_data_to_yaml(data, "graph")

    tasklist, taskmap, datamap = read_graph("graph")

    populate_dependents(taskmap)

    networkx_graph, networkx_label = build_networkx_graph(taskmap)
    plot_pydot(networkx_graph)


run()
