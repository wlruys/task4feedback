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

    def custom_placement(data_id: DataID) -> Devices:
        return (
            Device(Architecture.GPU, data_id.idx[0] % 4),
            Device(Architecture.CPU, 0),
        )

    data_config = CholeskyDataGraphConfig()
    # data_config.initial_placement = partial(
    #     round_robin_data_initial_placement_gpu, n_devices=4, index=0
    # )
    data_config.initial_placement = custom_placement

    def custom_tasks(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()

        import random

        time = random.randint(10000, 20000)
        device_tuple = (Device(Architecture.GPU, -1),)
        runtime_info = TaskRuntimeInfo(task_time=time)

        placement_info.add(device_tuple, runtime_info)

        return placement_info

    config = CholeskyConfig(blocks=4, task_config=custom_tasks)
    tasks, data = make_graph(config, data_config=data_config)

    data_objects = create_data_objects(data)

    for data in data_objects.values():
        print(data)


run()
