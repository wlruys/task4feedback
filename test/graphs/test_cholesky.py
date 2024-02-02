from task4feedback.graphs import *
from task4feedback.load import *

from rich import print

# from utility.execute import run
from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *

from task4feedback.simulator.analysis.recorder import *
from task4feedback.simulator.analysis.plot import *
from task4feedback.simulator.interface import *
from task4feedback.simulator.verify import *

from time import perf_counter as clock


def test_data():
    cpu = Device(Architecture.CPU, 0)
    gpu0 = Device(Architecture.GPU, 0)
    gpu1 = Device(Architecture.GPU, 1)

    def initial_data_placement(data_id: DataID) -> Devices:
        return Device(Architecture.CPU, 0)

    def sizes(data_id: DataID) -> int:
        return 100

    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        if task_id.task_idx[0] % 2 == 0:
            device_tuple = (gpu0,)
        else:
            device_tuple = (gpu1,)

        runtime_info = TaskRuntimeInfo(task_time=10000, device_fraction=1, memory=0)
        placement_info = TaskPlacementInfo()
        placement_info.add(device_tuple, runtime_info)

        return placement_info

    data_config = CholeskyDataGraphConfig()
    data_config.initial_placement = initial_data_placement
    data_config.initial_sizes = sizes

    config = CholeskyConfig(blocks=10, task_config=task_placement)
    tasks, data = make_graph(config, data_config=data_config)

    topology = TopologyManager().generate("frontera", config=None)

    simulator_config = SimulatorConfig(
        topology=topology,
        tasks=tasks,
        data=data,
        scheduler_type="parla",
        recorders=[],
    )
    simulator = create_simulator(config=simulator_config)

    start_t = clock()
    simulator.run()
    end_t = clock()

    print(f"Time to Simulate: {end_t - start_t}")
    print(f"Simulated Time: {simulator.time}")

    # print("Tasks: ")
    # print(summarize_dependencies(simulator_config.simulated_tasks))

    # for task in simulator_config.simulated_tasks.values():
    #    print(f"{task.name} {task.times}")

    # verify_order(simulator_config.simulated_tasks)
    # verify_runtime_resources(simulator_config.simulated_tasks, simulator.devicemap)

    # print(simulator.recorders)

    # make_plot(simulator.recorders.recorders[0])


test_data()
