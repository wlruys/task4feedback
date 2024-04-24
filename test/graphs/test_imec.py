# Blocked GEMM + cannon gemm
from task4feedback.graphs import *
from task4feedback.load import *

import math
from rich import print

# from utility.execute import run
from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *

from task4feedback.simulator.analysis.recorder import *
from task4feedback.simulator.analysis.plot import *
from task4feedback.simulator.interface import *
from time import perf_counter as clock


def test_data():
    cpu = Device(Architecture.CPU, 0)
    n_gpus = 4
    levels = 3
    blocks = 4
    B = 2
    gpus = [Device(Architecture.GPU, i) for i in range(n_gpus)]
    # gpu0 = Device(Architecture.GPU, 0)
    # gpu1 = Device(Architecture.GPU, 1)
    # gpu2 = Device(Architecture.GPU, 2)
    # gpu3 = Device(Architecture.GPU, 3)

    def initial_data_placement(data_id: DataID) -> Devices:
        # return Device(Architecture.CPU, 0)
        step = math.sqrt(n_gpus)
        if(data_id.idx[0][0] == 0):
            return Device(Architecture.GPU, data_id.idx[0][1])
        return Device(Architecture.GPU, int(data_id.idx[0][1] // step))

    def sizes(data_id: DataID) -> int:
        return 256 * 16

    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        device_tuple = (gpus[task_id.task_idx[4] % n_gpus],)
        # if task_id.task_idx[1] % 2 == 0:
        #     device_tuple = (gpu0,)
        # else:
        #     device_tuple = (gpu1,)

        runtime_info = TaskRuntimeInfo(task_time=10000, device_fraction=1, memory=0)
        placement_info = TaskPlacementInfo()
        placement_info.add(device_tuple, runtime_info)

        return placement_info

    # data_config = ImecHierDataGraphConfig(levels=levels, blocks=blocks, n_devices=n_gpus)
    # data_config.initial_placement = initial_data_placement
    # data_config.initial_sizes = sizes
    data_config = NoDataGraphConfig()
    config = ImecConfig(levels=levels, blocks=blocks, B=B, task_config=task_placement)
    tasks, data = make_graph(config, data_config=data_config)

    topology = TopologyManager().generate("imec", config=None)
    # write_tasks_to_yaml(tasks, "graph")
    # write_data_to_yaml(data, "graph")

    # tasklist, taskmap, datamap = read_sim_graph("graph")
    # populate_dependents(taskmap)
    # tasklist, taskmap = make_graph(tasks, data, use_data=True)

    # networkx_graph, networkx_label = build_networkx_graph(taskmap)
    # plot_pydot(networkx_graph)

    simulator_config = SimulatorConfig(
        topology=topology,
        tasks=tasks,
        data=data,
        scheduler_type="parla",
        recorders=[DataValidRecorder, ComputeTaskRecorder, DataTaskRecorder],
    )
    simulator = create_simulator(config=simulator_config)

    start_t = clock()
    simulator.run()
    end_t = clock()

    print(f"Time to Simulate: {end_t - start_t}")
    print(f"Simulated Time: {simulator.time}")


    # data_ids = []
    # for i in range(blocks):
    #     data_ids.append(DataID(((0, i),)))

    # for i in range(2 * blocks):
    #     data_ids.append(DataID(((levels, i),)))

    # intervals = simulator.recorders.recorders[0].intervals
    # data_task_recorder = simulator.recorders.get(DataTaskRecorder)
    # total_communication_energy = 0
    # for task in data_task_recorder.tasks.values():
    #     total_communication_energy += task.communication_energy
    # print(f"Communication Energy: {total_communication_energy} pJ")
    # make_data_plot(
    #     simulator.recorders,
    #     True,
    #     True,
    #     data_ids=data_ids,
    # )

test_data()