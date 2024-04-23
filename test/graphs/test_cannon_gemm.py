from task4feedback.graphs import *
from task4feedback.load import *

from rich import print
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# from utility.execute import run
from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *

from task4feedback.simulator.analysis.recorder import *
from task4feedback.simulator.analysis.plot import *
from task4feedback.simulator.interface import *
from time import perf_counter as clock


def test_data(n_gpus, blocks, n, data_size=None, m=None, energy=0.01):
    cpu = Device(Architecture.CPU, 0, energy)
    levels = int(math.sqrt(n_gpus) + 1)
    a = 8
    gpus = [Device(Architecture.GPU, i, energy) for i in range(n_gpus)]
    print(gpus)
    data_size = data_size if data_size is not None else (a * n * n) / n_gpus
    def initial_data_placement(data_id: DataID) -> Devices:
        if(data_id.idx[0][0] == 0):
            return Device(Architecture.GPU, data_id.idx[0][1], energy)
        return Device(Architecture.GPU, int(data_id.idx[0][1] // 2), energy)
    def sizes(data_id: DataID) -> int:
        return data_size
    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        device_tuple = (gpus[task_id.task_idx[1] % n_gpus],)
        # if task_id.task_idx[1] % 2 == 0:
        #     device_tuple = (gpu0,)
        # else:
        #     device_tuple = (gpu1,)
        runtime_info = TaskRuntimeInfo(task_time=10000, device_fraction=1, memory=0)
        placement_info = TaskPlacementInfo()
        placement_info.add(device_tuple, runtime_info)
        return placement_info
    #data_config = CannonGemmDataGraphConfig(levels=levels, blocks=blocks, n_devices=n_gpus, data_size=data_size, energy=energy)
    print("data graph config")
    data_config = NoDataGraphConfig()
    data_config.initial_placement = initial_data_placement
    data_config.initial_sizes = sizes
    print("cannon gemm graph config")
    config = CannonGemmConfig(levels=levels, blocks=blocks, task_config=task_placement)
    tasks, data = make_graph(config, data_config=data_config)
    topology = TopologyManager().generate("mesh", config={"N": n_gpus})
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
    data_ids = []
    for i in range(blocks):
        data_ids.append(DataID(((0, i),)))
    for i in range(2 * blocks):
        data_ids.append(DataID(((levels, i),)))
    intervals = simulator.recorders.recorders[0].intervals
    data_task_recorder = simulator.recorders.get(DataTaskRecorder)
    total_communication_energy = 0
    for task in data_task_recorder.tasks.values():
        total_communication_energy += task.communication_energy
    # print(data_task_recorder)
    print(f"Communication Energy: {total_communication_energy} pJ")
    make_data_plot(
        simulator.recorders,
        True,
        True,
        data_ids=data_ids,
    )
    return total_communication_energy

test_data(n_gpus=8100, blocks=8100, n=180)
