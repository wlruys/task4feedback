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
    energy = [0.2, 0.2]
    total_n = 40
    hier_levels = 2
    num_gpus = [4, 36]
    p_per_mesh = [4, 9]
    num_meshes = [1, 4]
    # num_gpus = [100, 8100]
    # p_per_mesh = [100, 81]
    # num_meshes = [1, 10]
    a = 8
    n = 360
    levels = []
    for i in range(hier_levels):
        levels.append(int(math.sqrt(p_per_mesh[i])))
    blocks = p_per_mesh
    B = 2
    gpus = []
    
    end = 0
    start = 0
    for i in range(hier_levels):
        idx = hier_levels - i - 1
        end += num_gpus[i]
        for j in range(start, end):
            #gpus.append(Device(Architecture.GPU, i))
            gpus.append(Device(Architecture.GPU, j, energy[idx]))
            # print("test create: ", j, " ", energy[idx])
        start = end
    gpus.append(Device(Architecture.GPU, total_n, 4.24)) #HBM
    def initial_data_placement(data_id: DataID) -> Devices:
        # return Device(Architecture.CPU, 0)
        return Device(Architecture.GPU, total_n, 4.24) # everyting is on HBM at the start

    def sizes(data_id: DataID) -> int:
        hier_level = data_id.idx[0][3]
        idx = hier_levels - hier_level - 1
        num_proc = num_gpus[idx]
        n_on_each_proc = (n * n) / num_proc
        return n_on_each_proc * a

    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        #device_tuple = (gpus[task_id.task_idx[6] % p_per_mesh[task_id.task_idx[4]]],)
        # if task_id.task_idx[1] % 2 == 0:
        #     device_tuple = (gpu0,)
        # else:
        #     device_tuple = (gpu1,)
        idx = hier_levels - task_id.task_idx[3] - 1

        runtime_info = TaskRuntimeInfo(task_time=10000, device_fraction=1, memory=0)
        placement_info = TaskPlacementInfo()
        start_gpu = 0
        for i in range(idx - 1, -1, -1):
            start_gpu += num_gpus[i]
        pos = start_gpu + task_id.task_idx[4] * p_per_mesh[idx] + task_id.task_idx[6] % p_per_mesh[idx]
        device_tuple = (gpus[pos],)
        placement_info.add(device_tuple, runtime_info)
        # print(str(task_id), " ", str(device_tuple), " ", str(start_gpu))
        return placement_info

    # data_config = ImecHierDataGraphConfig(levels=levels, blocks=blocks, n_devices=n_gpus)
    # data_config.initial_placement = initial_data_placement
    # data_config.initial_sizes = sizes
    data_config = ImecHierDataGraphConfig(hier_levels=hier_levels, levels=levels, p_per_mesh=p_per_mesh, blocks=blocks, energy=energy, num_gpus=num_gpus, n=n, a=a)
    config = ImecHierConfig(levels=levels, blocks=blocks, B=B, num_gpus=num_gpus, num_meshes=num_meshes, task_config=task_placement)
    tasks, data = make_graph(config, data_config=data_config)

    topology = TopologyManager().generate("imec_hier", config=None)
    # write_tasks_to_yaml(tasks, "graph")
    # write_data_to_yaml(data, "graph")

    # tasklist, taskmap = create_sim_graph(tasks, data, use_data=False)
    # # populate_dependents(taskmap)
    # #tasklist, taskmap = make_graph(tasks, data, use_data=True)

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