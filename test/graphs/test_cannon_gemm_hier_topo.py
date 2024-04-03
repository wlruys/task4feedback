from task4feedback.graphs import *
from task4feedback.load import *
import argparse
from task4feedback.simulator.utility import parse_size
from rich import print
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import test_util
# from utility.execute import run
from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *

from task4feedback.simulator.analysis.recorder import *
from task4feedback.simulator.analysis.plot import *
from task4feedback.simulator.interface import *
from time import perf_counter as clock


def test_data(n_gpus, blocks, n, p, hier_levels, a=8, data_size=None, m=None, energy=[0.01, 1, 100, 1000], dram=False):
    cpu = Device(Architecture.CPU, 0, energy[0])
    dram_cpus = []
    print("test: ", cpu)
    if dram:
        for i in range(hier_levels):
            dram_cpus.append(Device(Architecture.CPU, i + 1, energy[i]))
            # print("test: ", i + 1, " ", energy[i])
        # print("test: ", dram_cpus)
    
    #cpu = Device(Architecture.CPU, 0, energy[3])
    levels = int(math.sqrt(p[0]) + 1)
    gpus = []
    end = 0
    start = 0
    # f = open("pos.txt", "w+")
    for i in range(hier_levels):
        idx = hier_levels - i - 1
        num_gpus = int(pow(p[0], i + 1))
        end += num_gpus
        for j in range(start, end):
            #gpus.append(Device(Architecture.GPU, i))
            gpus.append(Device(Architecture.GPU, j, energy[idx]))
            # print("test create: ", j, " ", energy[idx])
        start = end

    # gpus = [Device(Architecture.GPU, i, energy) for i in range(n_gpus)]
    # print(gpus)
    def initial_data_size(data_id: DataID) -> Devices:
        num_proc = pow(p[0], (hier_levels - data_id.idx[0][0]))
        n_on_each_proc = (n * n) / num_proc
        return n_on_each_proc * a
    # data_size = data_size if data_size is not None else (a * n * n) / p[0]
    def initial_data_placement(data_id: DataID) -> Devices:
        # print(data_id.idx)
        # print(data_id.idx[0])
        if dram:
            if(data_id.idx[0][2] == levels):
                cpu = Device(Architecture.CPU, data_id.idx[0][0] + 1, energy[data_id.idx[0][0]])
                #print("test: data_place ", str(data_id.idx[0]), " ", cpu, " ", energy[data_id.idx[0][0]])
                #print("test_ in initial_data_place [2]: ", data_id.idx[0][0] + 1, " ", energy[data_id.idx[0][0]])
                return cpu #read from DRAM
            elif(data_id.idx[0][2] == 0):
                if(data_id.idx[0][0] == hier_levels - 1):
                    cpu = Device(Architecture.CPU, data_id.idx[0][0] + 1, energy[data_id.idx[0][0]])
                    #print("test: data_place ", str(data_id.idx[0]), " ", cpu, " ", energy[data_id.idx[0][0]])
                    #print("test_ in initial_data_place [0]: ", data_id.idx[0][0] + 1, " ", energy[data_id.idx[0][0]])
                    return cpu #highest level writes to its own DRAM
                else:
                    cpu = Device(Architecture.CPU, data_id.idx[0][0] + 2, energy[data_id.idx[0][0] + 1])
                    #print("test: data_place ", str(data_id.idx[0]), " ", cpu, " ", energy[data_id.idx[0][0]])
                    #print("test_ in initial_data_place else: ", data_id.idx[0][0] + 2, " ", energy[data_id.idx[0][0] + 1])
                    return cpu #lower levels write to next level's DRAM
        
        start_gpu = 0
        for i in range(hier_levels - data_id.idx[0][0] - 1, 0 , -1):
            start_gpu += int(pow(p[0], i))
        pos = start_gpu + data_id.idx[0][1] * p[0]
        if(data_id.idx[0][2] == 0):
            pos += data_id.idx[0][3]
        else:
            idx_pos = data_id.idx[0][3] % (2 * blocks)
            if(data_id.idx[0][3] >= 2 * blocks):
                pos += int(idx_pos) # To accomodate C
            else:
                # pos += int(data_id.idx[0][3] // 2) #If only A,B read
                pos += int(idx_pos // 2) # To accomodate A & B
        #print(data_id.idx, " ", pos)
        #return Device(Architecture.GPU, pos)
        # if(data_id.idx[0][0] == 0):
        #     print("Data id: " + str(data_id.idx) + " " + str(pos))
        #f.write("Data id: " + str(data_id.idx) + " " + str(pos))
        # print("test: data_place ", str(data_id.idx[0]), " ", pos, " ", gpus[pos], " ", energy[data_id.idx[0][0]])
        return Device(Architecture.GPU, pos, energy[data_id.idx[0][0]])
    # def sizes(data_id: DataID) -> int:
    #     return data_size
    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        runtime_info = TaskRuntimeInfo(task_time=10000, device_fraction=1, memory=0)
        placement_info = TaskPlacementInfo()
        if dram:
            if(task_id.task_idx[2] == 0):
                device_tuple = (dram_cpus[task_id.task_idx[0]],)
                placement_info.add(device_tuple, runtime_info)
                # print(placement_info)
                return placement_info
        #print("Task_id: ", task_id.task_idx)
        start_gpu = 0
        for i in range(hier_levels - task_id.task_idx[0] - 1, 0 , -1):
            start_gpu += int(pow(p[0], i))
        pos = start_gpu + task_id.task_idx[1] * p[0] + task_id.task_idx[3] % p[0]
        device_tuple = (gpus[pos],)
        
        # if(task_id.task_idx[0] == 0):
        #     print("Task ID: " + str(task_id.task_idx) + " " + str(pos))
        #     print(pos, "; ", device_tuple)
        # print("Task ID: " + str(task_id.task_idx) + " " + str(pos) + " " + str(gpus[pos]))
        
        placement_info.add(device_tuple, runtime_info)
        # print(placement_info)
        return placement_info
    #     # if task_id.task_idx[1] % 2 == 0:
    #     #     device_tuple = (gpu0,)
    #     # else:
    #     #     device_tuple = (gpu1,)

    data_config = CannonGemmHierDataGraphConfig(hier_levels=hier_levels, levels=levels, blocks=blocks, energy=energy, dram=dram, n=n, a=a, p=p[0])
    data_config.initial_placement = initial_data_placement
    data_config.initial_sizes = initial_data_size
    config = CannonGemmHierConfig(hier_levels=hier_levels, levels=levels, blocks=blocks, n = n, p=p[0], task_config=task_placement)
    tasks, data = make_graph(config, data_config=data_config)
    topology = TopologyManager().generate("mesh_hier", config={"N": p[0], "TOTAL_N": n_gpus, "HIER_LEVELS": hier_levels, "ENERGY": tuple(energy), "DRAM": dram})
    # write_tasks_to_yaml(tasks, "graph")
    # write_data_to_yaml(data, "graph")
    #tasklist, taskmap, datamap = read_sim_graph("graph")
    # populate_dependents(taskmap)
    #tasklist, taskmap = make_graph(tasks, data)
    #networkx_graph, networkx_label = build_networkx_graph(taskmap)
    #plot_pydot(networkx_graph)
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
    data_task_recorder = simulator.recorders.get(DataTaskRecorder)
    total_communication_energy = 0
    count = 0
    for task in data_task_recorder.tasks.values():
        if(task.data_size != 0 and task.communication_energy != 0):
            count += 1
            print(str(task.data.idx[0][2]) + " " + str(task.name) + " " + str(task.data_size) + "D: " + str(task.devices) + "S: " + str(task.source) + " " + str(task.communication_energy))
            
        #if(task.data.idx[0][2] == 0):
        #print(str(task.data.idx[0][2]) + " " + str(task.name) + " " + str(task.data_size) + "D: " + str(task.devices) + "S: " + str(task.source) + " " + str(task.communication_energy))
        total_communication_energy += task.communication_energy
    comm_energy_in_pJ = total_communication_energy * 0.064
    #print(data_task_recorder)
    print(f"Communication Energy: {total_communication_energy}")
    print(f"Communication Energy: {comm_energy_in_pJ} pJ")
    print(count)
    # make_data_plot(
    #     simulator.recorders,
    #     True,
    #     True,
    #     data_ids=data_ids,
    # )
    return comm_energy_in_pJ

def heatmap(data, row_labels, col_labels, vmax, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, vmin = 0, vmax = data[-1,-1], **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    im.set_clim(0, vmax)
    # cbar.set_ticks(ticks=[0, 10000000, 1000000000000000])

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True, fontsize=14)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True, fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylabel("m", fontsize=15)
    ax.set_xlabel("n", fontsize=15)

    return im
#test_data(16, 16, 256)
# def plot_graph(plot_n, p_sizes, energy, m, max_energy):
#     '''
#     m: int
#     '''
#     fig, ax = plt.subplots()
#     ax.set_title(f'E = f(n,p) for p = {m}', fontsize=15)
#     im = heatmap(energy, p_sizes, plot_n, max_energy, ax=ax,
#                        cmap="Wistia", cbarlabel="Energy [pJ]")
#     # texts = annotate_heatmap(im, valfmt="{x:.2e}")
#     valfmt = matplotlib.ticker.StrMethodFormatter("{x}")

#     for i in range(len(p_sizes)):
#         for j in range(len(plot_n)):
#             if(energy[i,j] > 1000):
#                 valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2e}")
#             elif(energy[i,j]% 1 != 0):
#                 valfmt = matplotlib.ticker.StrMethodFormatter("{x:.4f}")
#             else:
#                 valfmt = matplotlib.ticker.StrMethodFormatter("{x}")
#             text = valfmt(energy[i, j], None)
#             text1 = ax.text(j, i, text.rstrip('0').rstrip('.') if '.' in text else text,
#                            ha="center", va="center", color="black",size=14)

#     fig.tight_layout()
#     plt.show()
#     plt.savefig('cannon_gemm.png')



# p_sizes = [4096, 256, 128, 64, 32, 16, 8]
# num_blocks = [4096, 256, 128, 64, 32, 16, 8]    
# p_sizes = [64, 32, 16, 8]
# num_blocks = [64, 32, 16, 8]    
# matrix_sizes = [8, 10, 12, 14, 16, 18, 20, 22]
# energy = []
# max_energy = []
# for i in range(len(p_sizes)):
#     np_energy = []
#     m_val = 0
#     for j in range(len(matrix_sizes)):
#         n = 2 ** matrix_sizes[j]
#         n_energy = test_data(p_sizes[i], num_blocks[i], n)
#         np_energy.append(n_energy)
#         m_val = max(m_val, n_energy)
#     energy.append(np_energy)
#     max_energy.append(m_val)

# plot_n = []
# for n in matrix_sizes:
#     plot_n.append(2 ** n)
# for y_axis in range(len(energy)):
#     print(energy[y_axis])
#     plot_graph(plot_n, p_sizes, energy[y_axis], m_val[y_axis], max_energy[y_axis])
parser = argparse.ArgumentParser()
#Size of matrix
parser.add_argument('-n', type=int, default=8, help='Size of matrix')
#Number of processors
parser.add_argument('-p', type=int, default=4)
#Memory for max_level possible (level 4 by default)
parser.add_argument('-m', type=str, default='2 KB')
#Maximum levels possible in hierarchy
parser.add_argument('-max_levels', type=int, default='4')

args = parser.parse_args()

n = args.n
p_per_level = args.p
m4 = parse_size(args.m)
m = [m4]

#n = 256
levels = 2
#p_per_level = 4
p = []
a = 8
energy = [0.01, 1, 100, 1000] 
for i in range(args.max_levels):
    p.append(p_per_level)
for i in range(1, args.max_levels):
    m.append(m[i - 1] * p[i - 1])

#levels = test_util.calc_num_levels(a, n, m, p)
print(levels)
p, total_p = test_util.get_total_p_hier_mesh(levels, p_per_level)
# for i in range(levels):
#     p.append(p_per_level)
#     total_p += int(pow(p_per_level, i + 1))
print(total_p)
print(p)
dram = True
test_data(total_p, p_per_level, n, p, levels, a=a, energy=energy, dram=dram)