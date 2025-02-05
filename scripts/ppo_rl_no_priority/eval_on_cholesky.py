from task4feedback.types import *
from task4feedback.graphs import *
from task4feedback.fastsim.interface import (
    SimulatorHandler,
    Simulator,
    uniform_connected_devices,
    TNoiseType,
    CMapperType,
    RoundRobinPythonMapper,
    Phase,
    PythonMapper,
    Action,
)
from task4feedback.fastsim.models import TaskAssignmentNetDeviceOnly
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--devices", type=int, default=4)
parser.add_argument("--vcus", type=int, default=1)
parser.add_argument("--blocks", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", type=str, default="models/model.pth")
parser.add_argument("--use_eft", type=int, default=1)
args = parser.parse_args()


def initialize_simulator(blocks=3, seed=0):

    def task_config(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()
        placement_info.add(
            (Device(Architecture.GPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=args.vcus),
        )
        placement_info.add(
            (Device(Architecture.CPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=args.vcus),
        )
        return placement_info

    data_config = CholeskyDataGraphConfig(data_size=1 * 1024 * 1024 * 1024)
    config = CholeskyConfig(blocks=blocks, task_config=task_config)
    tasks, data = make_graph(config, data_config=data_config)

    mem = 1600 * 1024 * 1024 * 1024
    bandwidth = (20 * 1024 * 1024 * 1024) / 10**4
    latency = 1
    n_devices = 4
    devices = uniform_connected_devices(n_devices, mem, latency, bandwidth)

    H = SimulatorHandler(
        tasks,
        data,
        devices,
        noise_type=TNoiseType.LOGNORMAL,
        cmapper_type=CMapperType.EFT_DEQUEUE,
        pymapper=RoundRobinPythonMapper(n_devices),
        seed=seed,
    )
    sim = H.create_simulator()
    sim.initialize(use_data=True)
    sim.randomize_durations()
    sim.randomize_priorities()
    sim.enable_python_mapper()

    return H, sim


# Load the model
# Initialize a dummy simulator to get the graph features
H_dummy, sim_dummy = initialize_simulator(blocks=5)
candidates = sim_dummy.get_mapping_candidates()
local_graph = sim_dummy.observer.local_graph_features(candidates)

# Initialize the network and load the saved model
model = TaskAssignmentNetDeviceOnly(args.devices, args.hidden_dim, local_graph)
model.load_state_dict(
    torch.load(
        args.model,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
)
model.eval()


class GreedyNetworkMapper(PythonMapper):
    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)
        with torch.no_grad():
            d, v = self.model.forward(data)
            # Choose argmax of network output for priority and device assignment
            dev_per_task = torch.argmax(d, dim=-1)
            action_list = []
            for i in range(len(candidates)):
                # Check if p_per_task and dev_per_task are scalars
                if dev_per_task.dim() == 0:
                    dev_task = dev_per_task.item()
                else:
                    dev_task = dev_per_task[i].item()
                a = Action(
                    candidates[i],
                    i,
                    dev_task,
                    0,
                    0,
                )
                action_list.append(a)
        return action_list


def evaluate_model_on_graph(model, block, seed=0):
    # Initialize the simulator with the given seed and density
    # Run baseline
    H, sim = initialize_simulator(blocks=block, seed=seed)
    sim.randomize_durations()
    sim.randomize_priorities()
    baseline_sim = H.copy(sim)
    if args.use_eft:
        baseline_sim.disable_python_mapper()
        c_mapper = H.get_new_c_mapper()
        baseline_sim.set_c_mapper(c_mapper)
    baseline_sim.run()
    baseline_time = baseline_sim.get_current_time()

    # Run model with GreedyNetworkMapper
    model_sim = H.copy(sim)
    greedy_mapper = GreedyNetworkMapper(model)
    model_sim.set_python_mapper(greedy_mapper)
    model_sim.run()
    model_time = model_sim.get_current_time()

    # Compute accuracy (baseline_time divided by model_time)
    accuracy = baseline_time / model_time  # Higher is better

    return accuracy, model_time, baseline_time


# Evaluate the model on multiple test graphs and different densities
num_test_graphs = 100
blocks = [3, 4, 5, 6]
accuracies_per_size = {}

for block in blocks:
    accuracies_per_size[block] = []
    print(f"Testing on block {block}")
    for seed in range(10000, 10000 + num_test_graphs):

        accuracy, model_time, baseline_time = evaluate_model_on_graph(
            model, block, seed=seed
        )
        accuracies_per_size[block].append(accuracy)
        print(
            f"{block}x{block}, Graph seed {seed}: Model time {model_time}, Baseline time {baseline_time}, Accuracy (baseline/model) {accuracy}"
        )

# Plot the box plot with detailed statistics
data = [accuracies_per_size[d] for d in blocks]
# Increase font size by a factor of 2
plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 2})
fig, ax = plt.subplots(figsize=(10, 6))

boxprops = dict(
    linestyle="-", linewidth=2, color="darkgoldenrod", facecolor="gold", alpha=0.7
)
medianprops = dict(linestyle="-", linewidth=2, color="firebrick")
meanpointprops = dict(marker="D", markeredgecolor="black", markerfacecolor="firebrick")
flierprops = dict(marker="o", markerfacecolor="green", markersize=5, linestyle="none")

bp = ax.boxplot(
    data,
    labels=blocks,
    patch_artist=True,
    boxprops=boxprops,
    medianprops=medianprops,
    meanprops=meanpointprops,
    flierprops=flierprops,
    showmeans=True,
)

# Annotate each box with statistical details
for i, line in enumerate(bp["medians"]):
    # Get median value
    median = line.get_ydata()[0]
    x = i + 1  # x-position is the position of the boxplot
    ax.text(
        x,
        median,
        f"{median:.2f}",
        horizontalalignment="center",
        verticalalignment="bottom",
        color="firebrick",
        fontweight="bold",
    )

for i in range(len(bp["boxes"])):
    x = i + 1  # x-position is the position of the boxplot
    # Compute quartiles
    q1 = np.percentile(data[i], 25)
    q3 = np.percentile(data[i], 75)
    # Annotate quartiles
    ax.text(
        x,
        q1,
        f"Q1: {q1:.2f}",
        horizontalalignment="center",
        verticalalignment="top",
        color="blue",
    )
    ax.text(
        x,
        q3,
        f"Q3: {q3:.2f}",
        horizontalalignment="center",
        verticalalignment="bottom",
        color="blue",
    )
    # Annotate whiskers
    whisker_low = bp["whiskers"][i * 2].get_ydata()[1]
    whisker_high = bp["whiskers"][i * 2 + 1].get_ydata()[1]
    ax.text(
        x,
        whisker_low,
        f"Min: {whisker_low:.2f}",
        horizontalalignment="center",
        verticalalignment="top",
        color="green",
    )
    ax.text(
        x,
        whisker_high,
        f"Max: {whisker_high:.2f}",
        horizontalalignment="center",
        verticalalignment="bottom",
        color="green",
    )

# Set labels and title
plt.xlabel("Cholesky Block Size")
plt.ylabel("Speedup (model/baseline)")
if args.use_eft:
    plt.title("Model speedup compared to EFT")
else:
    plt.title("Model speedup compared to RoundRobin")
plt.tight_layout()
# Save the plot
if args.use_eft:
    plt.savefig("speedup_boxplot_cholesky_eft.png")
else:
    plt.savefig("speedup_boxplot_cholesky_roundrobin.png")
