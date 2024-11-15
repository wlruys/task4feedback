from task4feedback.types import *
from task4feedback.graphs import *
from task4feedback.fastsim.interface import (
    SimulatorHandler,
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


@dataclass
class Args:
    devices: int = 4
    vcus: int = 1
    blocks: int = 4
    hidden_dim: int = 64
    seed: int = 0


def initialize_simulator(seed=0, args=None, density=0.3):
    config = RandomConfig(
        n_devices=args.devices,
        seed=seed,
        nodes=15,
        density=density,
        no_data=False,
        z3_solver=False,
        ccr=1,
        num_gpus=4,
    )
    tasks, data = make_random_graph(config)

    mem = 1600 * 1024 * 1024 * 1024
    bandwidth = (20 * 1024 * 1024 * 1024) / 10**4
    latency = 1
    n_devices = args.devices
    devices = uniform_connected_devices(n_devices, mem, latency, bandwidth)

    H = SimulatorHandler(
        tasks,
        data,
        devices,
        noise_type=TNoiseType.NONE,
        cmapper_type=CMapperType.EFT_DEQUEUE,
        pymapper=RoundRobinPythonMapper(n_devices),
        seed=seed,
    )
    sim = H.create_simulator()
    sim.initialize(use_data=True)
    sim.randomize_durations()
    sim.enable_python_mapper()

    return H, sim


# Load the model
args = Args()
args.hidden_dim = 64  # Use the same hidden dimension as during training
args.devices = 4

# Initialize a dummy simulator to get the graph features
H_dummy, sim_dummy = initialize_simulator(seed=0, args=args)
candidates = sim_dummy.get_mapping_candidates()
local_graph = sim_dummy.observer.local_graph_features(candidates)

# Initialize the network and load the saved model
model = TaskAssignmentNetDeviceOnly(args.devices, args.hidden_dim, local_graph)
model.load_state_dict(
    torch.load(
        "model.pth",
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


def evaluate_model_on_graph(model, seed, args, density):
    # Initialize the simulator with the given seed and density
    H, sim = initialize_simulator(seed=seed, args=args, density=density)

    # Run baseline
    baseline_sim = H.copy(sim)
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
densities = [0.1, 0.2, 0.3, 0.4, 0.5]
accuracies_per_density = {}

for density in densities:
    accuracies_per_density[density] = []
    print(f"Testing on density {density}")
    for seed in range(10000, 10000 + num_test_graphs):
        accuracy, model_time, baseline_time = evaluate_model_on_graph(
            model, seed, args, density
        )
        accuracies_per_density[density].append(accuracy)
        print(
            f"Density {density}, Graph seed {seed}: Model time {model_time}, Baseline time {baseline_time}, Accuracy (baseline/model) {accuracy}"
        )

# Plot the box plot with detailed statistics
data = [accuracies_per_density[d] for d in densities]

fig, ax = plt.subplots(figsize=(10, 6))

boxprops = dict(
    linestyle="-", linewidth=2, color="darkgoldenrod", facecolor="gold", alpha=0.7
)
medianprops = dict(linestyle="-", linewidth=2, color="firebrick")
meanpointprops = dict(marker="D", markeredgecolor="black", markerfacecolor="firebrick")
flierprops = dict(marker="o", markerfacecolor="green", markersize=5, linestyle="none")

bp = ax.boxplot(
    data,
    labels=densities,
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
plt.xlabel("Graph Density")
plt.ylabel("Speedup (model/baseline)")
plt.title("Model speedup compared to EFT")
plt.tight_layout()
# Save the plot
plt.savefig("speedup_boxplot.png")
