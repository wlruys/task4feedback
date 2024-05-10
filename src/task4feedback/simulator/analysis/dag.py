import networkx as nx

from ...types import *
from ..topology import *


def print_graph_info(G):
    # Print number of edges in G
    print("Number of edges in G:", G.number_of_edges())
    # Print number of nodes in G
    print("Number of nodes in G:", G.number_of_nodes())
    # Print the width of the graph level by level
    generations = nx.topological_generations(G)
    num_gens = 0
    widths = []
    for gen in generations:
        num_gens += 1
        widths.append(len(gen))
    # Print the average width of the graph
    print("Number of generations(height):", num_gens)
    # print("Widths of the graph:", widths)
    print("Widths of the graph:", max(widths))
    print("Average width of the graph:", sum(widths) / num_gens)


def build_networkx_graph_from_infos(
    tasks: List
) -> Tuple[nx.DiGraph, Dict[TaskID, str]]:
    """
    Build a networkx graph with nodes representing tasks and edges representing dependencies.
    Edges are weighted by the duration of the task such that the critical path length can be computed.
    """

    G = nx.DiGraph()

    for name, task_info in tasks.items():

        runtime_infos = task_info.runtime[task_info.runtime.locations[0]]

        max_time = max([runtime_info.task_time for runtime_info in runtime_infos])
        duration = max_time

        # print(f"Adding node: {name}, duration: {duration}")
        name = str(name)
        G.add_node(name, label=name, info=task_info, duration=duration)

        d = 2 if len(task_info.dependencies) == 0 else 1

        for dep_id in task_info.dependencies:
            dep_info = tasks[dep_id]
            # print(f"Dep: ", type(dep_info))

            dep_runtime_infos = dep_info.runtime[dep_info.runtime.locations[0]]
            dep_duration = max(
                [runtime_info.task_time for runtime_info in dep_runtime_infos]
            )

            c = 2 if len(dep_info.dependencies) == 0 else 1
            edge_weight = (d * duration + c * dep_duration) / 2

            G.add_edge(str(dep_info.id), str(name), weight=edge_weight)
    return G


def calculate_critical_path(G, num_gpus):
    critical_path = nx.dag_longest_path(G)
    critical_path_time = 0
    generation_time = 0
    averaged_generation_time = 0

    for p in critical_path:
        critical_path_time += G.nodes[p].get("duration")

    other_critical_path_time = 0
    generations = nx.topological_generations(G)
    # print(generations)
    for g in generations:
        number_of_tasks = len(g)
        batches = int(np.ceil(number_of_tasks / num_gpus))
        for r in range(int(batches)):
            tasks_in_batch = g[r * num_gpus : (r + 1) * num_gpus]
            generation_time += max(
                [G.nodes[t].get("duration", 0) for t in tasks_in_batch]
            )
        durations = [G.nodes[t].get("duration", 0) for t in g]
        averaged_generation_time += max(
            min(durations), sum([G.nodes[t].get("duration", 0) for t in g]) / num_gpus
        )
        other_critical_path_time += max([G.nodes[t].get("duration", 0) for t in g])

    total_work_in_graph = sum([G.nodes[t].get("duration", 0) for t in G.nodes])

    print(f"Critical Path Time: {critical_path_time / 10**6}")
    print(f"Generation Time: {generation_time / 10**6}")
    print(f"BSP,simtime,{generation_time / 10**6}")
    print(f"Averaged Generation Time: {averaged_generation_time / 10**6}")
    # print(f"Other Critical Path Time: {other_critical_path_time / 10**6}")
    print(f"Independent Estimate: {total_work_in_graph / (10**6 * num_gpus)}")
