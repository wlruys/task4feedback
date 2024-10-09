from task4feedback.fastsim.simulator import (
    PyTasks,
    PyDevices,
    PySimulator,
    PyAction,
    PyExecutionState,
    PyStaticMapper,
    PyEventType,
)
from task4feedback.graphs import *
import argparse
import time
import networkx as nx


def build_networkx_graph_from_infos(
    tasks: Mapping[TaskID, TaskInfo],
) -> nx.DiGraph:
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
    independent_time = 0
    serial_time = 0
    averaged_generation_time = 0

    # Calculate serial/independent task times
    for n in G:
        serial_time += G.nodes[n].get("duration", 0)
    independent_time = serial_time / num_gpus

    for p in critical_path:
        critical_path_time += G.nodes[p].get("duration")

    other_critical_path_time = 0
    generations = nx.topological_generations(G)
    # print(generations)
    for g in generations:
        number_of_tasks = len(g)
        print(f"GEN WIDTH: {number_of_tasks}")
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
    print(f"Independent,simtime,{independent_time / 10**6}")
    print(f"Serial,simtime,{serial_time / 10**6}")
    print(f"Averaged Generation Time: {averaged_generation_time / 10**6}")
    # print(f"Other Critical Path Time: {other_critical_path_time / 10**6}")
    print(f"Independent Estimate: {total_work_in_graph / (10**6 * num_gpus)}")
    print("N Tasks", len(G.nodes))


parser = argparse.ArgumentParser()
parser.add_argument("--blocks", type=int, default=20)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--vcus", type=int, default=100)
args = parser.parse_args()


def task_config(task_id: TaskID) -> TaskPlacementInfo:
    placement_info = TaskPlacementInfo()
    placement_info.add(
        (Device(Architecture.GPU, -1),), TaskRuntimeInfo(task_time=100000)
    )
    return placement_info


data_config = CholeskyDataGraphConfig()
config = CholeskyConfig(blocks=args.blocks, task_config=task_config)
tasks, data = make_graph(config, data_config=data_config)

G = build_networkx_graph_from_infos(tasks)

calculate_critical_path(G, args.devices)


def create_data_ids(data):
    data_to_ids = {}
    ids_to_data = {}

    for i, d in enumerate(data):
        data_to_ids[d] = i
        ids_to_data[i] = d

    return data_to_ids, ids_to_data


def create_task_ids(tasks):
    tasks_to_ids = {}
    ids_to_tasks = {}

    for i, task in enumerate(tasks):
        tasks_to_ids[task] = i
        ids_to_tasks[i] = task

    return tasks_to_ids, ids_to_tasks


def build_c_graph(tasks):
    n_tasks = len(tasks)

    tasks_to_ids, ids_to_task = create_task_ids(tasks)
    data_to_ids, ids_to_data = create_data_ids(data)

    s = PyTasks(n_tasks)

    for task_name, task in tasks.items():
        id = tasks_to_ids[task_name]
        dependencies = [tasks_to_ids[dep] for dep in task.dependencies]
        s.create_task(id, str(task_name), dependencies)
        s.add_variant(id, 0, args.vcus, 0, 1000)

        read = [d.id for d in task.data_dependencies.read]
        write = [d.id for d in task.data_dependencies.write]
        read_write = [d.id for d in task.data_dependencies.read_write]

        read_set = set(read).union(set(read_write))
        write_set = set(write).union(set(read_write))

        read_ids = [data_to_ids[d] for d in read_set]
        write_ids = [data_to_ids[d] for d in write_set]

        print(f"Task {task_name} Read {read_ids} Write {write_ids}")

        s.add_read_set(id, read_ids)
        s.add_write_set(id, write_ids)

    return s


def build_devices():
    n_devices = args.devices
    d = PyDevices(n_devices)
    mem = 1e8
    vcu = 1000
    arch = 0

    for i in range(n_devices):
        name = f"device_{i}"
        d.create_device(i, name, arch, vcu, mem)

    return d


def run_simulator(simulator):
    flag = 1
    while flag == 1:
        flag = simulator.run()

        if flag == PyExecutionState.BREAKPOINT:
            print("Breakpoint Reached", simulator.get_current_time())
            flag = 1

        if flag == 4:
            candidates = simulator.get_mappable_candidates()

            action_list = []
            for i, candidate in enumerate(candidates):
                device = candidate % args.devices
                action_list.append(PyAction(candidate, i, device, 0, 0))

            simulator.map_tasks(action_list)
            flag = 1
    print("Exit Flag", flag)


pytasks = build_c_graph(tasks)
pyydevices = build_devices()
pymapper = PyStaticMapper()
simulator = PySimulator(pytasks, pyydevices, pymapper)

start_t = time.perf_counter()
simulator.initialize(0, True)
end_t = time.perf_counter()
print(f"Init Time: {end_t - start_t}")
# simulator.add_task_breakpoint(PyEventType.COMPLETER, 1)
# simulator.add_task_breakpoint(PyEventType.COMPLETER, 4)
# start_t = time.perf_counter()
# run_simulator(simulator)
# end_t = time.perf_counter()
# print(f"Time: {end_t - start_t}")
# print(f"Simulator Time: {simulator.get_current_time()}")

# start_t = time.perf_counter()
# exit_flag = simulator.run()
# end_t = time.perf_counter()
# print(f"Time: {end_t - start_t}")
# print(f"Simulator Time {simulator.get_current_time()}")
# print(f"Simulator Exit {exit_flag}")

for i in range(pytasks.n_tasks()):

    if pytasks.is_compute(i):
        name = pytasks.get_name(i)
        depth = pytasks.get_depth(i)
        dependencies = pytasks.get_dependencies(i)
        dependencies = [pytasks.get_name(dep) for dep in dependencies]
        dependents = pytasks.get_dependents(i)
        dependents = [pytasks.get_name(dep) for dep in dependents]
        print(f"Task {name} Depth {depth}")
        print(f"- Dependencies {dependencies}")
        print(f"- Dependents {dependents}")

        read_set = pytasks.get_read_set(i)
        write_set = pytasks.get_write_set(i)
        print(f"- Read Set {pytasks.get_read_set(i)}")
        print(f"- Write Set {pytasks.get_write_set(i)}")
        data_dependencies = pytasks.get_data_dependencies(i)
        data_dependencies = [pytasks.get_name(dep) for dep in data_dependencies]
        print(f"- Data Dependencies {data_dependencies}")

        data_dependents = pytasks.get_data_dependents(i)
        data_dependents = [pytasks.get_name(dep) for dep in data_dependents]

        print(f"- Data Dependents {data_dependents}")
    else:
        name = pytasks.get_name(i)
        dependencies = pytasks.get_dependencies(i)
        dependencies = [pytasks.get_name(dep) for dep in dependencies]
        dependents = pytasks.get_dependents(i)
        dependents = [pytasks.get_name(dep) for dep in dependents]
        print(f"Task {name}")
        print(f"- Dependencies {dependencies}")
        print(f"- Dependents {dependents}")
        print(f"- DataID {pytasks.get_data_id(i)}")
