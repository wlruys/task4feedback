from task4feedback.fastsim.simulator import PyTasks, PyDevices, PySimulator
from task4feedback.graphs import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--block", type=int, default=4)
args = parser.parse_args()


def task_config(task_id: TaskID) -> TaskPlacementInfo:
    placement_info = TaskPlacementInfo()
    placement_info.add(
        (Device(Architecture.GPU, -1),), TaskRuntimeInfo(task_time=100000)
    )
    return placement_info


data_config = CholeskyDataGraphConfig(data_size=1024 * 1024 * 1024)
config = CholeskyConfig(blocks=args.block, task_config=task_config)
tasks, data = make_graph(config)

print(tasks)


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

    s = PyTasks(n_tasks)

    for task_name, task in tasks.items():
        id = tasks_to_ids[task_name]
        dependencies = [tasks_to_ids[dep] for dep in task.dependencies]
        s.create_task(id, str(task_name), dependencies)
        s.add_variant(id, 0, 1, 0, 1000)

    s.initialize_dependents()

    return s


def build_devices():
    d = PyDevices(1)
    mem = 1e9
    d.create_device(0, "CPU", 0, 1, mem)

    return d


pytasks = build_c_graph(tasks)
pyydevices = build_devices()

simulator = PySimulator(pytasks, pyydevices)
