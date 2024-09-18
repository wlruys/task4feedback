from random import Random
from tabnanny import check
from typing_extensions import runtime

from numpy import average
from sympy import limit

from task4feedback.simulator import data
from .utilities import *
from ..types import *
from z3 import *

import matplotlib.pyplot as plt


@dataclass(slots=True)
class RandomConfig(GraphConfig):
    """
    Used to configure the generation of a random task graph.
    """

    nodes: int = 10  # Number of nodes in the graph
    max_width: int = 10
    density: float = 0.2  # Density of the graph
    seed: int = 42
    # Function to determine task placement information
    # These are here specifically for random graph generation
    # initial_sizes: Callable[[DataID], int] = default_data_sizes
    initial_placement: Callable[[DataID], Devices] = default_data_initial_placement
    gpu_size_limit: int = 16 * 1024 * 1024 * 1024  # 16 GB
    p2p_bw: int = 10 * 1024 * 1024 * 1024  # 10GB
    average_task_duration: int = 10000  # unit of us 10000us == 100MB / 10GB/start_time
    # 1us == 100000KB
    no_data: bool = True
    ccr: float = 1  # Computation to Communication Ratio. Higher, more data.
    num_gpus: int = 4
    z3_solver: bool = True
    plot_timeline: bool = False


@register_graph_generator
def make_random_graph(
    config: RandomConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)

    # Below 3 lines are specifically for z3 solver
    tid_to_int: dict[TaskID, int] = {}
    dag: dict[int, list[int]] = {i: [] for i in range(config.nodes)}
    task_times: list[int] = []
    transfer_times = [[0 for _ in range(config.nodes)] for _ in range(config.nodes)]

    random = Random(config.seed)
    average_data_size = int(
        (config.p2p_bw / 1024) * (config.average_task_duration / 1000) * config.ccr
    )
    print(f"Random avg data size: {average_data_size}")

    def task_placement(task_id: TaskID) -> tuple[TaskPlacementInfo, int]:
        runtime_info = TaskRuntimeInfo(
            task_time=random.randint(
                int(config.average_task_duration / 2), config.average_task_duration * 2
            ),
            device_fraction=1,
            memory=int(0),
        )
        placement_info = TaskPlacementInfo()

        for i in range(config.num_gpus):
            placement_info.add(Device(Architecture.GPU, i), runtime_info)

        return placement_info, runtime_info.task_time

    configurations = task_placement

    task_dict: dict[TaskID, TaskInfo] = {}
    data_dict: dict[DataID, DataInfo] = {}
    tasks_by_level: dict[int, list[TaskID]] = {}

    # Level 0 starts with only one node
    task_id = TaskID("T", (0, 0), 0)
    tid_to_int[task_id] = 0
    task_placement_info, time = configurations(task_id)
    task_times.append(time)
    if config.no_data:
        data_dependencies, data_dict = get_data_dependencies(
            task_id, data_dict, data_config
        )
        task_dict[task_id] = TaskInfo(
            task_id,
            task_placement_info,
            [],
            data_dependencies,
            get_mapping(config, task_id),
        )
    else:
        task_dict[task_id] = TaskInfo(
            task_id,
            task_placement_info,
            [],
            TaskDataInfo(),
            get_mapping(config, task_id),
        )
    tasks_by_level[0] = [task_id]

    # Generate subsequent levels
    nodes = 1
    total_level = 1
    while nodes != config.nodes:
        # Determine the number of tasks for this level
        if config.nodes - nodes < config.max_width:
            width = config.nodes - nodes
        else:
            width = random.randint(1, min(config.max_width, config.nodes - nodes))

        task_list = []

        for j in range(width):
            task_idx = (total_level, j)
            task_id = TaskID("T", task_idx, 0)
            tid_to_int[task_id] = nodes + j
            task_list.append(task_id)

            task_placement_info, time = configurations(task_id)
            task_times.append(time)
            dependency_list = []

            # Connect each task in level K to a random task in level K-1
            dep_task_idx = random.randint(0, len(tasks_by_level[total_level - 1]) - 1)
            dependency = tasks_by_level[total_level - 1][dep_task_idx]
            dependency_list.append(dependency)
            dag[tid_to_int[task_id]].append(tid_to_int[dependency])

            task_mapping = get_mapping(config, task_id)

            if config.no_data:
                data_dependencies, data_dict = get_data_dependencies(
                    task_id, data_dict, data_config
                )

                task_dict[task_id] = TaskInfo(
                    task_id,
                    task_placement_info,
                    dependency_list,
                    data_dependencies,
                    task_mapping,
                )
            else:
                task_dict[task_id] = TaskInfo(
                    task_id,
                    task_placement_info,
                    dependency_list,
                    TaskDataInfo(),
                    task_mapping,
                )
        # Store tasks of this level
        tasks_by_level[total_level] = task_list
        nodes += width
        total_level += 1
    print(f"Total levels: {total_level}")
    # Add random edges
    all_tasks = [task for level in tasks_by_level.values() for task in level]
    possible_tasks = [task for task in all_tasks if task.task_idx[0] < total_level - 1]
    success = 0
    num_random_edges = int(
        config.density * config.nodes * (config.nodes - 1) / 2 - config.nodes + 1
    )
    skipped_levels = []
    while success < num_random_edges:
        task1 = random.choice(possible_tasks)
        task1_level = task1.task_idx[0]

        # Select a task from a higher level
        higher_level_tasks = [
            task for task in all_tasks if task.task_idx[0] > task1_level
        ]
        if higher_level_tasks:
            task2 = random.choice(higher_level_tasks)
            if task1 not in task_dict[task2].dependencies:
                task_dict[task2].dependencies.append(task1)
                dag[tid_to_int[task2]].append(tid_to_int[task1])
                success += 1
                skipped_levels.append(task2.task_idx[0] - task1_level)
    if len(skipped_levels) > 0:
        print(f"Avg. skipped levels: {average(skipped_levels)}")
    else:
        print(f"Avg. skipped levels: {0}")
    if config.no_data is False:
        child_tracker: Dict[TaskID, int] = {}
        for task in task_dict.values():
            data_dependencies = TaskDataInfo()
            child_tracker[task.id] = 0
            if task.dependencies is []:
                # This is the first task
                continue
            for dep_task in task.dependencies:
                dep_data_id = DataID(
                    (
                        dep_task.task_idx[0],
                        dep_task.task_idx[1],
                        child_tracker[dep_task],
                    )
                )
                data_placement = config.initial_placement(dep_data_id)
                data_size = random.randint(
                    int(average_data_size / 2),
                    average_data_size * 2,
                )
                data_size = data_size - data_size % (1024 * 1024)

                data_dict[dep_data_id] = DataInfo(
                    dep_data_id, data_size, data_placement
                )

                data_dependencies.read.append(DataAccess(dep_data_id, device=0))

                task_dict[dep_task].data_dependencies.write.append(
                    DataAccess(dep_data_id, device=0)
                )
                transfer_times[tid_to_int[dep_task]][tid_to_int[task.id]] = int(
                    data_size * 1000 * 1000 / config.p2p_bw
                )
                child_tracker[dep_task] += 1

            task.data_dependencies = data_dependencies

    if config.z3_solver:
        print("Using Z3 solver")
        # print(f"# of gpu: {config.num_gpus}")
        # print(f"Task times: {task_times}")
        # print("Task times:")
        # for tid, i in tid_to_int.items():
        #     print(f"{tid} : {task_times[i]}")
        # print("Transfer Times:")
        # for i in range(config.nodes):
        #     for j in range(config.nodes):
        #         if transfer_times[i][j] != 0:
        #             # find tid based on int
        #             for tid, k in tid_to_int.items():
        #                 if k == i:
        #                     iid = tid
        #                 if k == j:
        #                     jid = tid
        #             print(f"{iid} -> {jid}: {transfer_times[i][j]}")
        # print(f"DAG: {dag}")

        M = len(task_times)  # Number of tasks
        N = config.num_gpus  # Number of devices

        # Define Z3 variables
        mapped = [Int(f"x_{i}") for i in range(M)]  # Device assignment for each task
        start_time = [Int(f"s_{i}") for i in range(M)]  # Start time for each task
        end_time = [Int(f"e_{i}") for i in range(M)]  # End time for each task
        T = Int("T")  # Makespan

        solver = Optimize()

        # Constraints for device assignment
        for i in range(M):
            solver.add(And(mapped[i] >= 0, mapped[i] < N))

        # Constraints for start and end times (ensure non-negative times)
        for i in range(M):
            solver.add(start_time[i] >= 0)
            solver.add(end_time[i] == start_time[i] + task_times[i])
            solver.add(end_time[i] >= start_time[i])

        # Precedence constraints
        for task in range(M):
            for dep1 in dag[task]:  # Iterate over all dependencies
                # If task is assigned to the same device as its dependency, ensure that it starts after the dependency ends
                solver.add(
                    If(
                        mapped[task] == mapped[dep1],
                        start_time[task] >= end_time[dep1],
                        start_time[task] >= end_time[dep1] + transfer_times[dep1][task],
                    )
                )

        # Makespan constraints
        for i in range(M):
            solver.add(T >= end_time[i])

        # Ensure that makespan is non-negative
        solver.add(T >= 0)

        # Mutual exclusion constraint: no two tasks on the same device can overlap in time
        for i in range(M):
            for j in range(i + 1, M):
                solver.add(
                    If(
                        mapped[i] == mapped[j],
                        Or(
                            end_time[i] <= start_time[j],
                            end_time[j] <= start_time[i],
                        ),  # Task i ends before Task j starts or vice versa
                        True,
                    )
                )  # If not on the same device, no constraint needed

        # Objective: minimize the makespan
        solver.minimize(T)

        # Check for solution
        if solver.check() == sat:
            model = solver.model()
            best_mapping = [model.evaluate(mapped[i]).as_long() for i in range(M)]  # type: ignore
            best_start_times = [model.evaluate(start_time[i]).as_long() for i in range(M)]  # type: ignore
            best_end_times = [model.evaluate(end_time[i]).as_long() for i in range(M)]  # type: ignore
            # best_makespan = model.evaluate(T).as_long()  # type: ignore
            best_makespan = max(best_end_times)
            # Store ranking of each starttime
            sorted_idx = sorted(
                range(len(best_start_times)), key=lambda k: best_start_times[k]
            )
            ranks = [0] * len(best_start_times)
            for i in range(len(sorted_idx)):
                ranks[sorted_idx[i]] = i

            print(f"Best Mapping: {best_mapping}")
            print(f"Optimal,simtime,{best_makespan/(1000**2)}")
            for tid, i in tid_to_int.items():
                task_dict[tid].z3_allocation = best_mapping[i]
                task_dict[tid].z3_order = ranks[i]
                print(
                    f"{tid} at {best_mapping[i]} Order:{ranks[i]} start:{best_start_times[i]} end:{best_end_times[i]}"
                )
        else:
            print("No solution found")

        if config.plot_timeline:
            device_colors = [
                "Red",
                "Green",
                "Blue",
                "Yellow",
                "Purple",
                "Orange",
                "Pink",
                "Brown",
            ]
            fig, ax = plt.subplots(figsize=(25, 10))
            fig.subplots_adjust(left=0.01, right=0.99)
            for tid, i in tid_to_int.items():
                start = best_start_times[i]
                end = best_end_times[i]
                bar = ax.barh(
                    y=best_mapping[i],
                    width=end - start,
                    left=start,
                    height=0.4,
                    color=device_colors[best_mapping[i]],
                    edgecolor="black",
                )
                ax.text(
                    start + (end - start) / 2,
                    best_mapping[i],
                    tid,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="xx-large",
                )
                plt.savefig("optimal_timeline.png")

    return task_dict, data_dict
