from z3 import *

from task4feedback.simulator.analysis import recorder
from task4feedback.simulator.analysis.recorder import ComputeTaskRecorder
from task4feedback.simulator.simulator import *
from ...legacy_types import *


def calculate_optimal(simulator: SimulatedScheduler):
    if simulator.scheduler_type == TaskOrderType.OPTIMAL:
        print(
            "For optimal scheduling order, optimal calculation is already done in graph generation."
        )
        return
    recorders = simulator.recorders
    compute_task_record: ComputeTaskRecorder = recorders.get(ComputeTaskRecorder)
    M = len(compute_task_record.tasks)
    N = simulator.topology.get_devices(device_type=Architecture.GPU)
    task_ids = []
    for taskid, task_record in compute_task_record.tasks.items():
        task_ids.append(taskid)

    # Define Z3 variables
    mapped = [Int(f"x_{i}") for i in range(M)]  # Device assignment for each task
    start_time = [Int(f"s_{i}") for i in range(M)]  # Start time for each task
    end_time = [Int(f"e_{i}") for i in range(M)]  # End time for each task
    T = Int("T")  # Makespan

    solver = Optimize()

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
            # Since the communication starts when the last dependent task ends, add the transfer time to the end of all dependent tasks
            for dep2 in dag[task]:
                if dep1 == dep2:
                    continue
                solver.add(
                    If(
                        mapped[task] == mapped[dep1],
                        True,
                        start_time[task] >= end_time[dep2] + transfer_times[dep1][task],
                    )
                )
                # Below is assuming perfect prefetcher.
                # Dependent data is moved to the successor when the task is finished.
                # This is not the case for current simulator.
                # Current simulator starts moving data when all the dependent tasks are finished.
                # solver.add(
                #     If(
                #         mapped[task] == mapped[dep1],
                #         If(mapped[task] == mapped[dep2],),
                #         start_time[dep1] >= end_time[task] + transfer_times[task][dep1],
                #     )
                # )

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
        best_makespan = model.evaluate(T).as_long()  # type: ignore

        # Store ranking of each starttime
        sorted_idx = sorted(
            range(len(best_start_times)), key=lambda k: best_start_times[k]
        )
        ranks = [0] * len(best_start_times)
        for i in range(len(sorted_idx)):
            ranks[sorted_idx[i]] = i

        print(f"Best Makespan: {best_makespan}")
    else:
        print("No solution found")
