from task4feedback.graphs import *
from task4feedback.load import *

from rich import print

# from utility.execute import run
from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *


def test_sim():
    cpu = Device(Architecture.CPU, 0)
    gpu0 = Device(Architecture.GPU, 0)
    gpu1 = Device(Architecture.GPU, 1)

    # For the moment, disable data movement
    # Hopefully, this will be integrated by end of day (1/30/2024)
    data_config = NoDataGraphConfig()

    #  Define some task placement lambda rule
    #  This places tasks on GPU0 if the task index is even and GPU1 if the task index is odd
    #  This is just for testing purposes
    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        if task_id.task_idx[0] % 2 == 0:
            device_tuple = (gpu0,)
        else:
            device_tuple = (gpu1,)

        runtime_info = TaskRuntimeInfo(task_time=100000, device_fraction=0.5, memory=0)
        placement_info = TaskPlacementInfo()
        placement_info.add(device_tuple, runtime_info)

        return placement_info

    # Define the task graph configuration with the task placement rule
    config = ChainConfig(steps=2, chains=2, task_config=task_placement)

    # Build the task graph (this can be used to launch a Parla program)
    tasks, data = make_graph(config, data_config=data_config)

    # Convert the task graph to a simulator input graph
    tasklist, task_map = create_sim_graph(tasks, data, use_data=False)

    # Build the device topology
    topology = TopologyManager().get_generator("frontera")(None)

    # Initialize a simulated scheduler
    scheduler = SimulatedScheduler(topology=topology, scheduler_type="parla")
    scheduler.register_taskmap(task_map)

    # Define a task ordering, and provide it to the scheduler
    topological_sort(tasklist, task_map)
    scheduler.add_initial_tasks(tasklist)

    # Run the scheduler
    scheduler.run()

    # Print the total time to execute the task graph (makespan)
    print(scheduler.time)


if __name__ == "__main__":
    test_sim()
