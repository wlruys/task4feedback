from rich import print

from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *
from task4feedback.graphs import *
from task4feedback.load import *
from task4feedback.execute import run

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-workers",
    type=int,
    default=1,
    help="How many workers to use. This will perform a sample of 1 to workers by powers of 2",
)
parser.add_argument(
    "-width",
    type=int,
    default=1,
    help="The width of the task graph. If not set this is equal to nworkers.",
)
parser.add_argument("-steps", type=int, default=1, help="The depth of the task graph.")
parser.add_argument(
    "-d", type=int, default=7, help="The size of the data if using numba busy kernel"
)
parser.add_argument(
    "-n",
    type=int,
    default=2**23,
    help="The size of the data if using numba busy kernel",
)
parser.add_argument(
    "-isync",
    type=int,
    default=0,
    help="Whether to synchronize (internally) using await at every timestep.",
)
parser.add_argument(
    "-vcus",
    type=int,
    default=1,
    help="Whether tasks use vcus to restrict how many can run on a single device",
)
parser.add_argument(
    "-deps",
    type=int,
    default=1,
    help="Whether tasks have dependencies on the prior iteration",
)
parser.add_argument("-verbose", type=int, default=0, help="Verbose!")

parser.add_argument(
    "-t",
    type=int,
    default=10,
    help="The task time in microseconds. These are hardcoded in this main.",
)
parser.add_argument(
    "-accesses",
    type=int,
    default=1,
    help="How many times the task stops busy waiting and accesses the GIL",
)
parser.add_argument(
    "-frac",
    type=float,
    default=0,
    help="The fraction of the total task time that the GIL is held",
)

parser.add_argument(
    "-strong",
    type=int,
    default=0,
    help="Whether to use strong (1) or weak (0) scaling of the task time",
)
parser.add_argument(
    "-sleep",
    type=int,
    default=1,
    help="Whether to use the synthetic sleep (1) or the numba busy kernel (0)",
)
parser.add_argument(
    "-restrict",
    type=int,
    default=0,
    help="This does two separate things. If using isync it restricts to only waiting on the prior timestep. If using deps, it changes the dependencies from being a separate chain to depending on all tasks in the prior timestep",
)

args = parser.parse_args()


def test():
    data_config = NoDataGraphConfig()

    def custom_tasks(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()
        device_tuple = (Device(Architecture.CPU, -1),)
        runtime_info = TaskRuntimeInfo(
            task_time=args.t,
            device_fraction=1 / args.workers,
            gil_fraction=args.frac,
            gil_accesses=args.accesses,
        )

        placement_info.add(device_tuple, runtime_info)

        return placement_info

    # config = IndependentConfig(
    #     task_count=args.steps, data_config=data_config, task_config=custom_tasks
    # )
    # config = ReductionConfig(levels=args.steps, branch_factor=2, data_config=data_config, task_config=custom_tasks)
    # config = StencilConfig(
    #     dimensions=1,
    #     width=args.width,
    #     steps=args.steps,
    #     data_config=data_config,
    #     task_config=custom_tasks,
    # )
    config = SweepConfig(
        width=args.width,
        steps=args.steps,
        dimensions=2,
        data_config=data_config,
        task_config=custom_tasks,
    )
    # config = SerialConfig(
    #     chains=1, steps=args.steps, data_config=data_config, task_config=custom_tasks
    # )
    # # config = MapReduceConfig(data_config=data_config)
    # config = ScatterReductionConfig(
    #     levels=args.steps,
    #     branch_factor=2,
    #     data_config=data_config,
    #     task_config=custom_tasks,
    # )
    tasks, data = make_graph(config)

    run_config = RunConfig(verbose=args.verbose)

    start_t = time.perf_counter()
    timing = run(tasks, data, run_config=run_config)
    end_t = time.perf_counter()
    print("Internal time:", timing)


test()
