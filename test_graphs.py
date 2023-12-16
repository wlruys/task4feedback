from task4feedback.graphs import *
from task4feedback.load import *

from rich import print

# from utility.execute import run
from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *


def run():
    cpu = Device(Architecture.CPU, 0)
    gpu = Device(Architecture.GPU, -1)

    gpu1 = Device(Architecture.GPU, 1)
    gpu2 = Device(Architecture.GPU, 2)

    # data_config = ChainDataGraphConfig(data_size=1000)
    # data_config = CholeskyDataGraphConfig(data_size=1000)
    # data_config = SweepDataGraphConfig(small_size=10, large_size=1000)
    # data_config = StencilDataGraphConfig(
    #    small_size=10, large_size=1000, dimensions=1, width=3, neighbor_distance=1
    # )

    data_config = NoDataGraphConfig()

    # config = CholeskyConfig(blocks=4, data_config=data_config)
    # config = ReductionConfig(levels=3, branch_factor=3, data_config=data_config)
    # config = StencilConfig(dimensions=1, width=3, steps=2, data_config=data_config)
    # config = SweepConfig(width=3, steps=2, dimensions=2, data_config=data_config)
    # config = SerialConfig(chains=2, steps=10, data_config=data_config)
    # config = MapReduceConfig(data_config=data_config)
    config = ScatterReductionConfig(levels=3, branch_factor=3, data_config=data_config)

    # tasks, data = make_sweep_graph(config)
    tasks, data = make_scatter_reduction_graph(config)

    # write_tasks_to_yaml(tasks, "graph")
    # write_data_to_yaml(data, "graph")

    tasklist, taskmap, datamap = read_graph("graph")

    populate_dependents(taskmap)

    networkx_graph, networkx_label = build_networkx_graph(taskmap)
    plot_pydot(networkx_graph)


run()
