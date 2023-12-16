from rich import print

from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *
from task4feedback.graphs import *
from task4feedback.load import *
from task4feedback.execute import run


def test():
    data_config = NoDataGraphConfig()
    config = IndependentConfig(task_count=10, data_config=data_config)
    tasks, data = make_independent_graph(config)

    print(tasks)

    run(tasks, data)


test()
