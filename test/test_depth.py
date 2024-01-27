from task4feedback.graphs import *
from task4feedback.simulator.preprocess import *
from rich import print

from rich.traceback import install


def test_serial_depth():
    length = 10
    chains = 10
    config = SerialConfig(steps=length, chains=chains)
    graph, data = make_graph(config)

    assert len(graph) == length * chains

    tasklist, taskmap = create_sim_graph(graph, data)

    for task in tasklist:
        assert task.task_idx[0] == taskmap[task].depth


def test_serial_depth_data():
    length = 4
    chains = 1
    config = SerialConfig(steps=length, chains=chains)
    data_config = ChainDataGraphConfig()
    graph, data = make_graph(config, data_config=data_config)

    # print(graph)

    assert len(graph) == length * chains

    tasklist, taskmap = create_sim_graph(graph, data, use_data=True)

    assert len(taskmap) != len(tasklist)

    for task in taskmap:
        if isinstance(taskmap[task], SimulatedDataTask):
            continue
        elif isinstance(taskmap[task], SimulatedComputeTask):
            assert (
                taskmap[task].depth == task.task_idx[0] * 2 + 1
            ), f"Compute Task {task} :: Depth {taskmap[task].depth}"


def test_reduction_depth():
    levels = 4
    branch = 2

    config = ReductionConfig(levels=levels, branch_factor=branch)

    graph, data = make_graph(config)

    assert len(graph) == (branch**levels - 1) / (branch - 1)

    tasklist, taskmap = create_sim_graph(graph, data)
    for task in tasklist:
        assert (
            task.task_idx[0] == levels - taskmap[task].depth - 1
        ), f"Task {task} :: Depth {levels - taskmap[task].depth - 1}"
