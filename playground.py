from task4feedback.graphs.mesh.base import *
from task4feedback.graphs.mesh.partition import *
from task4feedback.graphs.mesh.plot import *
from task4feedback.graphs.base import *
from task4feedback.graphs.jacobi import JacobiGraph, JacobiVariantGPUOnly, JacobiConfig
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *

from task4feedback.ml.ppo import *
import task4feedback.fastsim2 as fastsim
from task4feedback.interface.wrappers import (
    SimulatorFactory,
    create_graph_spec,
)
from torchrl.envs import StepCounter, TrajCounter, TransformedEnv, Compose
from task4feedback.ml.models import *
from itertools import permutations
from torch import multiprocessing
from task4feedback.fastsim2 import (
    EventType,
    ExecutionState,
    GraphExtractor,
)
from task4feedback.interface import start_logger

if __name__ == "__main__":
    start_logger()
    s = uniform_connected_devices(2, 1000000000, 0, int(1000))
    graph = Graph()

    graph.add_task("T0", 0)
    graph.graph.add_variant(
        0, DeviceType.GPU, 1000, 0, 1000
    )  # tag, device, vcus, memory, time
    graph.add_read_data(0, [0])

    graph.add_task("T1", 0)
    graph.graph.add_variant(
        1, DeviceType.CPU, 1000, 0, 1000
    )  # tag, device, vcus, memory, time

    graph.add_task("T2", 0)
    graph.graph.add_variant(
        2, DeviceType.GPU, 1000, 0, 1000
    )  # tag, device, vcus, memory, time
    graph.add_dependencies(2, [1])

    graph.add_task("T3", 0)
    graph.graph.add_variant(
        3, DeviceType.GPU, 1000, 0, 1000
    )  # tag, device, vcus, memory, time
    graph.add_dependencies(3, [2])

    graph.add_task("T4", 0)
    graph.graph.add_variant(
        4, DeviceType.GPU, 1000, 0, 1000
    )  # tag, device, vcus, memory, time
    graph.add_dependencies(4, [3])
    graph.add_read_data(4, [0])

    d = DataBlocks()
    d.add_block("B1", 7000 * 1000, 0)
    m = graph
    m.finalize_tasks()
    input = SimulatorInput(
        m,
        d,
        s,
        transition_conditions=fastsim.BatchTransitionConditions(2, 2, 2),
    )
    internal_mapper = fastsim.DequeueEFTMapper()
    print(type(internal_mapper))
    sim = fastsim.Simulator(input.to_input(), internal_mapper)
    sim.initialize()
    sim.initialize_data()
    sim.run()
    print(sim.get_current_time())
    exit()
