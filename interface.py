from task4feedback.fastsim.interface import (
    SimulatorHandler,
    uniform_connected_devices,
    TNoiseType,
    CMapperType,
    RoundRobinPythonMapper,
    Phase,
    PythonMapper,
    Action,
    start_logger,
    ExecutionState,
)
from typing import Optional, Self
from task4feedback.types import *
from task4feedback.graphs import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--blocks", type=int, default=3)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--vcus", type=int, default=1)
args = parser.parse_args()


def task_config(task_id: TaskID) -> TaskPlacementInfo:
    placement_info = TaskPlacementInfo()
    placement_info.add(
        (Device(Architecture.GPU, -1),),
        TaskRuntimeInfo(task_time=1000, device_fraction=args.vcus),
    )
    placement_info.add(
        (Device(Architecture.CPU, -1),),
        TaskRuntimeInfo(task_time=1000, device_fraction=args.vcus),
    )
    return placement_info


data_config = CholeskyDataGraphConfig(data_size=10000)
config = CholeskyConfig(blocks=args.blocks, task_config=task_config)
tasks, data = make_graph(config, data_config=data_config)

# 16GB in bytes
mem = 16 * 1024 * 1024 * 1024
bandwidth = 100
latency = 1
n_devices = args.devices
devices = uniform_connected_devices(n_devices, mem, latency, bandwidth)

# start_logger()

H = SimulatorHandler(tasks, data, devices, noise_type=TNoiseType.LOGNORMAL, seed=100)
sim = H.create_simulator()
sim.initialize(use_data=True)
sim.randomize_durations()
sim.set_python_mapper(RoundRobinPythonMapper(n_devices))
# sim.cmapper.set_mapping(
#     np.array([i % n_devices for i in range(len(tasks))], dtype=np.uint32)
# )
sim.enable_python_mapper()

samples = 10
for i in range(samples):
    current_sim = H.copy(sim)
    # current_sim.randomize_priorities()
    # current_sim.set_python_mapper(RoundRobinPythonMapper(n_devices))
    state = current_sim.run()
    print(i, state, current_sim.get_current_time())
