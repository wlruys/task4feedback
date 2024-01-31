from task4feedback.simulator.datapool import *
from task4feedback.simulator.data import *
from task4feedback.types import *
from task4feedback.simulator.device import *
from task4feedback.simulator.topology import *

import pytest

from rich import print


def initialize_data(num_blocks=10):
    datalist = []
    size = 32 * 1024 * 1024  # 32 MB
    for i in range(num_blocks):
        device = Device(Architecture.CPU, 0)
        data = DataInfo(id=DataID(idx=i), size=size, location=device)
        devices = [device]
        simdata = SimulatedData(info=data, system_devices=devices)
        datalist.append(simdata)
    return datalist


def test_datapool():
    topology_manager = TopologyManager()
    topology = topology_manager.get_generator("frontera")(None)

    pool = DataPool(devices=topology.devices)
    datalist = initialize_data()
    cpu = Device(Architecture.CPU, 0)

    for data in datalist:
        pool.add_data(cpu, data, DataState.EVICTABLE, initial=True)
    for data in datalist:
        

    for i in range(10):
        data = pool.get_next_evictable(cpu)
        assert data.idx[0] == i, f"{data.idx} != {i}"


test_datapool()
