from ..types import Architecture, Device, TaskID, DataID, DataInfo, ResourceType, Time
from typing import List, Dict, Set, Tuple, Optional, Callable, Sequence, Type
from .device import SimulatedDevice
from .resourceset import ResourceSet, FasterResourceSet
from dataclasses import dataclass, field, InitVar
from .utility import parse_size
import math
import numpy as np
from fractions import Fraction

NamedDevice = Device | SimulatedDevice


@dataclass(slots=True)
class ConnectionPool:
    host: SimulatedDevice
    devices: InitVar[Sequence[Device]]
    devices2index: Dict[Device, int] = field(init=False)
    connections: np.ndarray = field(init=False)
    active_connections: np.ndarray = field(init=False)
    bandwidth: np.ndarray = field(init=False)

    def __post_init__(self, devices: Sequence[Device]):
        """
        Create a connection pool for a set of devices.
        """
        self.devices2index = {}
        for i, device in enumerate(devices):
            self.devices2index[device] = i

        self.connections = np.zeros((len(devices), len(devices)), dtype=np.bool_)
        self.active_connections = np.zeros(
            (len(devices), len(devices)), dtype=np.uint32
        )
        self.bandwidth = np.zeros((len(devices), len(devices)), dtype=np.float32)

    def get_index(self, device: NamedDevice) -> int:
        if isinstance(device, SimulatedDevice):
            device = device.name
        return self.devices2index[device]

    def get_indicies(self, devices: Sequence[NamedDevice]) -> Sequence[int]:
        return [self.get_index(device) for device in devices]

    def check_connection_exists(self, source: NamedDevice, target: NamedDevice):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        return self.connections[source_idx, target_idx]

    def check_bandwidth_exists(self, source: NamedDevice, target: NamedDevice):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        return self.bandwidth[source_idx, target_idx] > 0

    def get_bandwidth(self, source: NamedDevice, target: NamedDevice):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        return self.bandwidth[source_idx, target_idx]

    def count_active_connections(
        self,
        source: NamedDevice,
        target: Optional[NamedDevice] = None,
        verbose: bool = False,
    ):
        source_idx = self.get_index(source)

        if target is not None:
            target_idx = self.get_index(target)
            return self.active_connections[source_idx, target_idx]
        else:
            return np.sum(self.active_connections[source_idx, :])

    def add_bandwidth(
        self,
        source: NamedDevice,
        target: NamedDevice,
        bandwidth: float,
        bidirectional: bool = True,
    ):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)

        self.bandwidth[source_idx, target_idx] = bandwidth

        if bidirectional:
            self.bandwidth[target_idx, source_idx] = bandwidth

    def add_connection(
        self, source: NamedDevice, target: NamedDevice, bidirectional: bool = True
    ):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)

        self.connections[source_idx, target_idx] = True

        if bidirectional:
            self.connections[target_idx, source_idx] = True

    def update_connection_usage(
        self,
        source: NamedDevice,
        target: NamedDevice,
        value: int,
        bidirectional: bool = False,
        verbose: bool = False,
    ) -> bool:
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)

        if source_idx == target_idx:
            # No connections needed
            return False

        if isinstance(source, SimulatedDevice):
            source = source.name
        if isinstance(target, SimulatedDevice):
            target = target.name

        self.active_connections[source_idx, target_idx] += value

        if bidirectional:
            self.active_connections[source_idx, target_idx] += value

        if self.connections[source_idx, target_idx] <= 0:
            # If no direct connection, route through the host device
            host_idx = self.get_index(self.host)

            self.active_connections[source_idx, host_idx] += value
            self.active_connections[host_idx, target_idx] += value

            if bidirectional:
                self.active_connections[target_idx, host_idx] += value
                self.active_connections[host_idx, source_idx] += value

        return True

    def acquire_connection(
        self, src: NamedDevice, dst: NamedDevice, verbose: bool = False
    ):
        self.update_connection_usage(src, dst, 1, verbose=verbose)

    def release_connection(
        self, src: NamedDevice, dst: NamedDevice, verbose: bool = False
    ):
        self.update_connection_usage(src, dst, -1, verbose=verbose)

    def check_connection_available(
        self,
        source: SimulatedDevice,
        target: SimulatedDevice,
        require_copy_engines: bool = True,
        require_symmetric=True,
        verbose: bool = False,
    ) -> bool:
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)

        if source_idx == target_idx:
            # No connection needed for a self copy
            return True

        # Is there a direct connection?
        direct_connection = self.check_connection_exists(source, target)

        # Check if copy engines are available (if required)
        if require_copy_engines:
            if self.count_active_connections(source) >= source.resources.copy:
                return False
            if (
                require_symmetric
                and self.count_active_connections(target) >= target.resources.copy
            ):
                return False

            if not direct_connection:
                if not self.check_connection_exists(source, self.host):
                    return False
                if not self.check_connection_exists(target, self.host):
                    return False

                if (
                    self.count_active_connections(self.host)
                    >= self.host[ResourceType.COPY]
                ):
                    return False

        return True

    def sort_by_bandwidth(
        self, target: NamedDevice, devices: Sequence[NamedDevice]
    ) -> Sequence[NamedDevice]:
        """
        Return a sorted list of devices by the bandwidth of the connection to the target
        """
        target_idx = self.get_index(target)
        bandwidths = self.bandwidth[target_idx, self.get_indicies(devices)]
        return [devices[i] for i in np.argsort(bandwidths)]

    def get_transfer_time(
        self, source: NamedDevice, target: NamedDevice, data_size: int
    ) -> Time:
        if source == target:
            return Time(0)

        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        bandwidth = self.bandwidth[source_idx, target_idx]
        if bandwidth <= 0:
            # If no direct connection, route through the host device
            host_idx = self.get_index(self.host)
            first_hop_bandwidth = self.bandwidth[source_idx, host_idx]
            second_hop_bandwidth = self.bandwidth[host_idx, target_idx]
            if first_hop_bandwidth <= 0 or second_hop_bandwidth <= 0:
                raise ValueError(
                    f"No connection between {source} and {target} or through the host"
                )
            time_in_seconds = (
                data_size / first_hop_bandwidth + data_size / second_hop_bandwidth
            )
            time_in_microseconds = int(time_in_seconds * 1e6)
            # print(
            #     f"Route through host: {data_size} bytes, {first_hop_bandwidth} -> {second_hop_bandwidth} = {time_in_microseconds} us"
            # )
        else:
            time_in_seconds = data_size / bandwidth
            time_in_microseconds = int(time_in_seconds * 1e6)
            # print(
            #     f"Direct connection: {data_size} bytes, {bandwidth} = {time_in_microseconds} us"
            # )
        return Time(time_in_microseconds)

    def get_connection_string(self, source: NamedDevice, target: NamedDevice) -> str:
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        return f"{source} -> {target} (bw={self.bandwidth[source_idx, target_idx]}, active={self.active_connections[source_idx, target_idx]})"

    def __str__(self) -> str:
        s = "ConnectionPool:\n"
        for source in self.devices2index.keys():
            for target in self.devices2index.keys():
                if self.check_connection_exists(source, target):
                    s += self.get_connection_string(source, target) + "\n"
        return s

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(slots=True)
class SimulatedTopology:
    devices: List[SimulatedDevice]
    name: str = "SimulatedTopology"
    connection_pool: ConnectionPool = field(init=False)

    def __post_init__(self):
        """
        Create a simulated topology.
        Assumes that the first device is the host device.
        """
        device_names = [device.name for device in self.devices]
        self.connection_pool = ConnectionPool(
            host=self.devices[0], devices=device_names
        )

    def add_connection(
        self, source: NamedDevice, target: NamedDevice, bidirectional: bool = True
    ):
        self.connection_pool.add_connection(source, target, bidirectional)

    def add_bandwidth(
        self,
        source: NamedDevice,
        target: NamedDevice,
        bandwidth: float,
        bidirectional: bool = True,
    ):
        self.connection_pool.add_bandwidth(source, target, bandwidth, bidirectional)

    def check_connection_exists(self, source: NamedDevice, target: NamedDevice):
        return self.connection_pool.check_connection_exists(source, target)

    def check_bandwidth_exists(self, source: NamedDevice, target: NamedDevice):
        return self.connection_pool.check_bandwidth_exists(source, target)

    def get_bandwidth(self, source: NamedDevice, target: NamedDevice):
        return self.connection_pool.get_bandwidth(source, target)

    def count_active_connections(
        self, source: NamedDevice, target: Optional[NamedDevice] = None
    ):
        return self.connection_pool.count_active_connections(source, target)

    def acquire_connection(
        self, src: NamedDevice, dst: NamedDevice, verbose: bool = False
    ):
        self.connection_pool.acquire_connection(src, dst, verbose)

    def release_connection(
        self, src: NamedDevice, dst: NamedDevice, verbose: bool = False
    ):
        self.connection_pool.release_connection(src, dst, verbose)

    def check_connection_available(
        self,
        source: SimulatedDevice,
        target: SimulatedDevice,
        require_copy_engines: bool = True,
        require_symmetric=True,
        verbose: bool = False,
    ) -> bool:
        return self.connection_pool.check_connection_available(
            source, target, require_copy_engines, require_symmetric, verbose
        )

    def nearest_valid_connection(
        self,
        target: SimulatedDevice,
        sources: Sequence[SimulatedDevice],
        require_copy_engines: bool = True,
        require_symmetric=True,
    ) -> Optional[NamedDevice]:
        sorted_sources = self.connection_pool.sort_by_bandwidth(target, sources)
        for source in sorted_sources:
            assert isinstance(source, SimulatedDevice)

            if self.check_connection_available(
                source, target, require_copy_engines, require_symmetric
            ):
                return source
        return None

    def get_devices(self, device_type: Architecture) -> List[SimulatedDevice]:
        return [
            device for device in self.devices if device.name.architecture == device_type
        ]

    def get_device_string(self, device: SimulatedDevice) -> str:
        return f"{device} (mem={device[ResourceType.MEMORY]})"

    def get_transfer_time(
        self, source: NamedDevice, target: NamedDevice, data_size: int
    ) -> Time:
        return self.connection_pool.get_transfer_time(source, target, data_size)

    def __str__(self) -> str:
        s = f"Topology: {self.name}\n"
        for device in self.devices:
            s += self.get_device_string(device) + "\n"
        s += str(self.connection_pool)
        return s

    def __repr__(self) -> str:
        return self.__str__()


class TopologyManager:
    generator_map: Dict[str, Callable[[Optional[Dict]], SimulatedTopology]] = {}

    @staticmethod
    def read_from_yaml(topology_name: str) -> SimulatedTopology:
        """
        Read topology from a YAML file.
        """
        raise NotImplementedError

    @staticmethod
    def register_generator(topology_name: str):
        """
        Register a topology generator.
        """

        def decorator(cls):
            if topology_name in TopologyManager.generator_map:
                raise ValueError(
                    f"Topology {topology_name} has already been registered."
                )
            TopologyManager.generator_map[topology_name] = cls
            return cls

        return decorator

    @staticmethod
    def get_generator(
        name: str,
    ) -> Callable[[Optional[Dict]], SimulatedTopology]:
        """
        Get a topology generator.
        """
        if name not in TopologyManager.generator_map:
            raise ValueError(f"Topology {name} is not registered.")
        return TopologyManager.generator_map[name]

    @staticmethod
    def generate(
        name: str, config: Optional[Dict[str, int]] = None
    ) -> SimulatedTopology:
        """
        Generate a topology.
        """
        generator = TopologyManager.get_generator(name)
        return generator(config)


@TopologyManager.register_generator("frontera")
def generate_ngpus_1cpu_toplogy(
    config: Optional[Dict[str, int]] = None
) -> SimulatedTopology:
    """
    This function creates 4 GPUs and 1 CPU architecture.

    The topology looks like below:

    gpu0 - gpu1
     | \   / |
     |  \ /  |
     |  / \  |
     | /   \ |
    gpu2 - gpu3

    gpu0-gpu1 and gpu2-gpu3 have bandwidth of 200 (we assume NVLinks),
    and other connections have bandiwdth of 100.

    All GPUs are connected to CPU by connections having bandwidth of 100.
    Each GPU is equipped with 16GB DRAM, and CPU is equipped with 130GB.
    """

    default_config = {
        "P2P_BW": parse_size("200 GB"),
        "H2D_BW": parse_size("100 GB"),
        "D2H_BW": parse_size("100 GB"),
        "GPU_MEM": parse_size("16 GB"),
        "CPU_MEM": parse_size("130 GB"),
        "GPU_COPY_ENGINES": 3,
        "CPU_COPY_ENGINES": 3,
        "NGPUS": 4,
    }

    if config is not None:
        for key, value in default_config.items():
            if key in config:
                default_config[key] = config[key]

    config = default_config

    P2P_BW = config["P2P_BW"]
    H2D_BW = config["H2D_BW"]
    D2H_BW = config["D2H_BW"]

    GPU_MEM = config["GPU_MEM"]
    CPU_MEM = config["CPU_MEM"]

    GPU_COPY_ENGINES = config["GPU_COPY_ENGINES"]
    CPU_COPY_ENGINES = config["CPU_COPY_ENGINES"]
    NGPUS = config["NGPUS"]

    # Create devices
    gpus = [
        SimulatedDevice(
            Device(Architecture.GPU, i), FasterResourceSet(1, GPU_MEM, GPU_COPY_ENGINES)
        )
        for i in range(4)
    ]
    cpus = [
        SimulatedDevice(
            Device(Architecture.CPU, 0), FasterResourceSet(1, CPU_MEM, CPU_COPY_ENGINES)
        )
    ]

    # Create device topology
    topology = SimulatedTopology(cpus + gpus, "Topology::4G-1C")

    for gpu in gpus:
        topology.add_connection(gpu, cpus[0], bidirectional=True)
        topology.add_bandwidth(gpu, cpus[0], D2H_BW)
        topology.add_bandwidth(cpus[0], gpu, H2D_BW)

    for i in range(NGPUS):
        for j in range(i + 1, NGPUS):
            topology.add_connection(gpus[i], gpus[j], bidirectional=True)
            topology.add_bandwidth(gpus[i], gpus[j], P2P_BW)

    # topology.add_connection(gpus[0], gpus[1], bidirectional=True)
    # topology.add_bandwidth(gpus[0], gpus[1], P2P_BW)

    # topology.add_connection(gpus[2], gpus[3], bidirectional=True)
    # topology.add_bandwidth(gpus[2], gpus[3], P2P_BW)

    return topology

@TopologyManager.register_generator("mesh")
def generate_mesh_toplogy(
    config: Optional[Dict[str, int]] = None
) -> SimulatedTopology:
    """
    This function creates n GPUs and 1 CPU architecture.

    The topology looks like below:

     -----------
    |           |
    -gpu0 --gpu2-
     | \   / |
     |  \ /  |
     |  / \  |
     | /   \ |
    -gpu1 -- gpu3-
    |           |
     -----------
   
    gpu0-gpu1 and gpu2-gpu3 have bandwidth of 200 (we assume NVLinks),
    and other connections have bandiwdth of 100.

    All GPUs are connected to CPU by connections having bandwidth of 100.
    Each GPU is equipped with 16GB DRAM, and CPU is equipped with 130GB.
    """
    energy = 0.01
    if config is not None:
        n = config["N"]
        # P2P_BW = config["P2P_BW"]
        # H2D_BW = config["H2D_BW"]
        # D2H_BW = config["D2H_BW"]

        # GPU_MEM = config["GPU_MEM"]
        # CPU_MEM = config["CPU_MEM"]

        # GPU_COPY_ENGINES = config["GPU_COPY_ENGINES"]
        # CPU_COPY_ENGINES = config["CPU_COPY_ENGINES"]
        # Default configuration for testing
        P2P_BW = parse_size("9 GB")  # 9 GB/s
        H2D_BW = parse_size("7 GB")  # 7 GB/s
        D2H_BW = parse_size("7 GB")  # 7 GB/s

        GPU_MEM = parse_size("6 GB")
        CPU_MEM = parse_size("130 GB")

        GPU_COPY_ENGINES = 3
        CPU_COPY_ENGINES = 3

    # Create devices
    gpus = [
        SimulatedDevice(
            Device(Architecture.GPU, i, energy), FasterResourceSet(1, GPU_MEM, GPU_COPY_ENGINES)
        )
        for i in range(n)
    ]
    cpus = [
        SimulatedDevice(
            Device(Architecture.CPU, 0, energy), FasterResourceSet(1, CPU_MEM, CPU_COPY_ENGINES)
        )
    ]

    # Create device topology
    topology = SimulatedTopology(cpus + gpus, "Topology::4G-1C")

    # add connections between all gpus and the cpu
    for gpu in gpus:
        # print(gpu)
        topology.add_connection(gpu, cpus[0], bidirectional=True)
        topology.add_bandwidth(gpu, cpus[0], D2H_BW)
        topology.add_bandwidth(cpus[0], gpu, H2D_BW)

    step = int(math.sqrt(n))

    # add connections between cols
    for i in range (step):
        mod = step * (i + 1)
        for j in range(step):
            curr_device = i * step + j
            if( j != step - 1):
                next_device = (curr_device + 1) % mod
            else:
                next_device = i * step
            # print("Adding col conn:")
            # print(curr_device, "-->", next_device)
            topology.add_connection(gpus[curr_device], gpus[next_device], bidirectional=True)
            topology.add_bandwidth(gpus[curr_device], gpus[next_device], P2P_BW)

    # add connection between rows
    for j in range(step):
        for i in range(step):
            curr_device = j + i * step
            next_device = (j + (i + 1) * step) % n
            # print("Adding row conn:")
            # print(curr_device, "-->", next_device)
            topology.add_connection(gpus[curr_device], gpus[next_device], bidirectional=True)
            topology.add_bandwidth(gpus[curr_device], gpus[next_device], P2P_BW)

    return topology

def generate_mesh(topology, gpus, bandwidth, start, step, n):
    # add connections between cols
    # print("STEP:", step)
    for i in range (step):
        mod = step * (i + 1) + start
        for j in range(step):
            curr_device = i * step + j + start
            if( j != step - 1):
                next_device = (curr_device + 1) % mod
            else:
                next_device = i * step + start
            #print("Adding col conn:")
            #print(curr_device, "-->", next_device)
            topology.add_connection(gpus[curr_device], gpus[next_device], bidirectional=True)
            topology.add_bandwidth(gpus[curr_device], gpus[next_device], bandwidth)

    # add connection between rows
    for j in range(step):
        for i in range(step):
            curr_device = j + i * step + start
            next_device = (j + (i + 1) * step) % n + start
            #print("Adding row conn:")
            #print(curr_device, "-->", next_device)
            topology.add_connection(gpus[curr_device], gpus[next_device], bidirectional=True)
            topology.add_bandwidth(gpus[curr_device], gpus[next_device], bandwidth)

    return topology


@TopologyManager.register_generator("imec")
def generate_imec_topology(
    config: Optional[Dict[str, int]] = None
) -> SimulatedTopology:
    """
    This function creates 2-level architecture. First level as 1 GPU. 
    The second level has 90 * 90 GPUs in mesh topology.
    """
    energy = [0.01, 1, 100, 1000]
    total_n = 1801
    n = [1, 1800]
    dram = False

    # if "N" in config:
    #     n = config["N"]

    if config is not None:
        n = config["N"]
        total_n = config["TOTAL_N"]
        hier_levels = config["HIER_LEVELS"]
        energy = config["ENERGY"]
        if "DRAM" in config:
            dram = config["DRAM"]
        P2P_BW = config["P2P_BW"]
        H2D_BW = config["H2D_BW"]
        D2H_BW = config["D2H_BW"]

        GPU_MEM = config["GPU_MEM"]
        CPU_MEM = config["CPU_MEM"]

        GPU_COPY_ENGINES = config["GPU_COPY_ENGINES"]
        CPU_COPY_ENGINES = config["CPU_COPY_ENGINES"]
        # Default configuration for testing
    P2P_BW = parse_size("9 GB")  # 9 GB/s
    H2D_BW = parse_size("7 GB")  # 7 GB/s
    D2H_BW = parse_size("7 GB")  # 7 GB/s

    GPU_MEM = parse_size("10 GB")
    CPU_MEM = parse_size("150 GB")

    GPU_COPY_ENGINES = 3
    CPU_COPY_ENGINES = 3

    cpus = [
        SimulatedDevice(
            Device(Architecture.CPU, 0, energy[0]), FasterResourceSet(1, CPU_MEM, CPU_COPY_ENGINES)
        )
    ]

    gpus_1 = [
        SimulatedDevice(
            Device(Architecture.GPU, i, energy[0]), FasterResourceSet(1, GPU_MEM, GPU_COPY_ENGINES)
        )
        for i in range(n[0])
    ]

    gpus_2 = [
        SimulatedDevice(
            Device(Architecture.GPU, i, energy[0]), FasterResourceSet(1, GPU_MEM, GPU_COPY_ENGINES)
        )
        for i in range(n[1])
    ]
    # print("topo ", cpus)
    #print(total_n)
    # Create device topology
    
    topology = SimulatedTopology(cpus + gpus_1 + gpus_2, "Topology::G-C")

    # add connections between all gpus and the cpu
    for gpu in gpus_2:
        # print(gpu)
        topology.add_connection(gpu, cpus[0], bidirectional=True)
        topology.add_bandwidth(gpu, cpus[0], D2H_BW)
        topology.add_bandwidth(cpus[0], gpu, H2D_BW)

    # add connection between level1 GPU and level2 GPUs
    for gpu_1 in gpus_1:
        for gpu_2 in gpus_2:
            topology.add_connection(gpu_1, gpu_2, bidirectional=True)
            topology.add_bandwidth(gpu_1, gpu_2, P2P_BW)

    # add connection between gpus and dram at each level, if dram is True
    # if(dram):
    #     start = 0
    #     end = 0
    #     # print(len(gpus))
    #     for i in range(len(dram_cpus)):
    #         idx = hier_levels - i - 1
    #         num_gpus = int(pow(n, i + 1))
    #         end += num_gpus
    #         # print("I: ", i, "num_gpus: ", num_gpus)
    #         for j in range(start, end):
    #             # print(j)
    #             topology.add_connection(gpus[j], dram_cpus[idx], bidirectional=True)
    #             topology.add_bandwidth(gpus[j], dram_cpus[idx], D2H_BW)
    #             topology.add_bandwidth(dram_cpus[idx], gpus[j], H2D_BW)
    #         start = end

    step = int(math.sqrt(n[1]))
    topology = generate_mesh(topology, gpus_2, P2P_BW, 0, step, n[1])
    return topology



@TopologyManager.register_generator("mesh_hier")
def generate_mesh_toplogy_hier(
    config: Optional[Dict[str, int]] = None
) -> SimulatedTopology:
    """
    This function creates n GPUs and 1 CPU architecture.

    The topology looks like below:

     -----------
    |           |
    -gpu0 --gpu2-
     | \   / |
     |  \ /  |
     |  / \  |
     | /   \ |
    -gpu1 -- gpu3-
    |           |
     -----------
   
    gpu0-gpu1 and gpu2-gpu3 have bandwidth of 200 (we assume NVLinks),
    and other connections have bandiwdth of 100.

    All GPUs are connected to CPU by connections having bandwidth of 100.
    Each GPU is equipped with 16GB DRAM, and CPU is equipped with 130GB.
    """
    energy = [0.01, 1, 100, 1000]
    n = 4
    total_n = 20
    hier_levels = 2
    dram = False

    if "N" in config:
        n = config["N"]

    if config is not None:
        n = config["N"]
        total_n = config["TOTAL_N"]
        hier_levels = config["HIER_LEVELS"]
        energy = config["ENERGY"]
        # P2P_BW = config["P2P_BW"]
        # H2D_BW = config["H2D_BW"]
        # D2H_BW = config["D2H_BW"]

        # GPU_MEM = config["GPU_MEM"]
        # CPU_MEM = config["CPU_MEM"]

        # GPU_COPY_ENGINES = config["GPU_COPY_ENGINES"]
        # CPU_COPY_ENGINES = config["CPU_COPY_ENGINES"]
        # Default configuration for testing
        P2P_BW = parse_size("9 GB")  # 9 GB/s
        H2D_BW = parse_size("7 GB")  # 7 GB/s
        D2H_BW = parse_size("7 GB")  # 7 GB/s

        GPU_MEM = parse_size("10 GB")
        CPU_MEM = parse_size("150 GB")

        GPU_COPY_ENGINES = 3
        CPU_COPY_ENGINES = 3

  
    if "DRAM" in config:
        dram = config["DRAM"]
    
    # Create devices
    if dram:
        dram_cpus = []
        for i in range(hier_levels):
            print("topo: ", i + 1, " ", energy[i])
            dram_cpus.append(SimulatedDevice(
                            Device(Architecture.CPU, i + 1, energy[i]), FasterResourceSet(1, CPU_MEM, CPU_COPY_ENGINES)
                        )   
                    )
        # print("topo: ", dram_cpus)
    gpus = []
    end = 0
    start = 0
    for i in range(hier_levels):
        idx = hier_levels - i - 1
        num_gpus = int(pow(n, i + 1))
        end += num_gpus
        for j in range(start, end):
            gpus.append(SimulatedDevice(
                            Device(Architecture.GPU, j, energy[idx]), FasterResourceSet(1, GPU_MEM, GPU_COPY_ENGINES)
                        )
                    )
            # print("topo: ", j, " ", energy[idx])
            #gpus.append(Device(Architecture.GPU, i))
            #gpus.append(Device(Architecture.GPU, i, energy[idx]))
        start = end

    cpus = [
        SimulatedDevice(
            Device(Architecture.CPU, 0, energy[0]), FasterResourceSet(1, CPU_MEM, CPU_COPY_ENGINES)
        )
    ]
    # print("topo ", cpus)
    #print(total_n)
    # Create device topology
    if dram:
        topology = SimulatedTopology(cpus + dram_cpus + gpus, "Topology::G-D-C")
    else:
        topology = SimulatedTopology(cpus + gpus, "Topology::G-C")
    # add connections between all gpus and the cpu
    for gpu in gpus:
        # print(gpu)
        topology.add_connection(gpu, cpus[0], bidirectional=True)
        topology.add_bandwidth(gpu, cpus[0], D2H_BW)
        topology.add_bandwidth(cpus[0], gpu, H2D_BW)

    # add connection between gpus and dram at each level, if dram is True
    if(dram):
        start = 0
        end = 0
        # print(len(gpus))
        for i in range(len(dram_cpus)):
            idx = hier_levels - i - 1
            num_gpus = int(pow(n, i + 1))
            end += num_gpus
            # print("I: ", i, "num_gpus: ", num_gpus)
            for j in range(start, end):
                # print(j)
                topology.add_connection(gpus[j], dram_cpus[idx], bidirectional=True)
                topology.add_bandwidth(gpus[j], dram_cpus[idx], D2H_BW)
                topology.add_bandwidth(dram_cpus[idx], gpus[j], H2D_BW)
            start = end

    gpu = 0
    mesh = 0
    step = int(math.sqrt(n))
    while(gpu < total_n):
        #print("MESH: ", mesh)
        topology = generate_mesh(topology, gpus, P2P_BW, mesh * n, step, n)
        mesh += 1
        gpu += n
    return topology

@TopologyManager.register_generator("imec_hier")
def generate_imec_toplogy_hier(
    config: Optional[Dict[str, int]] = None
) -> SimulatedTopology:
    """
    This function creates n GPUs and 1 CPU architecture.

    The topology looks like below:

     -----------
    |           |
    -gpu0 --gpu2-
     | \   / |
     |  \ /  |
     |  / \  |
     | /   \ |
    -gpu1 -- gpu3-
    |           |
     -----------
   
    gpu0-gpu1 and gpu2-gpu3 have bandwidth of 200 (we assume NVLinks),
    and other connections have bandiwdth of 100.

    All GPUs are connected to CPU by connections having bandwidth of 100.
    Each GPU is equipped with 16GB DRAM, and CPU is equipped with 130GB.
    """
    energy = [0.2, 0.2]
    
    hier_levels = 2
    # total_n = 8200
    # num_gpus = [100, 8100]
    # p_per_mesh = [100, 81]
    total_n = 40
    num_gpus = [4, 36]
    p_per_mesh = [4, 9]
    
    if config is not None:
        if "N" in config:
            n = config["N"]
        total_n = config["TOTAL_N"]
        hier_levels = config["HIER_LEVELS"]
        energy = config["ENERGY"]
        # P2P_BW = config["P2P_BW"]
        # H2D_BW = config["H2D_BW"]
        # D2H_BW = config["D2H_BW"]

        # GPU_MEM = config["GPU_MEM"]
        # CPU_MEM = config["CPU_MEM"]

        # GPU_COPY_ENGINES = config["GPU_COPY_ENGINES"]
        # CPU_COPY_ENGINES = config["CPU_COPY_ENGINES"]
        # Default configuration for testing
    P2P_BW = parse_size("9 GB")  # 9 GB/s
    H2D_BW = parse_size("7 GB")  # 7 GB/s
    D2H_BW = parse_size("7 GB")  # 7 GB/s

    GPU_MEM = parse_size("10 GB")
    CPU_MEM = parse_size("150 GB")

    GPU_COPY_ENGINES = 3
    CPU_COPY_ENGINES = 3

    gpus = []
    end = 0
    start = 0
    for i in range(hier_levels):
        idx = hier_levels - i - 1
        end += num_gpus[i]
        for j in range(start, end):
            gpus.append(SimulatedDevice(
                            Device(Architecture.GPU, j, energy[idx]), FasterResourceSet(1, GPU_MEM, GPU_COPY_ENGINES)
                        )
                    )
            # print("topo: ", j, " ", energy[idx])
            #gpus.append(Device(Architecture.GPU, i))
            #gpus.append(Device(Architecture.GPU, i, energy[idx]))
        start = end

    gpus.append(SimulatedDevice(
                            Device(Architecture.GPU, total_n, 4.24), FasterResourceSet(1, GPU_MEM, GPU_COPY_ENGINES)
                        )
                    )
    cpus = [
        SimulatedDevice(
            Device(Architecture.CPU, 0, energy[0]), FasterResourceSet(1, CPU_MEM, CPU_COPY_ENGINES)
        )
    ]

    topology = SimulatedTopology(cpus + gpus, "Topology::G-C")
    # add connections between all gpus and the cpu
    for gpu in gpus:
        # print(gpu)
        topology.add_connection(gpu, cpus[0], bidirectional=True)
        topology.add_bandwidth(gpu, cpus[0], D2H_BW)
        topology.add_bandwidth(cpus[0], gpu, H2D_BW)
    
    # add connections between GPU[total_n] (HBM) and other GPUs
    for gpu in gpus[0:-2]:
        topology.add_connection(gpu, gpus[-1], bidirectional=True)
        topology.add_bandwidth(gpus[-1], gpu, P2P_BW)

    gpu = 0
    mesh = 0
    
    while(gpu < total_n):
        idx = 0 if gpu == 0 else 1 # because only 1 10 * 10 mesh
        step = int(math.sqrt(p_per_mesh[idx]))
        #print("MESH: ", mesh)
        topology = generate_mesh(topology, gpus, P2P_BW, gpu, step, p_per_mesh[idx])
        mesh += 1
        gpu += p_per_mesh[idx]
    return topology