from typing import Optional
import warnings
from task4feedback.trip import (
    Devices,
    Topology,
    DeviceType,
)
from .types import (
    DeviceTuple,
    _bytes_to_readable,
)

class System:
    def __init__(self, fastest_flops=11e12, slowest_flops=11e12, gpu_flop=11e12, fastest_gmbw=443e9, slowest_gmbw=443e9):
        # Default specs are based on (the old) RTX5000s (Frontera).
        self.devices = Devices()
        self.topology = None
        self.slowest_bandwidth = float("inf")
        self.fastest_bandwidth = 0
        self.fastest_flops = fastest_flops
        self.slowest_flops = slowest_flops
        self.arch_to_flops = {
            DeviceType.CPU: 0,  # 0 GFLOPS for CPU (assume it cannot do work, this affects variant generation)
            DeviceType.GPU: gpu_flop,  # 11 TFLOPS
        }
        self.arch_to_gmbw = {
            DeviceType.CPU: 0,
            DeviceType.GPU: fastest_gmbw,  # 443 GB/s
        }
        self.fastest_gmbw = fastest_gmbw
        self.slowest_gmbw = slowest_gmbw
        self.arch_to_maxmem = {DeviceType.CPU: 0, DeviceType.GPU: 0}

    def create_device(self, name, arch, copy, memory, flops: Optional[int] = None, gmbw: Optional[int] = None):
        id = self.devices.append_device(name, arch, copy, memory)
        if flops is not None:
            self.fastest_flops = max(self.fastest_flops, flops)
            self.slowest_flops = min(self.slowest_flops, flops)
        self.arch_to_flops[arch] = flops if flops is not None else self.arch_to_flops.get(arch, 11e12)  # Default to 11 TFLOPs if not set
        self.arch_to_gmbw[arch] = gmbw if gmbw is not None else self.arch_to_gmbw.get(arch, 443e9)  # Default to 443 GB/s if not set
        self.arch_to_maxmem[arch] = max(self.arch_to_maxmem.get(arch, 0), memory)
        return DeviceTuple(name, id, self.devices.get_local_id(id), arch, memory)

    def finalize_devices(self):
        self.topology = Topology(self.devices.size())

    def get_global_id(self, architecture: DeviceType, local_id: int):
        return self.devices.get_global_id(architecture, local_id)

    def get_local_id(self, global_id: int):
        return self.devices.get_local_id(global_id)

    def get_type(self, global_id: int):
        return self.devices.get_type(global_id)

    def get_flops(self, architecture: DeviceType):
        return int(self.arch_to_flops.get(architecture, 1e9))  # Default to 1 GFLOPS if not set

    def get_flop_ms(self, architecture: DeviceType):
        flops = self.get_flops(architecture)
        return flops / 1e6

    def get_gmbw(self, architecture: DeviceType):
        return int(self.arch_to_gmbw.get(architecture, 1e9))

    def get_gmbw_ms(self, architecture: DeviceType):
        gmbw = self.get_gmbw(architecture)
        return gmbw / 1e6

    def get_device(self, global_id: int):
        local_id = self.get_local_id(global_id)
        name = self.devices.get_name(global_id)
        arch = self.devices.get_type(global_id)
        dev = self.devices.get_device(global_id)
        vcu = dev.get_vcu()
        memory = dev.get_mem()
        flops = self.arch_to_flops.get(arch, 1e9)
        return DeviceTuple(name, global_id, local_id, arch, memory, vcu, int(flops))

    def add_connection(self, s_gid, d_gid, bandwidth, latency, max_connections=2):
        if self.topology is None:
            # Fix: Raise RuntimeError instead of Warning
            raise RuntimeError("Devices must be finalized before adding connections. Call finalize_devices() first.")

        bandwidth = bandwidth / 1e6  # Convert to per microsecond
        self.slowest_bandwidth = min(self.slowest_bandwidth, bandwidth)
        self.fastest_bandwidth = max(self.fastest_bandwidth, bandwidth)
        self.topology.set_bandwidth(s_gid, d_gid, int(bandwidth))
        self.topology.set_latency(s_gid, d_gid, latency)
        self.topology.set_max_connections(s_gid, d_gid, max_connections)

    def __str__(self):
        result = []

        device_count = self.devices.size()
        result.append(f"SystemWrapper with {device_count} devices:")

        for i in range(device_count):
            device = self.get_device(i)
            result.append(f"  Device {device.global_id}: {device.name} (Type: {device.arch}, Memory: {_bytes_to_readable(device.memory)}, VCU: {device.vcu})")

        return "\n".join(result)

    def __len__(self):
        return self.devices.size()

    def connection_table(self):
        if self.topology is None:
            return "No topology defined. Call finalize_devices() first."

        result = []
        result.append("-" * 80)
        result.append(f"{'Source':<20} {'Dest':<20} {'BW':<15} {'L':<15} {'C':<15}")
        result.append("-" * 80)

        device_count = self.devices.size()

        for src in range(device_count):
            src_device = self.get_device(src)
            for dst in range(device_count):
                if src == dst:
                    continue

                if self.topology.get_bandwidth(src, dst) > 0:
                    dst_device = self.get_device(dst)
                    bandwidth = self.topology.get_bandwidth(src, dst)
                    latency = self.topology.get_latency(src, dst)
                    max_conn = self.topology.get_max_connections(src, dst)

                    bandwidth_str = _bytes_to_readable(bandwidth) + "/s" if bandwidth > 0 else "N/A"

                    result.append(
                        f"{src_device.name} (ID: {src})".ljust(20) + f"{dst_device.name} (ID: {dst})".ljust(20) + f"{bandwidth_str}".ljust(15) + f"{latency} ms".ljust(15) + f"{max_conn}".ljust(15)
                    )

        if len(result) == 3:  # Only header rows present
            result.append("No connections found between devices.")

        return "\n".join(result)

def uniform_connected_devices(
    n_devices: int, mem: int | float, latency: int, h2d_bw: int, d2d_bw: int, h2d_links: int = 2, d2d_links: int = 2, cpu_copyengines: int = 2, device_copyengines: int = 4, system_specs: dict = None
) -> System:
    """
    Creates a system with a uniform connection of devices including one CPU and multiple GPUs.
    """
    assert n_devices > 1
    if system_specs is None:
        s = System()
    else:
        s = System(**system_specs)
    n_gpus = n_devices - 1

    s.create_device("CPU:0", DeviceType.CPU, cpu_copyengines, int(2**62))
    for i in range(n_gpus):
        s.create_device(f"GPU:{i}", DeviceType.GPU, device_copyengines, int(mem) if mem != float("inf") else int(2**62))

    s.finalize_devices()

    for i in range(n_gpus):
        s.add_connection(0, i + 1, h2d_bw, latency, max_connections=h2d_links)
        s.add_connection(i + 1, 0, h2d_bw, latency, max_connections=h2d_links)

    for i in range(n_gpus):
        for j in range(n_gpus):
            if i == j:
                continue
            s.add_connection(i + 1, j + 1, d2d_bw, latency, max_connections=d2d_links)
            s.add_connection(j + 1, i + 1, d2d_bw, latency, max_connections=d2d_links)

    return s
