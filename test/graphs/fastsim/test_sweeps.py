import argparse

from task4feedback.graphs import *

from json import load
from task4feedback.types import *

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
)

from task4feedback.simulator.utility import parse_size

parser = argparse.ArgumentParser(prog="Sweep")

parser.add_argument("-t", "--time", type=int, help="time", default=4000)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    help="testing, training, parla, heft, loadbalance, eft_with_data, eft_without_data, random",
)
parser.add_argument(
    "-n", "--noise", help="Set if task duration noise is enabled", action="store_true"
)
parser.add_argument(
    "-ns", "--noise_scale", type=float, help="task duration noise scale", default=0.2
)
parser.add_argument(
    "-e",
    "--episode",
    type=int,
    help="the number of episodes (-1 for inifite loop)",
    default=-1,
)
parser.add_argument("-s", "--steps", type=int, help="sweep steps", default=10)
parser.add_argument("-w", "--width", type=int, help="sweep width", default=10)
parser.add_argument("-dm", "--dimensions", type=int, help="sweep dimensions", default=1)
parser.add_argument(
    "-i",
    "--ignore_initial_placement",
    help="ignore initial placement during HEFT calculation",
    action="store_true",
)
parser.add_argument(
    "-o",
    "--sort",
    type=str,
    help="task sorting method (random, heft, default)",
    default="default",
)
parser.add_argument(
    "-si",
    "--sorting_interval",
    type=int,
    help="task random sorting interval",
    default=0,
)
parser.add_argument(
    "-so",
    "--save_order",
    help="save task mapping order (saved in replay.order)",
    action="store_true",
)
parser.add_argument(
    "-lo",
    "--load_order",
    help="load task mapping order (saved in replay.order)",
    action="store_true",
)
parser.add_argument(
    "-sn",
    "--save_noise",
    help="save task mapping duration noise (saved in replay.noise)",
    action="store_true",
)
parser.add_argument(
    "-ln",
    "--load_noise",
    help="load task mapping duration noise (saved in replay.noise)",
    action="store_true",
)
parser.add_argument("-g", "--gpus", type=int, help="number of gpus", default=4)
parser.add_argument("-pb", "--p2p", type=str, help="P2P bandwidth", default="10")
parser.add_argument(
    "-dd", "--data_size", type=float, help="per-task data size in GB", default="1"
)
parser.add_argument(
    "-d",
    "--distribution",
    type=str,
    help="rr: distributing data to gpus in rr, cpu: distributing data from cpu, random: randomly distributing data to gpus",
    default="rr",
)


args = parser.parse_args()

args = parser.parse_args()


@dataclass(slots=True)
class DataPlacer:
    cpu_size: int = 0
    gpu_size: int = 0
    num_gpus: int = 0
    data_size: Callable[[DataID], int] = default_data_sizes

    device_data_sizes: dict = field(default_factory=dict, init=False)
    device_data_limit: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.device_data_limit[Device(Architecture.CPU, 0)] = self.cpu_size
        self.device_data_sizes[Device(Architecture.CPU, 0)] = 0
        for i in range(self.num_gpus):
            self.device_data_limit[Device(Architecture.GPU, i)] = self.gpu_size
            self.device_data_sizes[Device(Architecture.GPU, i)] = 0

    def rr_gpu_placement(self, data_id: DataID) -> Devices:
        chosen = data_id.idx[-1] % self.num_gpus
        if (
            self.device_data_sizes[Device(Architecture.GPU, chosen)]
            + self.data_size(data_id)
            >= self.device_data_limit[Device(Architecture.GPU, chosen)]
        ):
            return Device(Architecture.CPU, 0)
        else:
            self.device_data_sizes[Device(Architecture.GPU, chosen)] += self.data_size(
                data_id
            )
            return Device(Architecture.GPU, chosen)

    def random_gpu_placement(self, data_id: DataID) -> Devices:
        np.random.seed(None)
        chosen = np.random.randint(0, self.num_gpus)
        if (
            self.device_data_sizes[Device(Architecture.GPU, chosen)]
            + self.data_size(data_id)
            >= self.device_data_limit[Device(Architecture.GPU, chosen)]
        ):
            return Device(Architecture.CPU, 0)
        else:
            self.device_data_sizes[Device(Architecture.GPU, chosen)] += self.data_size(
                data_id
            )
            return Device(Architecture.GPU, chosen)


def test_data():
    def cpu_data_placement(data_id: DataID) -> Devices:
        return Device(Architecture.CPU, 0)

    def random_gpu_placement(data_id: DataID) -> Devices:
        np.random.seed(None)
        return Device(Architecture.GPU, np.random.randint(0, args.gpus))

    def rr_gpu_placement(data_id: DataID) -> Devices:
        return Device(Architecture.GPU, data_id.idx[-1] % args.gpus)

    def sizes(data_id: DataID) -> int:
        interior_size = args.data_size * 1024 * 1024 * 1024
        elements = interior_size / 4
        d = args.dimensions + 1
        boundary_elements = int(elements ** ((d - 1) / d))
        boundary_size = boundary_elements * 4

        if data_id.idx[0] == 0:
            return interior_size
        else:
            return boundary_size

    def homog_task_duration():
        return args.time

    def func_type_id(task_id: TaskID):
        return 0

    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        runtime_info = TaskRuntimeInfo(
            task_time=homog_task_duration(), device_fraction=1, memory=int(0)
        )
        placement_info = TaskPlacementInfo()

        for i in range(args.gpus):
            placement_info.add(Device(Architecture.GPU, i), runtime_info)

        return placement_info

    def get_task_sorting_method(episode: int) -> TaskOrderType:
        if args.sort == "heft" or args.mode == "heft":
            return TaskOrderType.HEFT
        elif args.sort == "random":
            si = args.sorting_interval
            if si <= 0 or (si > 0 and episode % si == 0):
                return TaskOrderType.RANDOM
            else:
                return TaskOrderType.REPLAY_LAST_ITER
        elif args.sort == "replay":
            return TaskOrderType.REPLAY_FILE
        else:
            return TaskOrderType.DEFAULT

    topo_config = {
        "P2P_BW": parse_size(args.p2p + " GB"),
        "H2D_BW": parse_size("10 GB"),
        "D2H_BW": parse_size("10 GB"),
        "GPU_MEM": parse_size("1300 GB"),
        "CPU_MEM": parse_size("1300 GB"),
        "GPU_COPY_ENGINES": 3,
        "CPU_COPY_ENGINES": 3,
        "NGPUS": args.gpus,
    }

    placer = DataPlacer(
        cpu_size=topo_config["CPU_MEM"],
        gpu_size=topo_config["GPU_MEM"],
        num_gpus=args.gpus,
        data_size=sizes,
    )

    data_config = SweepDataGraphConfig()
    data_config.initial_sizes = sizes
    data_config.n_devices = args.gpus
    data_config.large_size = args.data_size
    data_config.small_size = args.data_size

    if args.distribution == "rr":
        data_config.initial_placement = placer.rr_gpu_placement
    elif args.distribution == "cpu":
        data_config.initial_placement = cpu_data_placement
    elif args.distribution == "random":
        data_config.initial_placement = placer.random_gpu_placement

    config = SweepConfig(
        steps=args.steps,
        width=args.width,
        dimensions=args.dimensions,
        task_config=task_placement,
        func_id=func_type_id,
    )
    tasks, data = make_graph(config, data_config=data_config)

    devices = uniform_connected_devices(
        n_devices=args.gpus + 1,
        mem=topo_config["GPU_MEM"],
        latency=1,
        bandwidth=topo_config["P2P_BW"],
    )

    H = SimulatorHandler(
        tasks,
        data,
        devices,
        noise_type=TNoiseType.LOGNORMAL,
        cmapper_type=CMapperType.EFT_DEQUEUE,
        pymapper=RoundRobinPythonMapper(args.gpus + 1),
        seed=1,
    )
    sim = H.create_simulator()
    sim.initialize(use_data=True)
    sim.enable_python_mapper()

    sim.run()
    print(sim.get_current_time())
