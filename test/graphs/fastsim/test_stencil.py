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

parser = argparse.ArgumentParser(prog="Stencil")

parser.add_argument("-t", "--time", type=int, help="time", default=138)
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
parser.add_argument("-s", "--steps", type=int, help="stencil steps", default=15)
parser.add_argument("-w", "--width", type=int, help="stencil width", default=8)
parser.add_argument(
    "-dm", "--dimensions", type=int, help="stencil dimensions", default=2
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
    "-dd", "--data_size", type=float, help="per-task data size in GB", default="4"
)
parser.add_argument(
    "-d",
    "--distribution",
    type=str,
    help="rr: distributing data to gpus in rr, cpu: distributing data from cpu, random: randomly distributing data to gpus",
    default="rr",
)
parser.add_argument(
    "-i",
    "--ignore_initial_placement",
    help="ignore initial placement during HEFT calculation",
    action="store_true",
)

parser.add_argument(
    "-id",
    "--interior_data_size",
    type=str,
    help="interior data size in MB",
    default="63",
)

parser.add_argument(
    "-bd",
    "--boundary_data_size",
    type=str,
    help="boundary data size in MB",
    default="0.03",
)

args = parser.parse_args()


def test_data():

    interior_size = parse_size(args.interior_data_size + " MB")  #  63 MB
    boundary_size = parse_size(args.boundary_data_size + " MB")  # ~0.03 MB

    def sizes(data_id: DataID) -> int:
        return boundary_size if data_id.idx[1] == 1 else interior_size

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
        if args.sort == "heft":
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
        "GPU_MEM": parse_size("16 GB"),
        "CPU_MEM": parse_size("130000 GB"),
        "GPU_COPY_ENGINES": 3,
        "CPU_COPY_ENGINES": 3,
        "NGPUS": args.gpus,
    }

    data_config = StencilDataGraphConfig()
    data_config.initial_sizes = sizes
    data_config.n_devices = args.gpus
    data_config.dimensions = args.dimensions
    data_config.width = args.width

    placer = DataPlacer(
        cpu_size=topo_config["CPU_MEM"],
        gpu_size=topo_config["GPU_MEM"],
        num_gpus=args.gpus,
        data_size=sizes,
        stencil_width=args.width,
    )

    if args.distribution == "rr":
        data_config.initial_placement = placer.rr_gpu_placement
    elif args.distribution == "cpu":
        data_config.initial_placement = placer.cpu_data_placement
    elif args.distribution == "random":
        data_config.initial_placement = placer.random_gpu_placement
    elif args.distribution == "opt":
        data_config.initial_placement = placer.optimal_placement

    config = StencilConfig(
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
