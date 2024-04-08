import argparse

from task4feedback.graphs import *
from task4feedback.load import *

from rich import print

# from utility.execute import run
from task4feedback.visualize import *
from task4feedback.types import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *
from task4feedback.simulator.mapper import *

from task4feedback.simulator.analysis.recorder import *
from task4feedback.simulator.interface import *
from time import perf_counter as clock

from task4feedback.simulator.rl.models.env import *
from task4feedback.simulator.rl.models.agent_using_oracle import *


parser = argparse.ArgumentParser(prog="Sweep")

parser.add_argument("-m", "--mode",
                    type=str,
                    help="testing, training, parla, worst, loadbalance, random")
parser.add_argument("-n", "--noise",
                    type=bool,
                    help="True if task duration noise is enabled", default=False)
parser.add_argument("-ns", "--noise_scale",
                    type=float,
                    help="task duration noise scale", default=0.05)
parser.add_argument("-e", "--episode",
                    type=int,
                    help="the number of episodes (-1 for inifite loop)", default=-1)
parser.add_argument("-s", "--steps",
                    type=int,
                    help="sweep steps", default=20)
parser.add_argument("-w", "--width",
                    type=int,
                    help="sweep width", default=20)
parser.add_argument("-d", "--dimensions",
                    type=int,
                    help="sweep dimensions", default=1)
parser.add_argument("-o", "--sort",
                    type=str,
                    help="task sorting method (random, heft, default)", default="default")
parser.add_argument("-si", "--sorting_interval",
                    type=int,
                    help="task random sorting interval", default=0)
parser.add_argument("-so", "--save_order",
                    type=bool,
                    help="save task mapping order (saved in replay.order)", default=False)
parser.add_argument("-lo", "--load_order",
                    type=bool,
                    help="load task mapping order (saved in replay.order)", default=False)
parser.add_argument("-sn", "--save_noise",
                    type=bool,
                    help="save task mapping duration noise (saved in replay.noise)", default=False)
parser.add_argument("-ln", "--load_noise",
                    type=bool,
                    help="load task mapping duration noise (saved in replay.noise)", default=False)
parser.add_argument("-g", "--gpus",
                    type=int,
                    help="number of gpus", default=4)
parser.add_argument("-pb", "--p2p",
                    type=str,
                    help="P2P bandwidth", default="200")
parser.add_argument("-dd", "--data_size",
                    type=float,
                    help="per-task data size in GB", default="1")


args = parser.parse_args()


def test_data():

    def initial_data_placement(data_id: DataID) -> Devices:
        return Device(Architecture.CPU, 0)

    def sizes(data_id: DataID) -> int:
        return args.data_size * 1024 * 1024 * 1024

    def homog_task_duration():
        return 40000

    def func_type_id(task_id: TaskID):
        return 0

    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        """
        if task_id.task_idx[0] % 2 == 0:
            device_tuple = (gpu0,)
        else:
            device_tuple = (gpu1,)
        """
        device_tuple = Device(Architecture.GPU, -1)

        runtime_info = TaskRuntimeInfo(
            task_time=homog_task_duration(), device_fraction=1, memory=0)
        placement_info = TaskPlacementInfo()
        placement_info.add(device_tuple, runtime_info)

        return placement_info

    def get_task_sorting_method(episode: int) -> TaskOrderType:
        if args.sort == "heft" or args.mode == "heft":
            return TaskOrderType.HEFT
        elif args.sort == "random":
            si = args.sorting_interval
            if si <= 0 or (
               si > 0 and episode % si == 0):
                return TaskOrderType.RANDOM
            else:
                return TaskOrderType.REPLAY_LAST_ITER
        elif args.sort == "replay":
            return TaskOrderType.REPLAY_FILE
        else:
            return TaskOrderType.DEFAULT

    data_config = SweepDataGraphConfig()
    data_config.initial_placement = initial_data_placement
    data_config.initial_sizes = sizes
    print("data size:", args.data_size, " GB")

    config = SweepConfig(
        steps=args.steps, width=args.width, dimensions=args.dimensions,
        task_config=task_placement, func_id=func_type_id)
    tasks, data = make_graph(config, data_config=data_config)

    num_gpus = args.gpus
    rl_env = None
    rl_agent = None
    if args.mode != "parla":
        exec_mode = ExecutionMode.TESTING if args.mode == "testing" else ExecutionMode.TRAINING
        rl_env = RLEnvironment(num_gpus)
        rl_agent = SimpleAgent(rl_env, oracle_function=LoadbalancingPolicy(), exec_mode=exec_mode)

    episode = 0
    cum_wallclock_t = 0
    task_order_log = None
    si = args.sorting_interval

    topo_config = {
      "P2P_BW": parse_size(args.p2p + " GB"),
      "H2D_BW": parse_size("100 GB"),
      "D2H_BW": parse_size("100 GB"),
      "GPU_MEM": parse_size("10000 GB"),
      "CPU_MEM": parse_size("10000 GB"),
      "GPU_COPY_ENGINES": 3,
      "CPU_COPY_ENGINES": 3,
      "NGPUS": num_gpus,
    }

    mapper = TaskMapper()

    while True:
        if episode >= args.episode and args.episode != -1:
            break

        task_order_mode = get_task_sorting_method(episode)

        cpu = Device(Architecture.CPU, 0)
        gpu0 = Device(Architecture.GPU, 0)
        gpu1 = Device(Architecture.GPU, 1)

        topology = TopologyManager().generate("frontera", config=topo_config)

        mapper_mode = args.mode
        if args.mode == "testing" or args.mode == "training":
            mapper_mode = "rl"
            mapper = RLTaskMapper()

        simulator_config = SimulatorConfig(
            topology=topology,
            tasks=tasks,
            data=data,
            task_order_log=task_order_log,
            scheduler_type="parla",
            mapper_type=mapper_mode,
            randomizer=Randomizer(),
            task_order_mode=task_order_mode,
            use_duration_noise=args.noise,
            noise_scale=args.noise_scale,
            save_task_noise=args.save_noise,
            load_task_noise=args.load_noise,
            mapper=mapper,
            rl_env=rl_env,
            rl_mapper=rl_agent,
            recorders=[DataValidRecorder],
        )
        simulator = create_simulator(config=simulator_config)

        start_t = clock()
        episode += 1
        simulated_time, task_order_log = simulator.run()
        end_t = clock()
        # if not rl_agent.is_training_mode():
        cum_wallclock_t += end_t - start_t
        print("Wallclock,",episode,",",cum_wallclock_t)


if __name__ == "__main__":
    test_data()
