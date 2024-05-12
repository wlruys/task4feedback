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
from task4feedback.simulator.analysis.plot import *
from task4feedback.simulator.interface import *
from time import perf_counter as clock

from task4feedback.simulator.rl.models.env import *
from task4feedback.simulator.rl.models.agent_using_oracle import *


parser = argparse.ArgumentParser(prog="Serial")

parser.add_argument("-m", "--mode",
                    type=str,
                    help="testing, training, parla, loadbalance, random")
parser.add_argument("-n", "--noise",
                    help="Set if task duration noise is enabled", action="store_true")
parser.add_argument("-ns", "--noise_scale",
                    type=float,
                    help="task duration noise scale", default=0.05)
parser.add_argument("-e", "--episode",
                    type=int,
                    help="the number of episodes (-1 for inifite loop)", default=-1)
parser.add_argument("-s", "--steps",
                    type=int,
                    help="chain steps", default=4)
parser.add_argument("-c", "--chains",
                    type=int,
                    help="chain chains", default=4)
parser.add_argument("-o", "--sort",
                    type=str,
                    help="task sorting method (random, heft, default)", default="default")
parser.add_argument("-si", "--sorting_interval",
                    type=int,
                    help="task random sorting interval", default=0)
parser.add_argument("-so", "--save_order",
                    help="save task mapping order (saved in replay.order)", action="store_true")
parser.add_argument("-lo", "--load_order",
                    help="load task mapping order (saved in replay.order)", action="store_true")
parser.add_argument("-sn", "--save_noise",
                    help="save task mapping duration noise (saved in replay.noise)", action="store_true")
parser.add_argument("-ln", "--load_noise",
                    help="load task mapping duration noise (saved in replay.noise)", action="store_true")
parser.add_argument("-g", "--gpus",
                    type=int,
                    help="number of gpus", default=4)
parser.add_argument("-pb", "--p2p",
                    type=str,
                    help="P2P bandwidth", default="200")
parser.add_argument("-dd", "--data_size",
                    type=float,
                    help="per-task data size in GB", default="1")
parser.add_argument("-d", "--distribution",
                    type=str,
                    help="rr: distributing data to gpus in rr, cpu: distributing data from cpu, random: randomly distributing data to gpus", default="rr")


args = parser.parse_args()


def test_data():

    def cpu_data_placement(data_id: DataID) -> Devices:
        return Device(Architecture.CPU, 0)

    def random_gpu_placement(data_id: DataID) -> Devices:
        np.random.seed(None)
        return Device(Architecture.GPU, np.random.randint(0, args.gpus))

    def rr_gpu_placement(data_id: DataID) -> Devices:
        return Device(Architecture.GPU, data_id.idx[-1] % args.gpus)

    def sizes(data_id: DataID) -> int:
        return args.data_size * 1024 * 1024 * 1024  # 1 GB

    def homog_task_duration():
        return 40000

    def func_type_id(task_id: TaskID):
        return 0

    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        runtime_info = TaskRuntimeInfo(
            task_time=homog_task_duration(), device_fraction=1,
            memory=int(0))
        placement_info = TaskPlacementInfo()

        for i in range(args.gpus):
            placement_info.add(Device(Architecture.GPU, i), runtime_info)

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

    data_config = ChainDataGraphConfig()

    if args.distribution == "rr":
        data_config.initial_placement = rr_gpu_placement
    elif args.distribution == "cpu":
        data_config.initial_placement = cpu_data_placement
    elif args.distribution == "random":
        data_config.initial_placement = random_gpu_placement

    data_config.initial_sizes = sizes

    config = ChainConfig(
        steps=args.steps, chains=args.chains, task_config=task_placement,
        func_id=func_type_id)
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
            save_task_order=args.save_order,
            load_task_order=args.load_order,
            save_task_noise=args.save_noise,
            load_task_noise=args.load_noise,
            mapper=mapper,
            rl_env=rl_env,
            rl_mapper=rl_agent,
            recorders=[LaunchedResourceUsageListRecorder],
        )
        simulator = create_simulator(config=simulator_config)

        start_t = clock()
        episode += 1
        simulated_time, task_order_log, success = simulator.run()
        end_t = clock()
        # if not rl_agent.is_training_mode():
        cum_wallclock_t += end_t - start_t
        print("Wallclock,",episode,",",cum_wallclock_t)
        print(f"Time to Simulate: {end_t - start_t}")
        print(f"Simulated Time: {simulator.time}")
        print(f"Success: {success}")


if __name__ == "__main__":
    print("Mode:", args.mode)
    print("Noise enabled?:", args.noise)
    print("Noise scale:", args.noise_scale)
    print("# episodes:", args.episode)
    print("# steps:", args.steps)
    print("# chains:", args.chains)
    print("Sorting enabled?:", args.sort)
    print("Sorting interval:", args.sorting_interval)
    print("Saving task processing order?:", args.save_order)
    print("Loading task processing order stored?:", args.load_order)
    print("Saving task execution time noise?:", args.save_noise)
    print("Loading task execution time noise?:",  args.load_noise)
    print("# GPUs?:", args.gpus)
    print("p2p bandwidth?:", args.p2p)
    print("data size:", args.data_size)
    print("data distribution:", args.distribution)

    test_data()
