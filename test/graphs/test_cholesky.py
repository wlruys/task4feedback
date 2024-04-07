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
from task4feedback.simulator.analysis.export import *
from task4feedback.simulator.interface import *
from task4feedback.simulator.verify import *

from task4feedback.simulator.rl.models.a2c import *
from task4feedback.simulator.rl.models.env import *
from task4feedback.simulator.rl.models.oracles import *
from task4feedback.simulator.rl.models.agent_using_oracle import *
from task4feedback.simulator.rl.models.agent_a2c import *

from time import perf_counter as clock


parser = argparse.ArgumentParser(prog="Cholesky")

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
parser.add_argument("-b", "--block",
                    type=int,
                    help="bxb blocks")
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


args = parser.parse_args()


def test_data():

    def initial_data_placement(data_id: DataID) -> Devices:
        return Device(Architecture.CPU, 0)

    def sizes(data_id: DataID) -> int:
        return 1 * 1024 * 1024 * 1024  # 1 GB

    def task_duration_per_func(task_id: TaskID):
        duration = 40000
        if task_id.taskspace == "POTRF":
            duration = 60000
        elif task_id.taskspace == "SYRK":
            duration = 40000
        elif task_id.taskspace == "SOLVE":
            duration = 30000
        elif task_id.taskspace == "GEMM":
            duration = 40000
        return duration

    def homog_task_duration():
        return 80000

    def func_type_id(task_id: TaskID):
        func_id = 0
        if task_id.taskspace == "POTRF":
            func_id = 0
        elif task_id.taskspace == "SYRK":
            func_id = 1
        elif task_id.taskspace == "SOLVE":
            func_id = 2
        elif task_id.taskspace == "GEMM":
            func_id = 3
        return func_id

    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        device_tuple = Device(Architecture.GPU, -1)

        runtime_info = TaskRuntimeInfo(
            task_time=task_duration_per_func(task_id), device_fraction=1,
            # task_time=homog_task_duration(), device_fraction=1,
            memory=int(0))
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
                
    data_config = CholeskyDataGraphConfig()
    # data_config = NoDataGraphConfig()
    data_config.initial_placement = initial_data_placement
    data_config.initial_sizes = sizes

    config = CholeskyConfig(blocks=args.block, task_config=task_placement,
                            func_id=func_type_id)
    tasks, data = make_graph(config, data_config=data_config)

    # Execution mode configuration
    # TODO(hc): Readys testing/training
    #           Parla testing
    #           RL testing/training
    num_gpus = 4
    rl_env = None
    rl_agent = None
    if args.mode != "parla":
        exec_mode = ExecutionMode.TESTING if args.mode == "testing" else ExecutionMode.TRAINING
        rl_env = RLEnvironment(num_gpus)
        rl_agent = SimpleAgent(rl_env, oracle_function=LoadbalancingPolicy(), exec_mode=exec_mode)
        #rl_agent = A2CAgent(rl_env)

    episode = 0
    cum_wallclock_t = 0

    task_order_log = None
    si = args.sorting_interval

    mapper = TaskMapper()
    while True:
        if episode > args.episode and args.episode != -1:
            break

        task_order_mode = get_task_sorting_method(episode)

        cpu = Device(Architecture.CPU, 0)
        gpu0 = Device(Architecture.GPU, 0)
        gpu1 = Device(Architecture.GPU, 1)

        topology = TopologyManager().generate("frontera", config=None)

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
        )
        simulator = create_simulator(config=simulator_config)

        start_t = clock()
        episode += 1
        simulated_time, task_order_log = simulator.run()
        end_t = clock()
        # if not rl_agent.is_training_mode():
        cum_wallclock_t += end_t - start_t
        print("Wallclock,",episode,",",cum_wallclock_t)

    # print(
    #     simulator.recorders.get(LaunchedResourceUsageListRecorder).vcu_usage[
    #         Device(Architecture.GPU, 0)
    #     ]
    # )

    # make_resource_plot(
    #     recorder=simulator.recorders,
    #     resource_type=ResourceType.MEMORY,
    #     phase=TaskState.LAUNCHED,
    # )

    # print("Tasks: ")
    # print(summarize_dependencies(simulator_config.simulated_tasks))

    # for task in simulator_config.simulated_tasks.values():
    #    print(f"{task.name} {task.times}")

    # verify_order(simulator_config.simulated_tasks)
    # verify_runtime_resources(simulator_config.simulated_tasks, simulator.devicemap)

    # print(simulator.recorders)

    # make_plot(
    #     simulator.recorders.recorders[0],
    #     simulator.recorders.recorders[1],
    #     simulator.recorders.recorders[2],
    #     data_ids=[DataID((4, 1))],
    # )

    # export_task_records(
    #     simulator.recorders.get(ComputeTaskRecorder),
    #     simulator.recorders.get(DataTaskRecorder),
    #     filename="task_records.json",
    # )


if __name__ == "__main__":
    test_data()
