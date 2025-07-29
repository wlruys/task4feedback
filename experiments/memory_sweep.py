# src/train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *
from task4feedback.ml.algorithms import *
from task4feedback.interface.wrappers import start_logger
import wandb
import gmsh
from hydra.utils import instantiate
from hydra.experimental.callbacks import Callback

from helper.graph import make_graph_builder, GraphBuilder
from helper.env import make_env
from helper.model import *

from task4feedback.interface import uniform_connected_devices
from task4feedback.graphs.jacobi import (
    JacobiGraph,
    LevelPartitionMapper,
    JacobiRoundRobinMapper,
)
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
import matplotlib.pyplot as plt


class GitInfo(Callback):
    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        try:
            repo = git.Repo(search_parent_directories=True)
            outdir = Path(config.hydra.runtime.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "git_sha.txt").write_text(repo.head.commit.hexsha)
            (outdir / "git_dirty.txt").write_text(str(repo.is_dirty()))
            diff = repo.git.diff(None)
            (outdir / "git_diff.patch").write_text(diff)

            print(
                "Git SHA:",
                repo.head.commit.hexsha,
                " (dirty)" if repo.is_dirty() else " (clean)",
                flush=True,
            )

        except Exception as e:
            print(f"GitInfo callback failed: {e}")


def create_system(cfg: DictConfig):
    system = hydra.utils.instantiate(cfg.system)
    return system


def create_conditions(cfg: DictConfig):
    transition_conditions = hydra.utils.instantiate(cfg.runtime)
    return transition_conditions


def create_runtime_reward(cfg: DictConfig):
    runtime_env_t = hydra.utils.instantiate(cfg.reward)
    print(cfg.reward)
    return runtime_env_t


def create_observer_factory(cfg: DictConfig):
    graph_spec = hydra.utils.instantiate(cfg.feature.observer.spec)
    if (
        hasattr(cfg.feature.observer, "width")
        and hasattr(cfg.feature.observer, "prev_frames")
        and hasattr(cfg.feature.observer, "batched")
    ):
        if cfg.feature.observer.batched:
            graph_spec.max_candidates = cfg.graph.config.n**2
        observer_factory = hydra.utils.instantiate(
            cfg.feature.observer,
            spec=graph_spec,
            width=cfg.graph.config.n,
            prev_frames=cfg.feature.observer.prev_frames,
            batched=cfg.feature.observer.batched,
        )
    else:
        observer_factory = hydra.utils.instantiate(cfg.feature.observer)
    return observer_factory, graph_spec


def make_env(
    graph_builder: GraphBuilder,
    cfg: DictConfig,
) -> RuntimeEnv:
    gmsh.initialize()

    s = create_system(cfg)
    graph = graph_builder.function()

    d = graph.get_blocks()
    m = graph

    transition_conditions = create_conditions(cfg)
    runtime_env_t = create_runtime_reward(cfg)
    observer_factory, graph_spec = create_observer_factory(cfg)
    input = SimulatorInput(m, d, s, transition_conditions=transition_conditions)

    env = runtime_env_t(
        SimulatorFactory(input, graph_spec, observer_factory),
        device="cpu",
        change_priority=cfg.graph.env.change_priority,
        change_locations=cfg.graph.env.change_locations,
        seed=cfg.graph.env.seed,
        max_samples_per_iter=(
            len(graph) + 1
            if cfg.algorithm.rollout_steps == 0
            else cfg.algorithm.rollout_steps + 1
        ),
        location_list=[0, 1, 2, 3, 4],
    )
    return env


def test(cfg: DictConfig):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    graph_builder = make_graph_builder(cfg)
    env = make_env(graph_builder=graph_builder, cfg=cfg)
    mem = []

    # Define mapper functions for experiments
    from task4feedback.graphs.dynamic_jacobi import (
        DynamicJacobiGraph,
    )  # ensure import is present

    def naive_mapper(graph):
        assert isinstance(graph, DynamicJacobiGraph)
        graph.mincut_per_levels(
            bandwidth=cfg.system.d2d_bw,
            mode="dynamic",
            offset=1,
            level_chunks=cfg.graph.config.steps // 5,
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    def rr_mapper(graph):
        return JacobiRoundRobinMapper(
            n_devices=cfg.system.n_devices - 1,
        )

    def global_min_cut_mapper(graph):
        assert isinstance(graph, DynamicJacobiGraph)
        graph.mincut_per_levels(
            bandwidth=cfg.system.d2d_bw,
            mode="dynamic_metis",
            offset=1,
            level_chunks=1,  # Global min-cut
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    def dynamic_metis_mapper(graph, level_chunks=14):
        assert isinstance(graph, DynamicJacobiGraph)
        graph.mincut_per_levels(
            bandwidth=cfg.system.d2d_bw,
            mode="dynamic_metis",
            offset=1,
            level_chunks=level_chunks,
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    # Map experiment names to mapper functions (None means no mapping)
    experiment_mappers = {
        "EFT": None,
        "naive": naive_mapper,
        "RoundRobin": rr_mapper,
        "GlobalMinCut": global_min_cut_mapper,
        "Oracle": dynamic_metis_mapper,
    }
    colors = {}
    dynamic_metis_mapper_k_results = []
    res = {}
    mem_usage = {}
    for k, v in experiment_mappers.items():
        res[k] = []
        mem_usage[k] = []
        # Automatically assign colors to each mapper
    for i, k in enumerate(experiment_mappers.keys()):
        colors[k] = plt.cm.tab10(i)

    # Factorize cfg.graph.steps
    f = []
    for i in range(
        int("GlobalMinCut" in experiment_mappers.keys()),
        cfg.graph.config.steps // 3,
    ):
        # for i in [0, cfg.graph.config.steps // 5 - 1]:
        if cfg.graph.config.steps % (i + 1) == 0:
            f.append(i + 1)
    print(f"Factors of {cfg.graph.config.steps}: {f}")

    # Initialize memory size and problem size
    assert isinstance(env.simulator_factory.input.graph, DynamicJacobiGraph)
    data_stat = env.simulator_factory.input.graph.data.data_stat
    # mem_size = data_stat["average_step_data"] // 4
    # level_total_mem = data_stat["average_step_data"] * 3
    c = 1
    with_retire = (cfg.graph.config.n**2) * (
        1 * data_stat["interior_average"]  # Task's
        + 4 * data_stat["boundary_average"]  # Task's
        + 4 * data_stat["boundary_average"]  # Neighbor's data
        + 1 * data_stat["interior_average"]  # Next Step
        + 4 * data_stat["boundary_average"]  # Next Step
        + 1 * data_stat["interior_average"] * c  # Retire Data
        + 4 * data_stat["boundary_average"] * c  # Retire Data
    ) - 2 * (cfg.graph.config.n * 4) * data_stat[
        "boundary_average"
    ]  # Correcting boundary
    without_retire = (cfg.graph.config.n**2) * (
        1 * data_stat["interior_average"]  # Task's
        + 4 * data_stat["boundary_average"]  # Task's
        + 4 * data_stat["boundary_average"]  # Neighbor's data
        + 1 * data_stat["interior_average"]  # Next Step
        + 4 * data_stat["boundary_average"]  # Next Step
    ) - 2 * (cfg.graph.config.n * 4) * data_stat[
        "boundary_average"
    ]  # Correcting boundary

    print(f"with Retire Memory Size: {with_retire}")
    print(f"without Retire Memory Size: {without_retire}")
    # problem_size = without_retire
    # problem_size = 3.2 * data_stat["average_step_data"]
    problem_size = 0.9 * with_retire
    mem_size = int(data_stat["average_step_data"] // 4)
    mem_size = int((mem_size // 10000 + 1) * 10000)
    # mem_size = 480000
    step_size = (problem_size // 4 - mem_size) / 10
    step_size = int((step_size // 10000 + 1) * 10000)
    graph: JacobiGraph = env.simulator_factory.input.graph
    num_samples = 1

    sweep_list = []
    while 4 * mem_size / problem_size < 1.5:
        sweep_list.append(mem_size)
        mem_size += step_size
    sweep_list.append(int(problem_size // 4))
    sweep_list = sorted(sweep_list)
    print(f"Sweep List: {sweep_list}")
    print(
        f"Memory,{str.join(',',[str(m) for m in res.keys()])}",
        flush=True,
    )
    for mem_size in sweep_list:
        mem.append(4 * mem_size)
        s = uniform_connected_devices(
            n_devices=cfg.system.n_devices,
            h2d_bw=cfg.system.h2d_bw,
            d2d_bw=cfg.system.d2d_bw,
            latency=cfg.system.latency,
            mem=mem_size,
        )
        env.simulator_factory.input.system = s
        for k in res.keys():
            res[k].append(0)
            mem_usage[k].append(0)
        dynamic_metis_mapper_time_results = {}
        dynamic_metis_mapper_mem_results = {}
        for level_chunks in f:
            dynamic_metis_mapper_time_results[level_chunks] = 0
            dynamic_metis_mapper_mem_results[level_chunks] = 0
        for i in range(num_samples):
            for name, mapper_fn in experiment_mappers.items():
                # reset environment and get graph
                env._reset()
                graph = env.simulator_factory.input.graph
                sim = env.simulator_factory.create()
                # apply mapping if provided
                if mapper_fn is None:  # EFT
                    sim.disable_external_mapper()
                    sim.run()
                    res[name][-1] += sim.time
                    mem_usage[name][-1] += sim.max_mem_usage
                elif mapper_fn == dynamic_metis_mapper:  # Oracle
                    for level_chunks in f:
                        sim = env.simulator_factory.create()
                        mapper = mapper_fn(graph, level_chunks=level_chunks)
                        sim.external_mapper = mapper
                        sim.enable_external_mapper()
                        # run and record
                        sim.run()
                        dynamic_metis_mapper_time_results[level_chunks] += sim.time
                        dynamic_metis_mapper_mem_results[
                            level_chunks
                        ] += sim.max_mem_usage
                else:
                    mapper = mapper_fn(graph)
                    sim.external_mapper = mapper
                    sim.enable_external_mapper()
                    sim.run()
                    res[name][-1] += sim.time
                    mem_usage[name][-1] += sim.max_mem_usage

        if "Oracle" in experiment_mappers.keys():
            if 1 not in f:
                min_time = res["GlobalMinCut"][-1]
                min_mem = mem_usage["GlobalMinCut"][-1]
                min_k = 1
            else:
                min_time = dynamic_metis_mapper_time_results[1]
                min_mem = dynamic_metis_mapper_mem_results[1]
                min_k = 1
            for level_chunks in f:
                if level_chunks > 1:
                    if dynamic_metis_mapper_time_results[level_chunks] < min_time:
                        min_time = dynamic_metis_mapper_time_results[level_chunks]
                        min_mem = dynamic_metis_mapper_mem_results[level_chunks]
                        min_k = level_chunks
            res["Oracle"][-1] = min_time
            mem_usage["Oracle"][-1] = min_mem
            dynamic_metis_mapper_k_results.append(min_k)
        # for level_chunks in f:
        #     print(
        #         f"{level_chunks}: {dynamic_metis_mapper_time_results[level_chunks] / num_samples}"
        #     )
        print(f"{mem_size},", end="")
        # average memory usage over samples
        for k in res.keys():
            res[k][-1] /= num_samples
            mem_usage[k][-1] /= num_samples
            if k == "Oracle":
                print(
                    f"{int(res[k][-1])}({dynamic_metis_mapper_k_results[-1]})", end=","
                )
            else:
                print(int(res[k][-1]), end=",")
        print("")

    # ml_speedups = [
    #     0.926,
    #     0.908,
    #     0.92,
    #     0.97,
    #     1.08,
    #     1.31,
    #     1.25,
    #     1.68879,
    #     1.73629,
    #     0.93648,
    #     0.80549,
    #     0.87296,
    #     0.92076,
    #     1.04714,
    #     0.97467,
    #     0.94498,
    #     0.85624,
    #     0.95391,
    #     0.99315,
    #     0.93683,
    # ]  # Placeholder for ML speedups
    # ml_times = []
    # for i in range(len(res["1"])):
    #     ml_times.append(res["1"][i] / ml_speedups[i])
    # for idx in range(len(res["eft"])):
    #     ml_speedups[idx] = res["eft"][idx] / ml_times[idx]
    speedup_keys = ["RoundRobin", "GlobalMinCut", "Oracle"]
    speedup = {}
    for k in speedup_keys:
        speedup[k] = []
        for idx in range(len(res["EFT"])):
            if res["EFT"][idx] > 0:
                speedup[k].append(res["EFT"][idx] / res[k][idx])
            else:
                speedup[k].append(0)
    # get maximum speedup idx for the Oracle
    if "Oracle" in res.keys():
        max_speedup_idx = np.argmax(speedup["Oracle"])
        print(
            f"Maximum Oracle Speedup: {speedup['Oracle'][max_speedup_idx]} at Problem Size: {mem[max_speedup_idx]}"
        )
        problem_size = mem[max_speedup_idx]

    mem_x_axis = [m / problem_size for m in mem]
    # Plot execution times and relative speedup
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Primary axis: execution times
    for k in res.keys():
        ax1.plot(
            mem_x_axis,
            res[k],
            label=k,
            linestyle="dotted",
            color=colors[k],
        )
    # ax1.plot(mem, ml_times, label="RL", linestyle="dotted", color="purple")
    ax1.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left")

    # Secondary axis: speedup vs EFT
    ax2 = ax1.twinx()
    for k in speedup_keys:
        ax2.plot(
            mem_x_axis,
            speedup[k],
            label=f"{k}'s Speedup",
            color=colors[k],
            linestyle="-",
        )
    # ax2.plot(
    #     mem,
    #     ml_speedups,
    #     label="RL's Speedup",
    #     color="black",
    # )
    ax2.set_ylabel("Relative Speedup vs EFT")
    ax2.legend(loc="upper right")

    ax1.set_title("Execution Time vs Memory Size and Speedup")
    ax1.grid()
    # Indicate thresholds for with and without retire memory sizes
    # ax1.axvline(x=with_retire / without_retire, linestyle=":", color="gray")
    # ax1.axvline(x=without_retire / without_retire, linestyle=":", color="gray")

    # Separate figure for max memory usage
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for k in res.keys():
        ax2.plot(
            mem_x_axis,
            mem_usage[k],
            label=k,
        )

    ax2.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax2.set_ylabel("Max Memory Usage")
    ax2.legend(loc="upper right")
    ax2.set_title("Max Memory Usage vs Memory Size")
    ax2.grid()
    # Save plots to a file
    file_name = f"{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}"
    file_name += f"_{cfg.graph.config.workload_args.scale}"
    file_name += f"_{cfg.graph.config.workload_args.upper_bound/cfg.graph.config.workload_args.lower_bound}"
    file_name += f"_ibr{cfg.graph.config.interior_boundary_ratio}_ccr{cfg.graph.config.comm_compute_ratio}"
    file_name += f"_{cfg.graph.config.workload_args.traj_type}"
    for k, v in cfg.graph.config.workload_args.traj_specifics.items():
        file_name += f"_{k}{v}"

    fig.savefig(f"outputs/time_{file_name}.png")
    fig2.savefig(f"outputs/mem_{file_name}.png")

    # Save results to a file
    with open(
        f"outputs/result_{file_name}.txt",
        "w",
    ) as text_file:
        text_file.write(OmegaConf.to_yaml(cfg.graph))
        text_file.write(
            f"Problem Size: {problem_size/data_stat['average_step_data']}xAvgStep\n"
        )
        text_file.write(f"with Retire Memory Size: {with_retire}\n")
        text_file.write(f"without Retire Memory Size: {without_retire}\n")
        # text_file.write(
        #     "Memory,EFT," + ",".join([str(i) for i in f]) + ",oracle,oracle_best_f\n"
        # )
        text_file.write("Memory," + ",".join([str(i) for i in res.keys()]) + "\n")
        for idx in range(len(res["EFT"])):
            line = f"{int(mem[idx]*problem_size/4)}"
            for k in res.keys():
                if k == "Oracle":
                    line += f",{res[k][idx]}({dynamic_metis_mapper_k_results[idx]})"
                else:
                    line += f",{res[k][idx]}"
            line += "\n"
            text_file.write(line)


@hydra.main(config_path="conf", config_name="dynamic_batch", version_base=None)
def main(cfg: DictConfig):
    test(cfg)


if __name__ == "__main__":
    main()
