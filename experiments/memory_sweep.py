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

from task4feedback.graphs.jacobi import (
    JacobiGraph,
    LevelPartitionMapper,
    JacobiRoundRobinMapper,
)
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
import matplotlib.pyplot as plt
import matplotlib as mpl

font_scale = 1.75
mpl.rcParams["font.size"] = mpl.rcParams["font.size"] * font_scale
from matplotlib.lines import Line2D


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
    graph = graph_builder.function(s)

    d = graph.get_blocks()
    m = graph

    transition_conditions = create_conditions(cfg)
    runtime_env_t = create_runtime_reward(cfg)
    observer_factory, graph_spec = create_observer_factory(cfg)
    input = SimulatorInput(m, d, s, transition_conditions=transition_conditions)

    env = runtime_env_t(
        SimulatorFactory(input, graph_spec, observer_factory),
        device="cpu",
        change_priority=cfg.graph.env.change_priority if hasattr(cfg.graph.env, "change_priority") else False,
        change_location=cfg.graph.env.change_location if hasattr(cfg.graph.env, "change_location") else False,
        change_duration=cfg.graph.env.change_duration if hasattr(cfg.graph.env, "change_duration") else False,
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
    env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
    total_mem_list = []
    d2d_bandwidth = cfg.system.d2d_bw

    def naive_mapper(graph):
        assert isinstance(graph, DynamicJacobiGraph)
        graph.mincut_per_levels(
            bandwidth=d2d_bandwidth,
            mode="dynamic",
            offset=1,
            level_chunks=cfg.graph.config.steps // 5,
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    def rr_mapper(graph):
        return JacobiRoundRobinMapper(
            n_devices=cfg.system.n_devices - 1,
            offset=1,
        )

    def global_min_cut_mapper(graph):
        assert isinstance(graph, DynamicJacobiGraph)
        graph.mincut_per_levels(
            bandwidth=d2d_bandwidth,
            mode="metis",
            offset=1,
            level_chunks=1,  # Global min-cut
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    def dynamic_metis_mapper(graph, level_chunks=14):
        assert isinstance(graph, DynamicJacobiGraph)
        graph.mincut_per_levels(
            bandwidth=d2d_bandwidth,
            mode="metis",
            offset=1,
            level_chunks=level_chunks,
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    # Map experiment names to mapper functions (None means no mapping)
    experiment_mappers = {
        "EFT": None,
        "Naïve": naive_mapper,
        "Cyclic": rr_mapper,
        "GlobalMinCut": global_min_cut_mapper,
        "Oracle": dynamic_metis_mapper,
    }
    colors = {}
    dynamic_metis_mapper_k_results = []
    # Initialize metrics for logging
    metric_keys = ["time", "mem_usage", "total_mem_movement", "eviction_movement"]

    def add_metric(metrics, name, sim: SimulatorDriver):
        metrics[name]["time"][-1] += sim.time
        metrics[name]["mem_usage"][-1] += sim.max_mem_usage
        metrics[name]["total_mem_movement"][-1] += (
            sum(list(sim.total_data_movement())[1:]) / 4
        )
        metrics[name]["eviction_movement"][-1] += (
            sum(list(sim.total_eviction_movement())[1:]) / 4
        )

    metrics = {
        name: {key: [] for key in metric_keys} for name in experiment_mappers.keys()
    }
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
    assert isinstance(env.simulator_factory[0].input.graph, DynamicJacobiGraph)
    data_stat = env.simulator_factory[0].input.graph.data.data_stat
    # single_gpu_mem_size = data_stat["average_step_data"] // 4
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
    print(f"Average Step Data Size: {data_stat['average_step_data']}")
    # problem_size = without_retire
    # problem_size = 3.2 * data_stat["average_step_data"]
    problem_size = 0.9 * with_retire
    single_gpu_mem_size = int(data_stat["average_step_data"])
    single_gpu_mem_size = int((single_gpu_mem_size // 10000 + 1) * 10000)
    # single_gpu_mem_size = 480000
    step_size = (problem_size - single_gpu_mem_size) / 20
    step_size = int((step_size // 10000 + 1) * 10000)
    graph: JacobiGraph = env.simulator_factory[0].input.graph
    num_samples = 1

    sweep_list = []
    while 4 * single_gpu_mem_size / problem_size < 3:
        sweep_list.append(single_gpu_mem_size)
        single_gpu_mem_size += step_size
    sweep_list.append(int(problem_size // 4))
    sweep_list = sorted(sweep_list)
    print(f"Sweep List: {sweep_list}")
    print(
        f"Memory,{str.join(',',[str(m) for m in experiment_mappers.keys()])}",
        flush=True,
    )

    for single_gpu_mem_size in sweep_list:
        total_mem_list.append(4 * single_gpu_mem_size)
        s = uniform_connected_devices(
            n_devices=cfg.system.n_devices,
            latency=cfg.system.latency,
            h2d_bw=cfg.system.h2d_bw,
            d2d_bw=cfg.system.d2d_bw,
            mem=single_gpu_mem_size,
        )
        env.simulator_factory[0].input.system = s
        metis_metrics = {name: {key: [] for key in metric_keys} for name in f}

        for name in metrics:
            for key in metric_keys:
                metrics[name][key].append(0)
        for name in f:
            for key in metric_keys:
                metis_metrics[name][key].append(0)

        for i in range(num_samples):
            env._reset()
            for name, mapper_fn in experiment_mappers.items():
                # reset environment and get graph
                graph = env.simulator_factory[0].input.graph
                sim = env.simulator_factory[0].create()
                # apply mapping if provided
                if mapper_fn is None:  # EFT
                    sim.disable_external_mapper()
                    sim.run()
                    print(name, sim.time)
                    add_metric(metrics, name, sim)
                elif mapper_fn == dynamic_metis_mapper:  # Oracle
                    for level_chunks in f:
                        sim = env.simulator_factory[0].create()
                        mapper = mapper_fn(graph, level_chunks=level_chunks)
                        sim.external_mapper = mapper
                        sim.enable_external_mapper()
                        # run and record
                        sim.run()
                        print(name, f"{level_chunks}", sim.time)
                        add_metric(metis_metrics, level_chunks, sim)
                else:
                    mapper = mapper_fn(graph)
                    sim.external_mapper = mapper
                    sim.enable_external_mapper()
                    sim.run()
                    print(name, sim.time)
                    add_metric(metrics, name, sim)

        if "Oracle" in experiment_mappers.keys():
            if 1 not in f:
                min_time = metrics["GlobalMinCut"]["time"][-1]
                min_k = 1
            else:
                min_time = metis_metrics[1]["time"]
                min_k = 1
            for level_chunks in f:
                if level_chunks > 1:
                    if metis_metrics[level_chunks]["time"][-1] < min_time:
                        min_time = metis_metrics[level_chunks]["time"][-1]
                        min_k = level_chunks
            dynamic_metis_mapper_k_results.append(min_k)
            if min_k != 1:
                for key in metric_keys:
                    metrics["Oracle"][key][-1] = metis_metrics[min_k][key][-1]
            else:
                for key in metric_keys:
                    metrics["Oracle"][key][-1] = metrics["GlobalMinCut"][key][-1]
        # for level_chunks in f:
        #     print(
        #         f"{level_chunks}: {dynamic_metis_mapper_time_results[level_chunks] / num_samples}"
        #     )
        print(f"{single_gpu_mem_size},", end="")
        # average memory usage over samples
        for name in metrics:
            for key in metric_keys:
                metrics[name][key][-1] /= num_samples
            if name == "Oracle":
                print(
                    f"{int(metrics[name]['time'][-1])}({dynamic_metis_mapper_k_results[-1]})",
                    end=",",
                )
            else:
                print(int(metrics[name]["time"][-1]), end=",")
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
    speedup_keys = ["Cyclic", "GlobalMinCut", "Oracle"]
    speedup = {}
    for k in speedup_keys:
        speedup[k] = []
        for idx in range(len(metrics["EFT"]["time"])):
            if metrics["EFT"]["time"][idx] > 0:
                speedup[k].append(metrics["EFT"]["time"][idx] / metrics[k]["time"][idx])
            else:
                speedup[k].append(0)
    # get maximum speedup idx for the Oracle
    # if "Oracle" in res.keys():
    #     max_speedup_idx = np.argmax(speedup["Oracle"])
    #     print(
    #         f"Maximum Oracle Speedup: {speedup['Oracle'][max_speedup_idx]} at Problem Size: {total_mem_list[max_speedup_idx]}"
    #     )
    #     problem_size = total_mem_list[max_speedup_idx]
    for i in range(len(total_mem_list)):
        min_val = 2**60
        max_val = 1
        for name in metrics:
            if metrics[name]["mem_usage"][i] < min_val:
                min_val = metrics[name]["mem_usage"][i]
            if metrics[name]["mem_usage"][i] > max_val:
                max_val = metrics[name]["mem_usage"][i]
        print(
            f"Memory Usage at {total_mem_list[i]}: min={min_val}, max={max_val}, diff={(max_val - min_val)/min_val:.3f}"
        )
        if (max_val - min_val) >= 0.001 * min_val:
            problem_size = total_mem_list[i]
            break

    mem_x_axis = [m / problem_size for m in total_mem_list]
    # Plot execution times and relative speedup
    fig0, ax0 = plt.subplots(figsize=(6, 6))
    # Primary axis: execution times
    for k in metrics.keys():
        ax0.plot(
            mem_x_axis,
            metrics[k]["time"],
            label=k,
            linestyle="-",
            color=colors[k],
        )
    # ax0.plot(total_mem_list, ml_times, label="RL", linestyle="dotted", color="purple")
    ax0.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax0.set_ylabel("Execution Time (s)")
    ax0.set_yscale("log")
    ax0.legend(loc="upper right")

    # Secondary axis: speedup vs EFT
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    for k in speedup_keys:
        ax1.plot(
            mem_x_axis,
            speedup[k],
            label=f"{k}",
            color=colors[k],
        )
    # ax2.plot(
    #     total_mem_list,
    #     ml_speedups,
    #     label="RL's Speedup",
    #     color="black",
    # )
    ax1.set_ylabel("Relative Speedup vs EFT")
    ax1.legend(loc="upper right")
    ax1.grid()
    # Indicate thresholds for with and without retire memory sizes
    # ax1.axvline(x=with_retire / without_retire, linestyle=":", color="gray")
    # ax1.axvline(x=without_retire / without_retire, linestyle=":", color="gray")

    # Separate figure for max memory usage
    fig2, ax2 = plt.subplots(figsize=(6, 6))

    for i, m in enumerate(total_mem_list):
        for k in metrics.keys():
            metrics[k]["mem_usage"][i] = metrics[k]["mem_usage"][i] / (m / 4) * 80

    for k in speedup_keys:
        ax2.plot(
            mem_x_axis,
            metrics[k]["mem_usage"],
            label=k,
            color=colors[k],
        )

    ax2.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax2.set_ylabel("GB")
    ax2.legend(loc="upper right")
    ax2.set_title("Max Memory Usage")
    ax2.grid()

    fig3, ax3 = plt.subplots(figsize=(6, 6))

    for i, m in enumerate(total_mem_list):
        for k in metrics.keys():
            metrics[k]["total_mem_movement"][i] = (
                metrics[k]["total_mem_movement"][i] / (m / 4) * 80
            )
            metrics[k]["eviction_movement"][i] = (
                metrics[k]["eviction_movement"][i] / (m / 4) * 80
            )

    for k in speedup_keys:
        ax3.plot(
            mem_x_axis,
            metrics[k]["total_mem_movement"],
            label=k,
            linestyle="-",
            color=colors[k],
        )
        ax3.plot(
            mem_x_axis,
            metrics[k]["eviction_movement"],
            linestyle="--",
            color=colors[k],
        )

    ax3.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax3.set_ylabel("GB")
    ax3.set_title("")
    ax3.set_yscale("log")
    ax3.grid()

    # Add dual legends: one for mapper (color), one for movement type (style)

    # Primary legend for mapper colors
    legend1 = ax3.legend(loc="upper right")
    ax3.add_artist(legend1)

    # Secondary legend for line styles
    style_handles = [
        Line2D([0], [0], linestyle="-", color="black"),
        Line2D([0], [0], linestyle="--", color="black"),
    ]
    style_labels = ["Total", "Eviction"]
    ax3.legend(
        handles=style_handles,
        labels=style_labels,
        loc="center right",
    )

    # Combined 1×4 subplots with shared x-axis and a common x-axis label
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True)

    # 1) Execution Time (log scale)
    for k in metrics.keys():
        axes[0].plot(
            mem_x_axis, metrics[k]["time"], label=k, linestyle="-", color=colors[k]
        )
    axes[0].set_ylabel("Execution Time (s)")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", color="gray", linestyle="--", linewidth=0.5)
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("(a)", fontsize=mpl.rcParams["font.size"] / font_scale * 2.25)
    # 2) Relative Speedup vs EFT
    for k in speedup_keys:
        axes[1].plot(mem_x_axis, speedup[k], label=k, color=colors[k])
    axes[1].set_ylabel("Relative Speedup vs EFT")
    axes[1].legend(loc="upper right")
    axes[1].grid()
    axes[1].set_xlabel("(b)", fontsize=mpl.rcParams["font.size"] / font_scale * 2.25)
    # 3) Max Memory Usage
    speedup_keys = ["EFT", "Cyclic", "GlobalMinCut", "Oracle"]
    for k in speedup_keys:
        axes[2].plot(mem_x_axis, metrics[k]["mem_usage"], label=k, color=colors[k])
    axes[2].legend(loc="upper right")
    axes[2].set_ylabel("GB")
    axes[2].grid()
    axes[2].set_xlabel("(c)", fontsize=mpl.rcParams["font.size"] / font_scale * 2.25)
    # 4) Memory Movement (Total vs Eviction)
    for k in speedup_keys:
        axes[3].plot(
            mem_x_axis,
            metrics[k]["total_mem_movement"],
            linestyle="-",
            color=colors[k],
            label=k,
        )
        axes[3].plot(
            mem_x_axis, metrics[k]["eviction_movement"], linestyle="--", color=colors[k]
        )
    legend1 = axes[3].legend(loc="upper right")
    axes[3].add_artist(legend1)

    # Secondary legend for line styles
    style_handles = [
        Line2D([0], [0], linestyle="-", color="black"),
        Line2D([0], [0], linestyle="--", color="black"),
    ]
    style_labels = ["Total", "Eviction"]
    axes[3].legend(
        handles=style_handles,
        labels=style_labels,
        loc="lower left",
    )
    axes[3].set_ylabel("GB")
    axes[3].set_xlabel("(d)", fontsize=mpl.rcParams["font.size"] / font_scale * 2.25)
    axes[3].set_yscale("log")
    axes[3].grid()

    # Shared x-axis label and layout
    fig.supxlabel(
        "Total GPUs Memory Size / Problem Size",
        fontsize=mpl.rcParams["font.size"] / font_scale * 2.25,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])  # leave room at the bottom for the xlabel

    # Save plots to a file
    file_name = f"{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}"
    file_name += f"_{cfg.graph.config.workload_args.scale}"
    file_name += f"_{cfg.graph.config.workload_args.upper_bound/cfg.graph.config.workload_args.lower_bound}"
    file_name += f"_{cfg.graph.config.workload_args.traj_type}"
    # create a folder for outputs if not exists
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("outputs/" + file_name).mkdir(parents=True, exist_ok=True)
    Path("outputs/" + file_name + "/time").mkdir(parents=True, exist_ok=True)
    Path("outputs/" + file_name + "/speed").mkdir(parents=True, exist_ok=True)
    Path("outputs/" + file_name + "/mem").mkdir(parents=True, exist_ok=True)
    Path("outputs/" + file_name + "/log").mkdir(parents=True, exist_ok=True)
    Path("outputs/" + file_name + "/move").mkdir(parents=True, exist_ok=True)
    path_to_folder = Path("outputs") / file_name
    for k, v in cfg.graph.config.workload_args.traj_specifics.items():
        file_name += f"_{k}{v}"

    fig0.savefig(path_to_folder / f"time/time_{file_name}.png")
    fig1.savefig(path_to_folder / f"speed/speed_{file_name}.png")
    fig2.savefig(path_to_folder / f"mem/mem_{file_name}.png")
    fig3.savefig(path_to_folder / f"move/move_{file_name}.png")
    fig.savefig(path_to_folder / f"combined_{file_name}.png")

    # Save results to a file
    with open(
        path_to_folder / f"log/result_{file_name}.txt",
        "w",
    ) as text_file:
        text_file.write(OmegaConf.to_yaml(cfg.graph))
        text_file.write(
            f"Problem Size: {problem_size/data_stat['average_step_data']}xAvgStep\n"
        )
        text_file.write(f"Problem Size: {problem_size}\n")
        text_file.write(f"with Retire Memory Size: {with_retire}\n")
        text_file.write(f"without Retire Memory Size: {without_retire}\n")
        # text_file.write(
        #     "Memory,EFT," + ",".join([str(i) for i in f]) + ",oracle,oracle_best_f\n"
        # )
        text_file.write(
            "PerGPUMemory,TotalMem/ProblemSize,"
            + ",".join([str(i) for i in metrics.keys()])
            + "\n"
        )
        for idx in range(len(metrics["EFT"]["time"])):
            line = (
                f"{int(total_mem_list[idx]/4)},{total_mem_list[idx]/problem_size:.2f}"
            )
            for k in metrics.keys():
                if k == "Oracle":
                    line += f",{int(metrics[k]['time'][idx])}({dynamic_metis_mapper_k_results[idx]})"
                else:
                    line += f",{int(metrics[k]['time'][idx])}"
            line += "\n"
            text_file.write(line)


@hydra.main(config_path="conf", config_name="dynamic_batch", version_base=None)
def main(cfg: DictConfig):
    test(cfg)


if __name__ == "__main__":
    main()
