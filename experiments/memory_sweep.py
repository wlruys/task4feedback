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
from task4feedback.graphs.jacobi import JacobiGraph, LevelPartitionMapper
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph


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
    env = make_env(graph_builder=graph_builder, cfg=cfg)
    mem = []

    # Define mapper functions for experiments
    from task4feedback.graphs.dynamic_jacobi import (
        DynamicJacobiGraph,
    )  # ensure import is present

    def naive_mapper(graph):
        graph.mincut_per_levels(
            bandwidth=cfg.system.d2d_bw,
            mode="dynamic",
            offset=1,
            level_chunks=cfg.graph.config.steps // 5,
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    # Map experiment names to mapper functions (None means no mapping)
    experiment_mappers = {
        "eft": None,
        "naive": naive_mapper,
    }

    res = {"oracle": [], "oracle_f": []}
    mem_usage = {"global": [], "oracle": []}
    for k, v in experiment_mappers.items():
        res[k] = []
        mem_usage[k] = []
    # Factorize cfg.graph.steps
    f = []
    for i in range(cfg.graph.config.steps // 3):
        # for i in [0, cfg.graph.config.steps // 5 - 1]:
        if cfg.graph.config.steps % (i + 1) == 0:
            f.append(i + 1)
            res[str(i + 1)] = []
    print(f"Factors of {cfg.graph.config.steps}: {f}")

    assert isinstance(env.simulator_factory.input.graph, DynamicJacobiGraph)
    data_stat = env.simulator_factory.input.graph.data.data_stat
    mem_size = data_stat["average_step_data"] // 4
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
    problem_size = 1.125 * with_retire
    mem_size = int((mem_size // 10000 + 1) * 10000)
    # mem_size = 480000
    step_size = (0.5 * problem_size - mem_size) / 20
    step_size = int((step_size // 1000 + 1) * 1000)
    graph: JacobiGraph = env.simulator_factory.input.graph
    num_samples = 10
    print(f"Memory,EFT,{str.join(',',[str(m) for m in f])},oracle,oracle_f", flush=True)
    while True:
        mem.append(4 * mem_size / problem_size)
        s = uniform_connected_devices(
            n_devices=cfg.system.n_devices,
            h2d_bw=cfg.system.h2d_bw,
            d2d_bw=cfg.system.d2d_bw,
            latency=cfg.system.latency,
            mem=mem_size,
        )
        env.simulator_factory.input.system = s
        for k, v in res.items():
            if k not in ["oracle", "oracle_f"]:
                v.append(0)
        # initialize memory usage accumulators
        for k, v in mem_usage.items():
            v.append(0)
        # temp storage to accumulate per-factor memory usage for Oracle
        factor_mem_usage = {i: 0 for i in f}
        for i in range(num_samples):
            for name, mapper_fn in experiment_mappers.items():
                # reset environment and get graph
                env._reset()
                graph = env.simulator_factory.input.graph
                sim = env.simulator_factory.create()
                # apply mapping if provided
                if mapper_fn is None:
                    sim.disable_external_mapper()
                else:
                    mapper = mapper_fn(graph)
                    sim.external_mapper = mapper
                    sim.enable_external_mapper()
                # run and record
                sim.run()
                res[name][-1] += sim.time
                mem_usage[name][-1] += sim.max_mem_usage

            graph.mincut_per_levels(
                bandwidth=cfg.system.d2d_bw,
                mode="predict",
                offset=1,
                level_chunks=cfg.graph.config.steps // 5,
            )
            mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
            sim = env.simulator_factory.create()
            sim.external_mapper = mapper
            sim.enable_external_mapper()
            sim.run()
            res["naive"][-1] += sim.time

            for i in f:
                graph.mincut_per_levels(
                    bandwidth=cfg.system.d2d_bw,
                    level_chunks=i,
                    offset=1,
                )
                graph.align_partitions()
                mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
                sim = env.simulator_factory.create()
                sim.external_mapper = mapper
                sim.enable_external_mapper()
                sim.run()
                res[str(i)][-1] += sim.time
                factor_mem_usage[i] += sim.max_mem_usage
                # global min-cut corresponds to factor f[0] (usually 1)
                if i == f[0]:
                    mem_usage["global"][-1] += sim.max_mem_usage
        # average memory usage over samples
        mem_usage["eft"][-1] /= num_samples
        mem_usage["global"][-1] /= num_samples
        # compute Oracle memory usage based on oracle factor
        res["oracle"].append(res["eft"][-1])
        res["oracle_f"].append("-1")
        for k, v in res.items():
            if k in ["oracle", "oracle_f"]:
                continue
            v[-1] = int(v[-1] / num_samples)
            if k not in ["eft", "naive"] and v[-1] < res["oracle"][-1]:
                res["oracle"][-1] = v[-1]
                res["oracle_f"][-1] = k
        oracle_factor = int(res["oracle_f"][-1])
        mem_usage["oracle"][-1] = factor_mem_usage[oracle_factor] / num_samples

        print(f"{mem_size},{res['eft'][-1]}", end=" ")
        for i in f:
            print(f"{res[str(i)][-1]}", end=",")
        print(f"{res['oracle'][-1]},{res['oracle_f'][-1]}")
        mem_size += step_size
        if 4 * mem_size / problem_size > 2:
            break

    import matplotlib.pyplot as plt

    # Compute relative speedup vs EFT
    speedup = []
    for idx in range(len(res["eft"])):
        # find the fastest factor run at this memory size
        oracle_time = min(res[str(i)][idx] for i in f)
        speedup.append(res["eft"][idx] / oracle_time)

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

    # Plot execution times and relative speedup
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Primary axis: execution times
    ax1.plot(mem, res["eft"], label="EFT", linestyle="dotted", color="black")
    ax1.plot(mem, res["oracle"], label="Oracle", linestyle="dotted", color="green")
    ax1.plot(mem, res["1"], label="Global Min-Cut", linestyle="dotted", color="orange")
    ax1.plot(mem, res["naive"], label="Naive", linestyle="dotted", color="blue")
    # ax1.plot(mem, ml_times, label="RL", linestyle="dotted", color="purple")
    ax1.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax1.set_ylabel("Execution Time (s)")
    ax1.legend(loc="upper left")

    # Secondary axis: speedup vs EFT
    ax2 = ax1.twinx()
    ax2.plot(
        mem,
        speedup,
        label="Oracle's Speedup",
        color="red",
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
    ax2.plot(mem, mem_usage["eft"], label="EFT", marker="o")
    ax2.plot(
        mem,
        mem_usage["global"],
        label="Global Min-Cut",
        linestyle="dashed",
        color="orange",
    )
    ax2.plot(mem, mem_usage["oracle"], label="Oracle", marker="x", color="green")
    ax2.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax2.set_ylabel("Max Memory Usage")
    ax2.legend(loc="upper right")
    ax2.set_title("Max Memory Usage vs Memory Size")
    ax2.grid()
    # Save plots to a file
    fig.savefig(
        f"{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}_{cfg.graph.config.workload_args.max_angle}Cir_{cfg.graph.config.workload_args.scale}_{cfg.graph.config.workload_args.upper_bound/cfg.graph.config.workload_args.lower_bound}_ibr{cfg.graph.config.interior_boundary_ratio}_ccr{cfg.graph.config.comm_compute_ratio}_vs_memory.png"
    )
    fig2.savefig(
        f"{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}_{cfg.graph.config.workload_args.max_angle}Cir_{cfg.graph.config.workload_args.scale}_{cfg.graph.config.workload_args.upper_bound/cfg.graph.config.workload_args.lower_bound}_ibr{cfg.graph.config.interior_boundary_ratio}_ccr{cfg.graph.config.comm_compute_ratio}_usage.png"
    )

    # Save results to a file
    with open(
        f"{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}_{cfg.graph.config.workload_args.max_angle}Cir_{cfg.graph.config.workload_args.scale}_{cfg.graph.config.workload_args.upper_bound/cfg.graph.config.workload_args.lower_bound}_ibr{cfg.graph.config.interior_boundary_ratio}_ccr{cfg.graph.config.comm_compute_ratio}_results.txt",
        "w",
    ) as text_file:
        text_file.write(
            f"Problem Size: {problem_size/data_stat['average_step_data']}xAvgStep\n"
        )
        text_file.write(f"with Retire Memory Size: {with_retire}\n")
        text_file.write(f"without Retire Memory Size: {without_retire}\n")
        text_file.write(
            "Memory,EFT," + ",".join([str(i) for i in f]) + ",oracle,oracle_f\n"
        )
        for idx in range(len(res["eft"])):
            line = f"{int(mem[idx]*problem_size/4)},{res['eft'][idx]}"
            for i in f:
                line += f",{res[str(i)][idx]}"
            line += f",{res['oracle'][idx]},{res['oracle_f'][idx]}\n"
            text_file.write(line)


@hydra.main(config_path="conf", config_name="dynamic_batch", version_base=None)
def main(cfg: DictConfig):
    test(cfg)


if __name__ == "__main__":
    main()
