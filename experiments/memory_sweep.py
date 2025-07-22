# src/train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *
from task4feedback.ml.algorithms import *
from task4feedback.fastsim2 import start_logger
import wandb
import gmsh
from hydra.utils import instantiate

from helper.graph import make_graph_builder, GraphBuilder
from helper.env import make_env
from helper.model import *

from task4feedback.interface import uniform_connected_devices
from task4feedback.graphs.jacobi import JacobiGraph, LevelPartitionMapper
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph


def create_system(cfg: DictConfig):
    system = hydra.utils.instantiate(cfg.system)
    return system


def create_conditions(cfg: DictConfig):
    transition_conditions = hydra.utils.instantiate(cfg.runtime)
    return transition_conditions


def create_runtime_reward(cfg: DictConfig):
    runtime_env_t = hydra.utils.instantiate(cfg.reward)
    return runtime_env_t


def create_observer_factory(cfg: DictConfig):
    graph_spec = hydra.utils.instantiate(cfg.feature.observer.spec)
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
    res = {"eft": [], "best": [], "best_f": []}
    # track max memory usage for each strategy
    mem_usage = {"eft": [], "global": [], "oracle": []}

    # Factorize cfg.graph.steps
    f = []
    for i in range(cfg.graph.config.steps // 2):
        if cfg.graph.config.steps % (i + 1) == 0:
            f.append(i + 1)
            res[str(i + 1)] = []
    print(f"Factors of {cfg.graph.config.steps}: {f}")
    assert isinstance(env.simulator_factory.input.graph, DynamicJacobiGraph)
    data_stat = env.simulator_factory.input.graph.data.data_stat
    mem_size = data_stat["average_step_data"] // 4
    # level_total_mem = data_stat["average_step_data"] * 3

    with_retire = (cfg.graph.config.n**2) * (
        1 * data_stat["interior_average"]  # Task's
        + 4 * data_stat["boundary_average"]  # Task's
        + 4 * data_stat["boundary_average"]  # Neighbor's data
        + 1 * data_stat["interior_average"]  # Next Step
        + 4 * data_stat["boundary_average"]  # Next Step
        + 1 * data_stat["interior_average"]  # Retire Data
        + 4 * data_stat["boundary_average"]  # Retire Data
    )
    without_retire = (cfg.graph.config.n**2) * (
        1 * data_stat["interior_average"]  # Task's
        + 4 * data_stat["boundary_average"]  # Task's
        + 4 * data_stat["boundary_average"]  # Neighbor's data
        + 1 * data_stat["interior_average"]  # Next Step
        + 4 * data_stat["boundary_average"]  # Next Step
    ) - 1 * (cfg.graph.config.n * 4) * data_stat[
        "boundary_average"
    ]  # Correcting boundary

    print(f"with Retire Memory Size: {with_retire}")
    print(f"without Retire Memory Size: {without_retire}")
    mem_size = int((mem_size // 10000 + 1) * 10000)
    # mem_size = 480000
    step_size = 10000
    graph: JacobiGraph = env.simulator_factory.input.graph
    num_samples = 10
    print(f"Memory,EFT,{str.join(',',[str(m) for m in f])},best,best_f", flush=True)
    while True:
        mem.append(4 * mem_size / without_retire)
        s = uniform_connected_devices(
            n_devices=cfg.system.n_devices,
            bandwidth=cfg.system.bandwidth,
            latency=cfg.system.latency,
            mem=mem_size,
        )
        env.simulator_factory.input.system = s
        for k, v in res.items():
            if k not in ["best", "best_f"]:
                v.append(0)
        # initialize memory usage accumulators
        mem_usage["eft"].append(0)
        mem_usage["global"].append(0)
        mem_usage["oracle"].append(0)
        # temp storage to accumulate per-factor memory usage for Oracle
        factor_mem_usage = {i: 0 for i in f}
        for i in range(num_samples):
            env._reset()
            sim = env.simulator_factory.create()
            sim.disable_external_mapper()
            sim.run()
            res["eft"][-1] += sim.time
            mem_usage["eft"][-1] += sim.max_mem_usage
            for i in f:
                graph.mincut_per_levels(
                    bandwidth=cfg.system.bandwidth, level_chunks=i, offset=1
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
        # compute Oracle memory usage based on best factor
        res["best"].append(res["eft"][-1])
        res["best_f"].append("-1")
        for k, v in res.items():
            if k in ["best", "best_f"]:
                continue
            v[-1] = int(v[-1] / num_samples)
            if k != "eft" and v[-1] < res["best"][-1]:
                res["best"][-1] = v[-1]
                res["best_f"][-1] = k
        best_factor = int(res["best_f"][-1])
        mem_usage["oracle"][-1] = factor_mem_usage[best_factor] / num_samples

        print(f"{mem_size},{res['eft'][-1]}", end=" ")
        for i in f:
            print(f"{res[str(i)][-1]}", end=",")
        print(f"{res['best'][-1]},{res['best_f'][-1]}")
        mem_size += step_size
        if 4 * mem_size / without_retire > 2:
            break

    import matplotlib.pyplot as plt

    # Compute relative speedup vs EFT
    speedup = []
    for idx in range(len(res["eft"])):
        # find the fastest factor run at this memory size
        best_time = min(res[str(i)][idx] for i in f)
        speedup.append(res["eft"][idx] / best_time)

    # Plot execution times and relative speedup
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Primary axis: execution times
    ax1.plot(mem, res["eft"], label="EFT", marker="o")
    # for i in f:
    #     ax1.plot(mem, res[str(i)], label=f"f={i}", linestyle="dashed")
    ax1.plot(mem, res["best"], label="Oracle", marker="x", color="green")
    ax1.plot(mem, res["1"], label="Global Min-Cut", linestyle="dashed", color="orange")
    ax1.set_xlabel("Problem Size / Total GPUs Memory Size")
    ax1.set_ylabel("Execution Time (s)")
    ax1.legend(loc="upper left")

    # Secondary axis: speedup vs EFT
    ax2 = ax1.twinx()
    ax2.plot(
        mem,
        speedup,
        label="Speedup",
        color="red",
    )
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
    ax2.set_xlabel("Problem Size / Total GPUs Memory Size")
    ax2.set_ylabel("Max Memory Usage")
    ax2.legend(loc="upper right")
    ax2.set_title("Max Memory Usage vs Memory Size")
    ax2.grid()
    plt.show()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    test(cfg)


if __name__ == "__main__":
    main()
