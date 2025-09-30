from email import policy
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate
import matplotlib.pyplot as plt

from helper.graph import make_graph_builder
from helper.env import make_env
from helper.model import create_td_actor_critic_models
from helper.algorithm import create_optimizer, create_lr_scheduler

from task4feedback.ml.algorithms.ppo import run_ppo, run_ppo_lstm
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *
from task4feedback.graphs.jacobi import (
    JacobiGraph,
    LevelPartitionMapper,
    JacobiRoundRobinMapper,
    JacobiQuadrantMapper,
    BlockCyclicMapper,
    GraphMETISMapper,
)

# from task4feedback.graphs.mesh.plot_fast import *
# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")

from hydra.experimental.callbacks import Callback
from hydra.core.utils import JobReturn
from omegaconf import DictConfig, open_dict
from pathlib import Path
import git
import os
from hydra.core.hydra_config import HydraConfig
from helper.run_name import make_run_name, cfg_hash
import torch
import numpy
import random
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.mesh.plot import animate_mesh_graph
from task4feedback.ml.util import EvaluationConfig
from helper.parmetis import run_parmetis
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def configure_training(cfg: DictConfig):
    # start_logger()
    step = 5e9
    start = 30e9
    samples = 10
    level_chunks = 64
    mem_list = []
    const_policy_samples: list[list[float]] = []
    const_eft_samples: list[list[float]] = []
    inf_policy_samples: list[list[float]] = []
    inf_eft_samples: list[list[float]] = []

    inf_cfg = cfg.copy()
    inf_cfg.system.mem = 99999e9

    while True:
        cfg.graph.config.level_memory = start
        inf_cfg.graph.config.level_memory = start
        const_env = make_env(graph_builder=make_graph_builder(cfg), cfg=cfg, normalization=False)
        const_env.disable_reward()
        inf_env = make_env(graph_builder=make_graph_builder(inf_cfg), cfg=inf_cfg, normalization=False)
        inf_env.disable_reward()
        const_env.get_graph()
        inf_env.get_graph()

        const_samples_policy = []
        const_samples_eft = []
        inf_samples_policy = []
        inf_samples_eft = []

        for _ in range(samples):
            const_env._reset()
            inf_env._reset()

            # base = inf_env._get_baseline("Quad")
            # inf_samples_policy.append(1.0)
            # inf_samples_eft.append(inf_env._get_baseline("EFT") / base)
            # const_samples_policy.append(const_env._get_baseline("Quad") / base)
            # const_samples_eft.append(const_env._get_baseline("EFT") / base)
            inf_graph = inf_env.get_graph()
            inf_graph.mincut_per_levels(
                bandwidth=inf_cfg.system.d2d_bw,
                mode="metis",
                offset=1,
                level_chunks=level_chunks,
            )
            inf_graph.align_partitions()
            inf_sim = inf_env.simulator
            inf_sim.external_mapper = LevelPartitionMapper(level_cell_mapping=inf_graph.partitions)
            inf_sim.run()
            base = inf_sim.time

            const_graph = const_env.get_graph()
            const_graph.mincut_per_levels(
                bandwidth=cfg.system.d2d_bw,
                mode="metis",
                offset=1,
                level_chunks=level_chunks,
            )
            const_graph.align_partitions()
            const_sim = const_env.simulator
            const_sim.external_mapper = LevelPartitionMapper(level_cell_mapping=const_graph.partitions)
            const_sim.run()

            inf_samples_policy.append(1.0)
            inf_samples_eft.append(inf_env._get_baseline("EFT") / base)
            const_samples_policy.append(const_sim.time / base)
            const_samples_eft.append(const_env._get_baseline("EFT") / base)

        const_policy_samples.append(const_samples_policy)
        const_eft_samples.append(const_samples_eft)
        inf_policy_samples.append(inf_samples_policy)
        inf_eft_samples.append(inf_samples_eft)

        mem_list.append(start / 1e9)
        start += step
        if float(numpy.mean(const_samples_policy)) > 1.5:
            break

    fig, ax = plt.subplots()

    def plot_with_uncertainty(x_vals, samples_list, label, color):
        means = numpy.array([numpy.mean(s) for s in samples_list])
        stds = numpy.array([numpy.std(s) for s in samples_list])
        x_arr = numpy.array(x_vals)
        ax.plot(x_arr, means, marker="o", label=label, color=color)
        ax.fill_between(x_arr, means - stds, means + stds, alpha=0.2, color=color)

    colors = {
        "MinCut": "tab:blue",
        "EFT": "tab:orange",
        "MinCut(Inf)": "tab:green",
        "EFT(Inf)": "tab:red",
    }

    # show numbers for MinCut
    for i, mem in enumerate(mem_list):
        mean = numpy.mean(const_policy_samples[i])
        std = numpy.std(const_policy_samples[i])
        ax.text(
            mem,
            mean + 0.05,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=colors["MinCut"],
        )

    plot_with_uncertainty(mem_list, const_policy_samples, "MinCut", colors["MinCut"])
    plot_with_uncertainty(mem_list, const_eft_samples, "EFT", colors["EFT"])
    plot_with_uncertainty(mem_list, inf_policy_samples, "MinCut(Inf)", colors["MinCut(Inf)"])
    plot_with_uncertainty(mem_list, inf_eft_samples, "EFT(Inf)", colors["EFT(Inf)"])
    ax.set_xlabel("Level Memory(GB)")
    ax.set_ylabel("Normalized Runtime")
    ax.set_title(f"{cfg.graph.config.n}x{cfg.graph.config.n}")
    ax.legend()
    plt.tight_layout()
    out_path = "outputs/const_vs_inf.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")
    # plt.show()


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
