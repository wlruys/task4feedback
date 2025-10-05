from email import policy
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from helper.graph import make_graph_builder
from helper.env import make_env
from helper.model import create_td_actor_critic_models, load_policy_from_checkpoint
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
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def configure_training(cfg: DictConfig):
    # start_logger()
    mem_list = np.linspace(80e9, 100e9, 1)
    if rank == 0:
        model_path = "/home/cc/task4feedback_torchrl/experiments/saved_models/8x8x128_0.1-0.1-1_corners_CNN/models/8x8x128_0.1-0.1-1_corners_CNN_A_DeviceTrue_1Frames_1.409_1600000000.pt"
        if "_A_" in model_path:
            cfg.feature.observer.version = "A"
        elif "_B_" in model_path:
            cfg.feature.observer.version = "B"
        elif "_C_" in model_path:
            cfg.feature.observer.version = "C"
        elif "_D_" in model_path:
            cfg.feature.observer.version = "D"
        else:
            raise ValueError("Unknown model version")

        # mem_list = [70e9, 80e9, 90e9]
        # reverse the list
        # mem_list = mem_list[::-1]
        times = {
            "METIS": [0 for _ in mem_list],
            "ParMETIS": [0 for _ in mem_list],
            "BlockCyclic": [0 for _ in mem_list],
            "RL": [0 for _ in mem_list],
        }

        graph_builder = make_graph_builder(cfg)
        env, norm = make_env(graph_builder=graph_builder, cfg=cfg, normalization=None)
        observer = env.get_observer()
        feature_config = FeatureDimConfig.from_observer(observer)
        model, _, _ = create_td_actor_critic_models(cfg, feature_config)
        if not load_policy_from_checkpoint(model, model_path):
            raise ValueError("Failed to load model from checkpoint")

    for idx, gpu_mem in enumerate(mem_list):
        if rank == 0:
            cfg.system.mem = int(gpu_mem)
            graph_builder = make_graph_builder(cfg)
            env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=norm)
            env.disable_reward()
            graph = env.get_graph()
        for i in range(10):
            if rank == 0:
                td = env.rollout(max_steps=1000000, policy=model.actor)
                rl = td["observation", "aux", "time"][-1].item()
                times["RL"][idx] += rl
                base_sim = env.simulator.fresh_copy()
                base_sim.initialize()
                base_sim.initialize_data()
                base_sim.enable_external_mapper()
                graph = env.get_graph()

            if rank == 0:
                sim = base_sim.copy()
                graph.mincut_per_levels(
                    bandwidth=cfg.system.d2d_bw,
                    mode="metis",
                    offset=1,
                    level_chunks=64,
                )
                graph.align_partitions()
                sim.enable_external_mapper()
                sim.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
                sim.run()
                metis = sim.time
                times["METIS"][idx] += sim.time

                sim = base_sim.copy()
                sim.enable_external_mapper()
                sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=2, offset=1)
                sim.run()
                block_cyclic = sim.time
                times["BlockCyclic"][idx] += sim.time

                sim = base_sim.copy()
            run_parmetis(sim=sim if rank == 0 else None, cfg=cfg, itr=0.0001001, unbalance=1.02)

            if rank == 0:
                parmetis = sim.time
                times["ParMETIS"][idx] += sim.time

                print(
                    f"Memory: {gpu_mem/1e9}e9, Iteration {i+1}/10, RL time: {td['observation', 'aux', 'time'][-1].item()}, METIS time: {metis}, ParMETIS time: {parmetis}, BlockCyclic time: {block_cyclic}"
                )

    if rank == 0:
        # plot results
        norm_coef = min(min(times["METIS"]), min(times["ParMETIS"]), min(times["BlockCyclic"]))
        # plt.plot(mem_list / 80e9, np.array(time_sums) / min(time_sums))
        # plt.plot(mem_list / 80e9, np.array(ml_sums) / min(time_sums))
        for key in times:
            times[key] = np.array(times[key]) / norm_coef
            plt.plot(mem_list / 80e9, times[key], label=key)
        plt.legend()
        plt.xlabel("Required Memory / GPU Memory")
        plt.ylabel("Normalized Makespan")
        plt.title("Memory vs Makespan Trade-off")
        plt.grid()
        plt.savefig("memory_vs_makespan.png")


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)
    param = {}
    for interior in [0.1, 1, 10]:
        for boundary in [0.1, 1, 10]:
            if interior < boundary:
                continue
            param[(interior, boundary)] = (f"{595.5555555/interior:.7f}"[:-1], f"{0.25 / (interior / boundary)}")

    cfg.graph.config.arithmetic_intensity = float(param[(cfg.sweep.i, cfg.sweep.b)][0])
    cfg.graph.config.boundary_width = float(param[(cfg.sweep.i, cfg.sweep.b)][1])

    configure_training(cfg)


if __name__ == "__main__":
    main()
