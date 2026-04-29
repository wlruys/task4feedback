import os  # Added import
import pickle
import random

import hydra
import numpy
import torch
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf

from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.parmetis import (
    find_best_cfg,
    find_best_cfg_optuna,
    run_parmetis,
)
from task4feedback.experiment_helper.run_name import make_folder_name
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.graphs.jacobi import (
    BlockCyclicMapper,
    JacobiQuadrantMapper,
    JacobiRoundRobinMapper,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def configure_training(cfg: DictConfig):
    # start_logger()
    extend = cfg.get("extend", 1)
    num_samples = cfg.eval.samples
    parmetis = ParMETIS_wrapper()
    folder_name, graph_name, interior_str, boundary_str = make_folder_name(cfg)

    # Define the file path consistently
    file_path = f"./pickled_evaluation/{folder_name}.pkl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    cfg.graph.config.steps *= extend

    # --- Start: Check for existing file and matching config ---
    skip_execution = False
    if rank == 0 and os.path.exists(file_path):
        try:
            print(
                f"Found existing file at {file_path}. Checking config...",
                flush=True,
            )
            with open(file_path, "rb") as f:
                saved_state = pickle.load(f)

            # specific check: Compare current cfg YAML with saved cfg YAML
            current_cfg_yaml = OmegaConf.to_yaml(cfg)
            saved_cfg_yaml = saved_state.get("cfg", "")

            if current_cfg_yaml == saved_cfg_yaml:
                print(
                    "Configuration matches exactly. Skipping computation.",
                    flush=True,
                )
                skip_execution = True
            else:
                print(
                    "Configuration mismatch (file exists but cfg differs). Overwriting.",
                    flush=True,
                )
        except Exception as e:
            print(f"Error reading existing pickle (will overwrite): {e}", flush=True)

    # Broadcast decision to all ranks to ensure no rank hangs at a barrier
    skip_execution = comm.bcast(skip_execution, root=0)

    if skip_execution:
        return
    # --- End: Check for existing file ---

    eval_state = {
        "cfg": OmegaConf.to_yaml(cfg),
        "init_locs": [],
        "workloads": [],
        "eft_times": None,
        "policy_times": [],
        "reset_counter": [],
    }
    if rank == 0:
        graph_builder = make_graph_builder(cfg)
        env = make_env(
            graph_builder=graph_builder, cfg=cfg, normalization=False, eval=True
        )
        env.set_reset_counter(9999)
    else:
        env = None

    # First find the best configuration for parmetis
    best_cfg = None
    if best_cfg is None:
        best_cfg = find_best_cfg_optuna(
            cfg, parmetis, env=env, skip_search=True, mode="normal_optuna"
        )
    best_cfg = comm.bcast(best_cfg, root=0)
    assert best_cfg is not None, "Best configuration for ParMETIS not found!"

    for _i in range(num_samples):
        if rank == 0:
            eval_state["reset_counter"].append(env.resets)
            env.reset()
            copy_sim = env.simulator.copy()
            eval_state["init_locs"].append(
                env.get_graph().get_cell_locations(as_dict=False)
            )
            graph = env.get_graph()
            if isinstance(graph, DynamicJacobiGraph):
                eval_state["workloads"].append(
                    dict(graph.get_workload().level_workload)
                )
            else:
                eval_state["workloads"].append(None)
            # copy_sim.disable_external_mapper()
            # copy_sim.run()
            # eval_state["eft_times"].append(99999999999)
            # print(f"Eval {i}:\n EFT time {copy_sim.time}")

            copy_sim = env.simulator.copy()
        comm.barrier()
        run_parmetis(
            sim=(copy_sim if rank == 0 else None),
            cfg=cfg,
            itr=best_cfg[0],
            unbalance=best_cfg[1],
            n_compute_devices=cfg.system.n_devices - 1,
        )
        if rank == 0:
            # policy_time = min(copy_sim.time, eval_state["eft_times"][-1])
            policy_time = copy_sim.time
            print(f"ParMETIS time: {copy_sim.time}")

            # Block Cyclic 4x4
            if cfg.system.n_devices - 1 == 4:
                copy_sim = env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.external_mapper = BlockCyclicMapper(
                    geometry=copy_sim.input.graph.data.geometry,
                    n_devices=cfg.system.n_devices - 1,
                    block_size=4,
                    offset=1,
                )
                copy_sim.run()
                policy_time = min(copy_sim.time, policy_time)
                print(f"Block Cyclic 4x4 time: {copy_sim.time}")
            elif cfg.system.n_devices - 1 == 8:
                copy_sim = env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.external_mapper = JacobiQuadrantMapper(
                    graph=copy_sim.input.graph,
                    n_devices=cfg.system.n_devices - 1,
                    offset=1,
                )
                copy_sim.run()
                policy_time = min(copy_sim.time, policy_time)
                print(f"Block Cyclic 4x4 time: {copy_sim.time}")

            # Block Cyclic 2x2
            copy_sim = env.simulator.copy()
            copy_sim.enable_external_mapper()
            copy_sim.external_mapper = BlockCyclicMapper(
                geometry=copy_sim.input.graph.data.geometry,
                n_devices=cfg.system.n_devices - 1,
                block_size=2,
                offset=1,
            )
            copy_sim.run()
            policy_time = min(copy_sim.time, policy_time)
            print(f"Block Cyclic 2x2 time: {copy_sim.time}")

            # Block Cyclic 1x1
            copy_sim = env.simulator.copy()
            copy_sim.enable_external_mapper()
            copy_sim.external_mapper = BlockCyclicMapper(
                geometry=copy_sim.input.graph.data.geometry,
                n_devices=cfg.system.n_devices - 1,
                block_size=1,
                offset=1,
            )
            copy_sim.run()
            policy_time = min(copy_sim.time, policy_time)
            print(f"Block Cyclic 1x1 time: {copy_sim.time}")

            # RowCyclic
            copy_sim = env.simulator.copy()
            copy_sim.enable_external_mapper()
            copy_sim.external_mapper = JacobiRoundRobinMapper(
                n_devices=cfg.system.n_devices - 1, setting=1, offset=1
            )
            copy_sim.run()
            policy_time = min(copy_sim.time, policy_time)
            print(f"RowCyclic time: {copy_sim.time}")

            eval_state["policy_times"].append(policy_time)
            # print(f"{i}: EFT {eval_state['eft_times'][-1]:.4f}, {eval_state['policy_times'][-1]:.4f} ({eval_state['eft_times'][-1]/eval_state['policy_times'][-1]:.2f}x)")
    # print(eval_state)
    # pickle.dump(eval_state, open("4x4x16_static_1:1:1.pkl", "wb"))
    if rank == 0:
        # env.set_reset_counter(0)
        # env._reset()

        # # eval_state = pickle.load(open("dynamic_bump_eval.pkl", "rb"))
        # for i in range(num_samples):
        #     env.set_reset_counter(eval_state["reset_counter"][i])
        #     env.reset()
        #     # env.reset_to_state(saved_loc, workload)
        #     print(f"Eval {i}:")
        #     sim_time = env._get_baseline("EFT")
        #     if eval_state["eft_times"][i] != sim_time:
        #         print(f"  Warning: EFT time changed! {eval_state['eft_times'][i]} -> {sim_time}")
        #         raise ValueError("EFT time mismatch")
        #     else:
        #         print("EFT time matches.")
        #     # print("EFT:", eval_state["eft_times"][i])
        # else:
        #     # Modified to use the file_path variable defined earlier
        pickle.dump(eval_state, open(file_path, "wb"))


@hydra.main(config_path="conf", config_name="static_batch.yaml", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
