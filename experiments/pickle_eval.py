import hydra
from omegaconf import DictConfig, OmegaConf
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.run_name import make_folder_name

from task4feedback.graphs.jacobi import (
    JacobiRoundRobinMapper,
    BlockCyclicMapper,
)
import torch
import numpy
import random
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.experiment_helper.parmetis import run_parmetis
import pickle
import os  # Added import

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def configure_training(cfg: DictConfig):
    # start_logger()
    extend = cfg.extend
    num_samples = cfg.eval.samples

    folder_name, graph_name, interior_str, boundary_str = make_folder_name(cfg)

    # Define the file path consistently
    file_path = f"./pickled_evaluation/{folder_name}.pkl"

    cfg.graph.config.steps *= extend
    if cfg.graph.config.workload_args.traj_type == "circle":
        cfg.graph.config.workload_args.traj_specifics.max_angle *= extend

    # --- Start: Check for existing file and matching config ---
    skip_execution = False
    if rank == 0:
        if os.path.exists(file_path):
            try:
                print(f"Found existing file at {file_path}. Checking config...", flush=True)
                with open(file_path, "rb") as f:
                    saved_state = pickle.load(f)

                # specific check: Compare current cfg YAML with saved cfg YAML
                current_cfg_yaml = OmegaConf.to_yaml(cfg)
                saved_cfg_yaml = saved_state.get("cfg", "")

                if current_cfg_yaml == saved_cfg_yaml:
                    print("Configuration matches exactly. Skipping computation.", flush=True)
                    skip_execution = True
                else:
                    print("Configuration mismatch (file exists but cfg differs). Overwriting.", flush=True)
            except Exception as e:
                print(f"Error reading existing pickle (will overwrite): {e}", flush=True)

    # Broadcast decision to all ranks to ensure no rank hangs at a barrier
    skip_execution = comm.bcast(skip_execution, root=0)

    if skip_execution:
        return
    # --- End: Check for existing file ---

    eval_state = {"cfg": OmegaConf.to_yaml(cfg), "init_locs": [], "workloads": [], "eft_times": [], "policy_times": [], "reset_counter": []}
    if rank == 0:
        graph_builder = make_graph_builder(cfg)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False, eval=True)
        env.set_reset_counter(9999)

    # First find the best configuration for parmetis
    best_cfg = (None, None, float("inf"))  # (itr, ub, time)
    ub_cur = 1.0001
    for itr in [0.0001001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]:
        if rank == 0:
            temp = env.simulator.copy()
        comm.barrier()
        status = run_parmetis(sim=temp if rank == 0 else None, cfg=cfg, unbalance=ub_cur, itr=itr, n_compute_devices=cfg.system.n_devices - 1)
        if rank == 0 and temp.time < best_cfg[2]:
            best_cfg = (itr, ub_cur, temp.time)
            print(f"New best ITR {itr} with time {temp.time}", flush=True)

    best_cfg = comm.bcast(best_cfg, root=0)
    ub_list = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
    for ub in ub_list:
        if rank == 0:
            temp = env.simulator.copy()
        comm.barrier()
        status = run_parmetis(sim=(temp if rank == 0 else None), cfg=cfg, unbalance=ub, itr=best_cfg[0], n_compute_devices=cfg.system.n_devices - 1)
        if not status:
            break
        if rank == 0:
            print(f"Tried ub {ub:.2f} with time {temp.time}", flush=True)
            if temp.time < best_cfg[2]:
                # Improvement: accept move, keep direction, keep step
                ub_cur = ub
                best_cfg = (best_cfg[0], ub_cur, temp.time)
                print(f"New best ub {ub_cur:.2f} with time {temp.time}", flush=True)

    best_cfg = comm.bcast(best_cfg, root=0)

    for i in range(num_samples):
        if rank == 0:
            eval_state["reset_counter"].append(env.resets)
            env.reset()
            copy_sim = env.simulator.copy()
            eval_state["init_locs"].append(env.get_graph().get_cell_locations(as_dict=False))
            graph = env.get_graph()
            if isinstance(graph, DynamicJacobiGraph):
                eval_state["workloads"].append(dict(graph.get_workload().level_workload))
            else:
                eval_state["workloads"].append(None)
            copy_sim.disable_external_mapper()
            copy_sim.run()
            eval_state["eft_times"].append(copy_sim.time)
            print(f"Eval {i}:\n EFT time {copy_sim.time}")

            copy_sim = env.simulator.copy()
        comm.barrier()
        run_parmetis(sim=(copy_sim if rank == 0 else None), cfg=cfg, itr=best_cfg[0], unbalance=best_cfg[1], n_compute_devices=cfg.system.n_devices - 1)
        if rank == 0:
            policy_time = min(copy_sim.time, eval_state["eft_times"][-1])
            print(f"ParMETIS time: {copy_sim.time}")

            # Block Cyclic 4x4
            copy_sim = env.simulator.copy()
            copy_sim.enable_external_mapper()
            copy_sim.external_mapper = BlockCyclicMapper(geometry=copy_sim.input.graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=4, offset=1)
            copy_sim.run()
            policy_time = min(copy_sim.time, policy_time)
            print(f"Block Cyclic 4x4 time: {copy_sim.time}")

            # Block Cyclic 2x2
            copy_sim = env.simulator.copy()
            copy_sim.enable_external_mapper()
            copy_sim.external_mapper = BlockCyclicMapper(geometry=copy_sim.input.graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=2, offset=1)
            copy_sim.run()
            policy_time = min(copy_sim.time, policy_time)
            print(f"Block Cyclic 2x2 time: {copy_sim.time}")

            # Block Cyclic 1x1
            copy_sim = env.simulator.copy()
            copy_sim.enable_external_mapper()
            copy_sim.external_mapper = BlockCyclicMapper(geometry=copy_sim.input.graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=1, offset=1)
            copy_sim.run()
            policy_time = min(copy_sim.time, policy_time)
            print(f"Block Cyclic 1x1 time: {copy_sim.time}")

            # RowCyclic
            copy_sim = env.simulator.copy()
            copy_sim.enable_external_mapper()
            copy_sim.external_mapper = JacobiRoundRobinMapper(n_devices=cfg.system.n_devices - 1, setting=1, offset=1)
            copy_sim.run()
            policy_time = min(copy_sim.time, policy_time)
            print(f"RowCyclic time: {copy_sim.time}")

            eval_state["policy_times"].append(policy_time)
            print(f"{i}: EFT {eval_state['eft_times'][-1]:.4f}, {eval_state['policy_times'][-1]:.4f} ({eval_state['eft_times'][-1]/eval_state['policy_times'][-1]:.2f}x)")
    # print(eval_state)
    # pickle.dump(eval_state, open("4x4x16_static_1:1:1.pkl", "wb"))
    if rank == 0:

        env.set_reset_counter(0)
        env._reset()

        # eval_state = pickle.load(open("dynamic_bump_eval.pkl", "rb"))
        for i in range(num_samples):
            env.set_reset_counter(eval_state["reset_counter"][i])
            env.reset()
            # env.reset_to_state(saved_loc, workload)
            print(f"Eval {i}:")
            sim_time = env._get_baseline("EFT")
            if eval_state["eft_times"][i] != sim_time:
                print(f"  Warning: EFT time changed! {eval_state['eft_times'][i]} -> {sim_time}")
                raise ValueError("EFT time mismatch")
            else:
                print("EFT time matches.")
            # print("EFT:", eval_state["eft_times"][i])
        else:
            # Modified to use the file_path variable defined earlier
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
