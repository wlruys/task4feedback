from email import policy
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import create_td_actor_critic_models
from task4feedback.experiment_helper.algorithm import create_optimizer, create_lr_scheduler
from task4feedback.graphs.mesh.plot import _build_state
from task4feedback.ml.algorithms.ppo import run_ppo
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
from task4feedback.ml.env import RuntimeEnv

# from task4feedback.graphs.mesh.plot_fast import *
# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")

import torch
import numpy
import random

import pickle
import json


def export_tasks_trace(
    env: RuntimeEnv,
    out_file: str = "tasks_trace.json",
    process_name: str = "Simulator",
):
    """
    Export compute + data tasks to Chrome Trace / Perfetto format.

    - Compute tasks: single-lane execution
    - Data tasks: source + destination lanes + flow arrows
    """
    static_state, dynamic_state = _build_state(env)
    graph = env.get_graph()
    assert isinstance(graph, JacobiGraph)
    trace_events = []
    pid = 0

    # ------------------------------------------------------------
    # Process metadata
    # ------------------------------------------------------------
    trace_events.append(
        {
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "args": {"name": process_name},
        }
    )

    # ============================================================
    # Compute Tasks
    # ============================================================
    n_ct = static_state.n_compute_tasks

    for i in range(n_ct):
        launch = static_state.ct_launch_time[i]
        complete = static_state.ct_complete_time[i]

        if launch < 0 or complete < 0:
            continue

        duration = int(complete - launch)
        device = int(static_state.ct_device[i]) - 1
        cell = int(static_state.ct_cell[i])

        task = graph.get_task(i)
        used_datablocks = task.read

        trace_events.append(
            {
                "name": f"CT[{i}]",
                "cat": "compute_task",
                "ph": "X",
                "pid": pid,
                "tid": device,
                "ts": int(launch),
                "dur": duration,
                "args": {
                    "task_id": i,
                    "device": device,
                    "cell": cell,
                    "duration_us": float(static_state.ct_duration_us[i]),
                    "state": int(dynamic_state.ct_state[i]),
                    "changed": bool(dynamic_state.ct_changed[i]),
                    "used_datablocks": used_datablocks,
                },
            }
        )

    # ============================================================
    # Data Tasks
    # ============================================================
    n_dt = static_state.n_data_tasks

    for i in range(n_dt):
        launch = static_state.dt_launch_time[i]
        complete = static_state.dt_complete_time[i]

        if launch < 0 or complete < 0:
            continue

        if static_state.dt_virtual[i]:
            continue

        duration = int(complete - launch)

        src = int(static_state.dt_source[i]) - 1
        dst = int(static_state.dt_device[i]) - 1
        block = int(static_state.dt_block[i])

        # ---- Source side ----
        trace_events.append(
            {
                "name": f"DT[{i}] send",
                "cat": "data_task",
                "ph": "X",
                "pid": pid,
                "tid": src,
                "ts": int(launch),
                "dur": duration,
                "args": {
                    "task_id": i,
                    "role": "source",
                    "source_device": src,
                    "dest_device": dst,
                    "block_id": block,
                },
            }
        )

        # ---- Destination side ----
        trace_events.append(
            {
                "name": f"DT[{i}] recv",
                "cat": "data_task",
                "ph": "X",
                "pid": pid,
                "tid": dst,
                "ts": int(launch),
                "dur": duration,
                "args": {
                    "task_id": i,
                    "role": "destination",
                    "source_device": src,
                    "dest_device": dst,
                    "block_id": block,
                },
            }
        )

    # ------------------------------------------------------------
    # Write trace
    # ------------------------------------------------------------
    trace = {
        "traceEvents": trace_events,
        "displayTimeUnit": "us",
    }

    with open(out_file, "w") as f:
        json.dump(trace, f, indent=2)

    print(f"[✓] Trace written to: {out_file}")


class ReplayMapper:
    def __init__(self, history):
        self.history = history

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        candidates = torch.zeros((simulator.observer.graph_spec.max_candidates), dtype=torch.int64)
        num_candidates = simulator.simulator.get_mappable_candidates(candidates)
        mapping_result = []
        for i in range(num_candidates):
            global_task_id = candidates[i].item()
            device = self.history[global_task_id]
            mapping_priority = simulator.simulator.get_state().get_mapping_priority(global_task_id)
            mapping_result.append(fastsim.Action(i, device, mapping_priority, mapping_priority))
        return mapping_result


@hydra.main(config_path="conf", config_name="static_single.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)
    results = []
    for ir, br in [(0.1, 0.1), (1, 0.1), (1, 1), (10, 1)]:
        # for ir, br in [(1, 0.1)]:
        cfg.graph.config.n = 8
        cfg.graph.config.steps = 256
        cfg.graph.config.arithmetic_intensity = 0
        cfg.graph.env.change_location = False
        cfg.graph.init.randomize = False

        cfg.system.d2d_bw = 129 * 2**30

        cfg.graph.config.compute_time = 500

        cfg.graph.config.boundary_size = (cfg.graph.config.compute_time * cfg.system.d2d_bw / 1e6 * br / 4) // (2**20) * (2**20)
        cfg.graph.config.interior_size = (cfg.graph.config.compute_time * cfg.system.d2d_bw / 1e6 * ir) // (2**20) * (2**20)
        cfg.system.d2d_bw = 100 * 2**30

        # cfg.graph.config.compute_time = 1

        print(f"Interior size: {cfg.graph.config.interior_size / (2**20)}MB, Boundary size: {cfg.graph.config.boundary_size / (2**20)}MB")
        print(f"Interior movement time: {cfg.graph.config.interior_size / cfg.system.d2d_bw * 1e6}us")
        print(f"Boundary movement time: {cfg.graph.config.boundary_size * 4 / cfg.system.d2d_bw * 1e6}us")

        cfg.system.device_copyengines = 3
        cfg.system.d2d_links = 1
        cfg.system.n_devices = 5
        # cfg.system.n_devices = 2
        cfg.system.cpu_copyengines = 2
        cfg.system.h2d_links = 1
        cfg.system.h2d_bw = 55 * 2**30
        cfg.system.mem = 999e9
        # cfg.system.mem = 62e9
        cfg.system.latency = 0

        graph_builder = make_graph_builder(cfg)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
        graph = env.get_graph()
        assert isinstance(graph, JacobiGraph)
        info = graph.info_to_dict()
        cell_locations = graph.data.get_locations()
        cell_locations = {i: cell_locations[i] - 1 for i in range(len(cell_locations))}

        blockinfo = {}
        datablocks = graph.data.blocks
        for i in range(len(datablocks)):
            blockinfo[i] = {
                "location": datablocks.data.get_location(i) - 1,
                "size": datablocks.data.get_size(i),
            }

        # block_info

        sim = env.simulator
        copy_sim = env.simulator.copy()
        copy_sim.disable_external_mapper()
        copy_sim.run()
        runtime = copy_sim.state.get_task_runtime()
        # sim.external_mapper = JacobiQuadrantMapper(graph=graph, n_devices=4, offset=1)
        # history = {}
        # for task_id in range(cfg.graph.config.n**2 * cfg.graph.config.steps):
        #     history[task_id] = runtime.get_compute_task_mapped_device(task_id)
        # sim.external_mapper = ReplayMapper(history)

        for task_id in range(cfg.graph.config.n**2 * cfg.graph.config.steps):
            for block_id in info[task_id]["read"]:
                sim.initialize_data_replicate(block_id, runtime.get_compute_task_mapped_device(task_id))
                # sim.initialize_data_replicate(block_id, sim.external_mapper.mapping_from_id(task_id))
            for block_id in info[task_id]["write"]:
                sim.initialize_data_replicate(block_id, runtime.get_compute_task_mapped_device(task_id))
            # sim.initialize_data_replicate(block_id, sim.external_mapper.mapping_from_id(task_id))

        # sim.enable_external_mapper()
        sim.disable_external_mapper()
        sim.run()
        runtime = sim.state.get_task_runtime()
        sum_duration = []
        for k, v in info.items():
            info[k]["mapped_location"] = runtime.get_compute_task_mapped_device(k) - 1
            info[k]["duration"] = runtime.get_compute_task_duration(k)
            sum_duration.append(info[k]["duration"])
        # print(info)
        graph_info = {
            "task": info,
            "data": blockinfo,
            "time": sim.time,
            "interior_ratio": ir,
            "boundary_ratio": br,
            "interior_size": int(cfg.graph.config.interior_size // (2**20)),
            "boundary_size": int(cfg.graph.config.boundary_size // (2**20)),
            "bandwidth": int(cfg.system.d2d_bw // (2**30)),
            "mapper": "eft",
            "compute_time": int(cfg.graph.config.compute_time),
            "cell_locations": cell_locations,
        }
        pickle.dump(graph_info, open(f"{ir}_{br}.pkl", "wb"))
        results.append((ir, br, sim.time / 1e6, sum(sum_duration) / len(sum_duration)))
        export_tasks_trace(env, out_file=f"trace_ir{ir}_br{br}.json", process_name="StaticJacobiEval")
    print(results)


if __name__ == "__main__":
    main()
