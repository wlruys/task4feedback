import fcntl
import os
import random
from collections.abc import Callable
from dataclasses import dataclass

import hydra
import numpy
import task4feedback.fastsim2 as fastsim
import torch
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf

from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.mapper import ReplayMapper
from task4feedback.experiment_helper.parmetis import find_best_cfg_optuna, run_parmetis
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.graphs.jacobi import (
    BlockCyclicMapper,
    JacobiQuadrantMapper,
    JacobiRoundRobinMapper,
)
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
EVAL_GRAPH_STEPS = 256
PHASE_LENGTH = 128
SYSTEM_MEMORY = 96e9


@dataclass(frozen=True)
class MapperSpec:
    key: str
    label: str
    mode: str  # parmetis | eft | external | internal
    mapper_factory: Callable[[DynamicJacobiGraph, DictConfig], object] | None = None
    internal_mapper_factory: Callable[[DictConfig], fastsim.Mapper] | None = None
    transition_factory: Callable[[DictConfig], fastsim.TransitionConditions] | None = None


def write_results_atomic(path, lines):
    """
    Append lines to a file using an exclusive file lock.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def csv_existing_mapper_entries(path, key_tuple):
    """
    Return mapper keys that already exist for the base key in CSV.
    base key: (traj_type, level_memory, r_interior, r_boundary)
    """
    if not os.path.exists(path):
        return set()

    existing = set()
    key_tuple = tuple(map(str, key_tuple))
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            if tuple(parts[:4]) == key_tuple:
                existing.add(parts[4])
    return existing


def _cfg_list(cfg: DictConfig, path: str) -> list[str]:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def resolve_enabled_mapper_keys(
    cfg: DictConfig, available_keys: list[str]
) -> list[str]:
    enabled = _cfg_list(cfg, "run_all_mapper.enabled_keys")
    disabled = set(_cfg_list(cfg, "run_all_mapper.disabled_keys"))

    # Optional env override for quick local runs.
    env_disabled = os.getenv("DISABLED_MAPPERS", "").strip()
    if env_disabled:
        disabled.update(k.strip() for k in env_disabled.split(",") if k.strip())

    if not enabled:
        enabled = available_keys.copy()

    unknown_enabled = [k for k in enabled if k not in available_keys]
    if rank == 0 and unknown_enabled:
        print(
            f"[WARN] Unknown mapper keys in enabled list: {unknown_enabled}", flush=True
        )

    unknown_disabled = [k for k in disabled if k not in available_keys]
    if rank == 0 and unknown_disabled:
        print(
            f"[WARN] Unknown mapper keys in disabled list: {unknown_disabled}",
            flush=True,
        )

    return [k for k in enabled if k in available_keys and k not in disabled]


def build_mapper_specs(cfg: DictConfig) -> list[MapperSpec]:
    n_compute_devices = cfg.system.n_devices - 1

    def b4_factory(graph: DynamicJacobiGraph, local_cfg: DictConfig):
        if n_compute_devices == 4:
            return BlockCyclicMapper(
                geometry=graph.data.geometry,
                n_devices=local_cfg.system.n_devices - 1,
                block_size=4,
                offset=1,
            )
        return JacobiQuadrantMapper(
            graph=graph,
            n_devices=local_cfg.system.n_devices - 1,
            offset=1,
        )

    specs = [
        MapperSpec(key="parmetis", label="ParMETIS", mode="parmetis"),
        MapperSpec(key="eft", label="EFT", mode="eft"),
        MapperSpec(
            key="darts",
            label="DARTS",
            mode="internal",
            internal_mapper_factory=lambda _cfg: fastsim.DARTSMapper(),
            transition_factory=lambda _cfg: fastsim.DeviceThresholdTransitionConditions(
                0, -1
            ),
        ),
        MapperSpec(
            key="darts_pipeline",
            label="DARTS-Pipeline",
            mode="internal",
            internal_mapper_factory=lambda local_cfg: _make_darts_pipeline_mapper(
                local_cfg.system.n_devices - 1
            ),
            transition_factory=lambda local_cfg: fastsim.DARTSPipelineTransitionConditions(
                4, 4 * (local_cfg.system.n_devices - 1), 1
            ),
        ),
        MapperSpec(
            key="darts_extended",
            label="DARTS-Extended",
            mode="internal",
            internal_mapper_factory=lambda _cfg: _make_darts_extended_mapper(),
            transition_factory=lambda _cfg: fastsim.DeviceThresholdTransitionConditions(
                0, -1
            ),
        ),
        MapperSpec(
            key="b4",
            label="BlockCyclic(4x4)/Quadrant",
            mode="external",
            mapper_factory=b4_factory,
        ),
        MapperSpec(
            key="b2",
            label="BlockCyclic(2x2)",
            mode="external",
            mapper_factory=lambda graph, local_cfg: BlockCyclicMapper(
                geometry=graph.data.geometry,
                n_devices=local_cfg.system.n_devices - 1,
                block_size=2,
                offset=1,
            ),
        ),
        MapperSpec(
            key="rc",
            label="RowCyclic",
            mode="external",
            mapper_factory=lambda _graph, local_cfg: JacobiRoundRobinMapper(
                n_devices=local_cfg.system.n_devices - 1,
                offset=1,
                setting=1,
            ),
        ),
    ]

    if n_compute_devices not in (4, 8):
        specs = [s for s in specs if s.key != "b4"]

    return specs


def _make_darts_pipeline_mapper(n_compute_devices: int) -> fastsim.DARTSMapper:
    mapper = fastsim.DARTSMapper()
    mapper.pipeline_depth = 4
    mapper.starvation_threshold = 1
    mapper.max_in_flight = 4 * n_compute_devices
    return mapper


def _make_darts_extended_mapper() -> fastsim.DARTSMapper:
    mapper = fastsim.DARTSMapper()
    mapper.extended_frontier_enabled = True
    mapper.extended_batch_emission_enabled = True
    mapper.extended_batch_emission_cap = 2
    return mapper


def _sim_base_metrics(sim):
    eviction = sum(list(sim.total_eviction_movement())[1:])
    data_movement = sum(sim.total_data_movement())
    return sim.time, eviction, data_movement


def _extended_metrics(sim, inf_sim, hand_calculated_peak, single_device_peak):
    _, eviction, _ = _sim_base_metrics(sim)
    mapped_peak = inf_sim.max_mem_usage if eviction > 0 else sim.max_mem_usage
    return (
        *_sim_base_metrics(sim),
        mapped_peak,
        hand_calculated_peak,
        single_device_peak,
    )


def _inf_extended_metrics(inf_sim, hand_calculated_peak, single_device_peak):
    time, _, data_movement = _sim_base_metrics(inf_sim)
    return (
        time,
        0,
        data_movement,
        inf_sim.max_mem_usage,
        hand_calculated_peak,
        single_device_peak,
    )


def run_eft_once(env, infenv, hand_peak, single_peak):
    sim = env.simulator.copy()
    sim.disable_external_mapper()
    sim.run()

    inf_sim = infenv.simulator.copy()
    inf_sim.disable_external_mapper()
    inf_sim.run()

    print(f"EFT: {_sim_base_metrics(sim)}")
    return (
        _extended_metrics(sim, inf_sim, hand_peak, single_peak),
        _inf_extended_metrics(inf_sim, hand_peak, single_peak),
    )


def run_external_mapper_once(spec, cfg, graph, env, infenv, hand_peak, single_peak):
    assert spec.mapper_factory is not None

    sim = env.simulator.copy()
    sim.enable_external_mapper()
    sim.external_mapper = spec.mapper_factory(graph, cfg)
    sim.run()

    inf_sim = infenv.simulator.copy()
    inf_sim.enable_external_mapper()
    inf_sim.external_mapper = spec.mapper_factory(graph, cfg)
    inf_sim.run()

    print(f"{spec.label}: {_sim_base_metrics(sim)}")
    return (
        _extended_metrics(sim, inf_sim, hand_peak, single_peak),
        _inf_extended_metrics(inf_sim, hand_peak, single_peak),
    )


def _build_internal_driver(
    env,
    internal_mapper: fastsim.Mapper,
    transition_conditions: fastsim.TransitionConditions,
):
    base_input = env.simulator.input
    sim_input = SimulatorInput(
        base_input.graph,
        base_input.data,
        base_input.system,
        task_noise=base_input.task_noise,
        transition_conditions=transition_conditions,
        top_k_candidates=base_input.top_k_candidates,
    )
    driver = SimulatorDriver(
        sim_input,
        internal_mapper=internal_mapper,
        observer_factory=env.simulator.observer_factory,
    )
    driver.initialize()
    driver.initialize_data()
    driver.disable_external_mapper()
    return driver


def run_internal_mapper_once(spec, cfg, env, infenv, hand_peak, single_peak):
    assert spec.internal_mapper_factory is not None
    assert spec.transition_factory is not None

    sim = _build_internal_driver(
        env,
        spec.internal_mapper_factory(cfg),
        spec.transition_factory(cfg),
    )
    sim.run()

    inf_sim = _build_internal_driver(
        infenv,
        spec.internal_mapper_factory(cfg),
        spec.transition_factory(cfg),
    )
    inf_sim.run()

    print(f"{spec.label}: {_sim_base_metrics(sim)}")
    return (
        _extended_metrics(sim, inf_sim, hand_peak, single_peak),
        _inf_extended_metrics(inf_sim, hand_peak, single_peak),
    )


def run_parmetis_once(
    cfg,
    env,
    infenv,
    single_sim,
    graph,
    ParMETIS,
    best_cfg_state,
    hand_peak,
    single_peak,
):
    copy_sim = env.simulator.copy() if rank == 0 else None

    comm.barrier()
    if best_cfg_state["best"] is None:
        best_cfg_state["best"] = find_best_cfg_optuna(
            cfg,
            ParMETIS,
            env=env,
            skip_search=False,
            mode="normal_optuna",
        )
    comm.barrier()

    best_cfg = best_cfg_state["best"]
    if best_cfg is None:
        return None

    run_parmetis(
        sim=copy_sim,
        cfg=cfg,
        unbalance=best_cfg[1],
        itr=best_cfg[0],
        ParMETIS=ParMETIS,
        n_compute_devices=cfg.system.n_devices - 1,
        skip_error=True,
    )
    comm.barrier()

    if rank != 0:
        return None

    copy_inf_sim = infenv.simulator.copy()
    copy_inf_sim.external_mapper = ReplayMapper(copy_sim)
    copy_inf_sim.enable_external_mapper()
    copy_inf_sim.run()

    print(f"ParMETIS: {_sim_base_metrics(copy_sim)}")
    assert isinstance(graph, DynamicJacobiGraph)
    return (
        _extended_metrics(copy_sim, copy_inf_sim, hand_peak, single_peak),
        _inf_extended_metrics(copy_inf_sim, hand_peak, single_peak),
    )


def configure_training(cfg: DictConfig):
    if cfg.graph.env.change_duration:
        output_name = f"noise_level_sweep_results_{EVAL_GRAPH_STEPS}.csv"
    else:
        output_name = f"level_sweep_results_{EVAL_GRAPH_STEPS}.csv"
    # output_name = "parmetis_" + output_name
    out_file = os.path.join(f"./results/{cfg.system.n_devices - 1}gpus/", output_name)

    graph_name = cfg.graph.config.workload_args.traj_type
    base_key = (
        graph_name,
        cfg.graph.config.level_memory,
        cfg.graph.config.r_interior,
        cfg.graph.config.r_boundary,
    )

    specs = build_mapper_specs(cfg)
    available_by_key = {spec.key: spec for spec in specs}
    selected_keys = resolve_enabled_mapper_keys(cfg, [spec.key for spec in specs])
    selected_specs = [available_by_key[k] for k in selected_keys]

    if rank == 0:
        if not selected_specs:
            print(
                "[SKIP] No mappers selected after enabled/disabled filtering.",
                flush=True,
            )
            pending_keys = []
        else:
            existing_entries = csv_existing_mapper_entries(out_file, base_key)
            pending_keys = [
                spec.key
                for spec in selected_specs
                if spec.key not in existing_entries
                or f"{spec.key}_inf" not in existing_entries
            ]
            if not pending_keys:
                print(
                    f"[SKIP] All selected mapper entries already exist for {base_key}.",
                    flush=True,
                )
            else:
                print(f"[RUN] Pending mapper keys: {pending_keys}", flush=True)
    else:
        pending_keys = None

    pending_keys = comm.bcast(pending_keys, root=0)
    if not pending_keys:
        return

    pending_specs = [available_by_key[key] for key in pending_keys]

    cfg.system.mem = SYSTEM_MEMORY
    cfg.graph.config.steps = EVAL_GRAPH_STEPS
    cfg.graph.config.workload_args.traj_specifics.phase_length = PHASE_LENGTH
    num_runs = (
        20
        if cfg.graph.env.change_duration
        else 12
        if cfg.graph.env.change_workload
        else 1
    )

    ParMETIS = ParMETIS_wrapper()
    best_cfg_state = {"best": None}
    exceeded_memory = False
    if rank == 0:
        graph_builder = make_graph_builder(cfg)
        env = make_env(
            graph_builder=graph_builder, cfg=cfg, normalization=False, eval=True
        )
        graph = env.get_graph()
        hand_calculated_peak = (
            0.5 * graph.data.data_stat["average_step_data"] / (cfg.system.n_devices - 1)
        )
        print(f"Hand-calculated peak: {hand_calculated_peak / 1e9:.1f} GB")
        if hand_calculated_peak > SYSTEM_MEMORY * 1.5:
            exceeded_memory = True
        else:
            inf_cfg = cfg.copy()
            inf_cfg.system.mem = int(99999e9)
            inf_graph_builder = make_graph_builder(inf_cfg)
            infenv = make_env(
                graph_builder=inf_graph_builder,
                cfg=inf_cfg,
                normalization=False,
                eval=True,
            )

            single_device_cfg = inf_cfg.copy()
            single_device_cfg.system.h2d_bw = cfg.system.d2d_bw
            single_device_cfg.system.n_devices = 2
            single_graph_builder = make_graph_builder(single_device_cfg)
            single_device_env = make_env(
                graph_builder=single_graph_builder,
                cfg=single_device_cfg,
                normalization=False,
                eval=True,
            )

            results = {spec.key: [] for spec in pending_specs}
            inf_results = {spec.key: [] for spec in pending_specs}
    else:
        env = None
        infenv = None
        single_device_env = None
        graph = None
        hand_calculated_peak = None

    exceeded_memory = comm.bcast(exceeded_memory, root=0)
    if exceeded_memory:
        if rank == 0:
            print(
                f"[SKIP] Hand-calculated peak memory {hand_calculated_peak / 1e9:.1f} GB exceeds system memory. Skipping runs.",
                flush=True,
            )
        return

    needs_extended_metrics = any(
        spec.mode in ("parmetis", "external", "internal") for spec in pending_specs
    )

    for _ in range(num_runs):
        if rank == 0:
            env.reset()
            infenv.reset()
            single_device_env.reset()
            single_sim = single_device_env.simulator
            single_peak = None
            if needs_extended_metrics:
                single_sim.disable_external_mapper()
                single_sim.run()
                single_peak = single_sim.max_mem_usage
        else:
            single_sim = None
            single_peak = None

        for spec in pending_specs:
            if spec.mode == "parmetis":
                run_result = run_parmetis_once(
                    cfg=cfg,
                    env=env,
                    infenv=infenv,
                    single_sim=single_sim,
                    graph=graph,
                    ParMETIS=ParMETIS,
                    best_cfg_state=best_cfg_state,
                    hand_peak=hand_calculated_peak,
                    single_peak=single_peak,
                )
                if rank == 0 and run_result is not None:
                    normal_result, inf_result = run_result
                    results[spec.key].append(normal_result)
                    inf_results[spec.key].append(inf_result)
                continue

            if rank != 0:
                continue

            if spec.mode == "eft":
                normal_result, inf_result = run_eft_once(
                    env, infenv, hand_calculated_peak, single_peak
                )
            elif spec.mode == "internal":
                normal_result, inf_result = run_internal_mapper_once(
                    spec=spec,
                    cfg=cfg,
                    env=env,
                    infenv=infenv,
                    hand_peak=hand_calculated_peak,
                    single_peak=single_peak,
                )
            else:
                normal_result, inf_result = run_external_mapper_once(
                    spec=spec,
                    cfg=cfg,
                    graph=graph,
                    env=env,
                    infenv=infenv,
                    hand_peak=hand_calculated_peak,
                    single_peak=single_peak,
                )
            results[spec.key].append(normal_result)
            inf_results[spec.key].append(inf_result)

    if rank != 0:
        return

    averaged = {}
    inf_averaged = {}
    for mapper, values in results.items():
        if values:
            averaged[mapper] = tuple(
                sum(cols) / len(values) for cols in zip(*values, strict=False)
            )
    for mapper, values in inf_results.items():
        if values:
            inf_averaged[mapper] = tuple(
                sum(cols) / len(values) for cols in zip(*values, strict=False)
            )

    lines = []
    for mapper, avg_values in averaged.items():
        numeric_part = ",".join(f"{v:.0f}" for v in avg_values)
        lines.append(",".join(map(str, base_key)) + f",{mapper},{numeric_part}")

    for mapper, avg_values in inf_averaged.items():
        numeric_part = ",".join(f"{v:.0f}" for v in avg_values)
        lines.append(",".join(map(str, base_key)) + f",{mapper}_inf,{numeric_part}")

    if lines:
        write_results_atomic(out_file, lines)


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
