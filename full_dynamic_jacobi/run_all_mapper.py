import fcntl
import hashlib
import os
import pickle
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy
import torch
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf

import task4feedback.fastsim2 as fastsim
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.mapper import (
    DARTSConfig,
    EnhancedDARTSConfig,
    MemoryAwareEFTConfig,
    ReplayMapper,
    TransitionConfig,
    make_internal_mapper,
    make_transition_conditions,
)
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
PHASE_LENGTH = 128
SYSTEM_MEMORY = 96e9


@dataclass(frozen=True)
class MapperSpec:
    key: str
    label: str
    mode: str  # parmetis | eft | external | internal
    mapper_factory: Callable[[DynamicJacobiGraph, DictConfig], object] | None = None
    internal_mapper_factory: Callable[[DictConfig], fastsim.Mapper] | None = None
    transition_factory: Callable[[DictConfig], fastsim.TransitionConditions] | None = (
        None
    )


def write_results_atomic(path, lines):
    """
    Upsert lines to a file using an exclusive file lock.
    Each incoming line replaces any existing row with the same
    (graph, mem, interior, boundary, mapper) tuple.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    replacements = {}
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        replacements[tuple(parts[:5])] = line

    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        existing_lines = f.readlines()

        kept_lines = []
        seen_replacements = set()
        for raw_line in existing_lines:
            stripped = raw_line.rstrip("\n")
            parts = stripped.split(",")
            if len(parts) < 5:
                kept_lines.append(raw_line)
                continue

            key = tuple(parts[:5])
            if key in replacements:
                if key not in seen_replacements:
                    kept_lines.append(replacements[key] + "\n")
                    seen_replacements.add(key)
                continue

            kept_lines.append(raw_line)

        for key, line in replacements.items():
            if key not in seen_replacements:
                kept_lines.append(line + "\n")

        f.seek(0)
        f.truncate()
        f.writelines(kept_lines)
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


def csv_best_parmetis_rows(path):
    """
    Return best-known ParMETIS rows keyed by:
    (traj_type, level_memory, r_interior, r_boundary)
    """
    if not os.path.exists(path):
        return {}

    rows = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 11 or parts[4] != "parmetis":
                continue

            try:
                key = (parts[0], int(float(parts[1])), parts[2], parts[3])
                time_value = int(float(parts[5]))
            except ValueError:
                continue

            existing = rows.get(key)
            if existing is None or time_value < existing["time"]:
                rows[key] = {
                    "key": key,
                    "time": time_value,
                }

    return rows


def hash_graph_cfg(graph_cfg) -> str:
    data = OmegaConf.to_container(graph_cfg, resolve=True)
    serialized = repr(sorted(data.items())).encode()
    return hashlib.sha256(serialized).hexdigest()


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _parmetis_cache_path(
    cfg: DictConfig,
    cache_dir: str = "parmetis_cfg",
    mode: str = "normal_optuna",
) -> Path:
    cache_cfg = _clone_cfg(cfg)
    cache_cfg.graph.config.steps = 256
    cache_cfg.graph.env.change_duration = False
    graph_hash = hash_graph_cfg(cache_cfg.graph)
    return (
        Path(cache_dir) / f"{cfg.system.n_devices - 1}gpus" / mode / f"{graph_hash}.pkl"
    )


def maybe_seed_parmetis_from_larger_memory(
    cfg: DictConfig,
    out_file: str,
    base_key,
    existing_entries: set[str],
    cache_dir: str = "parmetis_cfg",
    mode: str = "normal_optuna",
):
    """
    If the current ParMETIS row already exists but a larger memory point has a
    lower runtime, copy that larger point's cached Optuna config to the current
    hash and request a ParMETIS rerun for this point.
    """
    state = {
        "rerun": False,
        "current_time": None,
        "target_cache_path": None,
        "original_cache": None,
        "target_cache_existed": False,
        "existing_parmetis_inf": "parmetis_inf" in existing_entries,
        "donor_key": None,
        "seeded_cfg": None,
    }

    if "parmetis" not in existing_entries:
        return state

    parmetis_rows = csv_best_parmetis_rows(out_file)
    current_key = (
        str(base_key[0]),
        int(float(base_key[1])),
        str(base_key[2]),
        str(base_key[3]),
    )
    current_row = parmetis_rows.get(current_key)
    if current_row is None:
        return state

    faster_larger = [
        row
        for key, row in parmetis_rows.items()
        if key[0] == current_key[0]
        and key[2] == current_key[2]
        and key[3] == current_key[3]
        and key[1] > current_key[1]
        and row["time"] < current_row["time"]
    ]
    if not faster_larger:
        return state

    donor_row = min(faster_larger, key=lambda row: (row["time"], row["key"][1]))

    donor_cfg = _clone_cfg(cfg)
    donor_cfg.graph.config.level_memory = donor_row["key"][1]
    donor_cache_path = _parmetis_cache_path(
        donor_cfg,
        cache_dir=cache_dir,
        mode=mode,
    )
    if not donor_cache_path.exists():
        print(
            f"[WARN] Found faster larger-memory ParMETIS row {donor_row['key']} "
            f"but no cached Optuna config at {donor_cache_path}.",
            flush=True,
        )
        return state

    target_cache_path = _parmetis_cache_path(
        cfg,
        cache_dir=cache_dir,
        mode=mode,
    )
    target_cache_path.parent.mkdir(parents=True, exist_ok=True)

    with donor_cache_path.open("rb") as f:
        donor_best_cfg = pickle.load(f)

    original_cache = None
    target_cache_existed = target_cache_path.exists()
    if target_cache_existed:
        with target_cache_path.open("rb") as f:
            original_cache = pickle.load(f)

    with target_cache_path.open("wb") as f:
        pickle.dump(donor_best_cfg, f)

    print(
        f"[RETRY] ParMETIS anomaly for {current_key}: current_time={current_row['time']} "
        f"donor={donor_row['key']} donor_time={donor_row['time']} "
        f"seeded itr={donor_best_cfg[0]} ub={donor_best_cfg[1]}",
        flush=True,
    )

    state.update(
        {
            "rerun": True,
            "current_time": current_row["time"],
            "target_cache_path": target_cache_path,
            "original_cache": original_cache,
            "target_cache_existed": target_cache_existed,
            "donor_key": donor_row["key"],
            "seeded_cfg": donor_best_cfg,
        }
    )
    return state


def restore_parmetis_cache(seed_state):
    target_cache_path = seed_state["target_cache_path"]
    if target_cache_path is None:
        return

    if seed_state["target_cache_existed"]:
        with target_cache_path.open("wb") as f:
            pickle.dump(seed_state["original_cache"], f)
    elif target_cache_path.exists():
        target_cache_path.unlink()


def persist_parmetis_cache(seed_state, best_cfg, new_time):
    target_cache_path = seed_state["target_cache_path"]
    if target_cache_path is None or best_cfg is None:
        return

    updated_cfg = (best_cfg[0], best_cfg[1], float(new_time))
    with target_cache_path.open("wb") as f:
        pickle.dump(updated_cfg, f)


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
    darts_transition_cfg = TransitionConfig()

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
        MapperSpec(
            key="memory_aware_eft",
            label="MemoryAwareEFT",
            mode="internal",
            internal_mapper_factory=lambda _cfg: make_internal_mapper(
                "memory_aware_eft",
                memory_aware_eft_config=MemoryAwareEFTConfig(),
            ),
        ),
        MapperSpec(
            key="darts",
            label="DARTS",
            mode="internal",
            internal_mapper_factory=lambda _cfg: make_internal_mapper(
                "darts",
                darts_config=DARTSConfig(),
            ),
            transition_factory=lambda _cfg: make_transition_conditions(
                "darts",
                top_k_candidates=1,
                config=darts_transition_cfg,
            ),
        ),
        MapperSpec(
            key="enhanced_darts",
            label="EnhancedDARTS",
            mode="internal",
            internal_mapper_factory=lambda _cfg: make_internal_mapper(
                "enhanced_darts",
                enhanced_darts_config=EnhancedDARTSConfig(),
            ),
            transition_factory=lambda _cfg: make_transition_conditions(
                "enhanced_darts",
                top_k_candidates=1,
                config=darts_transition_cfg,
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
            key="b1",
            label="BlockCyclic(1x1)",
            mode="external",
            mapper_factory=lambda _graph, local_cfg: JacobiRoundRobinMapper(
                n_devices=local_cfg.system.n_devices - 1,
                offset=1,
                setting=0,
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
    transition_conditions: fastsim.TransitionConditions | None,
):
    base_input = env.simulator.input
    sim_input = SimulatorInput(
        base_input.graph,
        base_input.data,
        base_input.system,
        task_noise=base_input.task_noise,
        transition_conditions=(
            transition_conditions
            if transition_conditions is not None
            else base_input.transition_conditions
        ),
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

    sim = _build_internal_driver(
        env,
        spec.internal_mapper_factory(cfg),
        spec.transition_factory(cfg) if spec.transition_factory is not None else None,
    )
    sim.run()

    inf_sim = _build_internal_driver(
        infenv,
        spec.internal_mapper_factory(cfg),
        spec.transition_factory(cfg) if spec.transition_factory is not None else None,
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
        output_name = f"noise_level_sweep_results_{cfg.graph.config.steps}.csv"
    else:
        output_name = f"level_sweep_results_{cfg.graph.config.steps}.csv"
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
    parmetis_seed_state = None

    if rank == 0:
        if not selected_specs:
            print(
                "[SKIP] No mappers selected after enabled/disabled filtering.",
                flush=True,
            )
            pending_keys = []
        else:
            existing_entries = csv_existing_mapper_entries(out_file, base_key)
            if "parmetis" in selected_keys:
                parmetis_seed_state = maybe_seed_parmetis_from_larger_memory(
                    cfg=cfg,
                    out_file=out_file,
                    base_key=base_key,
                    existing_entries=existing_entries,
                )

            pending_keys = []
            for spec in selected_specs:
                if spec.key == "parmetis" and parmetis_seed_state["rerun"]:
                    pending_keys.append(spec.key)
                    continue

                if (
                    spec.key not in existing_entries
                    or f"{spec.key}_inf" not in existing_entries
                ):
                    pending_keys.append(spec.key)

            if not pending_keys:
                print(
                    f"[SKIP] All selected mapper entries already exist for {base_key}.",
                    flush=True,
                )
            else:
                print(f"[RUN] Pending mapper keys: {pending_keys}", flush=True)
    else:
        pending_keys = None
        existing_entries = set()

    pending_keys = comm.bcast(pending_keys, root=0)
    if not pending_keys:
        return

    pending_specs = [available_by_key[key] for key in pending_keys]

    cfg.system.mem = SYSTEM_MEMORY
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

    if parmetis_seed_state is not None and parmetis_seed_state["rerun"]:
        new_parmetis = averaged.get("parmetis")
        if new_parmetis is None:
            print(
                f"[RETRY] No ParMETIS result produced for {base_key}; restoring original cache.",
                flush=True,
            )
            restore_parmetis_cache(parmetis_seed_state)
        else:
            new_time = new_parmetis[0]
            current_time = parmetis_seed_state["current_time"]
            if current_time is not None and new_time >= current_time:
                print(
                    f"[RETRY] Seeded ParMETIS config did not improve {base_key}: "
                    f"new_time={new_time:.0f} current_time={current_time}. "
                    f"Restoring original cache and keeping existing CSV row.",
                    flush=True,
                )
                restore_parmetis_cache(parmetis_seed_state)
                averaged.pop("parmetis", None)
                if parmetis_seed_state["existing_parmetis_inf"]:
                    inf_averaged.pop("parmetis", None)
            else:
                print(
                    f"[RETRY] Accepted seeded ParMETIS config for {base_key}: "
                    f"old_time={current_time} new_time={new_time:.0f}.",
                    flush=True,
                )
                persist_parmetis_cache(
                    parmetis_seed_state,
                    best_cfg_state["best"],
                    new_time,
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
