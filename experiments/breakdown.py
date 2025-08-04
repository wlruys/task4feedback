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

from task4feedback.interface import numa_devices
from task4feedback.graphs.jacobi import (
    JacobiGraph,
    LevelPartitionMapper,
    JacobiRoundRobinMapper,
)
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import numpy
from collections import defaultdict
import time


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


def get_device_stats_new(sim: SimulatorDriver, verbose: bool = False):
    def merge_sorted_intervals(intervals):
        """Assumes `intervals` is sorted by start; merges overlapping spans."""
        merged = []
        for s, e in intervals:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        return merged

    def interval_overlap_sum(raw, merged):
        """
        Sum total overlap between two lists of intervals:
        `raw` may overlap itself; `merged` must be non-overlapping.
        Both must be sorted by start.
        """
        total = 0
        i = j = 0
        while i < len(raw) and j < len(merged):
            rs, re = raw[i]
            ms, me = merged[j]
            # if they overlap...
            overlap = min(re, me) - max(rs, ms)
            if overlap > 0:
                total += overlap
            # advance the one that ends earlier
            if re < me:
                i += 1
            else:
                j += 1
        return total

    # --- 1. Bulk extract all tasks ---
    state = sim.get_state()
    rt = state.get_task_runtime()

    n_gpus = len(sim.input.system) - 1
    # raw interval storage
    compute_raw = []  # list of (dev, start, end)
    data_raw = []  # list of (src, dst, start, end)

    # collect compute tasks
    for i in range(rt.get_n_compute_tasks()):
        dev = rt.get_compute_task_mapped_device(i)
        assert dev != 0, "Compute tasks must be mapped to a device (not host)"
        start = rt.get_compute_task_launched_time(i)
        end = rt.get_compute_task_completed_time(i)
        compute_raw.append((dev, start, end))

    # collect data/eviction tasks
    for i in range(rt.get_n_data_tasks()):
        if rt.is_data_task_virtual(i):
            continue
        src = rt.get_data_task_source_device(i)
        dst = rt.get_data_task_mapped_device(i)
        start = rt.get_data_task_launched_time(i)
        end = rt.get_data_task_completed_time(i)
        data_raw.append((src, dst, start, end))

    # --- 2. Distribute into per-device lists & track end times ---
    DeviceEndTime = {d: 0 for d in range(1, n_gpus + 1)}
    compute_intervals = {d: [] for d in range(1, n_gpus + 1)}
    incoming_intervals = {d: [] for d in range(1, n_gpus + 1)}
    outgoing_intervals = {d: [] for d in range(1, n_gpus + 1)}
    eviction_intervals = {d: [] for d in range(1, n_gpus + 1)}

    # fill compute intervals
    for dev, s, e in compute_raw:
        compute_intervals[dev].append((s, e))
        DeviceEndTime[dev] = max(DeviceEndTime[dev], e)

    # fill data vs eviction intervals
    for src, dst, s, e in data_raw:
        # eviction if host (0) is one end
        if src == 0 and dst != 0:
            eviction_intervals[dst].append((s, e))
            DeviceEndTime[dst] = max(DeviceEndTime[dst], e)
        elif dst == 0 and src != 0:
            eviction_intervals[src].append((s, e))
            DeviceEndTime[src] = max(DeviceEndTime[src], e)
        else:
            incoming_intervals[dst].append((s, e))
            outgoing_intervals[src].append((s, e))
            DeviceEndTime[dst] = max(DeviceEndTime[dst], e)
            DeviceEndTime[src] = max(DeviceEndTime[src], e)

    # --- 3. Sort & merge each interval type once per device ---
    merged_compute = {}
    merged_comm = {}
    merged_evict = {}

    for d in DeviceEndTime:
        # compute
        ci = sorted(compute_intervals[d], key=lambda x: x[0])
        merged_compute[d] = merge_sorted_intervals(ci)

        # comm = incoming + outgoing
        comm_raw = incoming_intervals[d] + outgoing_intervals[d]
        ci2 = sorted(comm_raw, key=lambda x: x[0])
        merged_comm[d] = merge_sorted_intervals(ci2)

        # eviction
        ei = sorted(eviction_intervals[d], key=lambda x: x[0])
        merged_evict[d] = merge_sorted_intervals(ei)

    # --- 4. Overlap calculations in O(n) per device ---
    overlapped_in = {}
    overlapped_out = {}
    overlapped_evict = {}
    results = {}

    for d in DeviceEndTime:
        # sort raw for overlap routines
        in_raw = sorted(incoming_intervals[d], key=lambda x: x[0])
        out_raw = sorted(outgoing_intervals[d], key=lambda x: x[0])

        # data-comm overlapped with compute
        ov_in = interval_overlap_sum(in_raw, merged_compute[d])
        ov_out = interval_overlap_sum(out_raw, merged_compute[d])
        overlapped_data = ov_in + ov_out

        # create high-priority (compute+comm) merged timeline
        hp_raw = merged_compute[d] + merged_comm[d]
        hp_sorted = sorted(hp_raw, key=lambda x: x[0])
        merged_hp = merge_sorted_intervals(hp_sorted)

        # eviction overlapped by compute/comm
        ev = merged_evict[d]
        ov_evict = interval_overlap_sum(ev, merged_hp)

        # store
        overlapped_in[d] = ov_in
        overlapped_out[d] = ov_out
        overlapped_evict[d] = ov_evict

        # breakdown
        comp_time = sum(e - s for s, e in merged_compute[d])
        comm_time = sum(e - s for s, e in merged_comm[d])
        evict_time = sum(e - s for s, e in merged_evict[d])

        # total covered = union of all three merged lists
        all_busy_raw = merged_compute[d] + merged_comm[d] + merged_evict[d]
        all_busy = merge_sorted_intervals(sorted(all_busy_raw, key=lambda x: x[0]))
        covered = sum(e - s for s, e in all_busy)

        total = DeviceEndTime[d]
        idle = total - covered

        if verbose:
            print(f"\nDevice {d}:")
            print(f"  Compute (non-overlap): {comp_time} ({100*comp_time/total:.1f}%)")
            print(
                f"  Comm   (non-overlap): {comm_time - overlapped_data:.1f} ({100*(comm_time - overlapped_data)/total:.1f}%)"
            )
            print(
                f"  Evict  (non-overlap): {evict_time - ov_evict:.1f} ({100*(evict_time - ov_evict)/total:.1f}%)"
            )
            print(
                f"  Ov. Comm (hidden):   {overlapped_data:.1f} ({100*overlapped_data/total:.1f}%)"
            )
            print(f"  Ov. Evict (hidden):  {ov_evict:.1f} ({100*ov_evict/total:.1f}%)")
            print(f"  Idle:                {idle:.1f} ({100*idle/total:.1f}%)")

        results[d] = {
            "total_time": total,
            "compute": comp_time,
            "d2d": comm_time - overlapped_data,
            "eviction": evict_time - ov_evict,
            "idle": idle,
            "overlapped": {"d2d": overlapped_data, "eviction": ov_evict},
        }

    # --- 5. Compute averages ---
    avg = {
        "total_time": 0,
        "compute": 0,
        "d2d": 0,
        "eviction": 0,
        "idle": 0,
        "overlapped": {"d2d": 0, "eviction": 0},
    }
    for d, stats in results.items():
        avg["total_time"] += stats["total_time"]
        avg["compute"] += stats["compute"]
        avg["d2d"] += stats["d2d"]
        avg["eviction"] += stats["eviction"]
        avg["idle"] += stats["idle"]
        avg["overlapped"]["d2d"] += stats["overlapped"]["d2d"]
        avg["overlapped"]["eviction"] += stats["overlapped"]["eviction"]
    for k in ("total_time", "compute", "d2d", "eviction", "idle"):
        avg[k] //= n_gpus
    for k in ("d2d", "eviction"):
        avg["overlapped"][k] //= n_gpus

    results["avg"] = avg
    return results


@hydra.main(config_path="conf", config_name="dynamic_batch", version_base=None)
def main(cfg: DictConfig):
    # Seed for reproducibility
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Build environment and run simulation
    graph_builder = make_graph_builder(cfg)
    env = make_env(graph_builder=graph_builder, cfg=cfg)

    sim = env.simulator_factory.create()
    sim.disable_external_mapper()
    sim.run()

    print("Simulation completed successfully.")
    print(f"Final simulation time: {sim.time}")
    stats = get_device_stats_new(sim, verbose=False)
    print("Device statistics:")
    for dev, data in stats.items():
        print(f"Device {dev}: {data}")
    # Plot aggregated stacked bar chart for all devices
    devices = [dev for dev in stats if dev != "avg"]
    labels = ["compute", "d2d", "eviction", "idle"]
    data = {label: [stats[dev][label] for dev in devices] for label in labels}
    color_map = {
        "compute": "tab:blue",
        "d2d": "tab:orange",
        "eviction": "tab:green",
        "idle": "tab:red",
    }
    bottom = [0] * len(devices)
    plt.figure()
    for label in labels:
        values = data[label]
        bars = plt.bar(
            devices, values, bottom=bottom, label=label, color=color_map[label]
        )
        # annotate each segment with its percentage of total time
        for bar, val, dev in zip(bars, values, devices):
            total = stats[dev]["total_time"]
            pct = (val / total) * 100 if total else 0
            # skip labels below 0.1%
            if pct < 0.1:
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
            )
        bottom = [b + v for b, v in zip(bottom, values)]
    ax = plt.gca()
    # Set custom x-axis labels
    ax.set_xticks(devices)
    ax.set_xticklabels([f"GPU {d}" for d in devices])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0.5)
    ax.legend(title="Type", loc="lower left", bbox_to_anchor=(1, 0.5))
    # plt.xlabel("Device ID")
    plt.ylabel("Makespan")
    plt.title("Non-Overlapping Time Breakdown by GPU")
    # plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
