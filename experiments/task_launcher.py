import argparse
import json
import subprocess
import time
import sys
import os
from task4feedback.experiment_helper.run_name import calculate_ratio


# --- Hardware Detection Helpers (From your script) ---
def parse_cpulist(cpulist_str):
    """Convert strings like '0-3,8-11' -> [0,1,2,3,8,9,10,11]"""
    cores = []
    for part in cpulist_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))
    return cores


def get_cores_for_node(node_id):
    """Reads NUMA node CPU list from sysfs"""
    try:
        cmd = f"cat /sys/devices/system/node/node{node_id}/cpulist"
        cpulist = subprocess.check_output(cmd, shell=True).decode().strip()
        return parse_cpulist(cpulist)
    except subprocess.CalledProcessError:
        return []


def get_physical_cores(cores):
    """Filter only physical cores (one per hyperthread pair)."""
    physical = set()
    visited = set()

    for cpu in cores:
        if cpu in visited:
            continue

        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        if not os.path.exists(path):
            physical.add(cpu)
            continue

        with open(path) as f:
            siblings = parse_cpulist(f.read().strip())

        for s in siblings:
            visited.add(s)
        physical.add(min(siblings))  # Pick lowest ID as representative

    return sorted(list(physical))


# --- Scheduler Class ---
class Scheduler:
    def __init__(self):
        # 1. Auto-detect Hardware
        raw_node0 = get_cores_for_node(0)
        raw_node1 = get_cores_for_node(1)

        self.node0_phys = get_physical_cores(raw_node0)
        self.node1_phys = get_physical_cores(raw_node1)

        all_phys = self.node0_phys + self.node1_phys
        self.free_cores = set(all_phys)

        print(f"Detected Node0 Physical: {len(self.node0_phys)} cores")
        print(f"Detected Node1 Physical: {len(self.node1_phys)} cores")
        print(f"Total Available: {len(self.free_cores)}")

        self.running_jobs = []  # (proc, allocated_cores_list)
        self.last_node_used = 1  # Start ping-pong logic

    def allocate_cores(self, n):
        """Alternating Node Allocation Strategy"""
        node0_free = [c for c in self.node0_phys if c in self.free_cores]
        node1_free = [c for c in self.node1_phys if c in self.free_cores]

        # Determine preference based on last usage
        if self.last_node_used == 1:
            preferred_node = 0
            preferred_free = node0_free
            fallback_node = 1
            fallback_free = node1_free
        else:
            preferred_node = 1
            preferred_free = node1_free
            fallback_node = 0
            fallback_free = node0_free

        selected = None
        used_node_id = -1

        # Try preferred
        if len(preferred_free) >= n:
            selected = preferred_free[:n]
            used_node_id = preferred_node
        # Try fallback
        elif len(fallback_free) >= n:
            selected = fallback_free[:n]
            used_node_id = fallback_node

        if selected:
            for c in selected:
                self.free_cores.remove(c)
            self.last_node_used = used_node_id
            return selected

        return None

    def free_resources(self, cores):
        self.free_cores.update(cores)

    def tick(self):
        """Polls running processes and cleans up."""
        still_running = []
        for proc, cores in self.running_jobs:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, cores))
            else:
                print(f"[DONE] Job finished. Freed cores {cores}")
                self.free_resources(cores)
        self.running_jobs = still_running

    def launch(self, cmd_list, n_cores):
        cores = self.allocate_cores(n_cores)
        if cores is None:
            return False

        core_str = ",".join(map(str, cores))
        # Prepend taskset
        full_cmd = ["taskset", "-c", core_str] + cmd_list

        print(f"[START] Node? Cores {cores}: {' '.join(full_cmd)}")

        # Launch
        proc = subprocess.Popen(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.running_jobs.append((proc, cores))
        return True


# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs but do not run")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = json.load(f)

    jobs = []
    cmd_template = config["command_template"]
    cores_per_job = config.get("cores_per_job", 4)

    # Optional global overrides from JSON
    global_params = config.get("global_params", {})

    print(f"Generating jobs from {args.config}...")

    # Iterate over experiments defined in JSON
    for exp_name, run_list in config["experiments"].items():
        for params in run_list:
            interior = params.get("interior")
            boundary = params.get("boundary")

            # 1. Calculate Physics
            if interior and boundary:
                calc_int, calc_bound = calculate_ratio(interior, boundary)
                if calc_int is None:
                    continue
            else:
                raise ValueError("Interior and Boundary must be specified in params")

            # 2. Iterate Sweeps (e.g., dmem)
            # If "sweeps" is in params, use that, else use global "sweeps"
            sweeps = params.get("sweeps", config.get("sweeps", {}))

            # Cartesian product of all sweep lists
            import itertools

            keys = sweeps.keys()
            values = sweeps.values()

            for bundle in itertools.product(*values):
                sweep_context = dict(zip(keys, bundle))

                # 3. Build Context for String Formatting
                context = {
                    "exp_name": exp_name,
                    "seed_val": (args.seed + 10) * 100000000,
                    "calc_intensity": calc_int,
                    "calc_boundary": calc_bound,
                    **global_params,  # Global defaults
                    **params,  # Experiment specific
                    **sweep_context,  # Current sweep values
                }

                # Derived context (e.g. dmem_gb)
                if "dmem" in context:
                    context["dmem_gb"] = int(context["dmem"] / 1e9)
                    context["dmem_int"] = int(context["dmem"])

                # 4. Fill Template
                formatted_cmd = []
                for token in cmd_template:
                    formatted_cmd.append(str(token).format(**context))

                jobs.append((cores_per_job, formatted_cmd))

    print(f"Total jobs prepared: {len(jobs)}")

    if args.dry_run:
        for n, cmd in jobs:
            print(f"Dry Run ({n} cores): {' '.join(cmd)}")
        return

    # Run Scheduler
    scheduler = Scheduler()

    pending_jobs = jobs
    while pending_jobs or scheduler.running_jobs:
        scheduler.tick()

        next_pending = []
        for n_cores, cmd in pending_jobs:
            if not scheduler.launch(cmd, n_cores):
                next_pending.append((n_cores, cmd))
            # If launched, it drops out of pending

        pending_jobs = next_pending
        time.sleep(0.5)


if __name__ == "__main__":
    main()
