import argparse
import json
import subprocess
import time
import os
import shutil
import itertools
from tqdm import tqdm
from task4feedback.experiment_helper.run_name import calculate_ratio


# --- Hardware Detection Helpers ---
def parse_cpulist(cpulist_str):
    cores = []
    for part in cpulist_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))
    return cores


def get_cores_for_node(node_id):
    try:
        cmd = f"cat /sys/devices/system/node/node{node_id}/cpulist"
        cpulist = subprocess.check_output(cmd, shell=True).decode().strip()
        return parse_cpulist(cpulist)
    except subprocess.CalledProcessError:
        return []


def get_physical_cores(cores):
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
        physical.add(min(siblings))
    return sorted(list(physical))


# --- Scheduler Class ---
class Scheduler:
    def __init__(self, use_pinning=True, error_log_file="execution_errors.log"):
        # Hardware Detection
        raw_node0 = get_cores_for_node(0)
        raw_node1 = get_cores_for_node(1)
        self.node0_phys = get_physical_cores(raw_node0)
        self.node1_phys = get_physical_cores(raw_node1)

        # Fallback if detection fails
        if not self.node0_phys and not self.node1_phys:
            print("Warning: NUMA topology not detected. Using generic CPU count.")
            all_cores = list(range(os.cpu_count() or 1))
            self.node0_phys = all_cores
            self.node1_phys = []

        all_phys = self.node0_phys + self.node1_phys
        self.free_cores = set(all_phys)
        self.use_pinning = use_pinning

        # Logging Setup
        self.error_log_file = error_log_file
        self.temp_log_dir = ".temp_logs"
        os.makedirs(self.temp_log_dir, exist_ok=True)
        with open(self.error_log_file, "w") as f:
            f.write(f"--- Execution Started at {time.ctime()} ---\n")

        print(f"Detected Node0 Physical: {len(self.node0_phys)} cores")
        print(f"Detected Node1 Physical: {len(self.node1_phys)} cores")
        print(f"Total Available: {len(self.free_cores)}")

        self.running_jobs = []
        self.last_node_used = 1
        self.job_counter = 0

    def allocate_cores(self, n):
        if not self.use_pinning:
            # If not pinning, just reserve generic slots
            if len(self.free_cores) >= n:
                selected = []
                for _ in range(n):
                    selected.append(self.free_cores.pop())
                return selected
            return None

        # --- Strict NUMA Aware Allocation ---
        node0_free = [c for c in self.node0_phys if c in self.free_cores]
        node1_free = [c for c in self.node1_phys if c in self.free_cores]

        if self.last_node_used == 1:
            preferred_free, fallback_free = node0_free, node1_free
            preferred_node, fallback_node = 0, 1
        else:
            preferred_free, fallback_free = node1_free, node0_free
            preferred_node, fallback_node = 1, 0

        selected = None
        if len(preferred_free) >= n:
            selected = preferred_free[:n]
            self.last_node_used = preferred_node
        elif len(fallback_free) >= n:
            selected = fallback_free[:n]
            self.last_node_used = fallback_node

        if selected:
            for c in selected:
                self.free_cores.remove(c)
            return selected

        return None

    def free_resources(self, cores):
        self.free_cores.update(cores)

    def tick(self):
        finished_count = 0
        still_running = []

        for job in self.running_jobs:
            proc = job["proc"]
            ret = proc.poll()

            if ret is None:
                still_running.append(job)
            else:
                finished_count += 1
                self.free_resources(job["cores"])

        self.running_jobs = still_running
        return finished_count

    def launch(self, cmd_list, n_cores):
        cores = self.allocate_cores(n_cores)
        if cores is None:
            return False

        # --- Construct core command ---
        # 1. Start with OMP settings (Using 'env' wrapper)
        omp_prefix = ["env", f"OMP_NUM_THREADS={n_cores}"]

        # 2. Add numactl if pinning is enabled
        numa_prefix = []
        if self.use_pinning:
            core_str = ",".join(map(str, cores))

            # Determine NUMA node for memory binding
            mem_node = None
            if cores[0] in self.node0_phys:
                mem_node = 0
            elif cores[0] in self.node1_phys:
                mem_node = 1

            numa_prefix = ["numactl", "-C", core_str]
            if mem_node is not None:
                numa_prefix.extend(["-m", str(mem_node)])

        # 3. Combine parts: env VAR=VAL numactl ... python ...
        full_cmd = omp_prefix + numa_prefix + cmd_list
        cmd_str = " ".join(full_cmd)

        self.job_counter += 1

        try:
            # proc = subprocess.Popen(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            proc = subprocess.Popen(full_cmd, text=True)
            self.running_jobs.append({"proc": proc, "cores": cores, "cmd_str": cmd_str})
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to launch: {cmd_str}\n{e}")
            self.free_resources(cores)
            return False

        return True

    def cleanup(self):
        if os.path.exists(self.temp_log_dir):
            try:
                shutil.rmtree(self.temp_log_dir)
            except:
                pass


def parse_mem(mem_str):
    if isinstance(mem_str, (int, float)):
        return int(mem_str)
    return int(float(mem_str))


def is_sweep_sequence(value):
    return isinstance(value, (list, tuple))


def split_sweep_fields(mapping, skip_keys=None):
    skip_keys = set() if skip_keys is None else set(skip_keys)
    scalar_fields = {}
    sweep_fields = {}

    for key, value in mapping.items():
        if key in skip_keys:
            scalar_fields[key] = value
        elif is_sweep_sequence(value):
            sweep_fields[key] = value
        else:
            scalar_fields[key] = value

    return scalar_fields, sweep_fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--no-pinning", action="store_true", help="Disable numactl pinning")
    parser.add_argument("--nodes", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--node-number", type=int, default=0, help="Index of current node (0 to nodes-1)")
    args = parser.parse_args()

    if args.node_number >= args.nodes:
        raise ValueError(f"node_number ({args.node_number}) must be less than nodes ({args.nodes})")

    # Load Config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Global Config params
    start_mem = parse_mem(config.get("start_mem", 0))
    end_mem = parse_mem(config.get("end_mem", 0))
    step_mem = parse_mem(config.get("step_mem", 1e9))

    seed_start = config.get("seed_start", 0)
    seed_step = config.get("seed_step", 100000000)
    num_seeds = config.get("num_seeds", 1)

    # Generate Memory Points
    base_mem_range = []
    if start_mem > 0 and end_mem > 0:
        curr = start_mem
        while curr <= end_mem:
            base_mem_range.append(curr)
            curr += step_mem

    experiment_jobs = []

    # Iterate over experiments
    cmd_template = config["command_template"]
    cores_per_job = config.get("cores_per_job", 4)
    global_params = config.get("global_params", {})

    # Identify top-level keys that should be in context
    # exclude known structural keys
    exclude_keys = {"experiments", "command_template", "cores_per_job", "seed_start", "seed_step", "num_seeds", "start_mem", "end_mem", "step_mem", "global_params", "sweeps"}

    # Create a base context from top-level config items.
    # List-valued entries are treated as sweep dimensions instead of being
    # stringified directly into command templates.
    base_context_raw = {k: v for k, v in config.items() if k not in exclude_keys}
    base_context, base_sweeps = split_sweep_fields(base_context_raw)
    base_context.update(global_params)

    for seed_offset in range(num_seeds):
        current_seed = seed_start + seed_offset * seed_step

        for exp_name, run_list in config["experiments"].items():
            for params in run_list:
                # 1. Determine Memory Points

                # Start with the base range
                # Filter out points "close" to specified "mem" in params

                if "mem" in params and isinstance(params["mem"], list):
                    final_mem_points = [parse_mem(m) for m in params["mem"]]
                else:
                    job_specific_mem_points = []
                    final_mem_points = []
                    # Add base points that are NOT covered by specific points
                    # specific point covers range [spec_mem - step_mem, spec_mem + step_mem] ?
                    # User said: "if there is a memory that difference is less then step mem wrt specified "mem" aggregate that point into specified "mem""

                    # We want to iterate through base_mem_range.
                    # If a base_mem is within step_mem of ANY spec_mem, we skip it (it's "aggregated" into spec_mem)
                    # Then we add all spec_mem points.

                    # Wait, "aggregate" means the user wants to run the specific mem INSTEAD of the nearby base points.

                    for base_m in base_mem_range:
                        covered = False
                        # for spec_m in job_specific_mem_points:
                        #     if base_m - spec_m < step_mem and base_m - spec_m > 0:
                        #         covered = True
                        #         break
                        if not covered:
                            final_mem_points.append(base_m)

                    # Add specific points
                    final_mem_points.extend(job_specific_mem_points)
                    final_mem_points = sorted(list(set(final_mem_points)))

                    # If no memory config at all, just run once with params as is?
                    # The prompt implies we are generating sweeps over memory.
                    # If final_mem_points is empty (no start/end/step config), we just run params once

                if not final_mem_points:
                    final_mem_points = [None]  # Dummy to run loop once

                # Handle other sweeps if present, including list-valued config
                # fields like arch=["cnn", "gnn"].
                param_context, param_sweeps = split_sweep_fields(
                    params,
                    skip_keys={"mem", "sweeps"},
                )
                explicit_sweeps = params.get("sweeps", config.get("sweeps", {}))
                combined_sweeps = {
                    **base_sweeps,
                    **param_sweeps,
                    **explicit_sweeps,
                }

                sweep_keys = list(combined_sweeps.keys())
                sweep_values = [combined_sweeps[key] for key in sweep_keys]
                sweep_bundles = itertools.product(*sweep_values) if sweep_keys else [()]

                for bundle in sweep_bundles:
                    sweep_context = dict(zip(sweep_keys, bundle))

                    for mem_val in final_mem_points:
                        context = {
                            "exp_name": exp_name,
                            "seed_val": current_seed,
                            **base_context,
                            **param_context,
                            **sweep_context,
                        }

                        # Overwrite/Set memory context if valid
                        if mem_val is not None:
                            context["mem"] = int(mem_val)  # Ensure int format for template

                        # Derived memory params
                        if "dmem" in context:
                            if isinstance(context["dmem"], str):
                                context["dmem"] = parse_mem(context["dmem"])
                            context["dmem_gb"] = int(context["dmem"] / 1e9)
                            context["dmem_int"] = int(context["dmem"])

                        if "percentages" in context and mem_val is not None:
                            context["mem"] = int(float(mem_val) * context["percentages"] / 100)

                        formatted_cmd = []
                        for token in cmd_template:
                            formatted_cmd.append(str(token).format(**context))

                        experiment_jobs.append((cores_per_job, formatted_cmd))

    # Round Robin Distribution
    # Filter jobs for this node
    # experiments "circle": ..., num_seeds=32, nodes=32 -> every node should have 1 of the run
    # This implies we just slice the list

    my_jobs = experiment_jobs[args.node_number :: args.nodes]

    if not args.run:
        print(f"Total generate jobs: {len(experiment_jobs)}")
        print(f"Node {args.node_number}/{args.nodes} assigned {len(my_jobs)} jobs:")
        for n, cmd in my_jobs:
            print(f"{' '.join(cmd)}")
        return

    # Use args.no_pinning to toggle behavior
    scheduler = Scheduler(use_pinning=not args.no_pinning)

    pending_jobs = list(my_jobs)
    pbar = tqdm(total=len(my_jobs), desc="Processing Jobs", unit="job")

    try:
        while pending_jobs or scheduler.running_jobs:
            finished = scheduler.tick()
            if finished > 0:
                pbar.update(finished)

            active = len(scheduler.running_jobs)
            status = f"{active}/{args.max_jobs}" if args.max_jobs else f"{active}"
            pbar.set_description(f"Running: {status}")

            next_pending = []
            for i, (n_cores, cmd) in enumerate(pending_jobs):
                if args.max_jobs and len(scheduler.running_jobs) >= args.max_jobs:
                    next_pending.extend(pending_jobs[i:])
                    break

                if not scheduler.launch(cmd, n_cores):
                    next_pending.append((n_cores, cmd))

            pending_jobs = next_pending
            time.sleep(0.5)

    except KeyboardInterrupt:
        tqdm.write("\n[STOP] Interrupted.")
        for job in scheduler.running_jobs:
            job["proc"].kill()
    finally:
        pbar.close()
        scheduler.cleanup()
        print(f"\nErrors logged to: {scheduler.error_log_file}")


if __name__ == "__main__":
    main()
