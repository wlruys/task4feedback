import argparse
import json
import subprocess
import time
import os
import shutil
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

                # Check for errors
                job["err_file"].seek(0)
                err_content = job["err_file"].read()
                job["err_file"].close()

                if ret != 0:
                    tqdm.write(f"\n[FAILED] Exit {ret}: {job['cmd_str']}")
                    with open(self.error_log_file, "a") as log:
                        log.write(f"\n{'='*30}\nFAILED JOB: {job['cmd_str']}\n")
                        log.write(f"EXIT CODE: {ret}\n--- STDERR ---\n{err_content}\n{'='*30}\n")

                if os.path.exists(job["log_path"]):
                    os.remove(job["log_path"])

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
        log_path = os.path.join(self.temp_log_dir, f"job_{self.job_counter}.err")
        err_file = open(log_path, "w+")

        try:
            proc = subprocess.Popen(full_cmd, stdout=subprocess.DEVNULL, stderr=err_file)

            self.running_jobs.append({"proc": proc, "cores": cores, "cmd_str": cmd_str, "err_file": err_file, "log_path": log_path})
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to launch: {cmd_str}\n{e}")
            self.free_resources(cores)
            err_file.close()
            return False

        return True

    def cleanup(self):
        if os.path.exists(self.temp_log_dir):
            try:
                shutil.rmtree(self.temp_log_dir)
            except:
                pass


# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--no-pinning", action="store_true", help="Disable numactl pinning")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = json.load(f)

    jobs = []
    cmd_template = config["command_template"]
    cores_per_job = config.get("cores_per_job", 4)
    global_params = config.get("global_params", {})

    print(f"Generating jobs from {args.config}...")

    # Iterate over experiments
    for exp_name, run_list in config["experiments"].items():
        for params in run_list:
            interior = params.get("interior")
            boundary = params.get("boundary")

            if interior and boundary:
                calc_int, calc_bound = calculate_ratio(interior, boundary)
                if calc_int is None:
                    continue
            else:
                raise ValueError("Interior and Boundary must be specified")

            sweeps = params.get("sweeps", config.get("sweeps", {}))
            import itertools

            keys = sweeps.keys()
            values = sweeps.values()

            for bundle in itertools.product(*values):
                sweep_context = dict(zip(keys, bundle))
                context = {
                    "exp_name": exp_name,
                    "seed_val": (args.seed + 10) * 100000000,
                    "calc_intensity": calc_int,
                    "calc_boundary": calc_bound,
                    **global_params,
                    **params,
                    **sweep_context,
                }
                if "dmem" in context:
                    context["dmem_gb"] = int(context["dmem"] / 1e9)
                    context["dmem_int"] = int(context["dmem"])

                formatted_cmd = []
                for token in cmd_template:
                    formatted_cmd.append(str(token).format(**context))

                jobs.append((cores_per_job, formatted_cmd))

    print(f"Total jobs prepared: {len(jobs)}")

    if args.dry_run:
        for n, cmd in jobs:
            print(f"Dry Run: {' '.join(cmd)}")
        return

    # Use args.no_pinning to toggle behavior
    scheduler = Scheduler(use_pinning=not args.no_pinning)

    pending_jobs = jobs
    pbar = tqdm(total=len(jobs), desc="Processing Jobs", unit="job")

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
