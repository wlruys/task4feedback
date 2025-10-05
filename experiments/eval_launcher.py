import itertools
import subprocess
import time

list_of_ibs = [(0.1, 0.1), (1, 0.1), (1, 1), (10, 1), (10, 0.1), (100, 10)]

param = {}
for interior, boundary in list_of_ibs:
    if interior < boundary:
        continue
    param[(interior, boundary)] = (f"{595.5555555/interior:.7f}"[:-1], f"{0.25 / (interior / boundary)}")
for k, v in param.items():
    print(f"{k}: {v}")
print(param)


run_dict = {"corners": [], "circle": [], "bump": []}
# Interior, Boundary, Mem order
# run_dict["circle"].append((0.1, 0.1, "43e9", "ParMETIS(1.05,1)", "ColWise"))
# run_dict["circle"].append((1, 0.1, "110e9", "ParMETIS(1.0001,0.0001001)", "EFT"))
# run_dict["circle"].append((1, 1, "39e9", "ParMETIS(1.01,1)", "ColWise"))
# run_dict["circle"].append((10, 0.1, "125e9", "ParMETIS(1.01,1)", "Block(1x1)"))
# run_dict["circle"].append((10, 1, "109e9", "ParMETIS(1.0001,0.0001001)", "Block(1x1)"))
# run_dict["circle"].append((100, 10, "115e9", "ParMETIS(1.01,10)", "Block(1x1)"))

# run_dict["corners"].append((0.1, 0.1, "40e9", "BlockCyclic(2x2)", "Oracle(64)"))
# run_dict["corners"].append((1, 0.1, "103e9", "ParMETIS(1.03,0.0001001)", "EFT"))
# run_dict["corners"].append((1, 1, "37e9", "BlockCyclic(2x2)", "Colwise"))
# run_dict["corners"].append((10, 0.1, "120e9", "ParMETIS(1.05,0.0001001)", "BlockCyclic(2x2)"))
# run_dict["corners"].append((10, 1, "101e9", "BlockCyclic(2x2)", "ParMETIS(1.04,10)"))
# run_dict["corners"].append((100, 10, "107e9", "ParMETIS(1.04,10)", "BlockCyclic(2x2)"))

run_dict["bump"].append((0.1, 0.1, "47e9", "BlockCyclic(1x1)", "ColWise"))
run_dict["bump"].append((1, 0.1, "116e9", "BlockCyclic(1x1)", "ParMETIS(1.03, 0.0001001)"))
run_dict["bump"].append((1, 1, "45e9", "BlockCyclic(1x1)", "EFT"))
run_dict["bump"].append((10, 0.1, "127e9", "ParMETIS(1.03,0.0001001)", "BlockCyclic(1x1)"))
run_dict["bump"].append((10, 1, "110e9", "ParMETIS(1.0001,0.0001001)", "EFT"))
run_dict["bump"].append((100, 10, "112e9", "ParMETIS(1.01,10)", "BlockCyclic(1x1)"))

# --- Core pool setup ---
node0_cores = list(range(0, 160, 2))  # even = NUMA node0
node1_cores = list(range(1, 160, 2))  # odd = NUMA node1
all_cores = node0_cores + node1_cores
free_cores = set(all_cores)

running_jobs = []  # (proc, allocated_cores)

# --- Example data structures (replace with your real ones) ---
# run_dict = {...}
# param = {...}

# --- Build jobs list from run_dict ---
jobs = []
for k, v in run_dict.items():
    for interior, boundary, mem, policy, second_best in run_dict[k]:
        ib = param[(interior, boundary)]
        for obs_ver in ["D"]:
            for dmem in [56e9, 64e9, 72e9, 80e9, 88e9, 96e9, 104e9]:
                n_cores = 4
                cmd = [
                    "mpirun",
                    "-n",
                    "4",
                    "python3",
                    "run_one_model.py",
                    f"--config-name=8x8x128_dynamic_{k}_cnn",
                    f"graph.config.arithmetic_intensity={ib[0]}",
                    f"feature.observer.version={obs_ver}",
                    f"graph.config.boundary_width={ib[1]}",
                    f"graph.config.level_memory={mem}",
                    f"system.mem={int(dmem)}",
                ]
                jobs.append((n_cores, cmd))
                print(f"Prepared job with {n_cores} cores: {' '.join(cmd)}")

# sort jobs by number of cores (small to large)
jobs.sort(key=lambda x: x[0])

# ask to proceed
proceed = input(f"Prepared {len(jobs)} jobs. Proceed? (y/n): ")
if proceed.lower() != "y":
    print("Aborting.")
    exit(0)


# --- Scheduler helpers ---
def allocate_cores(n):
    """Allocate n free cores from pool."""
    global free_cores
    if len(free_cores) < n:
        return None
    selected = sorted(list(free_cores))[:n]
    for c in selected:
        free_cores.remove(c)
    return selected


def free_allocated(cores):
    """Return cores to pool."""
    global free_cores
    free_cores.update(cores)


# --- Job launcher loop ---
while jobs or running_jobs:
    # 1. Check for finished jobs
    still_running = []
    for proc, cores in running_jobs:
        ret = proc.poll()
        if ret is None:
            still_running.append((proc, cores))
        else:
            print(f"[DONE] Freed cores {cores}")
            free_allocated(cores)
    running_jobs = still_running

    # 2. Try to launch pending jobs
    pending = []
    for n, cmd in jobs:
        cores = allocate_cores(n)
        if cores is None:
            pending.append((n, cmd))
            continue
        core_str = ",".join(map(str, cores))
        full_cmd = ["taskset", "-c", core_str] + cmd
        print(f"[START] {' '.join(full_cmd)}")
        proc = subprocess.Popen(full_cmd)
        running_jobs.append((proc, cores))
    jobs = pending

    # 3. Avoid busy spinning
    time.sleep(2)
