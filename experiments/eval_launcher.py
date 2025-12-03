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


run_dict = {"corners": [], "circle": [], "noise": [], "bump": [], "ncircle": [], "lcorners": []}
# Interior, Boundary, Mem order
# run_dict["circle"].append((0.1, 0.1, "35e9", "ParMETIS(1.05,1)", "ColWise"))
# run_dict["circle"].append((1, 0.1, "100e9", "ParMETIS(1.0001,0.0001001)", "EFT"))
# run_dict["circle"].append((1, 1, "35e9", "ParMETIS(1.01,1)", "ColWise"))
# run_dict["circle"].append((10, 0.1, "105e9", "ParMETIS(1.01,1)", "Block(1x1)"))
# run_dict["circle"].append((10, 1, "100e9", "ParMETIS(1.0001,0.0001001)", "Block(1x1)"))

# run_dict["ncircle"].append((0.1, 0.1, "35e9", "ParMETIS(1.05,1)", "ColWise"))
# run_dict["ncircle"].append((1, 0.1, "100e9", "ParMETIS(1.0001,0.0001001)", "EFT"))
# run_dict["ncircle"].append((1, 1, "35e9", "ParMETIS(1.01,1)", "ColWise"))
# run_dict["ncircle"].append((10, 0.1, "105e9", "ParMETIS(1.01,1)", "Block(1x1)"))
# run_dict["ncircle"].append((10, 1, "100e9", "ParMETIS(1.0001,0.0001001)", "Block(1x1)"))
# run_dict["ncircle"].append((10, 1, "95e9", "ParMETIS(1.0001,0.0001001)", "Block(1x1)"))

# run_dict["corners"].append((0.1, 0.1, "35e9", "BlockCyclic(2x2)", "Oracle(64)"))
# run_dict["corners"].append((1, 0.1, "100e9", "ParMETIS(1.03,0.0001001)", "EFT"))
# run_dict["corners"].append((1, 1, "35e9", "BlockCyclic(2x2)", "Colwise"))
# run_dict["corners"].append((10, 0.1, "105e9", "ParMETIS(1.05,0.0001001)", "BlockCyclic(2x2)"))
# run_dict["corners"].append((10, 1, "95e9", "BlockCyclic(2x2)", "ParMETIS(1.04,10)"))

# run_dict["noise"].append((0.1, 0.1, "35e9", "BlockCyclic(2x2)", "Oracle(64)"))
# run_dict["noise"].append((1, 0.1, "100e9", "ParMETIS(1.03,0.0001001)", "EFT"))
# run_dict["noise"].append((1, 1, "35e9", "BlockCyclic(2x2)", "Colwise"))
# run_dict["noise"].append((10, 0.1, "105e9", "ParMETIS(1.05,0.0001001)", "BlockCyclic(2x2)"))
run_dict["noise"].append((10, 1, "95e9", "BlockCyclic(2x2)", "ParMETIS(1.04,10)"))

# run_dict["lcorners"].append((0.1, 0.1, "35e9", "BlockCyclic(2x2)", "Oracle(64)"))
# run_dict["lcorners"].append((1, 0.1, "100e9", "ParMETIS(1.03,0.0001001)", "EFT"))
# run_dict["lcorners"].append((1, 1, "35e9", "BlockCyclic(2x2)", "Colwise"))
# run_dict["lcorners"].append((10, 0.1, "105e9", "ParMETIS(1.05,0.0001001)", "BlockCyclic(2x2)"))
# run_dict["lcorners"].append((10, 1, "95e9", "BlockCyclic(2x2)", "ParMETIS(1.04,10)"))
# run_dict["lcorners"].append((100, 10, "49e9", "ParMETIS(1.04,10)", "BlockCyclic(2x2)"))

# --- Core pool setup ---
# node0_cores = list(range(0, 72))  # even = NUMA node0
# node1_cores = list(range(72, 144))  # odd = NUMA node1
node0_cores = list(range(0, 80, 2))  # even = NUMA node0
node1_cores = list(range(1, 80, 2))  # odd = NUMA node1
all_cores = node0_cores + node1_cores
free_cores = set(all_cores)

running_jobs = []  # (proc, allocated_cores)

# --- Example data structures (replace with your real ones) ---
# run_dict = {...}
# param = {...}

# --- Build jobs list from run_dict ---
jobs = []
for k in run_dict.keys():
    for interior, boundary, mem, policy, second_best in run_dict[k]:
        ib = param[(interior, boundary)]
        for obs_ver in ["D"]:
            for dmem in [64e9, 72e9, 80e9, 88e9, 96e9, 9999e9]:
                # for dmem in [96e9]:
                # for dmem in [64e9, 80e9, 9999e9]:
                n_cores = 4
                cmd = [
                    "mpirun",
                    "-n",
                    "4",
                    "python3",
                    "run_one_model_extend.py",
                    # "collect_stats.py",
                    # "visualize_model.py",
                    f"--config-name=8x8x{1024 if k == 'lcorners' else 128}_dynamic_{k}_cnn",
                    f"graph.config.arithmetic_intensity={ib[0]}",
                    f"feature.observer.version={obs_ver}",
                    f"graph.config.boundary_width={ib[1]}",
                    f"graph.config.level_memory={mem}",
                    f"system.mem={int(dmem)}",
                    f"runtime.batch_size=64",
                    f"runtime.queue_threshold=5",
                    f"runtime.max_in_flight=5",
                ]
                jobs.append((n_cores, cmd))
                print(f"Prepared job with {n_cores} cores: {' '.join(cmd)} {interior}-{boundary}-1_{int(dmem/1e9)}GB-{k}")

# exit()
# sort jobs by number of cores (small to large)
jobs.sort(key=lambda x: x[0])

# ask to proceed
proceed = input(f"Prepared {len(jobs)} jobs. Proceed? (y/n): ")
if proceed.lower() != "y":
    print("Aborting.")
    exit(0)


# --- Scheduler helpers ---
def allocate_cores(n):
    """Allocate n free cores from the same NUMA node."""
    global free_cores
    node0_free = [c for c in node0_cores if c in free_cores]
    node1_free = [c for c in node1_cores if c in free_cores]

    # Prefer node0 if enough free cores, else node1
    if len(node0_free) >= n:
        selected = node0_free[:n]
    elif len(node1_free) >= n:
        selected = node1_free[:n]
    else:
        return None

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
        proc = subprocess.Popen(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # proc = subprocess.Popen(full_cmd)
        running_jobs.append((proc, cores))
    jobs = pending

    # 3. Avoid busy spinning
    time.sleep(0.5)
