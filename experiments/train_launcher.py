import itertools
import subprocess
import time
import argparse


args = argparse.ArgumentParser()
args.add_argument("--seed", type=int, default=0)
args = args.parse_args()
seed = args.seed

list_of_ibs = [(0.1, 0.1), (1, 0.1), (1, 1), (10, 1), (10, 0.1), (100, 10)]

param = {}
for interior, boundary in list_of_ibs:
    if interior < boundary:
        continue
    param[(interior, boundary)] = (f"{595.5555555/interior:.7f}"[:-1], f"{0.25 / (interior / boundary)}")
for k, v in param.items():
    print(f"{k}: {v}")
print(param)


run_dict = {"corners": [], "circle": [], "noise": [], "bump": []}
# Interior, Boundary, Mem order
run_dict["circle"].append((0.1, 0.1, "35e9", "ParMETIS(1.05,1)", "ColWise"))
run_dict["circle"].append((1, 0.1, "100e9", "ParMETIS(1.0001,0.0001001)", "EFT"))
run_dict["circle"].append((1, 1, "35e9", "ParMETIS(1.01,1)", "ColWise"))
run_dict["circle"].append((10, 0.1, "105e9", "ParMETIS(1.01,1)", "Block(1x1)"))
run_dict["circle"].append((10, 1, "100e9", "ParMETIS(1.0001,0.0001001)", "Block(1x1)"))
run_dict["circle"].append((100, 10, "90e9", "ParMETIS(1.01,10)", "Block(1x1)"))

run_dict["corners"].append((0.1, 0.1, "35e9", "BlockCyclic(2x2)", "Oracle(64)"))
run_dict["corners"].append((1, 0.1, "100e9", "ParMETIS(1.03,0.0001001)", "EFT"))
run_dict["corners"].append((1, 1, "35e9", "BlockCyclic(2x2)", "Colwise"))
run_dict["corners"].append((10, 0.1, "105e9", "ParMETIS(1.05,0.0001001)", "BlockCyclic(2x2)"))
run_dict["corners"].append((10, 1, "95e9", "BlockCyclic(2x2)", "ParMETIS(1.04,10)"))
run_dict["corners"].append((100, 10, "95e9", "ParMETIS(1.04,10)", "BlockCyclic(2x2)"))

run_dict["noise"].append((0.1, 0.1, "35e9", "BlockCyclic(2x2)", "Oracle(64)"))
run_dict["noise"].append((1, 0.1, "100e9", "ParMETIS(1.03,0.0001001)", "EFT"))
run_dict["noise"].append((1, 1, "35e9", "BlockCyclic(2x2)", "Colwise"))
run_dict["noise"].append((10, 0.1, "105e9", "ParMETIS(1.05,0.0001001)", "BlockCyclic(2x2)"))
run_dict["noise"].append((10, 1, "95e9", "BlockCyclic(2x2)", "ParMETIS(1.04,10)"))
run_dict["noise"].append((100, 10, "95e9", "ParMETIS(1.04,10)", "BlockCyclic(2x2)"))

run_dict["bump"].append((0.1, 0.1, "45e9", "BlockCyclic(1x1)", "ColWise"))
run_dict["bump"].append((1, 0.1, "105e9", "BlockCyclic(1x1)", "ParMETIS(1.03, 0.0001001)"))
run_dict["bump"].append((1, 1, "35e9", "BlockCyclic(1x1)", "EFT"))
run_dict["bump"].append((10, 0.1, "100e9", "ParMETIS(1.03,0.0001001)", "BlockCyclic(1x1)"))
run_dict["bump"].append((10, 1, "105e9", "ParMETIS(1.0001,0.0001001)", "EFT"))
run_dict["bump"].append((100, 10, "55e9", "ParMETIS(1.01,10)", "BlockCyclic(1x1)"))
# --- Core pool setup ---
node0_cores = list(range(0, 72))  # even = NUMA node0
node1_cores = list(range(72, 144))  # odd = NUMA node1
all_cores = node0_cores + node1_cores
free_cores = set(all_cores)

running_jobs = []  # (proc, allocated_cores)

# --- Build jobs list from run_dict ---


jobs = []

for v in ["D"]:
    for k in run_dict.keys():
        for dmem in [64e9, 72e9, 80e9, 88e9, 96e9, 9999e9]:
            for interior, boundary, mem, _, _ in run_dict[k]:
                ib = param[(interior, boundary)]
                prj_name = f"8x8x256_{k}_{interior}-{boundary}-1_IPDPS_FINALLAST"
                n_cores = 4
                cmd = [
                    "python3",
                    "train.py",
                    "--config-name",
                    f"8x8x128_dynamic_{k}_cnn",
                    f"feature.observer.version={v}",
                    f"wandb.project={prj_name}",
                    "algorithm.ent_coef=0.00025",
                    "reward.gamma=0.99",
                    f"graph.config.level_memory={mem}",
                    "reward.uniform_reward_scale=10",
                    f"wandb.name={v}_{int(dmem/1e9)}",
                    f"wandb.group={v}_{int(dmem/1e9)}",
                    f"seed={(seed+10)*100000000}",
                    f"system.mem={dmem}",
                    f"graph.config.arithmetic_intensity={ib[0]}",
                    f"graph.config.boundary_width={ib[1]}",
                    "eval.eval_interval=0",
                ]
                jobs.append((n_cores, cmd))
                print(f"Prepared job with {n_cores} cores: {' '.join(cmd)}")


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
        print(full_cmd)
        running_jobs.append((proc, cores))
    jobs = pending

    # 3. Avoid busy spinning
    time.sleep(2)
