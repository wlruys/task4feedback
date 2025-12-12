import argparse
import math
import os
import subprocess

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -J temp_{id}
#SBATCH -o temp_{id}.o%j
#SBATCH -e temp_{id}.e%j
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 144
#SBATCH -t {hh}:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=jaeyoung@utexas.edu

source /scratch/09611/jaeyoung/set_nano.sh
cd /scratch/09611/jaeyoung/task4feedback/experiments

# ---- concurrency-limited launcher ----
run_cmd() {{
    while [ "$(jobs -p | wc -l)" -ge {max_concurrent} ]; do
        sleep 1
    done
    echo "[START] $1"
    bash -c "$1" &
}}

# ---- commands start here ----
{commands}

wait
exit
"""


def chunk_list(lst, n):
    """Split list lst into n chunks as evenly as possible."""
    size = math.ceil(len(lst) / n)
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM job scripts with per-node concurrency-limited execution.")
    parser.add_argument("cmdfile", help="Text file containing commands (one per line).")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes (scripts) to create.")
    parser.add_argument("--hh", type=int, required=True, help="Hours for each SLURM job.")
    parser.add_argument("--max-concurrent", type=int, default=30, help="Maximum number of commands running concurrently per node.")
    parser.add_argument("--outdir", default="jobs", help="Directory to write SLURM scripts.")
    parser.add_argument("--submit", action="store_true", help="Automatically run sbatch on generated scripts.")

    args = parser.parse_args()

    # Load command list
    with open(args.cmdfile, "r") as f:
        commands = [line.strip() for line in f if line.strip()]

    # Split commands across nodes
    chunks = chunk_list(commands, args.nodes)

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    submitted = []

    # Generate each node's script
    for i, cmdlist in enumerate(chunks, start=1):

        # Build command block using run_cmd
        cmd_block = ""
        for cmd in cmdlist:
            cmd_block += f'run_cmd "{cmd}"\n'

        script_text = SLURM_TEMPLATE.format(id=i, hh=args.hh, max_concurrent=args.max_concurrent, commands=cmd_block)

        script_path = os.path.join(args.outdir, f"job_{i}.slurm")
        with open(script_path, "w") as f:
            f.write(script_text)

        print(f"Generated {script_path} containing {len(cmdlist)} commands.")

        # Optionally submit
        if args.submit:
            print(f"Submitting job_{i}.slurm...")
            res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)

            if res.returncode == 0:
                print("  → Submitted:", res.stdout.strip())
                submitted.append(res.stdout.strip())
            else:
                print("  → ERROR:", res.stderr.strip())

    if args.submit:
        print("\n✓ All jobs submitted:")
        for line in submitted:
            print("   ", line)
    else:
        print("\n✓ Done. Submit with:")
        print(f"  sbatch {args.outdir}/job_1.slurm")


if __name__ == "__main__":
    main()
