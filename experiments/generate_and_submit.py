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
#SBATCH -t 12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=jaeyoung@utexas.edu

source /scratch/09611/jaeyoung/set_nano.sh
cd /scratch/09611/jaeyoung/task4feedback/experiments

# Commands follow
{commands}

wait
exit
"""


def chunk_list(lst, n):
    """Split list lst into n chunks as evenly as possible."""
    avg = math.ceil(len(lst) / n)
    return [lst[i : i + avg] for i in range(0, len(lst), avg)]


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM job scripts by splitting commands across nodes and auto-submit.")
    parser.add_argument("cmdfile", help="Text file with one command per line.")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes to use.")
    parser.add_argument("--outdir", default="jobs", help="Directory for job scripts.")
    parser.add_argument("--submit", action="store_true", help="Automatically submit the jobs with sbatch.")

    args = parser.parse_args()

    # Read commands
    with open(args.cmdfile, "r") as f:
        commands = [line.strip() for line in f if line.strip()]

    # Split into N chunks
    chunks = chunk_list(commands, args.nodes)

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    submitted_jobs = []

    for i, cmdlist in enumerate(chunks, start=1):
        cmd_block = "\n".join(cmd + " ;" for cmd in cmdlist)
        job_script = SLURM_TEMPLATE.format(id=i, commands=cmd_block)

        outpath = os.path.join(args.outdir, f"job_{i}.slurm")
        with open(outpath, "w") as f:
            f.write(job_script)

        print(f"Generated {outpath} with {len(cmdlist)} commands.")

        # Auto-submit job
        if args.submit:
            print(f"Submitting job_{i}.slurm...")
            result = subprocess.run(["sbatch", outpath], capture_output=True, text=True)

            if result.returncode == 0:
                print("  → Submitted:", result.stdout.strip())
                submitted_jobs.append(result.stdout.strip())
            else:
                print("  → Error submitting job:", result.stderr.strip())

    if args.submit:
        print("\n✓ All jobs submitted:")
        for job in submitted_jobs:
            print("   ", job)
    else:
        print("\n✓ Done! Submit manually using:")
        print(f"  sbatch {args.outdir}/job_1.slurm")


if __name__ == "__main__":
    main()
