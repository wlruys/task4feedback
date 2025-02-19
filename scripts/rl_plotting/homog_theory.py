import argparse
from dataclasses import dataclass


parser = argparse.ArgumentParser(prog="Others")

parser.add_argument("-n", "--num_tasks",
                    type=int,
                    help="number of tasks", default=4)
parser.add_argument("-g", "--gpus",
                    type=int,
                    help="number of gpus", default=4)
args = parser.parse_args()


serial = 0.08 * args.num_tasks
independent = serial / args.gpus

print(f"Independent,simtime,{independent}")
print(f"Serial,simtime,{serial}")
