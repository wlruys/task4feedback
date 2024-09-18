import argparse
from dataclasses import dataclass


parser = argparse.ArgumentParser(prog="Cholesky")
parser.add_argument("-b", "--block",
                    type=int,
                    help="bxb blocks", default=10)
parser.add_argument("-g", "--gpus",
                    type=int,
                    help="number of gpus", default=4)
parser.add_argument("-t", "--time",
                    help="per-task execution time (us)", default=None)

args = parser.parse_args()


@dataclass
class Bounds:
    independent: float
    level: float
    serial: float


def get_bounds(b: int, p: int) -> Bounds:
    potrf_tasks = 0
    trsm_tasks = 0
    gemm_tasks = 0
    level_time = 0

    potrf_weight = 160000
    trsm_weight = 150000
    gemm_weight = 140000

    if args.time is not None:
        potrf_weight = float(args.time)
        trsm_weight = float(args.time)
        gemm_weight = float(args.time)

    for k in range(1, b):
        new_potrf_tasks = 1
        new_trsm_tasks = b - k
        new_gemm_tasks = 0.5 * ((b - k) * (b - k + 1))

        potrf_tasks += new_potrf_tasks
        trsm_tasks += new_trsm_tasks
        gemm_tasks += new_gemm_tasks

        level_time += (
            new_potrf_tasks * potrf_weight
            + float(new_trsm_tasks) / p * trsm_weight
            + float(new_gemm_tasks) / p * gemm_weight
        )

    number_of_tasks = potrf_weight * potrf_tasks + trsm_weight * trsm_tasks + gemm_weight * gemm_tasks
    # print(f"Number of tasks: {number_of_tasks}")

    serial_time =  number_of_tasks
    independent_time = float(number_of_tasks) / p

    convert_to_ms = 1e3
    return Bounds(
        independent_time / convert_to_ms,
        level_time / convert_to_ms,
        serial_time / convert_to_ms,
    )

bounds=get_bounds(args.block, args.gpus)
print(f"Independent,simtime,{bounds.independent/1000}")
# print(f"BSP,simtime,{bounds.level/1000}")
print(f"Serial,simtime,{bounds.serial/1000}")

