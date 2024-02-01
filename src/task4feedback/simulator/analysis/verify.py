from .recorder import ComputeTaskRecorder, DataTaskRecorder
from ...types import *
from ..task import *


def verify_order(
    compute_tasks: ComputeTaskRecorder,
    data_tasks: DataTaskRecorder,
    taskmap: TaskMap | SimulatedTaskMap,
):
    pass
