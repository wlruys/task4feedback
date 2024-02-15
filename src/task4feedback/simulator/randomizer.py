from .task import *
from ..types import *
import numpy as np
import random
from .schedulers import SystemState
from .preprocess import *


def _random_set_pop(s: Set):
    index = random.randrange(0, len(s))
    element = list(s)[index]
    s.remove(element)
    return element


def _random_pop(s: Dict):
    index = random.randrange(0, len(s))
    element = list(s)[index]
    s.pop(element)
    return element


def _determinisitc_pop(s: Dict):
    index = 0
    element = list(s)[index]
    s.pop(element)
    return element


def gaussian_noise(task: SimulatedTask) -> int:
    duration = task.duration.duration
    stddev = duration * 0.05
    noise = np.random.normal(0, stddev)
    return int(noise)


def no_noise(task: SimulatedTask) -> int:
    return 0


def random_topological_sort(
    tasklist: List[TaskID], taskmap: SimulatedTaskMap, verbose: bool = False
) -> List[TaskID]:
    L = []
    S = dict.fromkeys(get_initial_tasks(taskmap))

    taskmapcopy = deepcopy(taskmap)
    i = 0
    while S:
        n = _random_pop(S)
        # n = _determinisitc_pop(S)
        L.append(n)

        if verbose:
            print(f"Step {i}, Adding {n} to L")

        for m in taskmapcopy[n].dependents:
            taskmapcopy[m].dependencies.remove(n)
            if not taskmapcopy[m].dependencies:
                if verbose:
                    print(f"Removed all dependencies from {m}, adding to S")
                S[m] = None

        i += 1
        taskmap[n].info.order = i

    return sort_tasks_by_order(tasklist, taskmap)


@dataclass(slots=True)
class Randomizer:
    seed: int = 0
    task_noise: Callable[[SimulatedTask], int] = no_noise
    task_order: Callable[[List[TaskID], SimulatedTaskMap], List[TaskID]] = (
        random_topological_sort
    )

    def __post_init__(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
