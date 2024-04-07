from .task import *
from ..types import *
import numpy as np
import random
#from .schedulers import SystemState
from .preprocess import *


def _random_set_pop(s: Set):
    index = random.randrange(0, len(s))
    element = list(s)[index]
    s.remove(element)
    return element


def _random_pop(s: Dict):
    index = np.random.randint(0, len(s))
    element = list(s)[index]
    s.pop(element)
    return element


def _determinisitc_pop(s: Dict):
    index = 0
    element = list(s)[index]
    s.pop(element)
    return element


def gaussian_noise(duration: float, scale: float = 0.05) -> int:
    stddev = duration * scale
    np.random.seed(None)
    noise = np.random.normal(0, stddev)
    return int(noise)


def no_noise(task: SimulatedTask) -> int:
    return 0


def random_topological_sort(
    tasklist: List[SimulatedTask], taskmap: SimulatedTaskMap, verbose: bool = False
) -> List[SimulatedTask]:
    L = []
    S = dict.fromkeys(get_initial_tasks(taskmap))

    random.seed(None)

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
    tasklist = sorted(tasklist, key=lambda t: t.info.order)

    return sort_tasks_by_order(tasklist, taskmap)


@dataclass(slots=True)
class Randomizer:
    seed: int = 0
    task_noise: Callable[[SimulatedTask], int] = no_noise
    task_order: Callable[[List[TaskID], SimulatedTaskMap], List[TaskID]] = (
        random_topological_sort
    )
    state: Optional[np.random.RandomState] = None
    init: bool = True

    def set_seed(self, seed: int):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def backup_state(self):
        self.state = np.random.get_state()

    def restore_state(self):
        np.random.set_state(self.state)

    def __post_init__(self):
        if self.init:
            np.random.seed(self.seed)
            random.seed(self.seed)
            self.backup_state()
            self.init = False

    def __deepcopy__(self, memo):
        return Randomizer(
            seed=self.seed,
            task_noise=self.task_noise,
            task_order=self.task_order,
            state=self.state,
            init=self.init,
        )
