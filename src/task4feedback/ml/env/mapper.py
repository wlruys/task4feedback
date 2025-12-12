import torch
import random
from tensordict import TensorDict
from typing import Optional
import task4feedback.trip as trip
from .base import RuntimeEnv
from task4feedback.trip import SchedulerState

MAPPER_SEED_OFFSET = int(1e7)

class MapperRuntimeEnv(RuntimeEnv):
    def __init__(
        self,
        simulator_factory,
        seed: int = 0,
        device="cpu",
        baseline_time=56000,
        use_external_mapper: bool = False,
        change_priority=True,
        change_duration=False,
        **kwargs,
    ):
        super().__init__(
            simulator_factory,
            seed,
            device,
            baseline_time,
            change_priority,
            change_duration,
            **kwargs,
        )
        self.use_external_mapper = use_external_mapper

    def _step(self, td: TensorDict) -> TensorDict:
        """
        Step the environment using actions from the configured mapper.

        """
        candidate_workspace = torch.zeros(
            self.simulator_factory[self.active_idx].graph_spec.max_candidates,
            dtype=torch.int64,
        )
        self.simulator.get_mappable_candidates(candidate_workspace)
        global_task_id = candidate_workspace[0].item()
        scheduler_state: SchedulerState = self.simulator.state

        if self.use_external_mapper:
            external_mapper = self.simulator.external_mapper
            action = external_mapper.map_tasks(
                self.simulator,
            )[0]
        else:
            internal_mapper = self.simulator.internal_mapper
            action = internal_mapper.map_task(
                global_task_id,
                scheduler_state,
            )

        mapper_td = td.clone()
        mapper_td.set("action", action.device - 1)
        return super()._step(mapper_td)

    def set_internal_mapper(self, internal_mapper):
        self.simulator.internal_mapper = internal_mapper

    def set_external_mapper(self, external_mapper):
        self.simulator.external_mapper = external_mapper

    def enable_external_mapper(self):
        self.use_external_mapper = True

    def disable_external_mapper(self):
        self.use_external_mapper = False

    def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
        """
        Set seed for mapper environment.
        """
        if seed is None:
            seed = 0
        else:
            seed = seed + MAPPER_SEED_OFFSET

        self.simulator_factory[self.active_idx].set_seed(priority_seed=seed)
        return seed

