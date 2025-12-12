import torch
from tensordict import TensorDict
from typing import Dict, Optional
import task4feedback.trip as trip
from .base import RuntimeEnv

class SanityCheckEnv(RuntimeEnv):
    """
    Environment for sanity checking task assignments against a known correct mapping.

    This environment is used for testing and validation, where we have a ground truth
    answer for which device each cell/task should be mapped to.

    Args:
        answer_mapping: Dictionary mapping cell IDs to correct device IDs
        **kwargs: Arguments passed to RuntimeEnv
    """

    def __init__(self, *args, answer_mapping: Optional[Dict[int, int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if answer_mapping is None:
            raise ValueError("SanityCheckEnv requires answer_mapping parameter")
        self.answer = answer_mapping

    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(policy="EFT")
            self.graph = self.simulator_factory[self.active_idx].input.graph
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory[self.active_idx].graph_spec.max_candidates,
            dtype=torch.int64,
        )

        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        self.simulator.simulator.map_tasks([trip.Action(0, chosen_device, mapping_priority, mapping_priority)])

        cell_id = self.graph.task_to_cell[global_task_id]

        # print(f"Cell ID: {cell_id}, Chosen Device: {chosen_device}")
        # print(f"Location List: {self.location_list}")

        answer = self.answer[cell_id]
        if answer == chosen_device:
            reward[0] = 1
        else:
            reward[0] = -1
        simulator_status = self.simulator.run_until_external_mapping()
        done[0] = simulator_status == trip.ExecutionState.COMPLETE

        obs = self._get_observation()
        time = obs["aux"]["time"].item()
        if done:
            obs, reward, time, improvement = self._handle_done(obs)

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        self.step_count += 1
        return out
