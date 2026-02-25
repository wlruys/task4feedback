import torch
from task4feedback.interface.wrappers import *


class ReplayMapper:
    """
    A mapper that replays the mapping decisions from a previous simulator execution.
    """

    def __init__(self, prev_simulator: "SimulatorDriver"):
        assert prev_simulator.status == fastsim.ExecutionState.COMPLETE, (
            "Previous simulator must be complete to create a replay mapper."
        )
        runtime = prev_simulator.state.get_task_runtime()
        self.history = {}
        for task_id, _ in prev_simulator.input.graph.tasks.items():
            self.history[task_id] = runtime.get_compute_task_mapped_device(task_id)

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        candidates = torch.zeros(
            (simulator.observer.graph_spec.max_candidates), dtype=torch.int64
        )
        num_candidates = simulator.simulator.get_mappable_candidates(candidates)
        mapping_result = []
        for i in range(num_candidates):
            global_task_id = candidates[i].item()
            device = self.history[global_task_id]
            mapping_priority = simulator.simulator.get_state().get_mapping_priority(
                global_task_id
            )
            mapping_result.append(
                fastsim.Action(i, device, mapping_priority, mapping_priority)
            )
        return mapping_result
