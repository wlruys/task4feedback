from task4feedback import fastsim2 as fastsim
from task4feedback.interface import *
import torch
from typing import Optional, List

from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import (
    DefaultObserverFactory,
    SimulatorFactory,
    create_graph_spec,
)

from torchrl.data import Composite, TensorSpec, Unbounded, Binary, Bounded
from torchrl.envs.utils import make_composite_from_td
from torchrl.envs import StepCounter, TrajCounter, TransformedEnv
import tensordict
from tensordict import TensorDict
from aim.pytorch import track_gradients_dists, track_params_dists
from task4feedback.graphs.base import Graph, DataBlocks, ComputeDataGraph, DataGeometry
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from task4feedback.legacy_graphs import *

# def make_test_stencil_graph():
#     interior_size = 1000 * 2000
#     boundary_size = 1000 * 2000

#     def task_config(task_id: TaskID) -> TaskPlacementInfo:
#         placement_info = TaskPlacementInfo()
#         placement_info.add(
#             (Device(Architecture.GPU, -1),),
#             TaskRuntimeInfo(task_time=1000, device_fraction=1),
#         )
#         placement_info.add(
#             (Device(Architecture.CPU, -1),),
#             TaskRuntimeInfo(task_time=1000, device_fraction=1),
#         )

#         return placement_info

#     def sizes(data_id: DataID) -> int:
#         return boundary_size if data_id.idx[1] == 1 else interior_size

#     def blocked_initial_placement(data_id: DataID) -> List[Device]:
#         batch = data_config.width // (4 // 2)

#         device_id_i = int(data_id.idx[-2] // batch)
#         device_id_j = int(data_id.idx[-1] // batch)
#         idx = device_id_i + (4 // 2) * device_id_j

#         dev_id = idx % 4
#         if dev_id == 0:
#             return Device(Architecture.CPU, 0)
#         else:
#             return Device(Architecture.GPU, dev_id - 1)

#     def cpu_placement(data_id: DataID) -> List[Device]:
#         return Device(Architecture.CPU, 0)

#     def random_placement(data_id: DataID) -> List[Device]:
#         dev_id = random.randint(0, 4 - 1)
#         if dev_id == 0:
#             return Device(Architecture.CPU, 0)
#         else:
#             return Device(Architecture.GPU, dev_id - 1)

#     data_config = StencilDataGraphConfig()
#     data_config.n_devices = 4
#     data_config.dimensions = 2
#     data_config.width = 4
#     data_config.initial_placement = random_placement
#     data_config.initial_sizes = sizes
#     config = StencilConfig(width=4, steps=20, task_config=task_config, dimensions=2)
#     tasks, data = make_graph(config, data_config=data_config)
#     return tasks, data

#     # Validate input length
#     if len(arr) != 16:
#         raise ValueError("Input array must have exactly 16 elements.")

#     # Reshape the list into a 4x4 numpy array
#     matrix = np.array(arr).reshape((4, 4))

#     # Define a color map: mapping 0->black, 1->red, 2->green, 3->blue.
#     cmap = ListedColormap(
#         ["black", "red", "green", "blue", "yellow", "purple", "orange", "cyan"]
#     )

#     # Create the plot
#     plt.imshow(matrix, cmap=cmap, vmin=0, vmax=7)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("Device Mapping Result")

#     # Create legend handles for each device color
#     patches = [
#         mpatches.Patch(color="black", label="Device 0"),
#         mpatches.Patch(color="red", label="Device 1"),
#         mpatches.Patch(color="green", label="Device 2"),
#         mpatches.Patch(color="blue", label="Device 3"),
#         mpatches.Patch(color="yellow", label="Device 4"),
#         mpatches.Patch(color="purple", label="Device 5"),
#         mpatches.Patch(color="orange", label="Device 6"),
#         mpatches.Patch(color="cyan", label="Device 7"),
#     ]

#     # Add the legend to the plot; position it outside the plot area
#     plt.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.15, 1))

#     plt.savefig(f"device_mapping_{number}.png")


def plot_8x8_matrix(arr, number):
    # Validate input length
    if len(arr) != 64:
        raise ValueError("Input array must have exactly 16 elements.")

    # Reshape the list into a 4x4 numpy array
    matrix = np.array(arr).reshape((8, 8))

    # Define a color map: mapping 0->black, 1->red, 2->green, 3->blue.
    cmap = ListedColormap(
        ["black", "red", "green", "blue", "yellow", "purple", "orange", "cyan"]
    )

    # Create the plot
    plt.imshow(matrix, cmap=cmap, vmin=0, vmax=7)
    plt.xticks([])
    plt.yticks([])
    plt.title("Device Mapping Result")

    # Create legend handles for each device color
    patches = [
        mpatches.Patch(color="black", label="Device 0"),
        mpatches.Patch(color="red", label="Device 1"),
        mpatches.Patch(color="green", label="Device 2"),
        mpatches.Patch(color="blue", label="Device 3"),
        mpatches.Patch(color="yellow", label="Device 4"),
        mpatches.Patch(color="purple", label="Device 5"),
        mpatches.Patch(color="orange", label="Device 6"),
        mpatches.Patch(color="cyan", label="Device 7"),
    ]

    # Add the legend to the plot; position it outside the plot area
    plt.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.savefig(f"device_mapping_{number}.png")


class RuntimeEnv(EnvBase):

    def __init__(
        self,
        simulator_factory: SimulatorFactory,
        seed: int = 0,
        device="cpu",
        baseline_time=56000,
        change_priority=False,
        change_duration=False,
        snapshot_interval=5,
    ):
        super().__init__(device=device)

        self.change_priority = change_priority
        self.change_duration = change_duration
        self.snapshot_interval = snapshot_interval

        self.simulator_factory = simulator_factory
        self.simulator: SimulatorDriver = simulator_factory.create(seed)

        self.buffer_idx = 0
        self.resets = 0

        self.observation_spec = self._create_observation_spec()
        self.action_spec = self._create_action_spec()
        self.reward_spec = self._create_reward_spec()
        self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

        self.workspace = self._prealloc_step_buffers(100)
        self.baseline_time = baseline_time
        self.mapping_history = [-1 for _ in range(64)]

    def _get_baseline(self, use_eft=True):
        if use_eft:
            simulator_copy = self.simulator.fresh_copy()
            simulator_copy.initialize()
            simulator_copy.initialize_data()
            simulator_copy.disable_external_mapper()
            final_state = simulator_copy.run()
            assert (
                final_state == fastsim.ExecutionState.COMPLETE
            ), f"Baseline returned unexpected final state: {final_state}"
            return simulator_copy.time
        return self.baseline_time

    def _create_observation_spec(self) -> TensorSpec:
        obs = self.simulator.observer.get_observation()
        comp = make_composite_from_td(obs)
        comp = Composite(observation=comp)
        return comp

    def _create_action_spec(self, ndevices: int = 5) -> TensorSpec:
        n_devices = self.simulator_factory.graph_spec.max_devices
        out = Bounded(
            shape=(1,),
            device=self.device,
            dtype=torch.int64,
            low=torch.tensor(0, device=self.device),
            high=torch.tensor(n_devices, device=self.device),
        )
        out = Composite(action=out)
        return out

    def _create_reward_spec(self) -> TensorSpec:
        return Unbounded(shape=(1,), device=self.device, dtype=torch.float32)

    def _get_observation(self, td: TensorDict = None) -> TensorDict:
        if td is None:
            obs = self.simulator.observer.get_observation()
            td = TensorDict(observation=obs)
        else:
            self.simulator.observer.get_observation(td["observation"])
        return td

    def _get_new_observation_buffer(self) -> TensorDict:
        obs = self.simulator.observer.new_observation_buffer()
        td = TensorDict(observation=obs)
        return td

    def _get_new_step_buffer(self) -> TensorDict:
        obs = self._get_new_observation_buffer()
        obs.set("reward", torch.tensor([0], device=self.device, dtype=torch.float32))
        obs.set("done", torch.tensor([False], device=self.device, dtype=torch.bool))
        return obs

    def _prealloc_step_buffers(self, n: int) -> List[TensorDict]:
        return [self._get_new_step_buffer() for _ in range(n)]

    def _get_preallocated_step_buffer(
        self, buffers: List[TensorDict], i: int
    ) -> TensorDict:
        if i >= len(buffers):
            buffers.extend(self._prealloc_step_buffers(2 * len(buffers)))
        return buffers[i]

    def _get_current_buffer(self):
        buf = self._get_preallocated_step_buffer(self.workspace, self.buffer_idx)
        self.buffer_idx += 1

    def _step(self, td: TensorDict) -> TensorDict:
        assert self.makespan > 0, "Makespan not set"
        chosen_device = td["action"].item()
        local_id = 0
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )
        self.simulator.get_mappable_candidates(candidate_workspace)
        global_task_id = candidate_workspace[local_id].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)
        reserving_priority = mapping_priority
        launching_priority = mapping_priority
        actions = [
            fastsim.Action(
                local_id, chosen_device, reserving_priority, launching_priority
            )
        ]
        self.simulator.simulator.map_tasks(actions)
        dummy_sim = self.simulator.copy()
        dummy_sim.disable_external_mapper()
        dummy_sim.run()
        simulator_status = self.simulator.run_until_external_mapping()
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        self.mapping_history[global_task_id % 64] = chosen_device
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE
        if dummy_sim.time > self.makespan:
            reward[0] = -1
        elif dummy_sim.time < self.makespan:
            reward[0] = 1
        else:
            reward[0] = 0
        self.makespan = dummy_sim.time

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()

        if done:
            baseline_time = self._get_baseline()
            print(f"Time: {time}, Baseline: {baseline_time}")

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.resets += 1
        if self.resets % (self.snapshot_interval * 2) == 0 and self.resets > 0:
            plot_8x8_matrix(self.mapping_history, int(self.resets / 2))

        # tasks, data = make_test_stencil_graph()
        # s = uniform_connected_devices(4, 1000000000, 0, bandwidth=2000)

        # self.simulator_factory.input.data = DataBlocks.create_from_legacy_data(data, s)
        current_priority_seed = self.simulator_factory.pseed
        current_duration_seed = self.simulator_factory.seed
        if self.change_priority:
            new_priority_seed = current_priority_seed + self.resets
        else:
            new_priority_seed = current_priority_seed
        if self.change_duration:
            new_duration_seed = current_duration_seed + self.resets
        else:
            new_duration_seed = current_duration_seed

        new_priority_seed = int(new_priority_seed)
        new_duration_seed = int(new_duration_seed)
        self.taskid_history = []
        self.simulator = self.simulator_factory.create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )
        dummy_sim = self.simulator.copy()
        dummy_sim.disable_external_mapper()
        dummy_sim.run()
        self.makespan = dummy_sim.time

        simulator_status = self.simulator.run_until_external_mapping()
        assert (
            simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING
        ), f"Unexpected simulator status: {simulator_status}"

        obs = self._get_observation()
        return obs

    @property
    def observer(self):
        return self.simulator.observer

    def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
        torch.manual_seed(seed)
        self.simulator_factory.set_seed(priority_seed=seed)


def make_simple_env_from_legacy(tasks, data):
    s = uniform_connected_devices(5, 1000000000, 1, 2000)
    d = DataBlocks.create_from_legacy_data(data, s)
    m = Graph.create_from_legacy_graph(tasks, data)
    m.finalize_tasks()
    spec = create_graph_spec()
    input = SimulatorInput(m, d, s)
    env = RuntimeEnv(
        SimulatorFactory(input, spec, DefaultObserverFactory), device="cpu"
    )
    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())
    return env


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
    ):
        super().__init__(
            simulator_factory,
            seed,
            device,
            baseline_time,
            change_priority,
            change_duration,
        )
        self.use_external_mapper = use_external_mapper

    def _step(self, td: TensorDict) -> TensorDict:
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
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

        new_action = torch.zeros((1,), dtype=torch.int64)
        new_action[0] = action.device
        td.set_("action", new_action)
        return super()._step(td)

    def set_internal_mapper(self, internal_mapper):
        self.simulator.internal_mapper = internal_mapper

    def set_external_mapper(self, external_mapper):
        self.simulator.external_mapper = external_mapper

    def enable_external_mapper(self):
        self.use_external_mapper = True

    def disable_external_mapper(self):
        self.use_external_mapper = False

    def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
        if seed is None:
            seed = 0
        else:
            seed = seed + 1e7

        self.simulator_factory.set_seed(priority_seed=seed)
        return seed


def make_simple_env(graph: ComputeDataGraph):
    s = uniform_connected_devices(5, 1000000000, 1, 2000)
    d = graph.get_blocks()
    m = graph
    m.finalize_tasks()

    spec = create_graph_spec()
    input = SimulatorInput(m, d, s)

    env = RuntimeEnv(
        SimulatorFactory(input, spec, DefaultObserverFactory), device="cpu"
    )

    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())

    return env
