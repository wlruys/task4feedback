from task4feedback import fastsim2 as fastsim
from task4feedback.fastsim2 import GraphExtractor, EventType
from task4feedback.interface import *
import torch
from typing import Optional, List
import numpy as np

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
from task4feedback.graphs.base import Graph, DataBlocks, ComputeDataGraph, DataGeometry
import random
from task4feedback.graphs.mesh.plot import *
from torchrl.data import Categorical
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from task4feedback.legacy_graphs import *
import os
import wandb
from task4feedback.graphs.jacobi import JacobiGraph


class RuntimeEnv(EnvBase):
    def __init__(
        self,
        simulator_factory: SimulatorFactory,
        seed: int = 0,
        device="cpu",
        baseline_time=4000 * 5,
        change_priority=False,
        change_duration=False,
        change_locations=False,
        only_gpu=True,
        location_seed=0,
        priority_seed=0,
        location_randomness=1,
        location_list=[1, 2, 3, 4],
    ):
        super().__init__(device=device)

        self.change_priority = change_priority
        self.change_duration = change_duration
        self.change_locations = change_locations
        self.location_seed = location_seed
        self.location_randomness = location_randomness
        self.location_list = location_list
        self.only_gpu = only_gpu

        self.simulator_factory = simulator_factory
        self.simulator: SimulatorDriver = simulator_factory.create(
            seed, priority_seed=priority_seed
        )

        self.buffer_idx = 0
        self.resets = 0

        if self.change_locations:
            graph = simulator_factory.input.graph
            assert hasattr(graph, "get_cell_locations")
            assert hasattr(graph, "set_cell_locations")
            assert hasattr(graph, "randomize_locations")
            self.initial_location_list = graph.get_cell_locations()
            self.location_randomness = location_randomness
            self.location_list = location_list
            random.seed(self.location_seed)

        self.observation_spec = self._create_observation_spec()
        self.action_spec = self._create_action_spec()
        self.reward_spec = self._create_reward_spec()
        self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

        self.workspace = self._prealloc_step_buffers(100)
        self.baseline_time = baseline_time

        if change_locations:
            graph.randomize_locations(
                self.location_randomness, self.location_list, verbose=False
            )

    def _get_baseline(self, use_eft=False):
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
        if self.only_gpu:
            n_devices -= 1
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

        if self.only_gpu:
            chosen_device = chosen_device + 1
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
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
        # print("Chosen device:", chosen_device)
        actions = [
            fastsim.Action(
                local_id, chosen_device, reserving_priority, launching_priority
            )
        ]
        self.simulator.simulator.map_tasks(actions)

        simulator_copy = self.simulator.copy()
        simulator_copy.disable_external_mapper()
        simulator_copy.run()
        if simulator_copy.time > self.makespan:
            reward[0] = -1
        elif simulator_copy.time < self.makespan:
            reward[0] = 1
        else:
            reward[0] = 0
        self.makespan = simulator_copy.time

        simulator_status = self.simulator.run_until_external_mapping()
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()
        if done:
            obs["observation"]["aux"]["improvement"][0] = self.EFT_baseline / time - 1
            print(
                f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {obs['observation']['aux']['improvement'][0]:.2f}"
            )

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.resets += 1

        current_priority_seed = self.simulator_factory.pseed
        current_duration_seed = self.simulator_factory.seed

        if self.change_locations:
            new_location_seed = self.location_seed + self.resets
            # Load initial location list
            graph = self.simulator_factory.input.graph
            graph.set_cell_locations(self.initial_location_list)

            random.seed(new_location_seed)
            graph.randomize_locations(
                self.location_randomness, self.location_list, verbose=False
            )

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
        if self.change_locations and isinstance(
            self.simulator_factory.input.graph, JacobiGraph
        ):
            self.simulator_factory.input.graph.randomize_locations(
                1, location_list=range(self.simulator_factory.graph_spec.max_devices)
            )
        self.simulator = self.simulator_factory.create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )
        self.EFT_baseline = self._get_baseline(use_eft=True)
        self.makespan = self.EFT_baseline

        self.makespan = self._get_baseline(use_eft=True)
        self.EFT_baseline = self.makespan

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
        np.random.seed(seed)
        random.seed(seed)
        if self.change_priority:
            self.simulator_factory.set_seed(priority_seed=seed)
        if self.change_duration:
            self.simulator_factory.set_seed(duration_seed=seed)


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

        new_action = torch.tensor([action.device - 1], dtype=torch.int64)
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


class EFTIncrementalEnv(EnvBase):
    def __init__(
        self,
        simulator_factory: SimulatorFactory,
        seed: int = 0,
        device="cpu",
        baseline_time=56000,
        change_priority=False,
        change_duration=False,
        change_locations=False,
        only_gpu=False,
        path=".",
    ):
        super().__init__(device=device)

        self.change_priority = change_priority
        self.change_duration = change_duration
        self.change_locations = change_locations
        self.path = path
        self.only_gpu = only_gpu

        self.simulator_factory = simulator_factory
        self.simulator: SimulatorDriver = simulator_factory.create(seed)
        self.graph_extractor: GraphExtractor = GraphExtractor(
            self.simulator.get_state()
        )

        self.buffer_idx = 0
        self.resets = 0

        self.observation_spec = self._create_observation_spec()
        self.action_spec = self._create_action_spec()
        self.reward_spec = self._create_reward_spec()
        self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

        self.workspace = self._prealloc_step_buffers(100)
        self.baseline_time = baseline_time

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
        if self.only_gpu:
            n_devices -= 1
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
        chosen_device = td["action"].item()
        if self.only_gpu:
            chosen_device = chosen_device + 1
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            1,
            dtype=torch.int64,
        )
        self.simulator.get_mappable_candidates(candidate_workspace)

        dependents = torch.zeros(16, dtype=torch.int64)
        dep_count = self.graph_extractor.get_k_hop_dependents(
            candidate_workspace, 2, dependents
        )

        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)
        reserving_priority = mapping_priority
        launching_priority = mapping_priority
        actions = [
            fastsim.Action(0, chosen_device, reserving_priority, launching_priority)
        ]
        sim_eft = self.simulator.copy()
        self.simulator.simulator.map_tasks(actions)
        sim_ml = self.simulator.copy()
        # Set Reward reward[0]
        sim_eft.disable_external_mapper()
        sim_ml.disable_external_mapper()
        for i in range(dep_count):
            sim_eft.set_task_breakpoint(EventType.COMPLETER, dependents[i])
            sim_ml.set_task_breakpoint(EventType.COMPLETER, dependents[i])
        for i in range(dep_count):
            sim_eft.run()
            sim_ml.run()
        eft_time = sim_eft.time - self.simulator.time
        ml_time = sim_ml.time - self.simulator.time
        reward[0] = (eft_time - ml_time) / self.EFT_baseline
        simulator_status = self.simulator.run_until_external_mapping()
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()
        if done:
            obs["observation"]["aux"]["improvement"][0] = self.EFT_baseline / time - 1
            print(
                f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {obs['observation']['aux']['improvement'][0]:.2f}"
            )

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.resets += 1

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
        if self.change_locations and isinstance(
            self.simulator_factory.input.graph, JacobiGraph
        ):
            self.simulator_factory.input.graph.randomize_locations(
                1, location_list=range(self.simulator_factory.graph_spec.max_devices)
            )
        self.simulator = self.simulator_factory.create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )
        self.graph_extractor: GraphExtractor = GraphExtractor(
            self.simulator.get_state()
        )
        self.EFT_baseline = self._get_baseline(use_eft=True)

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
        if self.change_priority:
            self.simulator_factory.set_seed(priority_seed=seed)
        if self.change_duration:
            self.simulator_factory.set_seed(duration_seed=seed)


class LookaheadEnv(EnvBase):
    """
    For each action, explore all the other actions using EFT and +1 if it was the best, 0 if it was the same, -1 if it was worse.
    """

    def __init__(
        self,
        simulator_factory: SimulatorFactory,
        seed: int = 0,
        device="cpu",
        baseline_time=4000 * 5,
        change_priority=False,
        change_duration=False,
        change_locations=False,
        only_gpu=True,
        location_seed=0,
        priority_seed=0,
        location_randomness=1,
        location_list=[1, 2, 3, 4],
        bonus_scale=0.0,
    ):
        super().__init__(device=device)

        self.change_priority = change_priority
        self.change_duration = change_duration
        self.change_locations = change_locations
        self.location_seed = location_seed
        self.location_randomness = location_randomness
        self.location_list = location_list
        self.only_gpu = only_gpu
        self.bonus_scale = bonus_scale

        self.simulator_factory = simulator_factory
        self.simulator: SimulatorDriver = simulator_factory.create(
            seed, priority_seed=priority_seed
        )

        self.buffer_idx = 0
        self.resets = 0
        self.action_candidates = range(
            int(only_gpu), self.simulator_factory.graph_spec.max_devices
        )

        if self.change_locations:
            graph = simulator_factory.input.graph
            assert hasattr(graph, "get_cell_locations")
            assert hasattr(graph, "set_cell_locations")
            assert hasattr(graph, "randomize_locations")
            self.initial_location_list = graph.get_cell_locations()
            self.location_randomness = location_randomness
            self.location_list = location_list
            random.seed(self.location_seed)

        self.observation_spec = self._create_observation_spec()
        self.action_spec = self._create_action_spec()
        self.reward_spec = self._create_reward_spec()
        self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

        self.workspace = self._prealloc_step_buffers(100)
        self.baseline_time = baseline_time

        if change_locations:
            graph.randomize_locations(
                self.location_randomness, self.location_list, verbose=False
            )

    def _get_baseline(self, use_eft=False):
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

    def _create_action_spec(self) -> TensorSpec:
        n_devices = self.simulator_factory.graph_spec.max_devices
        if self.only_gpu:
            n_devices -= 1
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
        chosen_device = td["action"].item()

        if self.only_gpu:
            chosen_device = chosen_device + 1
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
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
        # print("Chosen device:", chosen_device)
        actions = [
            fastsim.Action(
                local_id, chosen_device, reserving_priority, launching_priority
            )
        ]
        min_time = 999990000
        for i in self.action_candidates:
            if i == chosen_device:
                continue
            simulator_copy = self.simulator.copy()
            simulator_copy.simulator.map_tasks(
                [fastsim.Action(local_id, i, reserving_priority, launching_priority)]
            )
            simulator_copy.disable_external_mapper()
            simulator_copy.run()
            if simulator_copy.time < min_time:
                min_time = simulator_copy.time

        self.simulator.simulator.map_tasks(actions)
        simulator_copy = self.simulator.copy()
        simulator_copy.disable_external_mapper()
        simulator_copy.run()
        delta = ((simulator_copy.time - min_time) // 1000) * 1000
        if delta > 0:
            reward[0] = -1
        elif delta < 0:
            reward[0] = 1
        else:
            reward[0] = 0

        simulator_status = self.simulator.run_until_external_mapping()
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()
        if done:
            improvement = self.EFT_baseline / time - 1
            obs["observation"]["aux"]["improvement"][0] = improvement
            print(
                f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {obs['observation']['aux']['improvement'][0]:.2f}"
            )
            reward[0] += self.bonus_scale * improvement

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.resets += 1

        current_priority_seed = self.simulator_factory.pseed
        current_duration_seed = self.simulator_factory.seed

        if self.change_locations:
            new_location_seed = self.location_seed + self.resets
            # Load initial location list
            graph = self.simulator_factory.input.graph
            graph.set_cell_locations(self.initial_location_list)

            random.seed(new_location_seed)
            graph.randomize_locations(
                self.location_randomness, self.location_list, verbose=False
            )

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
        if self.change_locations and isinstance(
            self.simulator_factory.input.graph, JacobiGraph
        ):
            self.simulator_factory.input.graph.randomize_locations(
                1, location_list=range(self.simulator_factory.graph_spec.max_devices)
            )
        self.simulator = self.simulator_factory.create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )
        self.EFT_baseline = self._get_baseline(use_eft=True)

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
        np.random.seed(seed)
        random.seed(seed)
        if self.change_priority:
            self.simulator_factory.set_seed(priority_seed=seed)
        if self.change_duration:
            self.simulator_factory.set_seed(duration_seed=seed)


# class LookaheadBinaryEnv(EnvBase):
#     def __init__(
#         self,
#         simulator_factory: SimulatorFactory,
#         seed: int = 0,
#         device="cpu",
#         baseline_time=4000 * 5,
#         change_priority=False,
#         change_duration=False,
#         change_locations=False,
#         only_gpu=True,
#         location_seed=0,
#         priority_seed=0,
#         location_randomness=1,
#         location_list=[1, 2, 3, 4],
#     ):
#         super().__init__(device=device)

#         self.change_priority = change_priority
#         self.change_duration = change_duration
#         self.change_locations = change_locations
#         self.location_seed = location_seed
#         self.location_randomness = location_randomness
#         self.location_list = location_list
#         self.only_gpu = only_gpu
#         self.s = 0

#         self.simulator_factory = simulator_factory
#         self.simulator: SimulatorDriver = simulator_factory.create(
#             seed, priority_seed=priority_seed
#         )

#         self.buffer_idx = 0
#         self.resets = 0
#         self.action_candidates = range(
#             int(only_gpu), self.simulator_factory.graph_spec.max_devices
#         )

#         if self.change_locations:
#             graph = simulator_factory.input.graph
#             assert hasattr(graph, "get_cell_locations")
#             assert hasattr(graph, "set_cell_locations")
#             assert hasattr(graph, "randomize_locations")
#             self.initial_location_list = graph.get_cell_locations()
#             self.location_randomness = location_randomness
#             self.location_list = location_list
#             random.seed(self.location_seed)

#         self.observation_spec = self._create_observation_spec()
#         self.action_spec = self._create_action_spec()
#         self.reward_spec = self._create_reward_spec()
#         self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

#         self.workspace = self._prealloc_step_buffers(100)
#         self.baseline_time = baseline_time

#         if change_locations:
#             graph.randomize_locations(
#                 self.location_randomness, self.location_list, verbose=False
#             )

#     def _get_baseline(self, use_eft=False):
#         if use_eft:
#             simulator_copy = self.simulator.fresh_copy()
#             simulator_copy.initialize()
#             simulator_copy.initialize_data()
#             simulator_copy.disable_external_mapper()
#             final_state = simulator_copy.run()
#             assert (
#                 final_state == fastsim.ExecutionState.COMPLETE
#             ), f"Baseline returned unexpected final state: {final_state}"
#             return simulator_copy.time
#         return self.baseline_time

#     def _create_observation_spec(self) -> TensorSpec:
#         obs = self.simulator.observer.get_observation()
#         comp = make_composite_from_td(obs)
#         comp = Composite(observation=comp)
#         return comp

#     def _create_action_spec(self, ndevices: int = 5) -> TensorSpec:
#         n_devices = self.simulator_factory.graph_spec.max_devices
#         if self.only_gpu:
#             n_devices -= 1
#         out = Bounded(
#             shape=(1,),
#             device=self.device,
#             dtype=torch.int64,
#             low=torch.tensor(0, device=self.device),
#             high=torch.tensor(n_devices, device=self.device),
#         )
#         out = Composite(action=out)
#         return out

#     def _create_reward_spec(self) -> TensorSpec:
#         return Unbounded(shape=(1,), device=self.device, dtype=torch.float32)

#     def _get_observation(self, td: TensorDict = None) -> TensorDict:
#         if td is None:
#             obs = self.simulator.observer.get_observation()
#             td = TensorDict(observation=obs)
#         else:
#             self.simulator.observer.get_observation(td["observation"])
#         return td

#     def _get_new_observation_buffer(self) -> TensorDict:
#         obs = self.simulator.observer.new_observation_buffer()
#         td = TensorDict(observation=obs)
#         return td

#     def _get_new_step_buffer(self) -> TensorDict:
#         obs = self._get_new_observation_buffer()
#         obs.set("reward", torch.tensor([0], device=self.device, dtype=torch.float32))
#         obs.set("done", torch.tensor([False], device=self.device, dtype=torch.bool))
#         return obs

#     def _prealloc_step_buffers(self, n: int) -> List[TensorDict]:
#         return [self._get_new_step_buffer() for _ in range(n)]

#     def _get_preallocated_step_buffer(
#         self, buffers: List[TensorDict], i: int
#     ) -> TensorDict:
#         if i >= len(buffers):
#             buffers.extend(self._prealloc_step_buffers(2 * len(buffers)))
#         return buffers[i]

#     def _get_current_buffer(self):
#         buf = self._get_preallocated_step_buffer(self.workspace, self.buffer_idx)
#         self.buffer_idx += 1

#     def _step(self, td: TensorDict) -> TensorDict:
#         chosen_device = td["action"].item()

#         if self.only_gpu:
#             chosen_device = chosen_device + 1
#         done = torch.tensor((1,), device=self.device, dtype=torch.bool)
#         reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
#         local_id = 0

#         candidate_workspace = torch.zeros(
#             self.simulator_factory.graph_spec.max_candidates,
#             dtype=torch.int64,
#         )

#         self.simulator.get_mappable_candidates(candidate_workspace)
#         global_task_id = candidate_workspace[local_id].item()
#         mapping_priority = self.simulator.get_mapping_priority(global_task_id)
#         reserving_priority = mapping_priority
#         launching_priority = mapping_priority
#         # print("Chosen device:", chosen_device)
#         actions = [
#             fastsim.Action(
#                 local_id, chosen_device, reserving_priority, launching_priority
#             )
#         ]
#         for i in self.action_candidates:
#             if i == chosen_device:
#                 continue
#             simulator_copy = self.simulator.copy()
#             simulator_copy.simulator.map_tasks(
#                 [fastsim.Action(local_id, i, reserving_priority, launching_priority)]
#             )
#             simulator_copy.disable_external_mapper()
#             simulator_copy.run()
#             if simulator_copy.time // 1000 < self.min_time:
#                 self.min_time = simulator_copy.time // 1000

#         self.simulator.simulator.map_tasks(actions)
#         simulator_copy = self.simulator.copy()
#         simulator_copy.disable_external_mapper()
#         simulator_copy.run()
#         time = simulator_copy.time // 1000
#         if time > self.min_time:
#             if self.prev_time >= time:
#                 reward[0] = 0.1
#             else:
#                 reward[0] = -1
#         elif time <= self.min_time:
#             reward[0] = 1
#             self.min_time = time
#         self.prev_time = time

#         simulator_status = self.simulator.run_until_external_mapping()
#         done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

#         obs = self._get_observation()
#         if done:
#             time = obs["observation"]["aux"]["time"].item()
#             obs["observation"]["aux"]["improvement"][0] = self.EFT_baseline / time - 1
#             print(
#                 f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {obs['observation']['aux']['improvement'][0]:.2f}"
#             )

#         out = obs
#         out.set("reward", reward)
#         out.set("done", done)
#         return out

#     def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
#         self.resets += 1
#         current_priority_seed = self.simulator_factory.pseed
#         current_duration_seed = self.simulator_factory.seed
#         self.s = 0

#         if self.change_locations:
#             new_location_seed = self.location_seed + self.resets
#             # Load initial location list
#             graph = self.simulator_factory.input.graph
#             graph.set_cell_locations(self.initial_location_list)

#             random.seed(new_location_seed)
#             graph.randomize_locations(
#                 self.location_randomness, self.location_list, verbose=False
#             )

#         if self.change_priority:
#             new_priority_seed = current_priority_seed + self.resets
#         else:
#             new_priority_seed = current_priority_seed

#         if self.change_duration:
#             new_duration_seed = current_duration_seed + self.resets
#         else:
#             new_duration_seed = current_duration_seed

#         new_priority_seed = int(new_priority_seed)
#         new_duration_seed = int(new_duration_seed)
#         self.taskid_history = []
#         if self.change_locations and isinstance(
#             self.simulator_factory.input.graph, JacobiGraph
#         ):
#             self.simulator_factory.input.graph.randomize_locations(
#                 1, location_list=range(self.simulator_factory.graph_spec.max_devices)
#             )
#         self.simulator = self.simulator_factory.create(
#             priority_seed=new_priority_seed, duration_seed=new_duration_seed
#         )
#         self.EFT_baseline = self._get_baseline(use_eft=True)
#         self.min_time = 999999999
#         self.prev_time = self.EFT_baseline // 1000
#         simulator_status = self.simulator.run_until_external_mapping()
#         assert (
#             simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING
#         ), f"Unexpected simulator status: {simulator_status}"

#         obs = self._get_observation()
#         return obs

#     @property
#     def observer(self):
#         return self.simulator.observer

#     def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         random.seed(seed)
#         if self.change_priority:
#             self.simulator_factory.set_seed(priority_seed=seed)
#         if self.change_duration:
#             self.simulator_factory.set_seed(duration_seed=seed)


# class LookaheadFractionalEnv(EnvBase):
#     def __init__(
#         self,
#         simulator_factory: SimulatorFactory,
#         seed: int = 0,
#         device="cpu",
#         baseline_time=4000 * 5,
#         change_priority=False,
#         change_duration=False,
#         change_locations=False,
#         only_gpu=True,
#         location_seed=0,
#         priority_seed=0,
#         location_randomness=1,
#         location_list=[1, 2, 3, 4],
#     ):
#         super().__init__(device=device)

#         self.change_priority = change_priority
#         self.change_duration = change_duration
#         self.change_locations = change_locations
#         self.location_seed = location_seed
#         self.location_randomness = location_randomness
#         self.location_list = location_list
#         self.only_gpu = only_gpu
#         self.s = 0

#         self.simulator_factory = simulator_factory
#         self.simulator: SimulatorDriver = simulator_factory.create(
#             seed, priority_seed=priority_seed
#         )

#         self.buffer_idx = 0
#         self.resets = 0
#         self.action_candidates = range(
#             int(only_gpu), self.simulator_factory.graph_spec.max_devices
#         )

#         if self.change_locations:
#             graph = simulator_factory.input.graph
#             assert hasattr(graph, "get_cell_locations")
#             assert hasattr(graph, "set_cell_locations")
#             assert hasattr(graph, "randomize_locations")
#             self.initial_location_list = graph.get_cell_locations()
#             self.location_randomness = location_randomness
#             self.location_list = location_list
#             random.seed(self.location_seed)

#         self.observation_spec = self._create_observation_spec()
#         self.action_spec = self._create_action_spec()
#         self.reward_spec = self._create_reward_spec()
#         self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

#         self.workspace = self._prealloc_step_buffers(100)
#         self.baseline_time = baseline_time

#         if change_locations:
#             graph.randomize_locations(
#                 self.location_randomness, self.location_list, verbose=False
#             )

#     def _get_baseline(self, use_eft=False):
#         if use_eft:
#             simulator_copy = self.simulator.fresh_copy()
#             simulator_copy.initialize()
#             simulator_copy.initialize_data()
#             simulator_copy.disable_external_mapper()
#             final_state = simulator_copy.run()
#             assert (
#                 final_state == fastsim.ExecutionState.COMPLETE
#             ), f"Baseline returned unexpected final state: {final_state}"
#             return simulator_copy.time
#         return self.baseline_time

#     def _create_observation_spec(self) -> TensorSpec:
#         obs = self.simulator.observer.get_observation()
#         comp = make_composite_from_td(obs)
#         comp = Composite(observation=comp)
#         return comp

#     def _create_action_spec(self, ndevices: int = 5) -> TensorSpec:
#         n_devices = self.simulator_factory.graph_spec.max_devices
#         if self.only_gpu:
#             n_devices -= 1
#         out = Bounded(
#             shape=(1,),
#             device=self.device,
#             dtype=torch.int64,
#             low=torch.tensor(0, device=self.device),
#             high=torch.tensor(n_devices, device=self.device),
#         )
#         out = Composite(action=out)
#         return out

#     def _create_reward_spec(self) -> TensorSpec:
#         return Unbounded(shape=(1,), device=self.device, dtype=torch.float32)

#     def _get_observation(self, td: TensorDict = None) -> TensorDict:
#         if td is None:
#             obs = self.simulator.observer.get_observation()
#             td = TensorDict(observation=obs)
#         else:
#             self.simulator.observer.get_observation(td["observation"])
#         return td

#     def _get_new_observation_buffer(self) -> TensorDict:
#         obs = self.simulator.observer.new_observation_buffer()
#         td = TensorDict(observation=obs)
#         return td

#     def _get_new_step_buffer(self) -> TensorDict:
#         obs = self._get_new_observation_buffer()
#         obs.set("reward", torch.tensor([0], device=self.device, dtype=torch.float32))
#         obs.set("done", torch.tensor([False], device=self.device, dtype=torch.bool))
#         return obs

#     def _prealloc_step_buffers(self, n: int) -> List[TensorDict]:
#         return [self._get_new_step_buffer() for _ in range(n)]

#     def _get_preallocated_step_buffer(
#         self, buffers: List[TensorDict], i: int
#     ) -> TensorDict:
#         if i >= len(buffers):
#             buffers.extend(self._prealloc_step_buffers(2 * len(buffers)))
#         return buffers[i]

#     def _get_current_buffer(self):
#         buf = self._get_preallocated_step_buffer(self.workspace, self.buffer_idx)
#         self.buffer_idx += 1

#     def _step(self, td: TensorDict) -> TensorDict:
#         chosen_device = td["action"].item()

#         if self.only_gpu:
#             chosen_device = chosen_device + 1
#         done = torch.tensor((1,), device=self.device, dtype=torch.bool)
#         reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
#         local_id = 0

#         candidate_workspace = torch.zeros(
#             self.simulator_factory.graph_spec.max_candidates,
#             dtype=torch.int64,
#         )

#         self.simulator.get_mappable_candidates(candidate_workspace)
#         global_task_id = candidate_workspace[local_id].item()
#         mapping_priority = self.simulator.get_mapping_priority(global_task_id)
#         reserving_priority = mapping_priority
#         launching_priority = mapping_priority
#         # print("Chosen device:", chosen_device)
#         actions = [
#             fastsim.Action(
#                 local_id, chosen_device, reserving_priority, launching_priority
#             )
#         ]
#         min_time = 999999999
#         for i in self.action_candidates:
#             if i == chosen_device:
#                 continue
#             simulator_copy = self.simulator.copy()
#             simulator_copy.simulator.map_tasks(
#                 [fastsim.Action(local_id, i, reserving_priority, launching_priority)]
#             )
#             simulator_copy.disable_external_mapper()
#             simulator_copy.run()
#             if simulator_copy.time // 1000 < min_time:
#                 min_time = simulator_copy.time // 1000

#         self.simulator.simulator.map_tasks(actions)
#         simulator_copy = self.simulator.copy()
#         simulator_copy.disable_external_mapper()
#         simulator_copy.run()
#         if simulator_copy.time // 1000 > min_time:
#             reward[0] = -1
#         elif simulator_copy.time // 1000 < min_time:
#             reward[0] = 1
#         else:
#             reward[0] = 0

#         simulator_status = self.simulator.run_until_external_mapping()
#         done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

#         obs = self._get_observation()
#         time = obs["observation"]["aux"]["time"].item()
#         if done:
#             obs["observation"]["aux"]["improvement"][0] = self.EFT_baseline / time - 1
#             print(
#                 f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {obs['observation']['aux']['improvement'][0]:.2f}"
#             )

#         out = obs
#         out.set("reward", reward)
#         out.set("done", done)
#         return out

#     def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
#         self.resets += 1

#         current_priority_seed = self.simulator_factory.pseed
#         current_duration_seed = self.simulator_factory.seed
#         self.s = 0

#         if self.change_locations:
#             new_location_seed = self.location_seed + self.resets
#             # Load initial location list
#             graph = self.simulator_factory.input.graph
#             graph.set_cell_locations(self.initial_location_list)

#             random.seed(new_location_seed)
#             graph.randomize_locations(
#                 self.location_randomness, self.location_list, verbose=False
#             )

#         if self.change_priority:
#             new_priority_seed = current_priority_seed + self.resets
#         else:
#             new_priority_seed = current_priority_seed

#         if self.change_duration:
#             new_duration_seed = current_duration_seed + self.resets
#         else:
#             new_duration_seed = current_duration_seed

#         new_priority_seed = int(new_priority_seed)
#         new_duration_seed = int(new_duration_seed)
#         self.taskid_history = []
#         if self.change_locations and isinstance(
#             self.simulator_factory.input.graph, JacobiGraph
#         ):
#             self.simulator_factory.input.graph.randomize_locations(
#                 1, location_list=range(self.simulator_factory.graph_spec.max_devices)
#             )
#         self.simulator = self.simulator_factory.create(
#             priority_seed=new_priority_seed, duration_seed=new_duration_seed
#         )
#         self.EFT_baseline = self._get_baseline(use_eft=True)

#         simulator_status = self.simulator.run_until_external_mapping()
#         assert (
#             simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING
#         ), f"Unexpected simulator status: {simulator_status}"

#         obs = self._get_observation()
#         return obs

#     @property
#     def observer(self):
#         return self.simulator.observer

#     def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         random.seed(seed)
#         if self.change_priority:
#             self.simulator_factory.set_seed(priority_seed=seed)
#         if self.change_duration:
#             self.simulator_factory.set_seed(duration_seed=seed)
