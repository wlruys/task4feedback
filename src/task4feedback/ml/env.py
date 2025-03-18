from task4feedback import fastsim2 as fastsim
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
from aim.pytorch import track_gradients_dists, track_params_dists
from task4feedback.graphs.base import Graph, DataBlocks, ComputeDataGraph, DataGeometry
import random


class RuntimeEnv(EnvBase):
    def __init__(
        self,
        simulator_factory,
        seed: int = 0,
        device="cpu",
        baseline_time=56000,
        change_priority=False,
        change_duration=False,
    ):
        super().__init__(device=device)

        self.change_priority = change_priority
        self.change_duration = change_duration

        self.simulator_factory = simulator_factory
        self.simulator = simulator_factory.create(seed)
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
        local_id = 0
        device = chosen_device
        # print("Chosen device: ", chosen_device)
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
            fastsim.Action(local_id, device, reserving_priority, launching_priority)
        ]
        self.simulator.simulator.map_tasks(actions)
        simulator_status = self.simulator.run_until_external_mapping()
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)

        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE
        reward[0] = 0

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()

        if not done:
            assert (
                simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING
            ), f"Unexpected simulator status: {simulator_status}"
        else:
            # obs = self._reset()
            baseline_time = self._get_baseline()
            reward[0] = 1 + (baseline_time - time) / baseline_time
            # reward[0] = time
            print(
                f"Reward: {reward[0].item()}, Time: {time}, Baseline: {baseline_time}"
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
        # print("New priority seed: ", new_priority_seed)
        if self.change_duration:
            new_duration_seed = current_duration_seed + self.resets
        else:
            new_duration_seed = current_duration_seed

        new_priority_seed = int(new_priority_seed)
        new_duration_seed = int(new_duration_seed)

        self.simulator = self.simulator_factory.create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )

        simulator_status = self.simulator.run_until_external_mapping()
        assert (
            simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING
        ), f"Unexpected simulator status: {simulator_status}"

        obs = self._get_observation()
        # obs.set("time", obs["observation"]["aux"]["time"])
        # print("Reset: ", obs["observation"]["aux"]["candidates"]["idx"])
        # print("Reset: ", obs["observation"]["nodes"]["tasks"]["glb"])
        return obs

    @property
    def observer(self):
        return self.simulator.observer

    def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
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


class RandomLocationMapperRuntimeEnv(MapperRuntimeEnv):
    def __init__(
        self,
        simulator_factory,
        seed: int = 0,
        device="cpu",
        baseline_time=56000,
        use_external_mapper: bool = False,
        change_priority=True,
        change_duration=False,
        change_locations=False,
        location_seed=0,
        location_randomness=1,
        location_list=[1, 2, 3, 4],
    ):
        super().__init__(
            simulator_factory=simulator_factory,
            seed=seed,
            device=device,
            baseline_time=baseline_time,
            use_external_mapper=use_external_mapper,
            change_priority=change_priority,
            change_duration=change_duration,
        )
        self.change_locations = change_locations
        self.location_seed = 0

        graph = simulator_factory.input.graph
        assert hasattr(graph, "get_cell_locations")
        assert hasattr(graph, "set_cell_locations")
        assert hasattr(graph, "randomize_locations")
        self.initial_location_list = graph.get_cell_locations()
        self.location_randomness = location_randomness
        self.location_list = location_list

        random.seed(self.location_seed)

        if change_locations:
            graph.randomize_locations(
                self.location_randomness, self.location_list, verbose=True
            )

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
                self.location_randomness, self.location_list, verbose=True
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

        self.simulator = self.simulator_factory.create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )

        simulator_status = self.simulator.run_until_external_mapping()
        assert simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING, (
            f"Unexpected simulator status: {simulator_status}"
        )

        obs = self._get_observation()
        return obs

    def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
        s = super()._set_seed(seed, static_seed)
        # if s is not None:
        #     self.location_seed = s
        return s
