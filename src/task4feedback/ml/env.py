import task4feedback.fastsim2 as fastsim
from task4feedback.interface import *
import torch
from typing import Optional, List
import numpy as np

from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import (
    # DefaultObserverFactory,
    SimulatorFactory,
    create_graph_spec,
    observation_to_heterodata,
)
from task4feedback.fastsim2 import GraphExtractor, SchedulerState, EventType
from torchrl.data import Composite, TensorSpec, Unbounded, Binary, Bounded
from torchrl.envs.utils import make_composite_from_td
from torchrl.envs import StepCounter, TrajCounter, TransformedEnv
from tensordict import TensorDict
from task4feedback.graphs.base import Graph, DataBlocks, ComputeDataGraph
import random
from task4feedback.graphs.mesh.plot import *
import numpy as np
from task4feedback.legacy_graphs import *
from task4feedback.graphs.jacobi import JacobiGraph
from torch_geometric.data import HeteroData
from torch.profiler import record_function

MAX_BUFFERS = 2000


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
        location_list: Optional[List[int]] = None,
        answer: Optional[List[int]] = None,
        randomize_interval: int = 1,
    ):
        super().__init__(device=device)
        # print("Initializing environment")

        self.change_priority = change_priority
        self.change_duration = change_duration
        self.change_locations = change_locations
        self.location_seed = location_seed
        self.location_randomness = location_randomness
        self.randomize_interval = randomize_interval
        if location_list is None:
            location_list = range(
                int(only_gpu), simulator_factory.graph_spec.max_devices
            )
        self.location_list = location_list
        self.answer = answer
        self.only_gpu = only_gpu
        self.graph: JacobiGraph = simulator_factory.input.graph

        self.simulator_factory = simulator_factory
        self.simulator: SimulatorDriver = simulator_factory.create(
            seed, priority_seed=priority_seed
        )

        self.buffer_idx = 0
        self.resets = 0

        if self.change_locations:
            graph = simulator_factory.input.graph
            if self.only_gpu and (0 in self.location_list):
                print(
                    "Warning: CPU is in the location list. Although only_gpu is set to True, the CPU will be assigned data."
                )
            if (
                hasattr(graph, "get_cell_locations")
                and hasattr(graph, "set_cell_locations")
                and hasattr(graph, "randomize_locations")
            ):
                self.legacy_graph = False
            else:
                self.legacy_graph = True
                print(
                    "Warning: Randomizing locations on a legacy graph. This may not work as expected. location_randomness is ignored."
                )

        # print("Creating environment spec")
        self.observation_spec = self._create_observation_spec()
        # print("Observation spec created")
        self.action_spec = self._create_action_spec()
        self.reward_spec = self._create_reward_spec()
        self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

        self.workspace = self._prealloc_step_buffers(1000)
        self.baseline_time = baseline_time

        # if change_locations:
        #     graph.randomize_locations(
        #         self.location_randomness, self.location_list, verbose=False
        #     )

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
        # obs = Bounded(
        #     shape=(1,),
        #     device=self.device,
        #     dtype=torch.float32,
        #     low=torch.tensor(0, device=self.device),
        #     high=torch.tensor(1, device=self.device),
        # )
        # comp = obs
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
        td["observation", "aux", "progress"] = self.step_count / 80
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
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )
        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )

        simulator_copy = self.simulator.copy()
        simulator_copy.disable_external_mapper()
        simulator_copy.run()
        if simulator_copy.time > self.prev_makespan:
            reward[0] = -1
        elif simulator_copy.time < self.prev_makespan:
            reward[0] = 1
        else:
            reward[0] = 0
        self.prev_makespan = simulator_copy.time

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
        self.step_count += 1
        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        # print("Resetting environment")
        self.resets += 1
        self.step_count = 0
        current_priority_seed = self.simulator_factory.pseed
        current_duration_seed = self.simulator_factory.seed

        if self.change_locations and (
            (self.resets // 2) % self.randomize_interval == 0
        ):
            new_location_seed = self.location_seed + self.resets
            graph = self.simulator_factory.input.graph
            random.seed(new_location_seed)
            if self.legacy_graph:
                data = self.simulator_factory.input.data.data
                for i in range(data.size()):
                    data.set_location(i, random.choice(self.location_list))
            else:
                samples = [item for item in self.location_list for _ in range(4)]
                random.shuffle(samples)
                graph.set_cell_locations(samples, step=0)
                graph.set_cell_locations([-1 for i in samples], step=1)

        if self.change_priority and (self.resets % self.randomize_interval == 0):
            new_priority_seed = int(current_priority_seed + self.resets)
        else:
            new_priority_seed = int(current_priority_seed)

        if self.change_duration and (self.resets % self.randomize_interval == 0):
            new_duration_seed = int(current_duration_seed + self.resets)
        else:
            new_duration_seed = int(current_duration_seed)

        self.simulator = self.simulator_factory.create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )

        simulator_status = self.simulator.run_until_external_mapping()
        assert (
            simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING
        ), f"Unexpected simulator status: {simulator_status}"
        self.observer.reset()
        obs = self._get_observation()
        print(obs)
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
        if self.change_locations:
            self.location_seed = seed


class RunningAvgEnv(RuntimeEnv):
    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.eftsim = self.simulator.fresh_copy()
            self.eftsim.initialize()
            self.eftsim.initialize_data()
            self.eftsim.disable_external_mapper()
            final_state = self.eftsim.run()
            assert (
                final_state == fastsim.ExecutionState.COMPLETE
            ), f"Baseline returned unexpected final state: {final_state}"
            self.EFT_baseline = self.eftsim.time
            self.eft_history = []
            self.policy_history = []
            self.avg_eft = []
            self.avg_policy = []

        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )

        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )

        sim_ml = self.simulator.copy()
        sim_ml.disable_external_mapper()
        sim_ml.set_task_breakpoint(EventType.COMPLETER, global_task_id)
        sim_ml.run()

        self.eft_history.append(self.eftsim.task_finish_time(global_task_id))
        self.policy_history.append(sim_ml.task_finish_time(global_task_id))

        self.avg_eft.append(average(self.eft_history[-16:]))
        self.avg_policy.append(average(self.policy_history[-16:]))

        simulator_status = self.simulator.run_until_external_mapping()
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

        if self.step_count == 0:
            reward[0] = 0
        else:
            reward[0] = (self.avg_policy[-2] - self.avg_policy[-1]) / self.EFT_baseline

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()
        if done:
            obs["observation"]["aux"]["improvement"][0] = self.EFT_baseline / time
            print(
                f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {obs['observation']['aux']['improvement'][0]:.2f}"
            )

        out = obs

        if self.graph.task_to_level[global_task_id] < (self.graph.config.steps - 1):
            for next_id in self.graph.level_to_task[
                self.graph.task_to_level[global_task_id]
            ]:
                if (
                    self.graph.task_to_cell[global_task_id]
                    == self.graph.task_to_cell[next_id]
                ):
                    x, y = self.graph.xy_from_id(global_task_id)
                    self.observer.task_ids[x * self.graph.config.n + y] = next_id

        out.set("reward", reward)
        out.set("done", done)
        self.step_count += 1
        return out


class EFTIncrementalEnv(RuntimeEnv):
    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
            self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )

        sim_eft = self.simulator.copy()
        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )

        sim_ml = self.simulator.copy()
        sim_eft.disable_external_mapper()
        sim_ml.disable_external_mapper()
        sim_eft.run()
        sim_ml.run()
        eft_time = sim_eft.time
        ml_time = sim_ml.time
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
        self.step_count += 1
        return out


class TerminalEnv(RuntimeEnv):
    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )

        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )

        reward[0] = 0
        simulator_status = self.simulator.run_until_external_mapping()
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()
        if done:
            obs["observation"]["aux"]["improvement"][0] = self.EFT_baseline / time - 1
            reward[0] = obs["observation"]["aux"]["improvement"][0]
            print(
                f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {obs['observation']['aux']['improvement'][0]:.2f}"
            )

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        self.step_count += 1
        return out


class kHopEFTIncrementalEnv(RuntimeEnv):
    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
            self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )
        dependents = torch.zeros(16, dtype=torch.int64)

        sim_eft = self.simulator.copy()
        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )

        sim_ml = self.simulator.copy()
        sim_eft.disable_external_mapper()
        sim_ml.disable_external_mapper()
        dep_count = self.graph_extractor.get_k_hop_dependents(
            candidate_workspace, 2, dependents
        )
        for i in range(dep_count):
            sim_eft.set_task_breakpoint(fastsim.EventType.COMPLETER, dependents[i])
            sim_ml.set_task_breakpoint(fastsim.EventType.COMPLETER, dependents[i])
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
        self.step_count += 1
        return out


class EFTAllPossibleEnv(RuntimeEnv):
    """
    For each action, explore all the other actions using EFT and +1 if it was the best, 0 if it was the same, -1 if it was worse.
    """

    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
            self.action_candidates = range(
                int(self.only_gpu), self.simulator_factory.graph_spec.max_devices
            )
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )

        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        min_time = 999990000
        for i in self.action_candidates:
            if i == chosen_device:
                continue
            simulator_copy = self.simulator.copy()
            simulator_copy.simulator.map_tasks(
                [fastsim.Action(0, i, mapping_priority, mapping_priority)]
            )
            simulator_copy.disable_external_mapper()
            simulator_copy.run()
            if simulator_copy.time < min_time:
                min_time = simulator_copy.time

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )
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

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        self.step_count += 1
        return out


class RolloutEnv(RuntimeEnv):
    def set_policy(self, policy):
        self.policy = policy

    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
            self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )

        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )

        simulator_copy = self.simulator.copy()
        simulator_copy.disable_external_mapper()
        simulator_copy.run()
        eft_time = simulator_copy.time
        with torch.no_grad():
            simulator_copy = self.simulator.copy()
            state_copy = simulator_copy.run_until_external_mapping()
            copy_workspace = torch.zeros(
                self.simulator_factory.graph_spec.max_candidates,
                dtype=torch.int64,
            )
            copy_obs = TensorDict(
                observation=simulator_copy.observer.new_observation_buffer(
                    simulator_copy.observer.graph_spec
                )
            )
            while state_copy != fastsim.ExecutionState.COMPLETE:
                simulator_copy.observer.get_observation(copy_obs["observation"])
                action_logits = self.policy(copy_obs)["logits"]
                copy_action = torch.argmax(action_logits, dim=-1).item()
                if self.only_gpu:
                    copy_action = copy_action + 1
                simulator_copy.get_mappable_candidates(copy_workspace)
                copy_task_id = copy_workspace[0].item()
                copy_priority = simulator_copy.get_mapping_priority(copy_task_id)
                copy_actions = [
                    fastsim.Action(
                        0,
                        copy_action,
                        copy_priority,
                        copy_priority,
                    )
                ]
                simulator_copy.simulator.map_tasks(copy_actions)
                state_copy = simulator_copy.run_until_external_mapping()
        if simulator_copy.time - eft_time >= 1000:
            reward[0] = -1
        elif simulator_copy.time - eft_time <= -1000:
            reward[0] = 1
        else:
            reward[0] = 0
        eft_time = simulator_copy.time
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
        self.step_count += 1
        return out


class kHopRolloutEnv(RuntimeEnv):
    def set_policy(self, policy):
        self.policy = policy

    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
            self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )
        dependents = torch.zeros(50, dtype=torch.int64)

        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )

        dep_count = self.graph_extractor.get_k_hop_dependents(
            candidate_workspace, 2, dependents
        )
        simulator_copy = self.simulator.copy()
        for i in range(dep_count):
            simulator_copy.set_task_breakpoint(
                fastsim.EventType.LAUNCHER, dependents[i]
            )
        simulator_copy.disable_external_mapper()
        for i in range(dep_count):
            temp = simulator_copy.run()
            if temp == fastsim.ExecutionState.COMPLETE:
                break
        eft_time = simulator_copy.time
        with torch.no_grad():
            simulator_copy = self.simulator.copy()
            for i in range(dep_count):
                simulator_copy.set_task_breakpoint(
                    fastsim.EventType.LAUNCHER, dependents[i]
                )
            state_copy = simulator_copy.run_until_external_mapping()
            copy_workspace = torch.zeros(
                self.simulator_factory.graph_spec.max_candidates,
                dtype=torch.int64,
            )
            finished = 0
            copy_obs = TensorDict(observation=simulator_copy.observer.get_observation())
            while state_copy != fastsim.ExecutionState.COMPLETE:
                # print(state_copy)
                if state_copy == fastsim.ExecutionState.BREAKPOINT:
                    finished += 1
                    if finished >= dep_count:
                        break
                    state_copy = simulator_copy.run_until_external_mapping()
                    continue
                elif state_copy == fastsim.ExecutionState.EXTERNAL_MAPPING:
                    simulator_copy.observer.get_observation(copy_obs["observation"])
                    action_logits = self.policy(copy_obs)["logits"]
                    copy_action = torch.argmax(action_logits, dim=-1).item()
                    if self.only_gpu:
                        copy_action = copy_action + 1
                    simulator_copy.get_mappable_candidates(copy_workspace)
                    copy_task_id = copy_workspace[0].item()
                    copy_priority = simulator_copy.get_mapping_priority(copy_task_id)
                    copy_actions = [
                        fastsim.Action(
                            0,
                            copy_action,
                            copy_priority,
                            copy_priority,
                        )
                    ]
                    simulator_copy.simulator.map_tasks(copy_actions)
                    state_copy = simulator_copy.run_until_external_mapping()
                else:
                    print(f"Unexpected simulator status: {state_copy}")
                    assert False, f"Unexpected simulator status: {state_copy}"

        if simulator_copy.time - eft_time >= 1000:
            reward[0] = -1
        elif simulator_copy.time - eft_time <= -1000:
            reward[0] = 1
        else:
            reward[0] = 0

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
        self.step_count += 1
        return out


class GeneralizedIncrementalEFT(RuntimeEnv):
    def __init__(
        self,
        *args,
        gamma=0.4,
        flip=True,
        clip_total=False,
        clip_individual=False,
        binary=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.flip = flip
        n_tasks = len(self.simulator_factory.input.graph)
        self.eft_log = torch.zeros(n_tasks + 1, dtype=torch.int64)
        self.max_steps = n_tasks
        self.clip_total = clip_total
        self.clip_individual = clip_individual
        self.binary = binary

    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
            self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
            self.eft_log[self.step_count] = self.EFT_baseline

        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        candidate_workspace = torch.zeros(
            self.simulator_factory.graph_spec.max_candidates,
            dtype=torch.int64,
        )

        self.simulator.get_mappable_candidates(candidate_workspace)
        chosen_device = td["action"].item() + int(self.only_gpu)
        global_task_id = candidate_workspace[0].item()
        mapping_priority = self.simulator.get_mapping_priority(global_task_id)

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )

        sim_ml = self.simulator.copy()
        sim_ml.disable_external_mapper()
        sim_ml.run()
        ml_time = sim_ml.time
        self.eft_log[self.step_count + 1] = ml_time
        reward_sum = 0
        for i in range(0, self.step_count, 1):
            current = self.eft_log[i] - ml_time

            if self.binary:
                current = 1 if current > 0 else -1

            if self.clip_individual:
                current = max(current, 0)

            if not self.flip:
                discount = self.gamma * (1 - self.gamma) ** (
                    self.max_steps - 1 - (self.step_count - i)
                )
            else:
                discount = self.gamma * (1 - self.gamma) ** (self.step_count - 1 - i)

            reward_sum += current * discount

        if self.clip_total:
            reward_sum = min(reward_sum, 0)

        reward[0] = reward_sum / self.EFT_baseline

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
        # print("Reward: ", reward)
        out.set("reward", reward)
        out.set("done", done)
        self.step_count += 1
        return out


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
            # print(f"External mapper action: {action}")
        else:
            internal_mapper = self.simulator.internal_mapper
            action = internal_mapper.map_task(
                global_task_id,
                scheduler_state,
            )

        # new_action = torch.tensor([action.device - 1], dtype=torch.int64)
        td.set_("action", action.device - 1)
        # print(f"Action: {action.device - 1}")
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
        only_gpu=True,
        location_list=[1, 2, 3, 4],
        path=".",
    ):
        super().__init__(device=device)

        self.change_priority = change_priority
        self.change_duration = change_duration
        self.change_locations = change_locations
        self.path = path
        self.only_gpu = only_gpu
        self.location_list = location_list

        self.simulator_factory = simulator_factory
        self.simulator: SimulatorDriver = simulator_factory.create(seed)
        self.graph_extractor: GraphExtractor = GraphExtractor(
            self.simulator.get_state()
        )

        self.buffer_idx = 0
        self.resets = 0
        self.cum_time = 0

        self.observation_spec = self._create_observation_spec()
        self.action_spec = self._create_action_spec()
        self.reward_spec = self._create_reward_spec()
        self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

        self.workspace = self._prealloc_step_buffers(2000)
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
        sim_eft.run()
        sim_ml.run()
        eft_time = sim_eft.time
        ml_time = sim_ml.time
        reward[0] = (eft_time - ml_time) / self.EFT_baseline
        simulator_status = self.simulator.run_until_external_mapping()
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE
        self.cum_time += eft_time - ml_time
        # print("Difference to Step:", self.EFT_baseline - ml_time)
        # print("Cumulative Time:", self.cum_time)

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()
        if done:
            self.cum_time = 0
            # Did we beat the baseline?
            if time <= self.EFT_baseline:
                reward[0] += 2

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
                1, location_list=self.location_list
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


class IncrementalMappingEnv(EnvBase):
    def __init__(
        self,
        simulator_factory: SimulatorFactory,
        seed: int = 0,
        device="cpu",
        baseline_time=56000,
        change_priority=False,
        change_duration=False,
        change_locations=False,
        only_gpu=True,
        location_list=[1, 2, 3, 4],
        path=".",
    ):
        super().__init__(device=device)

        self.change_priority = change_priority
        self.change_duration = change_duration
        self.change_locations = change_locations
        self.path = path
        self.only_gpu = only_gpu
        self.location_list = location_list

        self.simulator_factory = simulator_factory
        self.simulator: SimulatorDriver = simulator_factory.create(seed)
        self.graph_extractor: GraphExtractor = GraphExtractor(
            self.simulator.get_state()
        )

        self.buffer_idx = 0
        self.resets = 0
        self.last_time = 0

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
        self.last_time = self.simulator.time
        self.simulator.simulator.map_tasks(actions)

        simulator_status = self.simulator.run_until_external_mapping()
        incremental_time = self.simulator.time - self.last_time
        reward[0] = -1 * incremental_time
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()
        if done:
            obs["observation"]["aux"]["improvement"][0] = self.EFT_baseline / time - 1
            print(
                f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {obs['observation']['aux']['improvement'][0]:.2f}"
            )
            self.last_time = 0

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.resets += 1

        current_priority_seed = self.simulator_factory.pseed
        current_duration_seed = self.simulator_factory.seed

        self.last_time = 0

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
                1, location_list=self.location_list
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
