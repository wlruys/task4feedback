import task4feedback.fastsim2 as fastsim
from task4feedback.interface import *
import torch
from typing import Optional, List
import numpy as np

from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import (
    DefaultObserverFactory,
    SimulatorFactory,
    create_graph_spec,
    observation_to_heterodata,
)
from task4feedback.fastsim2 import GraphExtractor, SchedulerState
from torchrl.data import Composite, TensorSpec, Unbounded, Binary, Bounded
from torchrl.envs.utils import make_composite_from_td
from torchrl.envs import StepCounter, TrajCounter, TransformedEnv
from tensordict import TensorDict
from task4feedback.graphs.base import TaskGraph, DataBlocks, ComputeDataGraph
import random
from task4feedback.graphs.mesh.plot import *
from task4feedback.legacy_graphs import *
from task4feedback.graphs.jacobi import JacobiGraph
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from torch_geometric.data import HeteroData
from torchrl.data import Categorical
from task4feedback.logging import training
from time import perf_counter
import sys 


class RuntimeEnv(EnvBase):
    def __init__(
        self,
        simulator_factory: SimulatorFactory | list[SimulatorFactory],
        seed: int = 0,
        device="cpu",
        baseline_time=4000 * 5,
        change_priority=False,
        change_duration=False,
        change_location=False,
        only_gpu=True,
        location_seed=0,
        priority_seed=0,
        location_randomness=1,
        location_list: Optional[List[int]] = None,
        max_samples_per_iter: int = 0,
        random_start: bool = False,
        verbose: bool = False,
        colorized: bool = False,
    ):
        super().__init__(device=device)
        # print("Initializing environment")
        self.verbose = verbose
        self.max_samples_per_iter = max_samples_per_iter
        self.change_priority = change_priority
        self.change_duration = change_duration
        self.change_location = change_location
        self.location_seed = location_seed
        self.location_randomness = location_randomness
        self.random_start = random_start
        if location_list is None:
            location_list = [
                i for i in range(int(only_gpu), len(simulator_factory.input.system))
            ]
        self.location_list = location_list
        self.only_gpu = only_gpu
        self.n_devices = len(self.location_list)

        if not isinstance(simulator_factory, list):
            simulator_factory = [simulator_factory]

        self.simulator_factory = simulator_factory
        self.active_idx = 0 # Index of the active simulator factory in case of multiple factories

        self.simulator: SimulatorDriver = simulator_factory[self.active_idx].create(
            seed, priority_seed=priority_seed
        )

        self.buffer_idx = 0
        self.resets = 0
        self.EFT_baseline = 1

        if self.change_location:
            graph = simulator_factory[self.active_idx].input.graph
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

        self.observation = self._get_new_observation_buffer()
        observation_spec = self._create_observation_spec(self.observation)

        action_spec = self._create_action_spec(n_devices=len(self.location_list))
        reward_spec = self._create_reward_spec()
        done_spec = self._create_done_spec()

        self.action_spec = action_spec
        self.observation_spec = Composite(observation=observation_spec)
        self.reward_spec = Composite(reward=reward_spec)

        spec = Composite(
            observation=observation_spec,
            reward=reward_spec,
            done=done_spec,
        )

        self.observations = []
        for _ in range(max(1, max_samples_per_iter)):
            obs = observation_spec.zero()
            self.observations.append(obs)

        self._buf = spec.zeros()
        self.candidate_workspace = torch.zeros(
            self.simulator_factory[self.active_idx].graph_spec.max_candidates, dtype=torch.int64
        )
        self.baseline_time = baseline_time

        if change_location:
            graph.randomize_locations(
                self.location_randomness, self.location_list, verbose=False
            )

        self.batch_size = torch.Size([])

        self.progress_key = ("aux", "progress")
        self.baseline_key = ("aux", "baseline")
        self.improvement_key = ("aux", "improvement")
        self.time_key = ("aux", "time")
        self.action_n = "action"
        self.reward_n = "reward"
        self.done_n = "done"
        self.observation_n = "observation"

    def size(self):
        """
        Return maximum number of steps in the environment.
        This is the number of tasks in the graph.
        """
        return int(
            len(self.simulator_factory[self.active_idx].input.graph)
            // self.simulator_factory[self.active_idx].graph_spec.max_candidates
        )

    def __len__(self):
        """
        Return maximum number of steps in the environment.
        This is the number of tasks in the graph.

        Note: May not be available in a TransformedEnv.
        """
        return self.size()

    def _get_baseline(self, use_eft=False):
        if use_eft:
            # print("Calculating EFT baseline...")
            simulator_copy = self.simulator.fresh_copy()
            simulator_copy.initialize()
            simulator_copy.initialize_data()
            simulator_copy.disable_external_mapper()
            final_state = simulator_copy.run()
            assert (
                final_state == fastsim.ExecutionState.COMPLETE
            ), f"Baseline returned unexpected final state: {final_state}"
            # cprint("EFT baseline calculated.")
            return simulator_copy.time
        return self.baseline_time

    def _create_observation_spec(self, td) -> TensorSpec:
        comp = make_composite_from_td(td, unsqueeze_null_shapes=False)
        return comp

    def _create_state_value_spec(self) -> TensorSpec:
        return Unbounded(shape=[1], device=self.device, dtype=torch.float32)

    def _create_action_spec(self, n_devices: int = 5) -> TensorSpec:
        n_candidates = self.simulator_factory[self.active_idx].graph_spec.max_candidates
        #print(f"Creating action spec with {n_devices} devices and {n_candidates} candidates", flush=True)
        out = Categorical(
            n=n_devices,
            shape=[n_candidates],
            device=self.device,
            dtype=torch.int64,
        )
        return out

    def _create_reward_spec(self) -> TensorSpec:
        return Unbounded(shape=[1], device=self.device, dtype=torch.float32)

    def _create_done_spec(self) -> TensorSpec:
        return Binary(n=1, device=self.device, dtype=torch.bool)

    def get_observer(self):
        return self.simulator.observer

    def _get_observation(self) -> TensorDict:
        step_count = self.step_count
        n_buffers = len(self.observations)

        obs = self.observations[step_count % n_buffers]
        obs.zero_()

        self.simulator.observer.get_observation(obs)
        n_tasks = len(self.simulator_factory[self.active_idx].input.graph)
        progress = step_count / n_tasks
        baseline = max(1.0, self.EFT_baseline)
        obs.set_at_(self.progress_key, progress, 0)
        obs.set_at_(self.baseline_key, baseline, 0)
        return obs

    def _get_new_observation_buffer(self) -> TensorDict:
        obs = self.simulator.observer.new_observation_buffer()
        return obs

    def _handle_done(self, obs):
        time = obs[self.time_key].item()
        improvement = (self.EFT_baseline - time) / self.EFT_baseline
        obs.set_at_(self.improvement_key, improvement, 0)
        reward = improvement
        if self.verbose:
            print(
                f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {improvement:.2f}",
                flush=True,
            )

        return obs, reward, time, improvement

    def map_tasks(self, actions: torch.Tensor):
        candidate_workspace = self.candidate_workspace
        num_candidates = self.simulator.get_mappable_candidates(candidate_workspace)
        graph = self.simulator_factory[self.active_idx].input.graph
        if num_candidates > 1:
            mapping_result = []
            assert isinstance(
                graph, JacobiGraph
            ), "Graph must be a JacobiGraph for batched mapping."
            for i in range(num_candidates):
                global_task_id = candidate_workspace[i].item()

                idx = graph.xy_from_id(global_task_id)
                chosen_device = actions[idx].item() + int(self.only_gpu)

                mapping_priority = self.simulator.get_mapping_priority(global_task_id)
                mapping_result.append(
                    fastsim.Action(
                        i,
                        chosen_device,
                        mapping_priority,
                        mapping_priority,
                    )
                )
            self.simulator.simulator.map_tasks(mapping_result)
        else:
            chosen_device = actions.item() + int(self.only_gpu)
            global_task_id = candidate_workspace[0].item()
            mapping_priority = self.simulator.get_mapping_priority(global_task_id)
            self.simulator.simulator.map_tasks(
                [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
            )

    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)

        self.step_count += 1
        #print(f"Action", td[self.action_n], flush=True)
        self.map_tasks(td[self.action_n])

        reward = 0
        simulator_status = self.simulator.run_until_external_mapping()
        done = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()

        # print(global_task_id, obs[("nodes", "tasks", "attr")])

        if done:
            obs, reward, time, improvement = self._handle_done(obs)

        buf = td.empty()
        obs = obs if self.max_samples_per_iter > 0 else obs.clone()
        buf.set(self.observation_n, obs)
        buf.set(
            self.reward_n, torch.tensor(reward, device=self.device, dtype=torch.float32)
        )
        buf.set(self.done_n, torch.tensor(done, device=self.device, dtype=torch.bool))
        return buf

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        # start_t = perf_counter()
        self.resets += 1
        self.step_count = 0
        current_priority_seed = self.simulator_factory[self.active_idx].pseed
        current_duration_seed = self.simulator_factory[self.active_idx].seed

        if self.change_location:
            new_location_seed = self.location_seed + self.resets
            graph = self.simulator_factory[self.active_idx].input.graph
            random.seed(new_location_seed)
            if self.legacy_graph:
                data = self.simulator_factory[self.active_idx].input.data.data
                for i in range(data.size()):
                    data.set_location(i, random.choice(self.location_list))
            else:
                assert hasattr(
                    graph, "randomize_locations"
                ), "Graph does not have randomize_locations method."

                if isinstance(graph, JacobiGraph):
                    if isinstance(graph, DynamicJacobiGraph):
                        graph.set_cell_locations([-1 for _ in range(graph.config.n**2)])
                        graph.randomize_workload(seed=new_location_seed)
                    graph.randomize_locations(
                        self.location_randomness,
                        self.location_list,
                        verbose=False,
                        step=0 if graph.dynamic else None,
                    )
                else:
                    graph.randomize_locations(
                        self.location_randomness,
                        self.location_list,
                        verbose=False,
                    )

        if self.change_priority:
            new_priority_seed = int(current_priority_seed + self.resets)
        else:
            new_priority_seed = int(current_priority_seed)

        if self.change_duration:
            new_duration_seed = int(current_duration_seed + self.resets)
        else:
            new_duration_seed = int(current_duration_seed)

        #print("New seeds - Priority: {}, Duration: {}".format(new_priority_seed, new_duration_seed))

        self.simulator = self.simulator_factory[self.active_idx].create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )
        self.simulator.observer.reset()
        if self.resets < 10 and self.random_start:
            # Run the simulator for a random number of steps
            n_steps = random.randint(1, self.size() - 1)
            self.simulator.disable_external_mapper()
            self.simulator.set_steps(n_steps)
            self.simulator.run()
            self.simulator.enable_external_mapper()

        simulator_status = self.simulator.run_until_external_mapping()
        assert (
            simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING
        ), f"Unexpected simulator status: {simulator_status}"

        if td is None:
            td = TensorDict()
        else:
            td = td.empty()

        obs = self._get_observation()
        td.set(self.observation_n, obs)
        # end_t = perf_counter()
        # print("Reset took %.2f ms", (end_t - start_t) * 1000, flush=True)
        return td

    @property
    def observer(self):
        return self.simulator.observer

    def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.change_priority:
            self.simulator_factory[self.active_idx].set_seed(priority_seed=seed)
        if self.change_duration:
            self.simulator_factory[self.active_idx].set_seed(seed=seed)
        if self.change_location:
            self.location_seed = seed

    def reset_for_evaluation(self, seed: int = 0):
        # save seeds from curret state
        self.saved_seeds = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        old_pseed = self.simulator_factory[self.active_idx].pseed
        old_seed = self.simulator_factory[self.active_idx].seed

        self.change_priority = False # Do not change priority in evaluation
        self.change_duration = False # Do not change duration in evaluation
        self._set_seed(seed)
        self.random_start = False # Do not random start in evaluation
        self.resets = 0
        self._reset()

        # Restore seeds
        torch.set_rng_state(self.saved_seeds["torch"])
        np.random.set_state(self.saved_seeds["numpy"])
        random.setstate(self.saved_seeds["random"])


class IncrementalEFT(RuntimeEnv):

    def __init__(self, *args, gamma: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
            self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
            self.eft_time = self.EFT_baseline

        self.step_count += 1

        self.map_tasks(td[self.action_n])

        start_time = perf_counter()
        sim_ml = self.simulator.copy()
        sim_ml.disable_external_mapper()
        end_time = perf_counter()
        # print(f"sim_ml.copy() took {(end_time - start_time) * 1000:.2f}ms")
        # print(f"Current sim time {sim_ml.time}", flush=True)
        start_time = perf_counter()
        sim_ml.run()
        end_time = perf_counter()
        # print(f"sim_ml.run() took {(end_time - start_time) * 1000:.2f}ms")

        ml_time = sim_ml.time

        reward = 8 * (self.eft_time - self.gamma * ml_time) / self.size()
        self.eft_time = ml_time
        simulator_status = self.simulator.run_until_external_mapping()
        done = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        if done:
            obs, reward, time, improvement = self._handle_done(obs)

        buf = td.empty()
        buf.set(
            self.observation_n, obs if self.max_samples_per_iter > 0 else obs.clone()
        )
        buf.set(
            self.reward_n, torch.tensor(reward, device=self.device, dtype=torch.float32)
        )
        buf.set(self.done_n, torch.tensor(done, device=self.device, dtype=torch.bool))
        return buf


class DelayIncrementalEFT(IncrementalEFT):

    def __init__(
        self,
        *args,
        delay: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.offset = 1

    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
            self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
            self.eft_time = self.EFT_baseline

        flag = (self.step_count + self.offset) % self.delay

        self.step_count += 1

        self.map_tasks(td[self.action_n])

        if flag == 0:
            sim_ml = self.simulator.copy()
            sim_ml.disable_external_mapper()
            sim_ml.run()
            ml_time = sim_ml.time
            reward = (self.eft_time - self.gamma**self.delay * ml_time) / self.size()
            self.eft_time = ml_time
        else:
            reward = 0.0

        simulator_status = self.simulator.run_until_external_mapping()
        done = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        if done:
            obs, reward, time, improvement = self._handle_done(obs)

        buf = td.empty()
        buf.set(
            self.observation_n, obs if self.max_samples_per_iter > 0 else obs.clone()
        )
        buf.set(
            self.reward_n, torch.tensor(reward, device=self.device, dtype=torch.float32)
        )
        buf.set(self.done_n, torch.tensor(done, device=self.device, dtype=torch.bool))
        return buf

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.offset = random.randint(1, self.delay)
        return super()._reset(td)


class BaselineImprovementEFT(RuntimeEnv):
    def __init__(self, *args, delay=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay

    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.prev_makespan = self.EFT_baseline
            self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())

        flag = (self.step_count + 1) % self.delay

        self.step_count += 1

        self.map_tasks(td[self.action_n])

        if flag == 0:
            sim_ml = self.simulator.copy()
            sim_ml.disable_external_mapper()
            sim_ml.run()
            ml_time = sim_ml.time
            reward = (self.EFT_baseline - ml_time) / self.size()
        else:
            reward = 0.0

        simulator_status = self.simulator.run_until_external_mapping()
        done = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        if done:
            obs, reward, time, improvement = self._handle_done(obs)

        buf = td.empty()
        buf.set(
            self.observation_n, obs if self.max_samples_per_iter > 0 else obs.clone()
        )
        buf.set(
            self.reward_n, torch.tensor(reward, device=self.device, dtype=torch.float32)
        )
        buf.set(self.done_n, torch.tensor(done, device=self.device, dtype=torch.bool))
        return buf


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
        n_tasks = len(self.simulator_factory[self.active_idx].input.graph)
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

        self.map_tasks(td[self.action_n])

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

        reward = reward_sum / self.EFT_baseline

        simulator_status = self.simulator.run_until_external_mapping()
        done = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        time = obs["aux"]["time"].item()
        if done:
            obs, _, time, improvement = self._handle_done(obs)

        buf = td.empty()
        buf.set(
            self.observation_n, obs if self.max_samples_per_iter > 0 else obs.clone()
        )
        buf.set(
            self.reward_n, torch.tensor(reward, device=self.device, dtype=torch.float32)
        )
        buf.set(self.done_n, torch.tensor(done, device=self.device, dtype=torch.bool))
        return buf


class SanityCheckEnv(RuntimeEnv):
    def _step(self, td: TensorDict) -> TensorDict:
        if self.step_count == 0:
            self.EFT_baseline = self._get_baseline(use_eft=True)
            self.graph: JacobiGraph = self.simulator_factory[self.active_idx].input.graph
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

        self.simulator.simulator.map_tasks(
            [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
        )

        cell_id = self.graph.task_to_cell[global_task_id]

        print(f"Cell ID: {cell_id}, Chosen Device: {chosen_device}")
        print(f"Location List: {self.location_list}")

        answer = self.answer[cell_id]
        if answer == chosen_device:
            reward[0] = 1
        else:
            reward[0] = -1
        simulator_status = self.simulator.run_until_external_mapping()
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

        obs = self._get_observation()
        time = obs["aux"]["time"].item()
        if done:
            obs, reward, time, improvement = self._handle_done(obs)

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        self.step_count += 1
        return out


# class kHopEFTIncrementalEnv(RuntimeEnv):
#     def _step(self, td: TensorDict) -> TensorDict:
#         if self.step_count == 0:
#             self.EFT_baseline = self._get_baseline(use_eft=True)
#             self.prev_makespan = self.EFT_baseline
#             self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
#         done = torch.tensor((1,), device=self.device, dtype=torch.bool)
#         reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
#         candidate_workspace = torch.zeros(
#             self.simulator_factory.graph_spec.max_candidates,
#             dtype=torch.int64,
#         )
#         dependents = torch.zeros(16, dtype=torch.int64)

#         sim_eft = self.simulator.copy()
#         self.simulator.get_mappable_candidates(candidate_workspace)
#         chosen_device = td["action"].item() + int(self.only_gpu)
#         global_task_id = candidate_workspace[0].item()
#         mapping_priority = self.simulator.get_mapping_priority(global_task_id)

#         self.simulator.simulator.map_tasks(
#             [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
#         )

#         sim_ml = self.simulator.copy()
#         sim_eft.disable_external_mapper()
#         sim_ml.disable_external_mapper()
#         dep_count = self.graph_extractor.get_k_hop_dependents(
#             candidate_workspace, 2, dependents
#         )
#         for i in range(dep_count):
#             sim_eft.set_task_breakpoint(fastsim.EventType.COMPLETER, dependents[i])
#             sim_ml.set_task_breakpoint(fastsim.EventType.COMPLETER, dependents[i])
#         for i in range(dep_count):
#             sim_eft.run()
#             sim_ml.run()
#         eft_time = sim_eft.time - self.simulator.time
#         ml_time = sim_ml.time - self.simulator.time
#         reward[0] = (eft_time - ml_time) / self.EFT_baseline
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
#         self.step_count += 1
#         return out


# class EFTAllPossibleEnv(RuntimeEnv):
#     """
#     For each action, explore all the other actions using EFT and +1 if it was the best, 0 if it was the same, -1 if it was worse.
#     """

#     def _step(self, td: TensorDict) -> TensorDict:
#         if self.step_count == 0:
#             self.EFT_baseline = self._get_baseline(use_eft=True)
#             self.prev_makespan = self.EFT_baseline
#             self.action_candidates = range(
#                 int(self.only_gpu), self.simulator_factory.graph_spec.max_devices
#             )
#         done = torch.tensor((1,), device=self.device, dtype=torch.bool)
#         reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
#         candidate_workspace = torch.zeros(
#             self.simulator_factory.graph_spec.max_candidates,
#             dtype=torch.int64,
#         )

#         self.simulator.get_mappable_candidates(candidate_workspace)
#         chosen_device = td["action"].item() + int(self.only_gpu)
#         global_task_id = candidate_workspace[0].item()
#         mapping_priority = self.simulator.get_mapping_priority(global_task_id)

#         min_time = 999990000
#         for i in self.action_candidates:
#             if i == chosen_device:
#                 continue
#             simulator_copy = self.simulator.copy()
#             simulator_copy.simulator.map_tasks(
#                 [fastsim.Action(0, i, mapping_priority, mapping_priority)]
#             )
#             simulator_copy.disable_external_mapper()
#             simulator_copy.run()
#             if simulator_copy.time < min_time:
#                 min_time = simulator_copy.time

#         self.simulator.simulator.map_tasks(
#             [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
#         )
#         simulator_copy = self.simulator.copy()
#         simulator_copy.disable_external_mapper()
#         simulator_copy.run()
#         delta = ((simulator_copy.time - min_time) // 1000) * 1000
#         if delta > 0:
#             reward[0] = -1
#         elif delta < 0:
#             reward[0] = 1
#         else:
#             reward[0] = 0

#         simulator_status = self.simulator.run_until_external_mapping()
#         done[0] = simulator_status == fastsim.ExecutionState.COMPLETE

#         obs = self._get_observation()
#         time = obs["observation"]["aux"]["time"].item()
#         if done:
#             improvement = self.EFT_baseline / time - 1
#             obs["observation"]["aux"]["improvement"][0] = improvement
#             print(
#                 f"Time: {time} / Baseline: {self.EFT_baseline} Improvement: {obs['observation']['aux']['improvement'][0]:.2f}"
#             )

#         out = obs
#         out.set("reward", reward)
#         out.set("done", done)
#         self.step_count += 1
#         return out


# class RolloutEnv(RuntimeEnv):
#     def set_policy(self, policy):
#         self.policy = policy

#     def _step(self, td: TensorDict) -> TensorDict:
#         if self.step_count == 0:
#             self.EFT_baseline = self._get_baseline(use_eft=True)
#             self.prev_makespan = self.EFT_baseline
#             self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
#         done = torch.tensor((1,), device=self.device, dtype=torch.bool)
#         reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
#         candidate_workspace = torch.zeros(
#             self.simulator_factory.graph_spec.max_candidates,
#             dtype=torch.int64,
#         )

#         self.simulator.get_mappable_candidates(candidate_workspace)
#         chosen_device = td["action"].item() + int(self.only_gpu)
#         global_task_id = candidate_workspace[0].item()
#         mapping_priority = self.simulator.get_mapping_priority(global_task_id)

#         self.simulator.simulator.map_tasks(
#             [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
#         )

#         simulator_copy = self.simulator.copy()
#         simulator_copy.disable_external_mapper()
#         simulator_copy.run()
#         eft_time = simulator_copy.time
#         with torch.no_grad():
#             simulator_copy = self.simulator.copy()
#             state_copy = simulator_copy.run_until_external_mapping()
#             copy_workspace = torch.zeros(
#                 self.simulator_factory.graph_spec.max_candidates,
#                 dtype=torch.int64,
#             )
#             copy_obs = TensorDict(
#                 observation=simulator_copy.observer.new_observation_buffer(
#                     simulator_copy.observer.graph_spec
#                 )
#             )
#             while state_copy != fastsim.ExecutionState.COMPLETE:
#                 simulator_copy.observer.get_observation(copy_obs["observation"])
#                 action_logits = self.policy(copy_obs)["logits"]
#                 copy_action = torch.argmax(action_logits, dim=-1).item()
#                 if self.only_gpu:
#                     copy_action = copy_action + 1
#                 simulator_copy.get_mappable_candidates(copy_workspace)
#                 copy_task_id = copy_workspace[0].item()
#                 copy_priority = simulator_copy.get_mapping_priority(copy_task_id)
#                 copy_actions = [
#                     fastsim.Action(
#                         0,
#                         copy_action,
#                         copy_priority,
#                         copy_priority,
#                     )
#                 ]
#                 simulator_copy.simulator.map_tasks(copy_actions)
#                 state_copy = simulator_copy.run_until_external_mapping()
#         if simulator_copy.time - eft_time >= 1000:
#             reward[0] = -1
#         elif simulator_copy.time - eft_time <= -1000:
#             reward[0] = 1
#         else:
#             reward[0] = 0
#         eft_time = simulator_copy.time
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
#         self.step_count += 1
#         return out


# class kHopRolloutEnv(RuntimeEnv):
#     def set_policy(self, policy):
#         self.policy = policy

#     def _step(self, td: TensorDict) -> TensorDict:
#         if self.step_count == 0:
#             self.EFT_baseline = self._get_baseline(use_eft=True)
#             self.prev_makespan = self.EFT_baseline
#             self.graph_extractor = fastsim.GraphExtractor(self.simulator.get_state())
#         done = torch.tensor((1,), device=self.device, dtype=torch.bool)
#         reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
#         candidate_workspace = torch.zeros(
#             self.simulator_factory.graph_spec.max_candidates,
#             dtype=torch.int64,
#         )
#         dependents = torch.zeros(50, dtype=torch.int64)

#         self.simulator.get_mappable_candidates(candidate_workspace)
#         chosen_device = td["action"].item() + int(self.only_gpu)
#         global_task_id = candidate_workspace[0].item()
#         mapping_priority = self.simulator.get_mapping_priority(global_task_id)

#         self.simulator.simulator.map_tasks(
#             [fastsim.Action(0, chosen_device, mapping_priority, mapping_priority)]
#         )

#         dep_count = self.graph_extractor.get_k_hop_dependents(
#             candidate_workspace, 2, dependents
#         )
#         simulator_copy = self.simulator.copy()
#         for i in range(dep_count):
#             simulator_copy.set_task_breakpoint(
#                 fastsim.EventType.LAUNCHER, dependents[i]
#             )
#         simulator_copy.disable_external_mapper()
#         for i in range(dep_count):
#             temp = simulator_copy.run()
#             if temp == fastsim.ExecutionState.COMPLETE:
#                 break
#         eft_time = simulator_copy.time
#         with torch.no_grad():
#             simulator_copy = self.simulator.copy()
#             for i in range(dep_count):
#                 simulator_copy.set_task_breakpoint(
#                     fastsim.EventType.LAUNCHER, dependents[i]
#                 )
#             state_copy = simulator_copy.run_until_external_mapping()
#             copy_workspace = torch.zeros(
#                 self.simulator_factory.graph_spec.max_candidates,
#                 dtype=torch.int64,
#             )
#             finished = 0
#             copy_obs = TensorDict(observation=simulator_copy.observer.get_observation())
#             while state_copy != fastsim.ExecutionState.COMPLETE:
#                 # print(state_copy)
#                 if state_copy == fastsim.ExecutionState.BREAKPOINT:
#                     finished += 1
#                     if finished >= dep_count:
#                         break
#                     state_copy = simulator_copy.run_until_external_mapping()
#                     continue
#                 elif state_copy == fastsim.ExecutionState.EXTERNAL_MAPPING:
#                     simulator_copy.observer.get_observation(copy_obs["observation"])
#                     action_logits = self.policy(copy_obs)["logits"]
#                     copy_action = torch.argmax(action_logits, dim=-1).item()
#                     if self.only_gpu:
#                         copy_action = copy_action + 1
#                     simulator_copy.get_mappable_candidates(copy_workspace)
#                     copy_task_id = copy_workspace[0].item()
#                     copy_priority = simulator_copy.get_mapping_priority(copy_task_id)
#                     copy_actions = [
#                         fastsim.Action(
#                             0,
#                             copy_action,
#                             copy_priority,
#                             copy_priority,
#                         )
#                     ]
#                     simulator_copy.simulator.map_tasks(copy_actions)
#                     state_copy = simulator_copy.run_until_external_mapping()
#                 else:
#                     print(f"Unexpected simulator status: {state_copy}")
#                     assert False, f"Unexpected simulator status: {state_copy}"

#         if simulator_copy.time - eft_time >= 1000:
#             reward[0] = -1
#         elif simulator_copy.time - eft_time <= -1000:
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
#         self.step_count += 1
#         return out


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

        self.simulator_factory[self.active_idx].set_seed(priority_seed=seed)
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
        change_location=False,
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
        self.change_location = change_location
        self.location_seed = 0

        graph = simulator_factory[self.active_idx].input.graph
        assert hasattr(graph, "get_cell_locations")
        assert hasattr(graph, "set_cell_locations")
        assert hasattr(graph, "randomize_locations")
        self.initial_location_list = graph.get_cell_locations()
        self.location_randomness = location_randomness
        self.location_list = location_list

        random.seed(self.location_seed)

        if change_location:
            graph.randomize_locations(
                self.location_randomness, self.location_list, verbose=False
            )

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.resets += 1
        current_priority_seed = self.simulator_factory[self.active_idx].pseed
        current_duration_seed = self.simulator_factory[self.active_idx].seed

        if self.change_location:
            new_location_seed = self.location_seed + self.resets
            # Load initial location list
            graph = self.simulator_factory[self.active_idx].input.graph
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

        self.simulator = self.simulator_factory[self.active_idx].create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )

        simulator_status = self.simulator.run_until_external_mapping()
        assert (
            simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING
        ), f"Unexpected simulator status: {simulator_status}"

        obs = self._get_observation()
        return obs

    def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
        s = super()._set_seed(seed, static_seed)
        # if s is not None:
        #     self.location_seed = s
        return s


class RandomLocationRuntimeEnv(RuntimeEnv):
    def __init__(
        self,
        simulator_factory,
        seed: int = 0,
        device="cpu",
        baseline_time=56000,
        change_priority=True,
        change_duration=False,
        change_location=False,
        location_seed=0,
        location_randomness=1,
        location_list=[1, 2, 3, 4],
    ):
        super().__init__(
            simulator_factory=simulator_factory,
            seed=seed,
            device=device,
            baseline_time=baseline_time,
            change_priority=change_priority,
            change_duration=change_duration,
        )
        self.change_location = change_location
        self.location_seed = 0

        graph = simulator_factory[self.active_idx].input.graph
        assert hasattr(graph, "get_cell_locations")
        assert hasattr(graph, "set_cell_locations")
        assert hasattr(graph, "randomize_locations")
        self.initial_location_list = graph.get_cell_locations()
        self.location_randomness = location_randomness
        self.location_list = location_list

        random.seed(self.location_seed)

        if change_location:
            graph.randomize_locations(
                self.location_randomness, self.location_list, verbose=False
            )

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.resets += 1
        current_priority_seed = self.simulator_factory[self.active_idx].pseed
        current_duration_seed = self.simulator_factory[self.active_idx].seed

        if self.change_location:
            new_location_seed = self.location_seed + self.resets
            # Load initial location list
            graph = self.simulator_factory[self.active_idx].input.graph
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

        self.simulator = self.simulator_factory[self.active_idx].create(
            priority_seed=new_priority_seed, duration_seed=new_duration_seed
        )

        simulator_status = self.simulator.run_until_external_mapping()
        assert (
            simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING
        ), f"Unexpected simulator status: {simulator_status}"

        obs = self._get_observation()
        return obs

    def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
        s = super()._set_seed(seed, static_seed)
        # if s is not None:
        #     self.location_seed = s
        return s


class IncrementalMappingEnv(EnvBase):
    def __init__(
        self,
        simulator_factory: SimulatorFactory,
        seed: int = 0,
        device="cpu",
        baseline_time=56000,
        change_priority=False,
        change_duration=False,
        change_location=False,
        only_gpu=True,
        location_list=[1, 2, 3, 4],
        path=".",
    ):
        super().__init__(device=device)

        self.change_priority = change_priority
        self.change_duration = change_duration
        self.change_location = change_location
        self.path = path
        self.only_gpu = only_gpu
        self.location_list = location_list

        self.simulator_factory = simulator_factory
        self.simulator: SimulatorDriver = simulator_factory[self.active_idx].create(seed)
        self.graph_extractor: GraphExtractor = GraphExtractor(
            self.simulator.get_state()
        )

        self.buffer_idx = 0
        self.resets = 0
        self.last_time = 0

        self.observation_spec = self._create_observation_spec()
        self.action_spec = self._create_action_spec(ndevices=5)
        self.reward_spec = self._create_reward_spec()
        self.done_spec = self._create_done_spec()

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
        n_devices = self.simulator_factory[self.active_idx].graph_spec.max_devices
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

    def _create_done_spec(self):
        return Binary(n=1, device=self.device, dtype=torch.bool)

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
            obs, reward, time, improvement = self._handle_done(obs)
            self.last_time = 0

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.resets += 1

        current_priority_seed = self.simulator_factory[self.active_idx].pseed
        current_duration_seed = self.simulator_factory[self.active_idx].seed

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
        if self.change_location and isinstance(
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
