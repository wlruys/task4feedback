import task4feedback.trip as trip
import torch
from typing import Optional, List
import numpy as np
import gc
import random

from torchrl.envs import EnvBase
from task4feedback.interface import DeviceType
from task4feedback.interface.wrappers import (
    DefaultObserverFactory,
    SimulatorFactory,
    create_graph_spec,
    System,
)
from task4feedback.trip import GraphExtractor, SchedulerState
from task4feedback.interface.simulator import SimulatorDriver
from torchrl.data import Composite, TensorSpec, Unbounded, Binary
from torchrl.envs.utils import make_composite_from_td
from tensordict import TensorDict
from task4feedback.graphs.jacobi import JacobiGraph, JacobiRoundRobinMapper, JacobiQuadrantMapper, LevelPartitionMapper
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from torchrl.data import Categorical
from task4feedback.logging import training

from .utils import sample_vector

# Constants
DEFAULT_BASELINE_TIME = 4000 * 5  # Default baseline time in time units
DEFAULT_BURN_IN_RESETS = 10  # Number of resets with random start before normal operation

class RuntimeEnv(EnvBase):
    """
    Base reinforcement learning environment for task scheduling.

    This environment wraps a task scheduler simulator and provides an RL interface
    for training agents to make scheduling decisions. The environment supports:
    - Randomization of task priorities, durations, locations, and workloads
    - Multiple baseline policies for comparison
    - Flexible reward computation through hook methods
    - Observation buffering for efficient training

    The environment follows the TorchRL EnvBase interface.

    Args:
        simulator_factory: Factory or list of factories for creating simulators
        seed: Random seed for reproducibility
        device: PyTorch device ("cpu" or "cuda")
        baseline_time: Default baseline time if baseline computation fails
        change_priority: Randomize task priorities on each reset
        change_duration: Randomize task durations on each reset
        change_location: Randomize data locations on each reset
        change_workload: Randomize workload on each reset (for dynamic graphs)
        only_gpu: Only use GPU devices for scheduling (skip CPU)
        location_seed: Seed for location randomization
        workload_seed: Seed for workload randomization
        priority_seed: Seed for priority randomization
        location_randomness: Fraction of locations to randomize (0.0-1.0)
        location_list: List of valid device IDs for location randomization
        max_samples_per_iter: Number of observation buffers to pre-allocate
        random_start: Start from random point in schedule during burn-in
        verbose: Print diagnostic information
        sample_z: Sample random RLE feature vectors
        burn_in_resets: Number of initial resets with random_start
        extra_logging_policy: Policy name for additional logging/comparison
    """

    def __init__(
        self,
        simulator_factory: SimulatorFactory | list[SimulatorFactory],
        seed: int = 0,
        device="cpu",
        baseline_time=DEFAULT_BASELINE_TIME,
        change_priority=False,
        change_duration=False,
        change_location=False,
        change_workload=False,
        only_gpu=True,
        location_seed=0,
        workload_seed=0,
        priority_seed=0,
        location_randomness=1,
        location_list: Optional[List[int]] = None,
        max_samples_per_iter: int = 0,
        random_start: bool = False,
        verbose: bool = True,
        sample_z: bool = False,
        burn_in_resets: int = DEFAULT_BURN_IN_RESETS,
        extra_logging_policy: str = "EFT",
        **_ignored,
    ):
        super().__init__(device=device)
        self.verbose = verbose
        self.max_samples_per_iter = max_samples_per_iter
        self.change_priority = change_priority
        self.change_duration = change_duration
        self.change_location = change_location
        self.change_workload = change_workload
        self.location_seed = location_seed
        self.workload_seed = workload_seed
        self.location_randomness = location_randomness
        self.random_start = random_start
        self.sample_z = sample_z
        self.burn_in_resets = burn_in_resets

        if location_list is None:
            location_list = [i for i in range(int(only_gpu), len(simulator_factory.input.system))]
        self.location_list = location_list
        self.only_gpu = only_gpu
        self.n_compute_devices = len(simulator_factory.input.system) - int(only_gpu)
        if verbose:
            print(
                f"""
    RuntimeEnv initialized with:
    - Seed: {seed}
    - Random Start: {self.random_start}
    - Change Priority: {self.change_priority}
    - Change Duration: {self.change_duration}
    - Change Location: {self.change_location}
    - Change Workload: {self.change_workload}
    - Only GPU: {self.only_gpu}
    - num Compute Devices: {self.n_compute_devices}
    - Location Randomness: {self.location_randomness}
    - Location List: {self.location_list}
    - Max Samples per Iteration: {self.max_samples_per_iter}
    - verbose: {self.verbose}
                """
            )

        if not isinstance(simulator_factory, list):
            simulator_factory = [simulator_factory]

        self.simulator_factory: list[SimulatorFactory] = simulator_factory
        self.active_idx = 0  # Index of the active simulator factory in case of multiple factories

        self.simulator: SimulatorDriver = simulator_factory[self.active_idx].create(seed, priority_seed=priority_seed)

        self.buffer_idx = 0
        self.resets = 0
        self.EFT_baseline = 1

        self.z_spa = sample_vector(sample=self.sample_z)
        self.z_ch = sample_vector(sample=self.sample_z)

        graph = self.get_graph()
        locs = self.get_graph().get_cell_locations(as_dict=False)
        self.get_graph().set_cell_locations([-1 for _ in range(graph.nx * graph.ny)])
        self.get_graph().set_cell_locations(locs, step=0)

        if self.change_location:
            graph = simulator_factory[self.active_idx].input.graph
            if self.only_gpu and (0 in self.location_list):
                print("Warning: CPU is in the location list. Although only_gpu is set to True, the CPU will be assigned data.")
            if hasattr(graph, "get_cell_locations") and hasattr(graph, "set_cell_locations") and hasattr(graph, "randomize_locations"):
                self.legacy_graph = False
            else:
                self.legacy_graph = True
                print("Warning: Randomizing locations on a legacy graph. This may not work as expected. location_randomness is ignored.")

        self.observation = self._get_new_observation_buffer()
        observation_spec = self._create_observation_spec(self.observation)

        action_spec = self._create_action_spec(n_devices=self.n_compute_devices)
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
        self.candidate_workspace = torch.zeros(self.simulator_factory[self.active_idx].graph_spec.max_candidates, dtype=torch.int64)
        self.candidate_mask = torch.zeros(self.simulator_factory[self.active_idx].graph_spec.max_candidates, dtype=torch.bool)
        self.baseline_time = baseline_time

        if change_location:
            graph.randomize_locations(self.location_randomness, self.location_list, verbose=False)

        self.batch_size = torch.Size([])

        self.progress_key = ("aux", "progress")
        self.baseline_key = ("aux", "baseline")
        self.improvement_key = ("aux", "improvement")
        self.z_ch_key = ("aux", "z_ch")
        self.z_spa_key = ("aux", "z_spa")
        self.time_key = ("aux", "time")
        self.last_action_key = ("aux", "last_action")
        self.action_n = "action"
        self.reward_n = "reward"
        self.done_n = "done"
        self.observation_n = "observation"
        self.disable_reward_flag = False

        self._last_action_onehot = self._zero_last_action_onehot()

    def size(self):
        """
        Return maximum number of steps in the environment.
        This is the number of tasks in the graph.
        """
        return int(len(self.simulator_factory[self.active_idx].input.graph) // self.simulator_factory[self.active_idx].graph_spec.max_candidates)

    def __len__(self):
        """
        Return maximum number of steps in the environment.
        This is the number of tasks in the graph.

        Note: May not be available in a TransformedEnv.
        """
        return self.size()

    def disable_reward(self):
        self.disable_reward_flag = True

    def enable_reward(self):
        self.disable_reward_flag = False

    def _get_baseline(self, policy="EFT"):
        if policy == "EFT":
            simulator_copy = self.simulator.fresh_copy()
            simulator_copy.initialize()
            simulator_copy.initialize_data()
            simulator_copy.disable_external_mapper()
            final_state = simulator_copy.run()
            assert final_state == trip.ExecutionState.COMPLETE, f"Baseline returned unexpected final state: {final_state}"
            return simulator_copy.time
        elif policy == "Cyclic":
            simulator_copy = self.simulator.fresh_copy()
            simulator_copy.initialize()
            simulator_copy.initialize_data()
            simulator_copy.enable_external_mapper()
            simulator_copy.external_mapper = JacobiRoundRobinMapper(n_devices=self.n_compute_devices, setting=0, offset=int(self.only_gpu))
            final_state = simulator_copy.run()
            assert final_state == trip.ExecutionState.COMPLETE, f"Baseline returned unexpected final state: {final_state}"
            return simulator_copy.time
        elif policy == "Quad":
            simulator_copy = self.simulator.fresh_copy()
            simulator_copy.initialize()
            simulator_copy.initialize_data()
            simulator_copy.enable_external_mapper()
            simulator_copy.external_mapper = JacobiQuadrantMapper(n_devices=self.n_compute_devices, graph=self.simulator.input.graph, offset=int(self.only_gpu))
            final_state = simulator_copy.run()
            assert final_state == trip.ExecutionState.COMPLETE, f"Baseline returned unexpected final state: {final_state}"
            return simulator_copy.time
        elif policy.startswith("Oracle(") and policy.endswith(")"):
            k = int(policy[len("Oracle(") : -1])
            simulator_copy = self.simulator.fresh_copy()
            simulator_copy.initialize()
            simulator_copy.initialize_data()
            simulator_copy.enable_external_mapper()
            graph: DynamicJacobiGraph = simulator_copy.input.graph
            graph.mincut_per_levels(
                bandwidth=450e9,
                level_chunks=k,
                n_parts=self.n_compute_devices,
                offset=1,
            )
            graph.align_partitions()
            simulator_copy.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
            final_state = simulator_copy.run()
            assert final_state == trip.ExecutionState.COMPLETE, f"Baseline returned unexpected final state: {final_state}"
            return simulator_copy.time
        else:
            raise ValueError(f"Unknown baseline policy: {policy}")
        return self.baseline_time

    def _create_observation_spec(self, td) -> TensorSpec:
        comp = make_composite_from_td(td, unsqueeze_null_shapes=False)
        return comp

    def get_graph(self, active_idx: Optional[int] = None):
        if active_idx is None:
            active_idx = self.active_idx
        return self.simulator_factory[active_idx].input.graph

    def _create_state_value_spec(self) -> TensorSpec:
        return Unbounded(shape=[1], device=self.device, dtype=torch.float32)

    def _create_action_spec(self, n_devices: int = 5) -> TensorSpec:
        out = Categorical(
            n=n_devices,
            shape=[self.simulator_factory[self.active_idx].graph_spec.max_candidates],
            device=self.device,
            dtype=torch.int64,
        )
        return out

    def _create_reward_spec(self) -> TensorSpec:
        spec = Unbounded(shape=[1], device=self.device, dtype=torch.float32)
        return spec

    def _create_done_spec(self) -> TensorSpec:
        return Binary(n=1, device=self.device, dtype=torch.bool)

    def get_observer(self):
        return self.simulator.observer

    def _get_observation(self, reset: bool = False) -> TensorDict:
        step_count = self.step_count
        n_buffers = len(self.observations)

        if n_buffers == 1:
            obs = self.observations[0].clone()
            obs.zero_()
        else:
            obs = self.observations[step_count % n_buffers].clone()
            obs.zero_()

        obs.set(self.z_ch_key, self.z_ch)
        obs.set(self.z_spa_key, self.z_spa)

        self.simulator.observer.get_observation(obs)
        obs.set(self.last_action_key, self._last_action_onehot)
        progress = step_count / self.size()
        baseline = max(1.0, self.EFT_baseline)
        obs.set_at_(self.progress_key, progress, 0)
        obs.set_at_(self.baseline_key, baseline, 0)
        return obs

    def _get_new_observation_buffer(self) -> TensorDict:
        obs = self.simulator.observer.new_observation_buffer()
        return obs

    def _zero_last_action_onehot(self) -> torch.Tensor:
        spec = self.simulator_factory[self.active_idx].graph_spec
        return torch.zeros(
            (spec.max_candidates, spec.max_devices),
            device=self.device,
            dtype=torch.float32,
        )

    def _handle_done(self, obs):
        time = obs[self.time_key].item()
        improvement = (self.EFT_baseline) / (time)
        obs.set_at_(self.improvement_key, improvement, 0)
        reward = (-time) / (self.EFT_baseline)
        if self.verbose:
            print(
                f"Time: {time} / EFT: {self.EFT_baseline} Improvement: {improvement:.2f}",
                flush=True,
            )

        return obs, reward, time, improvement

    def max_length(self) -> int:
        return max([len(self.simulator_factory[i].input.graph) for i in range(len(self.simulator_factory))])

    def map_tasks(self, td: Optional[TensorDict] = None):
        actions = td[self.action_n]

        candidate_workspace = self.candidate_workspace
        num_candidates = self.simulator.get_mappable_candidates(candidate_workspace)

        spec = self.simulator_factory[self.active_idx].graph_spec
        last_action_onehot = torch.zeros(
            (spec.max_candidates, spec.max_devices),
            device=self.device,
            dtype=torch.float32,
        )

        if num_candidates == 0:
            self._last_action_onehot = last_action_onehot
            return

        observer = self.get_observer() 

        mapping_result = []
        for i in range(num_candidates):
            global_task_id = candidate_workspace[i].item()
            idx = observer.get_candidate_to_action(i, global_task_id)
            chosen_device = actions[idx].item() + int(self.only_gpu)
            mapping_priority = self.simulator.get_mapping_priority(global_task_id)
            action = trip.Action(
                i,
                chosen_device,
                mapping_priority,
                mapping_priority,
            )
            mapping_result.append(action)
            if 0 <= idx < spec.max_candidates and 0 <= chosen_device < spec.max_devices:
                last_action_onehot[idx, chosen_device] = 1.0

        self._last_action_onehot = last_action_onehot
        self.simulator.simulator.map_tasks(mapping_result)

    def _compute_reward(self, td: TensorDict) -> float:
        """
        Hook for subclasses to calculate reward before simulation step.

        Called after mapping tasks but before running the simulator.
        Base implementation returns 0.0.

        Args:
            td: Current TensorDict with action and observation

        Returns:
            Reward value for this step
        """
        return 0.0

    def _init_step_zero(self):
        """
        Initialize environment state at the start of an episode (step 0).

        Computes the baseline policy performance for comparison.
        Subclasses should call super()._init_step_zero() and add their own initialization.
        """
        self.EFT_baseline = self._get_baseline(policy="EFT")

    def _post_process_reward(self, reward: float, final_reward: float, done: bool) -> float:
        """
        Hook to post-process reward after step execution.

        Useful for adding terminal rewards or applying transformations.

        Args:
            reward: Accumulated reward from _compute_reward and _compute_post_step_reward
            final_reward: Terminal reward if done=True, 0.0 otherwise
            done: Whether episode is complete

        Returns:
            Final processed reward
        """
        return reward

    def _compute_post_step_reward(self, td: TensorDict, done: bool) -> float:
        """
        Hook for subclasses to calculate reward after simulation step execution.

        Called after the simulator has run but before observations are collected.
        Useful for rewards based on simulation state changes.

        Args:
            td: Current TensorDict with action
            done: Whether the episode completed

        Returns:
            Reward value for this step
        """
        return 0.0

    def _step(self, td: TensorDict) -> TensorDict:
        # print(f"Step {self.step_count+1}/{self.size()}", flush=True)
        if self.step_count == 0:
            self._init_step_zero()

        self.step_count += 1

        self.map_tasks(td)

        # Calculate reward using the hook
        reward = self._compute_reward(td)

        simulator_status = self.simulator.run_until_external_mapping()
        done = simulator_status == trip.ExecutionState.COMPLETE

        reward += self._compute_post_step_reward(td, done)

        obs = self._get_observation()

        final_reward = 0.0
        if done:
            obs, final_reward, time, improvement = self._handle_done(obs)
        
        reward = self._post_process_reward(reward, final_reward, done)

        buf = td.empty()
        buf.set(self.observation_n, obs)
        buf.set(self.reward_n, torch.tensor(reward, device=self.device, dtype=torch.float32))
        buf.set(self.done_n, torch.tensor(done, device=self.device, dtype=torch.bool))
        return buf

    def reset_to_state(self, cell_location, workload_state=None) -> TensorDict:
        self.step_count = 0
        graph = self.simulator_factory[self.active_idx].input.graph
        graph.set_cell_locations([-1 for _ in range(graph.nx * graph.ny)])
        graph.set_cell_locations(cell_location, step=0)

        if workload_state is not None:
            graph.load_workload(self.simulator_factory[self.active_idx].input.system, workload_state)

        self.simulator = self.simulator_factory[self.active_idx].create()
        self.simulator.observer.reset()

        simulator_status = self.simulator.run_until_external_mapping()
        assert simulator_status == trip.ExecutionState.EXTERNAL_MAPPING, f"Unexpected simulator status: {simulator_status}"
        gc.collect()

    def set_reset_counter(self, count):
        self.resets = count

    def _handle_location_randomization(self) -> None:
        """
        Randomize data/task locations on reset.
        Can be overridden by subclasses to customize behavior.
        """
        if not self.change_location:
            return

        new_location_seed = self.location_seed + self.resets
        graph = self.simulator_factory[self.active_idx].input.graph
        random.seed(new_location_seed)

        if self.legacy_graph:
            data = self.simulator_factory[self.active_idx].input.data.data
            for i in range(data.size()):
                data.set_location(i, random.choice(self.location_list))
        else:
            assert hasattr(graph, "randomize_locations"), "Graph does not have randomize_locations method."

            if isinstance(graph, JacobiGraph):
                graph.set_cell_locations([-1 for _ in range(graph.nx * graph.ny)])
                graph.randomize_locations(
                    self.location_randomness,
                    self.location_list,
                    verbose=False,
                    step=0,
                )
            else:
                graph.randomize_locations(
                    self.location_randomness,
                    self.location_list,
                    verbose=False,
                )

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        # start_t = perf_counter()
        training.info("Resetting environment (reset count: {})".format(self.resets))
        self.resets += 1
        self.step_count = 0
        self._last_action_onehot = self._zero_last_action_onehot()
        current_priority_seed = self.simulator_factory[self.active_idx].pseed
        current_duration_seed = self.simulator_factory[self.active_idx].seed

        self._handle_location_randomization()

        if self.change_workload:
            graph = self.simulator_factory[self.active_idx].input.graph
            assert isinstance(graph, DynamicJacobiGraph), "Graph must be a DynamicJacobiGraph to randomize workload."
            new_workload_seed = self.workload_seed + self.resets
            random.seed(new_workload_seed)
            graph.randomize_workload(seed=new_workload_seed, system=self.simulator_factory[self.active_idx].input.system)
            partition = graph.initial_mincut_partition(
                arch=DeviceType.GPU,
                bandwidth=450e9,
                n_parts=self.n_compute_devices,
                offset=0,
            )
            partition = graph.maximize_matches(partition)
            partition = [p + 1 for p in partition]  # offset by 1 to ignore cpu
            graph.set_cell_locations([-1 for _ in range(len(partition))])
            graph.set_cell_locations(partition, step=0)

        if self.change_priority:
            new_priority_seed = int(current_priority_seed + self.resets)
        else:
            new_priority_seed = int(current_priority_seed)

        if self.change_duration:
            new_duration_seed = int(current_duration_seed + self.resets)
        else:
            new_duration_seed = int(current_duration_seed)

        self.simulator = self.simulator_factory[self.active_idx].create(priority_seed=new_priority_seed, duration_seed=new_duration_seed)
        self.simulator.observer.reset()
        if self.resets < self.burn_in_resets and self.random_start:
            # Run the simulator for a random number of steps
            n_steps = random.randint(1, self.size() - 1)
            self.simulator.disable_external_mapper()
            self.simulator.set_steps(n_steps)
            self.simulator.run()
            self.simulator.enable_external_mapper()

        simulator_status = self.simulator.run_until_external_mapping()
        assert simulator_status == trip.ExecutionState.EXTERNAL_MAPPING, f"Unexpected simulator status: {simulator_status}"

        if td is None:
            td = TensorDict()
        else:
            td = td.empty()

        obs = self._get_observation(reset=True).clone()

        td.set(self.observation_n, obs)
        # end_t = perf_counter()
        # print("Reset took %.2f ms", (end_t - start_t) * 1000, flush=True)
        gc.collect()
        return td

    @property
    def observer(self):
        return self.simulator.observer

    def _set_seed(self, seed: Optional[int] = None, static_seed: Optional[int] = None):
        if self.verbose:
            print(
                f"""
            RuntimeEnv initialized with:
            - Seed: {seed}"""
            )
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.resets = 0
        if self.change_priority:
            self.simulator_factory[self.active_idx].set_seed(priority_seed=seed)
        if self.change_duration:
            self.simulator_factory[self.active_idx].set_seed(seed=seed)
        if self.change_location:
            self.location_seed = seed
        if self.change_workload:
            self.workload_seed = seed

    def reset_for_evaluation(self, seed: int = 0):
        # save seeds from curret state
        self.saved_seeds = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        old_pseed = self.simulator_factory[self.active_idx].pseed
        old_seed = self.simulator_factory[self.active_idx].seed

        self.change_priority = False  # Do not change priority in evaluation
        self.change_duration = False  # Do not change duration in evaluation
        self._set_seed(seed)
        self.random_start = False  # Do not random start in evaluation
        self.change_location = False  # Do not change location in evaluation
        self.change_workload = False  # Do not change workload in evaluation
        self.resets = 0
        self._reset()

        # Restore seeds
        torch.set_rng_state(self.saved_seeds["torch"])
        np.random.set_state(self.saved_seeds["numpy"])
        random.setstate(self.saved_seeds["random"])
