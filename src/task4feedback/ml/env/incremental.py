import torch
import random
from tensordict import TensorDict
from typing import Optional
import task4feedback.trip as trip
from .base import RuntimeEnv
from .mixins import SimulationLookaheadMixin
from .utils import tasks_to_steps

class IncrementalEFT(SimulationLookaheadMixin, RuntimeEnv):
    """
    Environment with incremental EFT-based rewards.

    Reward is based on the difference between previous and current lookahead makespan,
    encouraging the agent to reduce completion time at each step.

    Args:
        gamma: Discount factor for current makespan
        scaling_factor: Scale factor for reward normalization
        terminal_reward: Add final makespan as terminal reward if True
    """

    def __init__(self, *args, gamma: float = 1.0, scaling_factor: float = 1.0, terminal_reward: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.scaling_factor = scaling_factor
        self.terminal_reward = terminal_reward

    def _init_step_zero(self):
        super()._init_step_zero()
        self.prev_makespan = self.EFT_baseline
        self.graph_extractor = trip.GraphExtractor(self.simulator.get_state())
        self.eft_time = self.EFT_baseline

    def _compute_reward(self, td: TensorDict) -> float:
        if not self.disable_reward_flag:
            ml_time = self.run_lookahead(steps=0, drain=False) # Just run to completion

            reward = (self.eft_time - self.gamma * ml_time) / (self.EFT_baseline / self.scaling_factor)
            self.eft_time = ml_time
            return reward
        return 0.0

    def _post_process_reward(self, reward: float, final_reward: float, done: bool) -> float:
        if done and self.terminal_reward:
            return reward + final_reward
        return reward


class LookbackKStep(SimulationLookaheadMixin, RuntimeEnv):

    def __init__(self, *args, gamma: float = 1.0, k: int = 5, terminal_reward: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.k = k
        # Removed self.step_count = 0 - use parent's step_count
        max_length = max([len(self.simulator_factory[i].input.graph) for i in range(len(self.simulator_factory))])
        self.kstep_record = torch.zeros(max_length + 2, dtype=torch.int64)
        self.current_record = torch.zeros(max_length + 2, dtype=torch.int64)
        self.terminal_reward = terminal_reward

    def _init_step_zero(self):
        super()._init_step_zero()
        self.graph_extractor = trip.GraphExtractor(self.simulator.get_state())

    def _compute_reward(self, td: TensorDict) -> float:
        if not self.disable_reward_flag:
            # K-step lookahead
            max_candidates = self.simulator_factory[self.active_idx].graph_spec.max_candidates
            steps = tasks_to_steps(self.k, max_candidates) if self.k > 0 else 0
            self.kstep_record[self.step_count] = self.run_lookahead(steps=steps, drain=True)

            # Current state drain
            self.current_record[self.step_count] = self.run_lookahead(steps=0, drain=True)

            if self.step_count < self.k:
                reward = 0.0
            else:
                predicted_time = self.kstep_record[self.step_count - self.k]
                agent_time = self.current_record[self.step_count]
                reward = (predicted_time - self.gamma * agent_time) / (self.EFT_baseline / self.size())
            return reward
        return 0.0

    def _post_process_reward(self, reward: float, final_reward: float, done: bool) -> float:
        if done and self.terminal_reward:
            return reward + final_reward
        return reward


class SparseLookbackKStep(SimulationLookaheadMixin, RuntimeEnv):

    def __init__(self, *args, gamma: float = 1.0, k: int = 5, delay: int = 5, terminal_reward: bool = False, random_offset: bool = False, offset: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.k = k
        # Removed self.step_count = 0 - use parent's step_count
        max_length = max([len(self.simulator_factory[i].input.graph) for i in range(len(self.simulator_factory))])
        self.kstep_record = torch.zeros(max_length + 2, dtype=torch.int64)
        self.current_record = torch.zeros(max_length + 2, dtype=torch.int64)
        self.terminal_reward = terminal_reward
        self.random_offset = random_offset
        self.offset = offset
        self.delay = min(delay, k)  # Ensure delay is not greater than k
        self.look_ahead = max(0, k - self.delay)  # Lookahead steps after the delay
        self.reference_steps = self.delay + self.look_ahead

    def _init_step_zero(self):
        super()._init_step_zero()
        self.graph_extractor = trip.GraphExtractor(self.simulator.get_state())

    def _compute_reward(self, td: TensorDict) -> float:
        flag_predict = (self.step_count + self.offset) % self.delay == 0
        flag_check = (self.step_count > self.delay) and ((self.step_count - self.delay + self.offset) % self.delay == 0)

        max_candidates = self.simulator_factory[self.active_idx].graph_spec.max_candidates

        if not self.disable_reward_flag and flag_predict:
            steps = tasks_to_steps(self.reference_steps, max_candidates) if self.k > 0 else 0
            self.kstep_record[self.step_count] = self.run_lookahead(steps=steps, drain=True)

        if not self.disable_reward_flag and flag_check:
            steps = tasks_to_steps(self.look_ahead, max_candidates) if self.look_ahead > 0 else 0
            self.current_record[self.step_count] = self.run_lookahead(steps=steps, drain=True)

            reward = (self.kstep_record[self.step_count - self.delay] - self.current_record[self.step_count]) / (self.EFT_baseline / self.size())
            return reward
        return 0.0

    def _post_process_reward(self, reward: float, final_reward: float, done: bool) -> float:
        if done and self.terminal_reward:
            return reward + final_reward
        return reward

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        if self.random_offset:
            self.offset = random.randint(1, self.k)
        return super()._reset(td)


class LookaheadKStep(SimulationLookaheadMixin, RuntimeEnv):

    def __init__(self, *args, gamma: float = 1.0, k: int = 5, terminal_reward: bool = False, chance: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.k = k
        # Removed self.step_count = 0 - use parent's step_count
        max_length = max([len(self.simulator_factory[i].input.graph) for i in range(len(self.simulator_factory))])
        self.kstep_record = torch.zeros(max_length + 2, dtype=torch.int64)
        self.current_record = torch.zeros(max_length + 2, dtype=torch.int64)
        self.terminal_reward = terminal_reward
        self.chance = chance

    def _init_step_zero(self):
        super()._init_step_zero()
        self.graph_extractor = trip.GraphExtractor(self.simulator.get_state())

    def _compute_reward(self, td: TensorDict) -> float:
        check_s = random.random() <= self.chance

        if not self.disable_reward_flag and check_s:
            max_candidates = self.simulator_factory[self.active_idx].graph_spec.max_candidates
            steps = tasks_to_steps(self.k, max_candidates) if self.k > 0 else 0
            self.kstep_record[self.step_count] = self.run_lookahead(steps=steps, drain=True)

            steps = tasks_to_steps(self.k - 1, max_candidates) if self.k > 0 else 0
            self.current_record[self.step_count] = self.run_lookahead(steps=steps, drain=True)

            reward = (self.kstep_record[self.step_count] - self.current_record[self.step_count]) / (self.EFT_baseline / self.size())
            return reward
        return 0.0

    def _post_process_reward(self, reward: float, final_reward: float, done: bool) -> float:
        if done and self.terminal_reward:
            return reward + final_reward
        return reward


class KStepIncrementalEFT(SimulationLookaheadMixin, RuntimeEnv):
    """
    I made sooo many mistakes when implementing this, compared to what was intended.
    1. Wrong baseline (should be 0)
    2. Compares unequal schedule lengths
    Yet, it is still rather performant... so I am keeping it as a record.
    """

    def __init__(self, *args, gamma: float = 1.0, k: int = 5, drain: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.k = k
        self.drain = drain

    def _init_step_zero(self):
        super()._init_step_zero()
        self.prev_makespan = self.EFT_baseline
        self.graph_extractor = trip.GraphExtractor(self.simulator.get_state())
        self.eft_time = self.EFT_baseline

    def _compute_reward(self, td: TensorDict) -> float:
        if not self.disable_reward_flag:
            max_candidates = self.simulator_factory[self.active_idx].graph_spec.max_candidates
            steps = tasks_to_steps(self.k, max_candidates)
            ml_time = self.run_lookahead(steps=steps, drain=self.drain)

            reward = (self.eft_time - self.gamma * ml_time) / (self.EFT_baseline / (self.size()))
            self.eft_time = ml_time
            return reward
        return 0.0

    def _post_process_reward(self, reward: float, final_reward: float, done: bool) -> float:
        if done:
            return reward + final_reward
        return reward


class IncrementalMakespan(SimulationLookaheadMixin, RuntimeEnv):

    def __init__(
        self,
        *args,
        gamma: float = 1.0,
        pbrs: bool = True,
        k: int = 0,
        terminal_reward: bool = True,
        chance: float = 1.0,
        dense_reward_scale: float = 1.0,
        sparse_reward_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.k = k
        self.chance = chance
        self.terminal_reward = terminal_reward
        self.dense_reward_scale = dense_reward_scale
        self.sparse_reward_scale = sparse_reward_scale
        self.pbrs = pbrs

        self.interval_flags = torch.zeros(self.max_length(), dtype=torch.bool)
        self._reinitialize_intervals()

    def _reinitialize_intervals(self):
        if self.chance >= 1.0:
            self.interval_flags = torch.ones(self.max_length() + 1, dtype=torch.bool)
            return

        sample = torch.rand(self.max_length())
        self.interval_flags = sample <= self.chance

    def _init_step_zero(self):
        super()._init_step_zero()
        self.prev_makespan = self.EFT_baseline
        self.graph_extractor = trip.GraphExtractor(self.simulator.get_state())
        self.eft_time = self.EFT_baseline

        max_candidates = self.simulator_factory[self.active_idx].graph_spec.max_candidates
        steps = tasks_to_steps(self.k, max_candidates) if self.k > 0 else 0
        sim_time = self.run_lookahead(steps=steps, drain=True)

        self.potential = [(-sim_time) / (self.EFT_baseline)]
        self.potential_sum = 0.0
        if self.chance < 1.0:
            self._reinitialize_intervals()

    def _compute_reward(self, td: TensorDict) -> float:
        if not self.disable_reward_flag and self.interval_flags[self.step_count - 1]:
            max_candidates = self.simulator_factory[self.active_idx].graph_spec.max_candidates
            steps = tasks_to_steps(self.k, max_candidates) if self.k > 0 else 0
            sim_time = self.run_lookahead(steps=steps, drain=True)

            self.potential.append((-sim_time) / (self.EFT_baseline))

            reward = self.dense_reward_scale * (self.gamma * self.potential[-1] - self.potential[-2]) 
            if self.verbose:
                print(f"Step {self.step_count} Reward: {reward:.4f} (P(s)={self.potential[-2]:.4f}, P(s+1)={self.potential[-1]:.4f})")
            return reward
        else:
            self.potential.append(0.0)
            return 0.0

    def _post_process_reward(self, reward: float, final_reward: float, done: bool) -> float:
        if done:
            self.potential_sum -= reward
            if self.terminal_reward:
                reward = self.sparse_reward_scale * final_reward
                if self.pbrs:
                    reward = reward + self.dense_reward_scale * (0 - self.potential[-2])
                    self.potential_sum += self.dense_reward_scale * (0 - self.potential[-2])
            if self.verbose:
                print(f"Terminal Step {self.step_count} Reward: {reward:.4f} Terminal: {final_reward:.4f} Sum(Potential): {self.potential_sum:.4f}")
                deltas = []
                for i in range(1, len(self.potential)):
                    deltas.append(self.dense_reward_scale * (self.gamma * self.potential[i] - self.potential[i - 1]))
                if not self.disable_reward_flag:
                    print(f"Max pbrs: {max(deltas):.4f}, Min pbrs: {min(deltas):.4f}")
        return reward


class DelayIncrementalEFT(IncrementalEFT):

    def __init__(
        self,
        *args,
        delay: int = 10,
        random_offset: bool = True,
        offset: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.offset = offset
        self.random_offset = random_offset

    def _compute_reward(self, td: TensorDict) -> float:
        flag = (self.step_count + self.offset) % self.delay

        if flag == 0:
            ml_time = self.run_lookahead(steps=0, drain=False)
            reward = (self.eft_time - self.gamma**self.delay * ml_time) / (self.EFT_baseline / self.size())
            self.eft_time = ml_time
            return reward
        return 0.0

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        if self.random_offset:
            self.offset = random.randint(1, self.delay)
        return super()._reset(td)


class BaselineImprovementEFT(SimulationLookaheadMixin, RuntimeEnv):
    def __init__(self, *args, delay=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay

    def _init_step_zero(self):
        super()._init_step_zero()
        self.prev_makespan = self.EFT_baseline
        self.graph_extractor = trip.GraphExtractor(self.simulator.get_state())

    def _compute_reward(self, td: TensorDict) -> float:
        flag = (self.step_count + 1) % self.delay

        if flag == 0:
            ml_time = self.run_lookahead(steps=0, drain=False)
            reward = (self.EFT_baseline - ml_time) / self.size()
            return reward
        return 0.0


class GeneralizedIncrementalEFT(SimulationLookaheadMixin, RuntimeEnv):
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

    def _init_step_zero(self):
        super()._init_step_zero()
        self.prev_makespan = self.EFT_baseline
        self.graph_extractor = trip.GraphExtractor(self.simulator.get_state())
        self.eft_log[self.step_count] = self.EFT_baseline

    def _compute_reward(self, td: TensorDict) -> float:
        ml_time = self.run_lookahead(steps=0, drain=False)
        self.eft_log[self.step_count + 1] = ml_time
        reward_sum = 0
        for i in range(0, self.step_count, 1):
            current = self.eft_log[i] - ml_time

            if self.binary:
                current = 1 if current > 0 else -1

            if self.clip_individual:
                current = max(current, 0)

            if not self.flip:
                discount = self.gamma * (1 - self.gamma) ** (self.max_steps - 1 - (self.step_count - i))
            else:
                discount = self.gamma * (1 - self.gamma) ** (self.step_count - 1 - i)

            reward_sum += current * discount

        if self.clip_total:
            reward_sum = min(reward_sum, 0)

        reward = reward_sum / self.EFT_baseline
        return reward
