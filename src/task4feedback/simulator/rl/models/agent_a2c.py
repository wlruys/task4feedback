from torch.distributions import Categorical

from typing import Dict, List, Tuple
from collections import namedtuple

from ..networks.a2c_fcn import *
from ...task import SimulatedTask
from ....types import TaskState, TaskType
from .globals import *
from .model import *
from .env import *
from .oracles import *

import torchviz
import math
import random


class A2CAgent(RLModel):
    def __init__(
        self,
        rl_env: RLBaseEnvironment,
        load_best_model: int = 0,
        execution_mode: str = "testing",
        lr: float = 0.999,
        eps_start=0.9,
        eps_end=0.03,
        eps_decay=1000,
        oracle_function: OraclePolicy = None,
    ):
        self.rl_env = rl_env
        self.num_actions = rl_env.get_out_dim()
        # S, (S, A) value networks
        self.network = A2CNetworkNoGCN(rl_env.get_state_dim(), rl_env.get_out_dim())
        self.optimizer = optim.RMSprop(
            self.network.parameters(), lr=0.0001
        )  # , weight_decay=0.5)
        # lr=0.0005)
        self.execution_mode = execution_mode
        self.episode = 0

        # Model file related information
        self.is_loaded_model_best = load_best_model
        self.fastest_execution_time = float("inf")
        self.net_fname = "network.pt"
        self.optimizer_fname = "optimizer.pt"
        self.best_net_fname = "best_network.pt"
        self.best_optimizer_fname = "best_optimizer.pt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Accumulated reward
        self.accumulated_reward = 0

        # Model parameters
        self.action_steps = 1
        self.steps = 1

        # Model output buffer for optimziation
        self.entropy_sum = 0
        self.log_probs = []
        self.values = []
        self.rewards = []

        # Track the last task to exploit a time-difference optimization
        self.terminal_task = None

        if not self.is_training_mode():
            print("[Testing mode] Load model..\n")
            if self.is_loaded_model_best == 1:
                self.load_best_model()
            else:
                self.load_model()

    def compute_returns(self, next_value, rewards):
        returns = []
        R = next_value
        # Rewards are stored in time sequence.
        # R on the deeper level should be used for more latest return
        # calculation.
        for step in reversed(range(len(rewards))):
            R = rewards[step] + 0.999 * R
            # Restore the time sequence order.
            returns.insert(0, R)
        return returns

    def select_device(
        self, task: SimulatedTask, state: torch.tensor, sched_state: "SystemState"
    ):
        """
        Select a device from pi.
        If a specified state has not been visited, select a device from a neural
        network.
        """
        actions, v = self.network(state)
        action_probs = F.softmax(actions, dim=0)
        dist = Categorical(action_probs)
        action = dist.sample()

        if self.is_training_mode():
            log_prob = dist.log_prob(action)
            self.entropy_sum += dist.entropy().mean()
            self.log_probs.append(log_prob)
            self.values.append(v)
            self.terminal_task = task

        return action.item()

    def add_reward(self, reward):
        """
        Add a reward to the list.
        """
        self.accumulated_reward += reward

    def optimize_model(self, reward: float, sched_state: "SystemState"):
        """
        Optimize a model.
        """
        self.steps += 1

        next_state = self.rl_env.create_state(self.terminal_task, sched_state)
        _, next_v = self.network(next_state)
        cat_log_probs = torch.cat([lp.unsqueeze(0) for lp in self.log_probs])
        cat_values = torch.cat(self.values)
        cat_rewards = torch.tensor([reward for i in range(0, len(self.values))]).to(
            self.device
        )

        returns = self.compute_returns(next_v, cat_rewards)
        returns = torch.cat(returns).detach()
        advantage = returns - cat_values
        actor_loss = -(cat_log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(cat_values.unsqueeze(-1), returns.unsqueeze(-1))
        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy_sum
        print("Loss,", self.steps - 1, ",", loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        # Note that gradients are generally (-1, 1) and so this operation is no-op.
        # But just in case, it is added.
        # for param in self.network.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.entropy_sum = 0
        self.log_probs = []
        self.values = []
        self.rewards = []

    def load_model(self):
        """Load a2c model and optimizer parameters from files;
        if a file doesn't exist, skip reading and use default parameters.
        """
        print("Load models..", flush=True)
        if os.path.exists(self.net_fname):
            self.network = torch.load(self.net_fname)
        else:
            print("A2C network does not exist, and so, not loaded", flush=True)
        if os.path.exists(self.optimizer_fname):
            # The optimizer needs to do two phases to correctly link it
            # to the policy network, and load parameters.
            loaded_optimizer = torch.load(self.optimizer_fname)
            self.optimizer.load_state_dict(loaded_optimizer.state_dict())
        else:
            print("Optimizer  does not exist, and so, not loaded", flush=True)

    def save_model(self):
        """Save a2c model and optimizer parameters to files."""
        if not self.is_training_mode():
            return
        print("Save models..", flush=True)
        torch.save(self.network, self.net_fname)
        torch.save(self.optimizer, self.optimizer_fname)

    def load_best_model(self):
        print("Load best models..", flush=True)
        if os.path.exists(self.best_net_fname):
            self.network = torch.load(self.best_net_fname)
        else:
            print("A2C network does not exist", flush=True)
            print("Load normal model if it exists", flush=True)
            self.load_model()
            return
        if os.path.exists(self.best_optimizer_fname):
            # The optimizer needs to do two phases to correctly link it
            # to the policy network, and load parameters.
            loaded_optimizer = torch.load(self.best_optimizer_fname)
            self.optimizer.load_state_dict(loaded_optimizer.state_dict())
        else:
            print("Optimizer  does not exist, and so, not loaded", flush=True)
            print("Load normal optimizer if it exists", flush=True)

    def save_best_network(self):
        if not self.is_training_mode():
            return
        print("Save best model:", self.fastest_execution_time, flush=True)
        torch.save(self.network, self.best_net_fname)
        torch.save(self.optimizer, self.best_optimizer_fname)

    def start_episode(self):
        """
        Start a new episode, and update (or initialize) the current state.
        """
        self.episode += 1
        # self.print_model("started")

    def complete_episode(self, execution_time):
        """
        Finalize the current episode.
        """
        # self.print_model("finished")

        print("reward,", self.steps - 1, ",", self.accumulated_reward)
        with open("log.out", "a") as fp:
            fp.write(
                str(self.episode) + " reward, " + str(self.accumulated_reward) + "\n"
            )

        if self.is_training_mode():
            self.save_model()
            if execution_time < self.fastest_execution_time:
                self.fastest_execution_time = execution_time
                self.save_best_network()
            self.accumulated_reward = 0

    def print_model(self, prefix: str):
        """
        Print model parameters to file.
        """
        with open("models/" + prefix + ".a2c_network.str", "w") as fp:
            for key, param in self.network.named_parameters():
                fp.write(key + " = " + str(param))
        with open("models/" + prefix + ".optimizer.str", "w") as fp:
            for key, param in self.optimizer.state_dict().items():
                fp.write(key + " = " + str(param))

    def set_training_mode(self):
        self.execution_mode = "training"

    def set_test_mode(self):
        self.execution_mode = "test"
        if self.is_loaded_model_best == 1:
            self.load_best_model()
        else:
            self.load_model()

    def is_training_mode(self):
        return "training" in self.execution_mode
