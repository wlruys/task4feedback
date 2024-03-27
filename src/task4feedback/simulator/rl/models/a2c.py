from torch.distributions import Categorical

from typing import Dict, List, Tuple
from collections import namedtuple

from ..networks.a2c_gcn_fcn import *
from ..networks.a2c_fcn import *
from ...task import SimulatedTask
from ....types import TaskState, TaskType
from .globals import *
from .model import *
from .env import *

import torchviz

MappingLogs = namedtuple("MappingLogs",
                         ("state", "action", "next_state"))

class A2CAgent(RLModel):

    def __init__(self, rl_env: RLBaseEnvironment, load_best_model: int = 0,
                 execution_mode: str = "testing", lr: float = 0.999):
        # Actor: Policy network that selects an action.
        # Critic: Value network that evaluates an action from the policy network.
        self.a2c_model = A2CNetworkNoGCN(rl_env.get_state_dim(), rl_env.get_out_dim())
        self.optimizer = optim.RMSprop(self.a2c_model.parameters(),
                                       lr=0.0001)
                                       #lr=0.0005)
        self.steps = 0
        self.execution_mode = execution_mode
        self.is_loaded_model_best = load_best_model
        self.fastest_execution_time = float("inf")
        # Interval to update the actor network parameter
        self.step_for_optim = 10
        self.lr = lr
        self.episode = 0
        self.a2cnet_fname = "a2c_network.pt"
        self.optimizer_fname = "optimizer.pt"
        self.best_a2cnet_fname = "best_a2c_network.pt"
        self.best_optimizer_fname = "best_optimizer.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_mapping_decision = dict()
        # Accumulated reward 
        self.accumulated_reward = 0
        # Log action selection
        self.entropy_sum = 0
        self.log_probs = []
        self.values = []
        self.rewards = []

        self.tmp_curr_state = None
        self.tmp_next_state = None
        self.tmp_action = None
        self.logs: List[MappingLogs] = []

        if not self.is_training_mode():
            print("Load model..\n")
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
            R = rewards[step] + self.lr * R
            # Restore the time sequence order.
            returns.insert(0, R)
        return returns

    def select_device(self, target_task: SimulatedTask, x: torch.tensor):
        model_input = NetworkInput(x, False, None, None)
        # This gets two values:
        # 1) transition probability of all the actions from the current state
        # 2) state value that evaluates the actor's policy;
        #    if a state value and probability distribution are corresponding,
        #    it is a good model.
        actions, value = self.a2c_model(model_input)
        action_probabilities = F.softmax(actions, dim=0)
        # Sample an action by using Categorical random distribution
        dist = Categorical(action_probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.entropy_sum += dist.entropy().mean()
        self.log_probs.append(log_prob)
        self.values.append(value)
        """
        print("Target task:", target_task)
        print("action:", action)
        print("action probs:", action_probabilities)
        print("select device:", target_task)
        print("model input:", model_input)
        print("value:", value)
        print("actions:", actions)
        print("entropy:", dist.entropy())
        print("sum:", self.entropy_sum)
        """
        return action

    def add_reward(self, reward):
        """
        Add a reward to the list.
        """
        self.accumulated_reward += reward.item()
        self.rewards.append(reward)

    def optimize_model(self, next_x, next_gcn_x, next_gcn_edgeindex):
        self.steps += 1
        if not self.is_training_mode():
            # Reset the model states
            self.steps = 0
            self.entropy_sum = 0
            self.log_probs = []
            self.values = []
            self.rewards = []
            return
        if self.steps == self.step_for_optim:
            assert len(self.log_probs) == self.step_for_optim
            assert len(self.values) == self.step_for_optim
            assert len(self.rewards) == self.step_for_optim

            # To perform TD to optimize the model, get a state value
            # of the expected next state from the critic netowrk
            _, next_value = self.a2c_model(
                NetworkInput(next_x, False, next_gcn_x, next_gcn_edgeindex))
            cat_log_probs = torch.cat(
                [lp.unsqueeze(0) for lp in self.log_probs])
            cat_values = torch.cat(self.values)
            cat_rewards = torch.cat(self.rewards).to(self.device)
            returns = self.compute_returns(next_value, cat_rewards)
            returns = torch.cat(returns).detach() 
            advantage = returns - cat_values
            actor_loss = -(cat_log_probs * advantage.detach()).mean()
            #critic_loss = advantage.pow(2).mean()
            critic_loss = 1 * F.mse_loss(cat_values.unsqueeze(-1), returns.unsqueeze(-1))
            loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy_sum
            self.optimizer.zero_grad()
            loss.backward()
            # torchviz.make_dot(loss, params=dict(self.a2c_model.named_parameters())).render("attacehd", format="png")
            for param in self.a2c_model.parameters():
               param.grad.data.clamp_(-1, 1)
            """
            print("next x:", next_x)
            print("next_gcn_x:", next_gcn_x)
            print("next gcn edgeindex:", next_gcn_edgeindex)
            print("next value:", next_value)
            print("lst_log_probs:", cat_log_probs)
            print("lst rewards:", cat_rewards)
            print("lst values:", cat_values)
            print("cat returns:", returns)
            print("actor loss:", actor_loss, ", and critic loss:", critic_loss, " advantage:", advantage)
            print("loss;", loss)
            """
            # Reset the model states
            self.steps = 0
            self.optimizer.step()
            self.entropy_sum = 0
            self.log_probs = []
            self.values = []
            self.rewards = []

    def load_model(self):
        """ Load a2c model and optimizer parameters from files;
            if a file doesn't exist, skip reading and use default parameters.
        """
        print("Load models..", flush=True)
        if os.path.exists(self.a2cnet_fname):
            self.a2c_model = torch.load(self.a2cnet_fname)
        else:
            print("A2C network does not exist, and so, not loaded",
                  flush=True)
        if os.path.exists(self.optimizer_fname):
            # The optimizer needs to do two phases to correctly link it
            # to the policy network, and load parameters.
            loaded_optimizer = torch.load(self.optimizer_fname)
            self.optimizer.load_state_dict(loaded_optimizer.state_dict())
        else:
            print("Optimizer  does not exist, and so, not loaded", flush=True)

    def save_model(self):
        """ Save a2c model and optimizer parameters to files. """
        if not self.is_training_mode():
            return
        print("Save models..", flush=True)
        torch.save(self.a2c_model, self.a2cnet_fname)
        torch.save(self.optimizer, self.optimizer_fname)

    def load_best_model(self):
        print("Load best models..", flush=True)
        if os.path.exists(self.best_a2cnet_fname):
            self.a2c_model = torch.load(self.best_a2cnet_fname)
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
        torch.save(self.a2c_model, self.best_a2cnet_fname)
        torch.save(self.optimizer, self.best_optimizer_fname)

    def is_training_mode(self):
        return "training" in self.execution_mode

    def set_training_mode(self):
        self.execution_mode = "training"

    def set_test_mode(self):
        self.execution_mode = "test"
        if self.is_loaded_model_best == 1:
            self.load_best_model()
        else:
            self.load_model()

    def start_episode(self):
        """ Start a new episode, and update (or initialize) the current state.
        """
        self.episode += 1
        #self.print_model("started")

    def complete_episode(self, execution_time):
        """ Finalize the current episode.
        """
        #self.print_model("finished")
        print("Episode total reward:", self.episode, ", ", self.accumulated_reward)
        with open("log.out", "a") as fp:
            fp.write(str(self.episode) + " reward, " + str(self.accumulated_reward) + "\n")
        if self.is_training_mode():
            self.save_model()
            if execution_time < self.fastest_execution_time:
                self.fastest_execution_time = execution_time
                self.save_best_network()
            self.accumulated_reward = 0

    def print_model(self, prefix: str):
        with open("models/" + prefix + ".a2c_network.str", "w") as fp:
            for key, param in self.a2c_model.named_parameters():
                fp.write(key + " = " + str(param))
        with open("models/" + prefix + ".optimizer.str", "w") as fp:
            for key, param in self.optimizer.state_dict().items():
                fp.write(key + " = " + str(param))

    def log_state(self, state: torch.tensor):
        """
        Log a current state (S).
        """
        self.tmp_curr_state = state

    def log_action(self, action: int):
        """
        Log a chosen action (A). 
        """
        self.tmp_action = action

    def log_next_state(self, next_state: torch.tensor):
        """
        Log a next action (S').
        """
        self.tmp_next_state = next_state

    def log_sans(self):
        """
        Buffer (S, A, S').
        All the buffered logs share the same terminal reward.
        """

        # print("S:", self.tmp_curr_state, " A:", self.tmp_action, " S':", self.tmp_next_state)
        self.logs.append(MappingLogs(self.tmp_curr_state, self.tmp_action, self.tmp_next_state))
        # print("logs length:", len(self.logs))
        self.tmp_curr_state = None
        self.tmp_action = None
        self.tmp_next_state = None
