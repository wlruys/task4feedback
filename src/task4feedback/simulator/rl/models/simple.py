from torch.distributions import Categorical

from typing import Dict, List, Tuple
from collections import namedtuple

from ..networks.a2c_fcn import *
from ...task import SimulatedTask
from ....types import TaskState, TaskType
from .globals import *
from .model import *
from .env import *

import torchviz
import math

MappingLogs = namedtuple("MappingLogs",
                         ("state", "next_state"))

class SimpleAgent(RLModel):

    def __init__(self, rl_env: RLBaseEnvironment, load_best_model: int = 0,
                 execution_mode: str = "testing", lr: float = 0.999):
        self.num_actions = rl_env.get_out_dim()
        # S, (S, A) value networks
        self.network = A2CNetworkNoGCN(rl_env.get_state_dim(), rl_env.get_out_dim())
        self.optimizer = optim.RMSprop(self.network.parameters(),
                                       lr=0.0001)
                                       #lr=0.0005)
        self.execution_mode = execution_mode
        self.is_loaded_model_best = load_best_model
        self.fastest_execution_time = float("inf")
        self.lr = lr
        self.episode = 0
        self.a2cnet_fname = "a2c_network.pt"
        self.optimizer_fname = "optimizer.pt"
        self.best_a2cnet_fname = "best_a2c_network.pt"
        self.best_optimizer_fname = "best_optimizer.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Accumulated reward 
        self.accumulated_reward = 0

        # pi(state) -> action probabilities
        self.pi = {}
        # lambda to update a target policy
        self.ld = 0.5

        # Buffer (S, A, S') until the terminal state
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

    def select_device(self, state: torch.tensor, oracle):
        """
        Select a device from pi.
        If a specified state has not been visited, select a device from a neural
        network.
        """

        state_tuple = tuple(state.tolist())
        with torch.no_grad():
            if state_tuple in self.pi:
                # If the current state is already in pi, sample an action from
                # the stored action probabilities
                dist = Categorical(self.pi[state_tuple])
                action = dist.sample()
                # print("State existed: ", action)
            else:
                actions, v = self.network(NetworkInput(state, False, None, None))
                self.pi[state_tuple] = F.softmax(actions, dim=0)
                dist = Categorical(self.pi[state_tuple])
                action = dist.sample()
                # print("action: ", action, " pi:", self.pi[state_tuple])
            if self.is_training_mode():
                self.update_pi(state, state_tuple, oracle)
        return action

    def update_pi(self, state: torch.tensor, state_tuple: Tuple[float], oracle):
        """ Update target policy, pi.
        """

        # Buffer probabilities from network
        actions, v = self.network(NetworkInput(state, False, None, None))
        actions = F.softmax(actions, dim=0)
        # print("oracle:", oracle, " softmax dnn:", actions)
        # Calculate loss
        self.pi[state_tuple] = (1 - self.ld) * actions + self.ld * oracle.to(self.device)

    def add_reward(self, reward):
        """
        Add a reward to the list.
        """
        self.accumulated_reward += reward.item()

    def optimize_model(self, reward: float):
        """
        Optimize a model.
        """
        print("Model optimization starts..")
        prob_loss = 0
        value_loss = 0
        num_computes = 0
        entropy_sum = 0
        plist = []
        vlist = []
        pilist = []
        zlist = []
        for log in self.logs:
            state, next_state = log
            state_tuple = tuple(state.tolist())
            p, v = self.network(NetworkInput(state, False, None, None))
            p = F.softmax(p, dim=0)
            pis = self.pi[state_tuple]
            # print("p:", p, " pis:", pis)
            plist.append(p)
            vlist.append(v)
            pilist.append(pis)
            zlist.append(torch.tensor([reward], dtype=torch.float))

        concat_p = torch.cat(
            [p.unsqueeze(0) for p in plist])
        # concat_p = torch.FloatTensor(plist)
        concat_v = torch.cat(vlist).to(self.device)
        # Pi is not updated through this
        concat_pi = torch.cat(
            [pi.unsqueeze(0) for pi in pilist]).to(self.device).detach()
        # concat_pi = torch.FloatTensor(pilist)
        concat_z = torch.cat(zlist).to(self.device)
        # print("plist:", plist, " pilist:", pilist)
        """
        print("concat p:", concat_p)
        print("concat v:", concat_v)
        print("concat pi:", concat_pi)
        print("concat z:", concat_z)

        # print("usq concat p:", concat_p.unsqueeze(-1))
        print("usq concat v:", concat_v.unsqueeze(-1))
        # print("usq concat pi:", concat_pi.unsqueeze(-1))
        print("usq concat z:", concat_z.unsqueeze(-1))
        with torch.no_grad():
            print("crossentropy:", F.cross_entropy(concat_p, concat_pi))
            print("crossentropy mean:", F.cross_entropy(concat_p, concat_pi).mean())
            # print("usq crossentropy:", F.cross_entropy(concat_p.unsqueeze(-1), concat_pi.unsqueeze(-1)))
            print("usq mse loss:", F.mse_loss(concat_v.unsqueeze(-1), concat_z.unsqueeze(-1)))
            print("mse loss:", F.mse_loss(concat_v, concat_z))
        """

        loss = -F.cross_entropy(concat_p, concat_pi, reduction='sum') + \
               F.mse_loss(concat_v.unsqueeze(-1), concat_z.unsqueeze(-1), reduction='sum')
        print("Loss:", loss)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        # print("loss:", loss)
        self.logs = []
        self.tmp_curr_state = None
        self.tmp_action = None
        self.tmp_next_state = None

    def load_model(self):
        """ Load a2c model and optimizer parameters from files;
            if a file doesn't exist, skip reading and use default parameters.
        """
        print("Load models..", flush=True)
        if os.path.exists(self.a2cnet_fname):
            self.network = torch.load(self.a2cnet_fname)
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
        torch.save(self.network, self.a2cnet_fname)
        torch.save(self.optimizer, self.optimizer_fname)

    def load_best_model(self):
        print("Load best models..", flush=True)
        if os.path.exists(self.best_a2cnet_fname):
            self.network = torch.load(self.best_a2cnet_fname)
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
        torch.save(self.network, self.best_a2cnet_fname)
        torch.save(self.optimizer, self.best_optimizer_fname)

    def start_episode(self):
        """
        Start a new episode, and update (or initialize) the current state.
        """
        self.episode += 1
        #self.print_model("started")

    def finalize_episode(self, execution_time):
        """
        Finalize the current episode.
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
        """
        Print model parameters to file.
        """
        with open("models/" + prefix + ".a2c_network.str", "w") as fp:
            for key, param in self.network.named_parameters():
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
        self.logs.append(MappingLogs(self.tmp_curr_state, self.tmp_next_state))
        # print("logs length:", len(self.logs))
        self.tmp_curr_state = None
        self.tmp_action = None
        self.tmp_next_state = None

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

