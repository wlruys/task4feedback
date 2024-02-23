from torch.distributions import Categorical

from time import perf_counter as clock
from typing import Dict, List, Tuple
from collections import namedtuple

from ..networks.a2c_fcn import *
from ..networks.pv_fcn import *
from ...task import SimulatedTask
from ....types import TaskState, TaskType, ExecutionMode
from .globals import *
from .model import *
from .env import *
from .oracles import *

import torchviz
import math
import random

MappingLogs = namedtuple("MappingLogs",
                         ("state", "p", "v", "pi"))

class SimpleAgent(RLModel):

    def __init__(self, rl_env: RLBaseEnvironment, load_best_model: int = 0,
                 exec_mode: ExecutionMode = ExecutionMode.TESTING, lr: float = 0.999,
                 eps_start = 0.9, eps_end = 0.03, eps_decay = 1000,
                 oracle_function: OraclePolicy = None):
        self.num_actions = rl_env.get_out_dim()
        self.oracle_function = oracle_function
        # S, (S, A) value networks
        self.network = PVFCN(rl_env.get_state_dim(), rl_env.get_out_dim())
        # self.network = A2CNetworkNoGCN(rl_env.get_state_dim(), rl_env.get_out_dim())
        self.optimizer = optim.RMSprop(self.network.parameters(),
                                       lr=0.00005)#, weight_decay=0.5)
                                       #lr=0.0005)
        self.exec_mode = exec_mode
        self.is_loaded_model_best = load_best_model
        self.fastest_execution_time = float("inf")
        self.episode = 0
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
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.sim_g_f = 0
        self.sim_g_f_freq_threshold = 10
        self.sim_g_f_threshold = 0.85
        self.random_enabled = False

        # Buffer (S, P, V, PI) until the terminal state
        self.logs: List[MappingLogs] = []

        self.num_consensus = 0
        self.num_selection = 0

        if not self.is_training_mode():
            print("[Testing mode] Load model..\n")
            if self.is_loaded_model_best == 1:
                self.load_best_model()
            else:
                self.load_model()

    def select_device_using_p(self, f_action_probs, o_action_probs):
        action = torch.tensor(max(enumerate(f_action_probs), key=lambda x: x[1])[0])
        o_action = max(enumerate(o_action_probs), key=lambda x: x[1])[0]
        self.num_selection += 1
        self.num_consensus += 1 if o_action == action else \
                              1 if o_action_probs[o_action] == o_action_probs[action] \
                              else 0
        return action

    def select_device_using_pqr(self, state, v, f_action_probs, o_action_probs):
        decayed_ld = 1 / ((self.steps)**(1/3))
        ld = 0 if not self.is_training_mode() else decayed_ld if decayed_ld > 0.2 else 0

        if self.random_enabled:
            eps_threshold = self.eps_end + (
                            self.eps_start - self.eps_end) * math.exp(
                            -1. * self.action_steps / self.eps_decay)
            sample = random.random()
            self.action_steps += 1

        if not self.random_enabled or (self.random_enabled and sample > eps_threshold):
            action_probs = (1 - ld) * f_action_probs + ld * o_action_probs.to(self.device)
            o_action = max(enumerate(o_action_probs), key=lambda x: x[1])[0]
            f_action = max(enumerate(f_action_probs), key=lambda x: x[1])[0]
            self.num_selection += 1
            self.num_consensus += 1 if o_action == f_action else \
                                  1 if o_action_probs[o_action] == o_action_probs[f_action] \
                                  else 0
            # print("o_action_probs action:", o_action, " f action:", f_action)
            action = torch.tensor(max(enumerate(action_probs), key=lambda x: x[1])[0])
            self.logs.append(MappingLogs(
                  state, f_action_probs, v, o_action_probs.to(self.device)))
        else:
            # print("Random chosen")
            random.seed()
            action = torch.tensor(
                     random.choice([d for d in range(self.num_actions)]),
                     dtype=int)
        return action

    def select_device(self, task:SimulatedTask, state: torch.tensor, sched_state: "SystemState"):
        """
        Select a device from pi.
        If a specified state has not been visited, select a device from a neural
        network.
        """
        actions, v = self.network(state)
        f_action_probs = F.softmax(actions, dim=0)
        o_action_probs = self.oracle_function.get_action(sched_state)

        if not self.is_training_mode():
            # Always uses a function approximator
            action = self.select_device_using_p(f_action_probs, o_action_probs)
        else:
            action = self.select_device_using_pqr(state, v, f_action_probs, o_action_probs)
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
        print("Model optimization starts..")

        plist = []
        vlist = []
        pilist = []
        zlist = []

        self.steps += 1

        for log in self.logs:
            state, p, v, pi = log
            plist.append(p)
            vlist.append(v)
            pilist.append(pi)
            # zlist.append(torch.tensor([reward*len(self.logs)], dtype=torch.float))
            zlist.append(torch.tensor([reward], dtype=torch.float))
            # print("P:", p)

        concat_p = torch.cat(
            [p.unsqueeze(0) for p in plist])
        concat_v = torch.cat(vlist).to(self.device)
        # Pi is not updated through this
        concat_pi = torch.cat(
            [pi.unsqueeze(0) for pi in pilist]).to(self.device).detach()
        concat_z = torch.cat(zlist).to(self.device)
        # print("concat p:", concat_p)
        # print("concat v:", concat_v)
        # print("concat pi:", concat_pi)
        # print("concat z:", concat_z)

        """
        # print("usq concat p:", concat_p.unsqueeze(-1))
        print("usq concat v:", concat_v.squeeze(-1))
        # print("usq concat pi:", concat_pi.unsqueeze(-1))
        print("usq concat z:", concat_z.squeeze(-1))
        """
        # with torch.no_grad():
            # print("crossentropy:", F.cross_entropy(concat_p.squeeze(-1), concat_pi.squeeze(-1)))
            # print("smooth:", torch.nn.SmoothL1Loss()(concat_p, concat_pi))
            # print("usq crossentropy:", F.cross_entropy(concat_p.unsqueeze(-1), concat_pi.unsqueeze(-1)))
            # print("usq mse loss:", F.mse_loss(concat_v.unsqueeze(-1), concat_z.unsqueeze(-1)))
            # print("prob:", -(concat_p.log() * concat_pi))
            # print("mse loss:", F.mse_loss(concat_v, concat_z))

        # loss = -F.cross_entropy(concat_p, concat_pi, reduction='mean')# + \
               # F.mse_loss(concat_v.unsqueeze(-1), concat_z.unsqueeze(-1), reduction='sum')
        # loss = torch.nn.SmoothL1Loss()(concat_p, concat_pi)

        # print("loss fun:", (concat_p.log()) * concat_pi)
        pc = 1 - 1/(math.exp(500/self.steps))
        print("Pc:", pc)
        loss = -(concat_p.log() * concat_pi).mean() * pc + F.mse_loss(
               concat_v.unsqueeze(-1), concat_z.unsqueeze(-1))

        print("Loss,", self.steps-1, ",", loss.item())
        if self.random_enabled == False:
            if self.num_consensus / float(self.num_selection) > self.sim_g_f_threshold:
                self.sim_g_f += 1
                if self.sim_g_f_freq_threshold <= self.sim_g_f:
                    self.random_enabled = True
            else:
                self.sim_g_f = 0

        self.optimizer.zero_grad()
        loss.backward()
        # Note that gradients are generally (-1, 1) and so this operation is no-op.
        # But just in case, it is added.
        # for param in self.network.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.logs = []
        self.add_reward(reward)

    def load_model(self):
        """ Load a2c model and optimizer parameters from files;
            if a file doesn't exist, skip reading and use default parameters.
        """
        print("Load models..", flush=True)
        if os.path.exists(self.net_fname):
            self.network = torch.load(self.net_fname)
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

    def complete_episode(self, execution_time):
        """
        Finalize the current episode.
        """
        #self.print_model("finished")
        # with open("log.out", "a") as fp:
        #     fp.write(str(self.episode) + " reward, " + str(self.accumulated_reward) + "\n")

        if not self.is_training_mode():
            print("reward,",self.steps-1, ",", self.accumulated_reward)
            print("consensus,", self.steps-1, ",", self.num_consensus, ",", self.num_selection)
            print("simtime,",self.steps-1, ",", execution_time / 1000)

        self.num_consensus = 0
        self.num_selection = 0

        if self.is_training_mode():
            self.save_model()
            if execution_time < self.fastest_execution_time:
                self.fastest_execution_time = execution_time
                self.save_best_network()
            self.accumulated_reward = 0
            # Enable an evaluation mode after for each training mode
            self.set_eval_mode()
        elif self.is_evaluating_mode():
            self.set_training_mode()

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

    def set_eval_mode(self):
        print("set evaluating mode")
        self.exec_mode = ExecutionMode.EVALUATION

    def set_training_mode(self):
        print("set training mode")
        self.exec_mode = ExecutionMode.TRAINING

    def set_test_mode(self):
        self.exec_mode = ExecutionMode.TESTING
        if self.is_loaded_model_best == 1:
            self.load_best_model()
        else:
            self.load_model()

    def is_training_mode(self):
        return self.exec_mode == ExecutionMode.TRAINING

    def is_evaluating_mode(self):
        return self.exec_mode == ExecutionMode.EVALUATION
