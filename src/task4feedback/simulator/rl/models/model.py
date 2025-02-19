from abc import ABC, abstractmethod

import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F

import os


class RLModel(ABC):
    @abstractmethod
    def select_device(self, target_task, x, sched_state):
        NotImplementedError()

    @abstractmethod
    def optimize_model(self):
        NotImplementedError()

    @abstractmethod
    def load_model(self):
        NotImplementedError()

    @abstractmethod
    def save_model(self):
        NotImplementedError()

    @abstractmethod
    def start_episode(self):
        NotImplementedError()

    @abstractmethod
    def complete_episode(self):
        NotImplementedError()

    @abstractmethod
    def is_training_mode(self):
        NotImplementedError()
