import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class A2CNetworkNoGCN(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.fcn1_indim = in_dim
        # self.fcn1_outdim = max(128, in_dim * 4)
        # self.fcn2_outdim = max(256, in_dim * 8)
        self.fcn1_outdim = max(64, in_dim * 2)
        self.fcn2_outdim = max(128, in_dim * 4)
        self.actor_outdim = out_dim
        self.critic_outdim = 1
        # Actor configuration
        self.actor_fcn1 = Linear(self.fcn1_indim, self.fcn1_outdim,
                                 device=self.device)
        self.actor_fcn2 = Linear(self.fcn1_outdim, self.fcn2_outdim,
                                 device=self.device)
        self.actor_out = Linear(self.fcn2_outdim, self.actor_outdim,
                                device=self.device)
        # Critic configuration
        self.critic_fcn1 = Linear(self.fcn1_indim, self.fcn1_outdim,
                                  device=self.device)
        self.critic_fcn2 = Linear(self.fcn1_outdim, self.fcn2_outdim,
                                  device=self.device)
        self.critic_out = Linear(self.fcn2_outdim, self.critic_outdim,
                                 device=self.device)

    def forward(self, x):
        x = x.to(self.device)
        # Actor forward
        a = self.actor_fcn1(x)
        # a = F.leaky_relu(a)
        # a = F.leaky_relu(self.actor_fcn2(a))
        a = F.leaky_relu(self.actor_fcn2(a))
        a = self.actor_out(a)

        # Critic forward
        # c = F.leaky_relu(self.critic_fcn1(x))
        c = self.critic_fcn1(x)
        c = F.leaky_relu(self.critic_fcn2(c))
        c = self.critic_out(c)
        return a,c
