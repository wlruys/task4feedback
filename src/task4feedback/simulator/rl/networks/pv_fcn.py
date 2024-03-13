import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class PVFCN(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.fcn1_indim = in_dim
        self.fcn1_outdim = max(64, in_dim * 2)
        self.fcn2_outdim = max(128, in_dim * 4)
        self.p_outdim = out_dim
        self.v_outdim = 1
        print("PVFCN is activated")

        self.common_fcn1 = Linear(self.fcn1_indim, self.fcn1_outdim,
                                  device=self.device)
        self.common_fcn2 = Linear(self.fcn1_outdim, self.fcn2_outdim,
                                  device=self.device)
  
        # P-network configuration
        self.p_fcn1 = Linear(self.fcn2_outdim, self.fcn2_outdim,
                             device=self.device)
        self.p_fcn2 = Linear(self.fcn2_outdim, self.fcn2_outdim,
                             device=self.device)
        self.p_out = Linear(self.fcn2_outdim, self.p_outdim,
                            device=self.device)
        # V-networkt configuration
        self.v_fcn1 = Linear(self.fcn2_outdim, self.fcn2_outdim,
                             device=self.device)
        self.v_fcn2 = Linear(self.fcn2_outdim, self.fcn2_outdim,
                             device=self.device)
        self.v_out = Linear(self.fcn2_outdim, self.v_outdim,
                            device=self.device)

    def forward(self, x):
        x = x.to(self.device)

        x = self.common_fcn1(x)
        x = self.common_fcn2(x)

        # p forward
        p = self.p_fcn1(x)
        p = F.leaky_relu(self.p_fcn2(p))
        p = self.p_out(p)

        # v forward
        v = self.v_fcn1(x)
        v = F.leaky_relu(self.v_fcn2(v))
        v = self.v_out(v)
        return p,v
