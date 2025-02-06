import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm


class PVFCN(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fcn1_indim = in_dim
        self.fcn1_outdim = max(64, in_dim * 2)
        self.fcn2_outdim = max(128, in_dim * 4)
        self.p_outdim = out_dim
        self.v_outdim = 1
        print("PVFCN is activated")

        self.common_fcn1 = Linear(self.fcn1_indim, self.fcn1_outdim, device=self.device)
        self.nl1 = LayerNorm(self.fcn1_outdim, device=self.device)
        self.common_fcn2 = Linear(
            self.fcn1_outdim, self.fcn2_outdim, device=self.device
        )
        self.nl2 = LayerNorm(self.fcn2_outdim, device=self.device)
        self.common_fcn3 = Linear(
            self.fcn2_outdim, self.fcn2_outdim, device=self.device
        )
        self.nl3 = LayerNorm(self.fcn2_outdim, device=self.device)
        self.common_fcn4 = Linear(
            self.fcn2_outdim, self.fcn2_outdim, device=self.device
        )
        self.nl4 = LayerNorm(self.fcn2_outdim, device=self.device)

        # P-network configuration
        self.p_fcn1 = Linear(self.fcn2_outdim, self.fcn2_outdim, device=self.device)
        self.p_nl1 = LayerNorm(self.fcn2_outdim, device=self.device)
        self.p_fcn2 = Linear(self.fcn2_outdim, self.fcn2_outdim, device=self.device)
        self.p_nl2 = LayerNorm(self.fcn2_outdim, device=self.device)
        self.p_out = Linear(self.fcn2_outdim, self.p_outdim, device=self.device)

        # V-networkt configuration
        self.v_fcn1 = Linear(self.fcn2_outdim, self.fcn2_outdim, device=self.device)
        self.v_nl1 = LayerNorm(self.fcn2_outdim, device=self.device)
        self.v_out = Linear(self.fcn2_outdim, self.v_outdim, device=self.device)

    def forward(self, x):
        x = x.to(self.device)

        x = F.leaky_relu(self.nl1(self.common_fcn1(x)))
        x = F.leaky_relu(self.nl2(self.common_fcn2(x)))
        x = F.leaky_relu(self.nl3(self.common_fcn3(x)))
        x = F.leaky_relu(self.nl4(self.common_fcn4(x)))

        # p forward
        p = F.leaky_relu(self.p_nl1(self.p_fcn1(x)))
        p = F.leaky_relu(self.p_nl2(self.p_fcn2(p)))
        p = self.p_out(p)

        # v forward
        v = F.leaky_relu(self.v_nl1(self.v_fcn1(x)))
        v = self.v_out(v)
        return p, v
