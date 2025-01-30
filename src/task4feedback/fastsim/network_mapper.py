from .interface import Action, PythonMapper
import numpy as np
import torch as th
from torch.distributions.categorical import Categorical
from torch.distributions.utils import logits_to_probs
from typing import Optional, TypeVar, Union

MaybeMasks = Union[th.Tensor, np.ndarray, None]


class MaskableCategorical(Categorical):
    """
    Modified PyTorch Categorical distribution with support for invalid action masking.

    To instantiate, must provide either probs or logits, but not both.

    :param probs: Tensor containing finite non-negative values, which will be renormalized
        to sum to 1 along the last dimension.
    :param logits: Tensor of unnormalized log probabilities.
    :param validate_args: Whether or not to validate that arguments to methods like lob_prob()
        and icdf() match the distribution's shape, support, etc.
    :param masks: An optional boolean ndarray of compatible shape with the distribution.
        If True, the corresponding choice's logit value is preserved. If False, it is set to a
        large negative value, resulting in near 0 probability.
    """

    def __init__(
        self,
        probs: Optional[th.Tensor] = None,
        logits: Optional[th.Tensor] = None,
        validate_args: Optional[bool] = None,
        masks: MaybeMasks = None,
    ):
        self.masks: Optional[th.Tensor] = None
        super().__init__(probs, logits, validate_args)
        self._original_logits = self.logits
        self.apply_masking(masks)

    def apply_masking(self, masks: MaybeMasks) -> None:
        """
        Eliminate ("mask out") chosen categorical outcomes by setting their probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, it is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """

        if masks is not None:
            device = self.logits.device
            self.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(
                self.logits.shape
            )
            HUGE_NEG = th.tensor(-1e8, dtype=self.logits.dtype, device=device)

            logits = th.where(self.masks, self._original_logits, HUGE_NEG)
        else:
            self.masks = None
            logits = self._original_logits

        # Reinitialize with updated logits
        super().__init__(logits=logits)

        # self.probs may already be cached, so we must force an update
        self.probs = logits_to_probs(self.logits)

    def entropy(self) -> th.Tensor:
        if self.masks is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0, device=device))
        return -p_log_p.sum(-1)


class GreedyNetworkMapper(PythonMapper):

    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator):
        data = simulator.observer.local_graph_features(candidates)
        with torch.no_grad():
            p, d, v = self.model.forward(data)

            # choose argmax of network output
            # This is e-greedy policy
            p_per_task = torch.argmax(p, dim=1)
            dev_per_task = torch.argmax(d, dim=1)
            action_list = []
            for i in range(len(candidates)):
                a = Action(
                    candidates[i],
                    i,
                    dev_per_task[i].item(),
                    p_per_task[i].item(),
                    p_per_task[i].item(),
                )
                action_list.append(a)
        return action_list


class RandomNetworkMapper(PythonMapper):

    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator, output=None):
        data = simulator.observer.local_graph_features(candidates)

        with torch.no_grad():
            self.model.eval()
            p, d, v = self.model.forward(data)
            self.model.train()

            # sample from network output
            p_per_task, plogprob, _ = logits_to_actions(p)
            dev_per_task, dlogprob, _ = logits_to_actions(d)

            if output is not None:
                output["candidates"] = candidates
                output["state"] = data
                output["plogprob"] = plogprob
                output["dlogprob"] = dlogprob
                output["value"] = v
                output["pactions"] = p_per_task
                output["dactions"] = dev_per_task

            action_list = []
            for i in range(len(candidates)):

                if p_per_task.dim() == 0:
                    a = Action(
                        candidates[i],
                        i,
                        dev_per_task,
                        p_per_task,
                        p_per_task,
                    )
                else:
                    a = Action(
                        candidates[i],
                        i,
                        dev_per_task[i].item(),
                        p_per_task[i].item(),
                        p_per_task[i].item(),
                    )
                action_list.append(a)
        return action_list

    def evaluate(self, obs, daction, paction):
        p, d, v = self.model.forward(obs)
        _, plogprob, pentropy = logits_to_actions(p, paction)
        _, dlogprob, dentropy = logits_to_actions(d, daction)
        return (p, plogprob, pentropy), (d, dlogprob, dentropy), v
