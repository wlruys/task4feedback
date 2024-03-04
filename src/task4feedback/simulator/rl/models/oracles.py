from abc import ABC,abstractmethod

import torch

from ....types import Device, Architecture

class OraclePolicy(ABC):

  def get_action(self, system_state: "SystemState"):
      NotImplementedError()


class LoadbalancingPolicy(ABC):


  def get_action(self, system_state: "SystemState"):
      """
      This method calculates action probabilities from load-balancing based
      heursitic policy.
      """

      devicemap = system_state.objects.devicemap

      action_probs = torch.zeros(len(devicemap) - 1)
      total_workload: float = 0

      with torch.no_grad():
          need_further_norm: bool = False
          for device in devicemap:
              # Ignore CPU device
              if device.architecture == Architecture.CPU:
                  continue

              workload = system_state.perdev_active_workload[device]

              if workload == 0:
                  # If any workload is 0, the probs should be renormalized
                  # to make the sum 1.
                  need_further_norm = True
              total_workload += workload
              action_probs[device.device_id] = workload

          total_workload_for_next_norm = 0
          for device in devicemap:
              if device.architecture == Architecture.CPU:
                  continue;

              if total_workload > 0:
                  action_probs[device.device_id] = (1 - action_probs[device.device_id].item() / total_workload)
                  if need_further_norm:
                      total_workload_for_next_norm += action_probs[device.device_id].item()
              else:
                  # If all workload is 0 (so all devices are idle), assign 1/(#devices) as its action prob.
                  action_probs[device.device_id] = 1 / float(len(devicemap) - 1)
          if total_workload > 0 and need_further_norm:
              for device in devicemap:
                  if device.architecture == Architecture.CPU:
                      continue;

                  action_probs[device.device_id] = action_probs[device.device_id].item() / total_workload_for_next_norm

      return action_probs
