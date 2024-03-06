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
      max_worload: float = -1
      with torch.no_grad():
          for device in devicemap:
              # Ignore CPU device
              if device.architecture == Architecture.CPU:
                  continue

              workload = system_state.perdev_active_workload[device]
              max_worload = max(max_worload, workload)

          for device in devicemap:
              # Ignore CPU device
              if device.architecture == Architecture.CPU:
                  continue

              business = max_worload - system_state.perdev_active_workload[device]
              total_workload += business
              action_probs[device.device_id] = business
          
          for device in devicemap:
              if device.architecture == Architecture.CPU:
                  continue;

              if total_workload > 0:
                  action_probs[device.device_id] = action_probs[device.device_id].item() / total_workload

      return action_probs
