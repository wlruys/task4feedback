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
          for device in devicemap:
              # Ignore CPU device
              if device.architecture == Architecture.CPU:
                  continue

              workload = system_state.perdev_active_workload[device]
              action_probs[device.device_id] = workload
              total_workload += workload

          for device in devicemap:
              if device.architecture == Architecture.CPU:
                  continue;

              if total_workload > 0:
                  action_probs[device.device_id] = (
                      1 - action_probs[device.device_id].item() / total_workload) / (
                      len(devicemap) - 2)
              else:
                  action_probs[device.device_id] = 1 / (len(devicemap) - 1)

      return action_probs
