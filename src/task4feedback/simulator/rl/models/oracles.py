from abc import ABC,abstractmethod

import torch

from ....types import Device, Architecture

class OraclePolicy(ABC):

  def get_action(self, system_state: "SystemState"):
      NotImplementedError()


class LoadbalancingPolicy(ABC):


  def get_action(self, system_state: "SystemState"):

      devicemap = system_state.objects.devicemap

      action_probs = torch.zeros(len(devicemap) - 1)
      total_workload: float = 0

      with torch.no_grad():

          for device in devicemap:
              # Ignore CPU device
              if device.architecture == Architecture.CPU:
                  continue

              workload = system_state.perdev_active_workload[device]
              total_workload += workload
              action_probs[device.device_id] = workload

          if total_workload > 0:
              for device in devicemap:
                  if device.architecture == Architecture.CPU:
                      continue;

                  action_probs[device.device_id] = (1 - action_probs[device.device_id].item() / total_workload)

      return action_probs
