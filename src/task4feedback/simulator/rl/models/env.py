import copy
import torch

from collections import namedtuple
from functools import partial
from typing import Dict, List, Tuple
from itertools import chain

from ...utility import convert_to_float
from ...task import SimulatedTask, SimulatedComputeTask
from ....types import TaskState, TaskType, Device, Architecture, RLInfo
from .globals import *


def get_indegree(task: SimulatedTask) -> int:
    return len(task.dependencies)


def get_outdegree(task: SimulatedTask) -> int:
    return len(task.dependents)


class RLBaseEnvironment:
    pass


class RLEnvironment(RLBaseEnvironment):
  """
    *** State features ***

    1. Global information (2):
      * Completed tasks / total tasks
      * Relative current wall-clock time

    2. Task-specific information (5 + # devices):
      * In-degree / max in-degree
      * Out-degree / max out-degree
      * Task type ID
      * Expected exectuion time / max expected exectuion timm
      * Depth (from a root to the current task)
      * # parent tasks mapped per device / total # parent tasks

    3. Device/network information (3 * # devices):
      * Normalized resource usage (memory, VCUs)
      * # mapped tasks per device / # active tasks
      * Relative per-device idle time so far

  """

  def __init__(self, num_devices: int):
      self.task_property_offset = 2
      self.device_utilization_state_offset = 5 + num_devices
      self.device_state_len = 3 * num_devices
      self.state_dim = self.task_property_offset + \
                       self.device_utilization_state_offset + \
                       self.device_state_len
      self.out_dim = num_devices
      # print("state dimension:", self.state_dim)

  def create_state(self, target_task: SimulatedTask, rl_info: RLInfo, sched_state: "SystemState"):
      print("create state is called")
      current_state = torch.zeros(self.state_dim, dtype=torch.float)
      self.create_global_info(current_state, rl_info)
      self.create_task_property_info(current_state, target_task, rl_info, sched_state)
      self.create_device_utilization_info(current_state, rl_info)
      return current_state

  def create_global_info(self, current_state, rl_info: RLInfo):
      print(">> completed tasks:", rl_info.total_num_completed_tasks)
      print(">> total tasks:", rl_info.total_num_tasks)
      # % of completed tasks
      current_state[0] = rl_info.total_num_completed_tasks / rl_info.total_num_tasks \
                         if rl_info.total_num_tasks > 0 else 0
      print("[0]: ", current_state[0])
      # Relative wall clock time
      current_state[1] = 0
      print("[1]: ", current_state[1])

  def create_task_property_info(self, current_state, target_task: SimulatedTask,
                                rl_info: RLInfo, sched_state: "SystemState"):
      """
        ** Task-specific information (5 + # devices):
          * In-degree / max in-degree
          * Out-degree / max out-degree
          * Task type ID
          * Expected exectuion time / max expected exectuion timm
          * Depth (from a root to the current task)
          * # parent tasks mapped per device / total # parent tasks
      """
      offset = self.task_property_offset

      # In-degree 
      if rl_info.max_indegree > 0:
          current_state[offset] = get_indegree(target_task) / rl_info.max_indegree

      # Out-degree
      offset += 1
      if rl_info.max_outdegree > 0:
          current_state[offset] = get_outdegree(target_task) / rl_info.max_outdegree

      # Task function type
      offset += 1
      current_state[offset] = target_task.info.func_id

      # Task expected execution time
      offset += 1
      if rl_info.max_duration > 0:
          # Gets expected task duration time on the 0th device.
          # For RL, we assume that all devices require the same duration
          # for a task, which means that in real systems, it can be variable.
          task_duration = convert_to_float(
              sched_state.get_task_duration(
                  target_task,
                  target_task.info.runtime.locations[0])[0].scale_to("us"))
          print("scale:", task_duration, " type:", type(task_duration))
          print("max duration:", rl_info.max_duration, " type:", type(rl_info.max_duration))
          current_state[offset] = task_duration / rl_info.max_duration
      
      # Task depth
      offset += 1
      if rl_info.max_depth > 0:
          current_state[offset] = target_task.info.depth / rl_info.max_depth

      # Parent task distribution
      offset += 1
      taskmap = sched_state.objects.taskmap
      for pid in target_task.dependencies:
          p = taskmap[pid]
          if isinstance(p, SimulatedComputeTask):
              p_dev_id = p.assigned_devices[0].device_id
              print(p, " --> ", p_dev_id)
              current_state[offset + p_dev_id] += 1 
 
  def create_device_utilization_info(self, current_state, rl_info):
      pass

  def finalize_epoch(self, execution_time):
      pass

  def get_state_dim(self):
      return self.state_dim

  def get_out_dim(self):
      return self.out_dim
