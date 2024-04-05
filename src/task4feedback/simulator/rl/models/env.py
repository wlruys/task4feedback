import copy
import torch

from collections import namedtuple
from functools import partial
from typing import Dict, List, Tuple
from itertools import chain

from .globals import *

from ...utility import convert_to_float
from ...task import SimulatedTask, SimulatedComputeTask

from ....types import TaskState, TaskType, Device, Architecture
from ....logging import logger


class RLBaseEnvironment:
    pass


class RLEnvironment(RLBaseEnvironment):
  """
    *** RL state features ***

    1. Global information (2):
      * Completed tasks / total tasks
      * Relative current wall-clock time (TODO(hc))

    2. Task-specific information (4 + # types + # devices):
      * In-degree / max in-degree
      * Out-degree / max out-degree
      * Task type ID (parameter)
      * Expected exectuion time / max expected exectuion timm
      * Depth (from a root to the current task)
      * # parent tasks mapped per device / total # parent tasks

    3. Device/network information (3 * # devices):
      * Normalized resource usage (memory, VCUs) (TODO(hc))
      * # mapped tasks per device / # active tasks
      * Relative per-device idle time so far (TODO(hc))

  """


  def __init__(self, num_devices: int):
      self.task_property_offset = 2
      self.num_task_types = 4
      self.task_property_len = 4 + self.num_task_types + num_devices
      self.device_utilization_state_offset = self.task_property_offset + \
                                             self.task_property_len
      self.device_state_len = 3 * num_devices
      self.state_dim = self.device_utilization_state_offset + \
                       self.device_state_len
      self.out_dim = num_devices
      self.num_devices = num_devices
      if logger.ENABLE_LOGGING:
          logger.runtime.info(f"RL state dimension: {self.state_dim}.")


  def create_state(
      self, target_task: SimulatedTask, simulator,
      mapper: "RLTaskMapper"
  ):
      current_state = torch.zeros(self.state_dim, dtype=torch.float)
      self.create_global_info(current_state, simulator, mapper)
      self.create_task_property_info(current_state, target_task, simulator, mapper)
      self.create_device_utilization_info(current_state, simulator)
      return current_state


  def create_global_info(
      self, current_state, simulator, mapper: "RLTaskMapper"
  ):
      """
        ** Global information (2):
          * Completed tasks / total tasks (TODO(hc))
          * Relative current wall-clock time (TODO(hc))
      """

      sched_state = simulator.state

      # % of completed tasks
      current_state[0] = sched_state.total_num_completed_tasks / mapper.total_num_tasks \
                         if mapper.total_num_tasks > 0 else 0

      # Relative wall clock time
      target_exec_time = sched_state.target_exec_time
      # print("Current time:",convert_to_float(sched_state.time.scale_to("ms")))
      # print("Target time:", target_exec_time)
      current_state[1] = convert_to_float(sched_state.time.scale_to("ms")) / (5 * target_exec_time) \
                         if target_exec_time > 0 else 0

      if logger.ENABLE_LOGGING:
          logger.runtime.debug(f"RL state [0]: {current_state[0].item()} "
                               f"(# completed tasks: {sched_state.total_num_completed_tasks}"
                               f", # total tasks: {mapper.total_num_tasks}).")
          logger.runtime.debug(f"RL state [1]: {current_state[1].item()} (wall clock).")


  def create_task_property_info(
      self, current_state, target_task: SimulatedTask, simulator, mapper: "RLTaskMapper"
  ):
      """
        ** Task-specific information (5 + # devices):
          * In-degree / max in-degree
          * Out-degree / max out-degree
          * Task type ID
          * Expected exectuion time / max expected exectuion timm
          * Depth (from a root to the current task)
          * # parent tasks mapped per device / total # parent tasks
      """

      sched_state = simulator.state

      offset = self.task_property_offset

      # In-degree 
      if mapper.max_indegree > 0:
          current_state[offset] = mapper.get_indegree(target_task) / mapper.max_indegree
      if logger.ENABLE_LOGGING:
          logger.runtime.debug(f"RL state [{offset}]: {current_state[offset].item()} "
                               f"(indegree {mapper.get_indegree(target_task)}, "
                               f"max indegree {mapper.max_indegree}).")

      # Out-degree
      offset += 1
      if mapper.max_outdegree > 0:
          current_state[offset] = mapper.get_outdegree(target_task) / mapper.max_outdegree
      if logger.ENABLE_LOGGING:
          logger.runtime.debug(f"RL state [{offset}]: {current_state[offset].item()} "
                               f"(outdegree {mapper.get_outdegree(target_task)}, "
                               f"max outdegree {mapper.max_outdegree}).")

      # Task function type
      offset += 1
      current_state[offset + target_task.info.func_id] = 1
      if logger.ENABLE_LOGGING:
          logger.runtime.debug(f"RL state [{offset}]: Task type vector base.")
          logger.runtime.debug(f"RL state [{offset + target_task.info.func_id}]: "
                               f"{current_state[offset + target_task.info.func_id].item()} "
                               f"(task func id {target_task.info.func_id}).")

      # Task expected execution time
      offset += self.num_task_types
      if mapper.max_duration > 0:
          # Gets expected task duration time on the 0th device.
          # For RL, we assume that all devices require the same duration
          # for a task, which means that in real systems, it can be variable.
          task_duration = convert_to_float(
              sched_state.get_task_duration(
                  target_task,
                  target_task.info.runtime.locations[0]).scale_to("us"))
          current_state[offset] = task_duration / mapper.max_duration
          if logger.ENABLE_LOGGING:
              logger.runtime.debug(f"RL state [{offset}]: "
                                   f"{current_state[offset].item()} "
                                   f"(duration {task_duration} "
                                   f"max duration: {mapper.max_duration}).")
      
      # Task depth
      offset += 1
      if mapper.max_depth > 0:
          current_state[offset] = target_task.info.depth / mapper.max_depth
      if logger.ENABLE_LOGGING:
          logger.runtime.debug(f"RL state [{offset}]: "
                               f"{current_state[offset].item()} "
                               f"(depth {target_task.info.depth} "
                               f"max depth: {mapper.max_depth}).")

      # Parent task distribution
      offset += 1
      taskmap = sched_state.objects.taskmap
      for pid in target_task.dependencies:
          p = taskmap[pid]
          if isinstance(p, SimulatedComputeTask):
              p_dev_id = p.assigned_devices[0].device_id
              current_state[offset + p_dev_id] += 1 
              if logger.ENABLE_LOGGING:
                  logger.runtime.debug(f"RL state [{offset + p_dev_id}]: "
                                       f"{current_state[offset + p_dev_id].item()} "
                                       f"(# parent tasks at device {p_dev_id}).")
      # Normalization
      devicemap = sched_state.objects.devicemap
      for device in devicemap:
          # Ignore CPU device
          if device.architecture == Architecture.CPU:
              continue

          dev_id = device.device_id
          if mapper.get_indegree(target_task) > 0:
              current_state[offset + dev_id] = \
                  current_state[offset + dev_id].item() / mapper.get_indegree(target_task)
          if logger.ENABLE_LOGGING:
              logger.runtime.debug(f"RL state [{offset + dev_id}]: "
                                   f"{current_state[offset + dev_id].item()} / {mapper.get_indegree(target_task)}."
                                   f" (normalization)")

 
  def create_device_utilization_info(self, current_state, simulator):
      """
        ** Device/network information (3 * # devices):
          * Normalized resource usage (memory, VCUs)
          * # mapped tasks per device / # active tasks
          * Relative per-device idle time so far
      """

      sched_state = simulator.state
      sched_arch = simulator.mechanisms

      offset = self.device_utilization_state_offset

      # Resource usage
      if logger.ENABLE_LOGGING:
          logger.runtime.debug(f"RL state [{offset}]: "
                               f"{current_state[offset].item()} (Resource usage).")

      # Num mapped tasks per device (normalized by the total number of active tasks)
      offset += self.num_devices
      if logger.ENABLE_LOGGING:
          logger.runtime.debug(f"RL state [{offset}]: "
                               f"{current_state[offset].item()} (Per-device workload).")
      devicemap = sched_state.objects.devicemap
      for device in devicemap:
          # Ignore CPU device
          if device.architecture == Architecture.CPU:
              continue

          dev_id = device.device_id
          current_state[dev_id + offset] = sched_state.perdev_active_workload[device]
          if sched_state.total_active_workload > 0:
              current_state[dev_id + offset] = \
                  current_state[dev_id + offset].item() / sched_state.total_active_workload
          if logger.ENABLE_LOGGING:
              logger.runtime.debug(f"RL state [{dev_id + offset}]: "
                                   f"{current_state[dev_id + offset].item()} "
                                   f"(# tasks on device {dev_id} "
                                   f", total # active tasks: {sched_state.total_active_workload}).")

      # Relative per-device idle time so far
      offset += self.num_devices
      target_exec_time = sched_state.target_exec_time

      for device, device_instance in devicemap.items():
          # Ignore CPU device
          if device.architecture == Architecture.CPU:
              continue

          dev_id = device.device_id
          is_nothing_launched = sched_arch.is_nothing_launched[device]
          if is_nothing_launched:
              idle_time = float(sched_state.time.scale_to("ms"))
              print("sched time:", sched_state.time, " idle time:", idle_time)
          else:
              idle_time = convert_to_float(device_instance.stats.idle_time.scale_to("ms"))
          norm_idle_time = (idle_time / (5 * target_exec_time) if target_exec_time else 0)
          current_state[dev_id + offset] = norm_idle_time
          if logger.ENABLE_LOGGING:
              logger.runtime.debug(f"RL state [{dev_id + offset}]: "
                                   f"{current_state[dev_id + offset].item()} "
                                   f"(device {dev_id} "
                                   f"idle time: {idle_time}, target exect ime: {target_exec_time} "
                                   f"norm idle time: {norm_idle_time}, flag: {is_nothing_launched}) ")

  def finalize_epoch(self, execution_time):
      pass

  def get_state_dim(self):
      return self.state_dim

  def get_out_dim(self):
      return self.out_dim
