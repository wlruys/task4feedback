import copy
import torch

from collections import namedtuple
from functools import partial
from typing import Dict, List, Tuple
from itertools import chain

from ...task import SimulatedTask
from ....legacy_types import TaskState, TaskType, Device, Architecture
from .globals import *

from .env_rl import *

# TODO(hc): make this an rl environment class


class READYSEnvironment(RLBaseEnvironment):
    # [Features]
    # 1. dependency per device (5, GCN)
    # 2. dependentedependency per state (6 due to dependents, GCN)
    # 3. num. of visible dependents (1, GCN)
    # 4. device per-state load (3 * 5 = 15, FCN)
    gcn_indim = 5
    fcn_indim = 1
    outdim = 4
    device_feature_dim = 3

    def __init__(self):
        self.max_heft = 0

    def get_heft_rank(self, taskmap, task):
        return taskmap[task].heft_rank

    def calculate_heft(self, tasklist, taskmap, devices):
        """
        Calculate HEFT (Heterogeneous Earliest Finish Time) for each task.
        This function assumes that the tasklist is already sorted by a topology.
        The below is the equation:

        HEFT_rank = task_duration + max(HEFT_rank(successors))
        """
        # Iterate from the leaf tasks, and assigns HEFT ranks
        for taskid in reversed(tasklist):
            task = taskmap[taskid]
            task_heft_agent = task.heft_agent
            max_dependent_rank = 0
            duration = 0
            # Get the max HEFT rank among dependent tasks
            for dependent_id in task.dependents:
                dependent = taskmap[dependent_id]
                max_dependent_rank = max(dependent.heft_rank, max_dependent_rank)
            duration = max(
                [
                    task_runtime_info.task_time
                    for task_runtime_info in task.get_runtime_info(
                        Device(Architecture.GPU, -1)
                    )
                ]
            )
            # Calculate the HEFT rank
            task.heft_rank = duration + max_dependent_rank

        heft_rank_returner = partial(self.get_heft_rank, taskmap)
        # Sort task list by heft rank
        heft_sorted_tasks = sorted(tasklist, key=heft_rank_returner)
        # print("heft sorted tasks:", heft_sorted_tasks)

        # After this, each task gets a heft rank (.heft_rank).
        agents = {agent.device_id: [] for agent in devices}
        # print("--->", agents)

        HEFTEvent = namedtuple("HEFTEvent", "task start end")
        # ft = lambda device:

        for taskid in reversed(heft_sorted_tasks):
            task = taskmap[taskid]
            duration = max(
                [
                    task_runtime_info.task_time
                    for task_runtime_info in task.get_runtime_info(
                        Device(Architecture.GPU, -1)
                    )
                ]
            )
            # Dependenices' makespan have already been calculated (should be).
            ready_time = 0
            if len(task.dependencies) > 0:
                ready_time = max(
                    [taskmap[dep].heft_makespan for dep in task.dependencies]
                )
            # Check second last task's end time and last task's start time.
            # If that gap fits to the target task's duration time, put that there.
            # If that gap doesn't fit, append the target task.
            earliest_start = -1.0
            earliest_start_agent = -1
            for agent_id in agents:
                agent = agents[agent_id]
                # TODO(hc): add communication time later
                if len(agent) > 0:
                    candidate_earliest_start = 0
                    any_slack_found = False
                    # Get the end time of the second last task; it tries to
                    # calculate a slack between this and the last task.
                    a = chain([HEFTEvent(None, None, 0)], agent[:-1])
                    for e1, e2 in zip(a, agent):
                        tmp_earliest_start = max(ready_time, e1.end)
                        if e2.start - tmp_earliest_start > duration:
                            # If the last and second lask tasks have enough slack,
                            # schedule that task to the slack.
                            candidate_earliest_start = tmp_earliest_start
                            any_slack_found = True
                            break
                    if not any_slack_found:
                        # If it failed to find a slack, append this task to the last task.
                        candidate_earliest_start = max(agent[-1].end, ready_time)
                else:
                    # If this agent (device) does not have mapped tasks, the earliest start
                    # time is 0.
                    candidate_earliest_start = 0

                if earliest_start == -1 or earliest_start > candidate_earliest_start:
                    earliest_start_agent = agent_id
                    earliest_start = candidate_earliest_start

            agents[earliest_start_agent].append(
                HEFTEvent(taskid, earliest_start, earliest_start + duration)
            )
            task.heft_makespan = earliest_start + duration
            if task.heft_makespan > self.max_heft:
                self.max_heft = task.heft_makespan
        """
      for key, value in agents.items():
         print("Key:", key)
         for vvalue in value:
             print("span:", taskmap[vvalue.task].heft_makespan, ", ", vvalue)
      """

    def create_gcn_task_workload_state(
        self,
        node_id_offset: int,
        target_task: SimulatedTask,
        devices: List,
        taskmap: Dict,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Create a state that shows task workload states.
        This function creates states not only of the current target task, but also
        its adjacent (possibly k-hops in the future) tasks, and its edge list.
        This state will be an input of the GCN layer.
        """
        lst_node_features = []
        lst_src_edge_index = []
        lst_dst_edge_index = []
        # Create a state of the current task, and append it to the features list.
        lst_node_features.append(
            self.create_task_workload_state(target_task, devices, taskmap)
        )
        # This function temporarily assigns an index to each task.
        # This should match the index on the node feature list and the edge list.
        node_id_offset += 1
        for dependency_id in target_task.dependencies:
            dependency = taskmap[dependency_id]
            # Add a dependency to the edge list
            lst_src_edge_index.append(node_id_offset)
            # 0th task is the target task
            lst_dst_edge_index.append(0)
            lst_node_features.append(
                self.create_task_workload_state(dependency, devices, taskmap)
            )
            node_id_offset += 1
        for dependent_id in target_task.dependents:
            dependent = taskmap[dependent_id]
            # 0th task is the target task
            lst_src_edge_index.append(0)
            # Add a dependent to the edge list
            lst_dst_edge_index.append(node_id_offset)
            lst_node_features.append(
                self.create_task_workload_state(dependent, devices, taskmap)
            )
            node_id_offset += 1
        edge_index = torch.Tensor([lst_src_edge_index, lst_dst_edge_index])
        edge_index = edge_index.to(torch.int64)
        node_features = torch.cat(lst_node_features)
        # Src/dst lists
        assert len(edge_index) == 2
        assert len(node_features) == node_id_offset
        return edge_index, node_features

    def create_task_workload_state(
        self, target_task: SimulatedTask, devices: List, taskmap: Dict
    ) -> torch.tensor:
        current_gcn_state = torch.zeros(self.gcn_indim)
        # 1) add number of successors
        num_succesors = len(target_task.dependents)

        # 2) add number of predecessors
        num_predecessors = len(target_task.dependencies)

        # 3) task type
        task_type = target_task.task_type

        # 4) ready or not (always 0)
        ready = 1

        # 5) Normalized F
        f_type = task_type
        for dependent_id in target_task.dependents:
            dependent = taskmap[dependent_id]
            dep_task_type = dependent.task_type
            num_dep_deps = len(dependent.dependencies)
            f_type = dep_task_type / float(num_dep_deps)

        current_gcn_state[0] = num_succesors
        current_gcn_state[1] = num_predecessors
        current_gcn_state[2] = task_type
        current_gcn_state[3] = ready
        current_gcn_state[4] = f_type
        print(
            f"task:{target_task}, 0:{num_succesors}, 1:{num_predecessors}, 2:{task_type}, 3:{ready}, 4:{f_type}"
        )
        return current_gcn_state.unsqueeze(0)

    def create_device_load_state(
        self,
        target_task: SimulatedTask,
        devices: List,
        reservable_tasks: Dict,
        launchable_tasks: Dict,
        launched_tasks: Dict,
    ) -> torch.tensor:
        """
        Create a state that shows devices' workload states.
        This state will be an input of the fully-connected layer.
        """
        current_state = torch.zeros(self.fcn_indim)
        # print("******** Create states:", target_task)
        idx = 0
        total_tasks = 0
        most_idle_device = -1
        most_idle_device_total_tasks = -1
        for device in devices:
            # Ignore CPU
            if device.architecture == Architecture.CPU:
                continue
            dev_num_tasks = (
                len(reservable_tasks[device])
                + len(launchable_tasks[device][TaskType.COMPUTE])
                + len(launchable_tasks[device][TaskType.DATA])
                + len(launched_tasks[device])
            )
            print("device:", device, ", id:", device.device_id, ", ", dev_num_tasks)
            if (
                most_idle_device_total_tasks == -1
                or dev_num_tasks < most_idle_device_total_tasks
            ):
                most_idle_device_total_tasks = dev_num_tasks
                most_idle_device = device
        current_state[0] = most_idle_device.device_id
        print(
            f"task:{target_task}, 0:{most_idle_device}, total: {most_idle_device_total_tasks}"
        )
        return current_state

    def create_state(
        self, target_task: SimulatedTask, devices: List, taskmap: Dict, parla_arch
    ):
        """
        Create the current state.
        The state consists of two feature types:
        1) Device load state: How many tasks of each state are mapped to each device
        2) Task workload state: How many dependencies are on each state, and how many
                                dependent tasks this task has?
        """
        spawned_tasks = parla_arch.spawned_tasks
        mappable_tasks = parla_arch.mappable_tasks
        reservable_tasks = parla_arch.reservable_tasks
        launchable_tasks = parla_arch.launchable_tasks
        launched_tasks = parla_arch.launched_tasks
        current_device_load_state = self.create_device_load_state(
            target_task, devices, reservable_tasks, launchable_tasks, launched_tasks
        )
        edge_index, current_workload_features = self.create_gcn_task_workload_state(
            0, target_task, devices, taskmap
        )
        return current_device_load_state, edge_index, current_workload_features

    def inspect_reward(self, task, completion_time):
        pass

    def calculate_reward(self, task, completion_time):
        print("task:", task, " assigned devices:", task.assigned_devices)
        if task.heft_makespan == 0 or task.info.is_terminal == False:
            print("task heft mksp:", task.heft_makespan)
            return torch.tensor([[0]], dtype=torch.float)
        else:
            # reward = (self.max_heft - completion_time) / self.max_heft
            # print("task heft mksp:", self.max_heft, " vs ", completion_time, " = reward:", reward)
            reward = (task.heft_makespan - completion_time) / task.heft_makespan
            print(
                "task heft mksp:",
                task.heft_makespan,
                " vs ",
                completion_time,
                " = reward:",
                reward,
            )
            return torch.tensor([[reward]], dtype=torch.float)

    def finalize_epoch(self, execution_time):
        pass
