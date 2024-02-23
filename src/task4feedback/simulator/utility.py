from collections import namedtuple
from functools import partial
from itertools import chain
from typing import Tuple, Dict

from .task import SimulatedDataTask
from ..types import Device, Architecture, TaskState, AccessType

import bisect
import random

units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12} 

def parse_size(size_str: str):
    number, unit = [string.strip() for string in size_str.split()]
    return int(float(number) * units[unit])


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def calculate_heft(
    tasklist, taskmap, num_devices: int,
    scheduler_state: "SystemState", in_place_update: bool = False
) -> float:
    """
    Calculate HEFT (Heterogeneous Earliest Finish Time) for each task.
    This function assumes that the tasklist is already sorted by a topology.
    The below is the equation:

    HEFT_rank = task_duration + max(HEFT_rank(successors))
    """

    def get_heft_rank(task):
        return task.info.heft_rank

    # Calculate HEFT ranks from bottom to top
    for task in reversed(tasklist):
        max_dependent_rank = 0

        # Do not calculate data move task's HEFT rank, but
        # calculate only compute task's HEFT rank.
        # Compute task's HEFT rank considers data transfer
        # overhead.
        if isinstance(task, SimulatedDataTask):
            continue

        # Get task's data with write and rw permission
        if task.info.data_dependencies is not None:
            write = task.info.data_dependencies[AccessType.WRITE]
            rw = task.info.data_dependencies[AccessType.READ_WRITE]
            src_data = write + rw

        # Get dependents' data and dependencies between them and the task
        # being mapped. Note that the dependencies are only between
        # write/rw permission data of the task and data of the dependents.
        for dep in task.dependents:
            dependent_instance = taskmap[dep]

            if isinstance(dependent_instance, SimulatedDataTask):
                continue

            # Compute communication overhead
            all_dep_data = dependent_instance.info.data_dependencies.all_ids()
            # intersected_data = []
            intersected_data_size = 0
            for sd in src_data:
                for dd in all_dep_data:
                    if sd.id == dd:
                        # intersected_data.append(sd)
                        intersected_data_size += scheduler_state.objects.datamap[sd.id].size 

            # Change it to gb
            average_comm_time : float = 0
            num_pairs: float = 0
            for d1 in range(len(scheduler_state.topology.devices)):
                for d2 in range(len(scheduler_state.topology.devices)):
                    if d1 == d2 or d1 == 0 or d2 == 0:
                        continue
                    bandwidth = scheduler_state.topology.connection_pool.bandwidth[d1, d2]
                    # TODO(hc): I assume that all GPUs have HW connections through either
                    # NVLink or P2P. But we can still use HW topo that might not have connections 
                    # between some GPUs, and in this case, we need to move data through host.
                    # But for now, I am not considering that case yet.
                    if bandwidth > 0:
                        # Change the unit to ms
                        curr_average_comm_time = (intersected_data_size / bandwidth) * 1000
                        average_comm_time += curr_average_comm_time
                        num_pairs += 1
            average_comm_time /= num_pairs

            # print(task, " vs ", dependent_instance, " src data:", src_data, " all data:", all_dep_data)
            # print(">> size:", intersected_data_size, " average comm time:",
            #      average_comm_time, " data size:", intersected_data_size)

            # Upward calculation
            max_dependent_rank = max(dependent_instance.info.heft_rank + average_comm_time,
                                     max_dependent_rank) 
               
        duration = convert_to_float(
            scheduler_state.get_task_duration(task, task.info.runtime.locations[0]).
            scale_to("ms"))

        # Calculate the HEFT rank
        task.info.heft_rank = duration + max_dependent_rank

    # Sort task list by heft rank
    heft_sorted_tasks = reversed(sorted(tasklist, key=get_heft_rank))

    agents = {agent: [] for agent in range(0, num_devices)}

    HEFTEvent = namedtuple('HEFTEvent', 'task start end')

    def get_start_time(heft_event):
        return heft_event.start

    max_heft = -1
    if in_place_update:
        tasklist[:] = []
        heft_events = []
    # Forward phase to allocate each task to each device
    for task in heft_sorted_tasks:
        duration = convert_to_float(
            scheduler_state.get_task_duration(task, task.info.runtime.locations[0]).
            scale_to("ms"))

        ready_time = 0
        earliest_start = -1.0 
        earliest_start_agent = -1
        # Try to insert each task to each agent (device)
        for agent_id, agent in agents.items():

            # Iterate all dependencies and check rank + communication overhead
            # and find task's ready time when the task is assigned to the current
            # agent
            for dep in task.dependencies:
                dependent_instance = taskmap[dep]

                if isinstance(dependent_instance, SimulatedDataTask):
                    continue

                # TODO(hc): We can reuse this information collected in the above later
                # Compute communication overhead
                all_dep_data = dependent_instance.info.data_dependencies.all_ids()
                intersected_data = []
                intersected_data_size = 0
                for sd in src_data: # SimulatedData
                    for dd in all_dep_data: # DataID
                        if sd.id == dd:
                            intersected_data.append(sd)
                            intersected_data_size += scheduler_state.objects.datamap[sd.id].size 

                assigned_agent_id = dependent_instance.info.heft_allocation

                # Calculate data transfer time from the dependency's device to
                # the current device (CPU is index 0)
                bandwidth = scheduler_state.topology.connection_pool.bandwidth[assigned_agent_id + 1, agent_id + 1]
                # milliseconds! 
                comm_time : float = (intersected_data_size / bandwidth) * 1000 if bandwidth > 0 else 0
                ready_time = max(taskmap[dep].info.heft_makespan + comm_time,
                                 ready_time)

                # print(">> size:", intersected_data_size, " actual comm time:", comm_time)


            # Find the earliest start time on this agent
            if len(agent) > 0:
                candidate_earliest_start = 0
                any_slack_found = False

                # Check second last task's end time and last task's start time.
                # If that gap fits to the target task's execution time + communication time, schedule it.
                # If that gap doesn't fit, schedule it after the last task
                a = chain([HEFTEvent(None, None, 0)], agent[:-1])
                for e1, e2 in zip(a, agent):
                    tmp_earliest_start = max(ready_time, e1.end)
                    if e2.start - tmp_earliest_start > duration:
                        # If the last and second lask tasks have enough slack,
                        # schedule that task to the slack.
                        candidate_earliest_start = tmp_earliest_start
                        any_slack_found = True
                        # print(task.info.id, " earliest start:", tmp_earliest_start, " e2 start:",
                        #     e2.start, " duration: ", duration, " on device", agent_id)
                        break 

                if not any_slack_found:
                    candidate_earliest_start = max(agent[-1].end, ready_time)
                    # print(task.info.id, " earlist estart:", candidate_earliest_start)
            else:
                # If this agent (device) does not have mapped tasks, the earliest start
                # time is 0.
                candidate_earliest_start = 0

            if earliest_start == -1 or earliest_start > candidate_earliest_start:
                earliest_start_agent = agent_id
                earliest_start = candidate_earliest_start

        heft_event = HEFTEvent(task, earliest_start, earliest_start + duration)
        if in_place_update:
            heft_events.append(heft_event)
        bisect.insort(agents[earliest_start_agent], heft_event, key=lambda x: x.start)
        task.info.heft_makespan = earliest_start + duration
        task.info.heft_allocation = earliest_start_agent
        if task.info.heft_makespan > max_heft:
            max_heft = task.info.heft_makespan

    """
    for key, value in agents.items():
        test = sorted(value, key=lambda x: x.start)
        print("Key:", key)
        for t in test:
          print(t.task.info.id, " :: ", t.start, " = ", t.end)
    """

    if in_place_update:
        heft_events = sorted(heft_events, key=get_start_time)
        tasklist[:] = [he.task for he in heft_events]
        order = 0
        for task in tasklist:
            task.info.order = order
            order += 1
    print("HEFT time:", max_heft)

    """
    for key, value in agents.items():
       print("Key:", key)
       for vvalue in value:
           print("span:", vvalue.task.info.id, ", ", vvalue.start, " ~ ", vvalue.end)
    for t in tasklist:
        print(t.info.id, "...")
    """
    return max_heft


def random_mapping(
    task: "SimulatedTask", sched_state: "SystemState"
) -> Tuple[Device, ...]:
    """
    """
    # -1 is not supported
    # devices = task.info.runtime.locations
    devices = sched_state.topology.devices
    gpu_devices = []
    for device in devices:
        if device.name.architecture != Architecture.CPU:
            gpu_devices.append(device.name)
    random.shuffle(gpu_devices)
    device = gpu_devices[0]

    if not isinstance(device, Tuple):
        device = (device,)

    return device


def load_balancing_mapping(
    task: "SimulatedTask", sched_state: "SystemState"
) -> Tuple[Device, ...]:
    """
    """
    devices = sched_state.topology.devices
    lowest_workload = 99999999999999
    best_device = None

    for device in devices:
        if device.name.architecture == Architecture.CPU:
            continue

        workload = sched_state.perdev_active_workload[device.name]
        if best_device is None or workload < lowest_workload:
            lowest_workload = workload
            best_device = device

    return (best_device.name,)


def parla_mapping(
    task: "SimulatedTask", sched_state: "SystemState"
) -> Tuple[Device, ...]:
    """
    """
    devices = sched_state.topology.devices
    taskmap = sched_state.objects.taskmap
    datamap = sched_state.objects.datamap
    best_score = -1
    best_device = -1
    total_workload = 0
    for device in devices:
        workload = sched_state.perdev_active_workload[device.name]
        total_workload += workload

    for device in devices:
        if device.name.architecture == Architecture.CPU:
            continue

        workload = sched_state.perdev_active_workload[device.name]
        norm_workload = (workload / total_workload
                         if total_workload != 0 else workload)

        local_data = 0
        nonlocal_data = 0
        total_data = 0
        if task.data_tasks is not None:
            for dtask_id in task.data_tasks:
                # print("task ", task, " did:", dtask_id)
                dtask = taskmap[dtask_id]
                # print("dtask:", dtask)
                for data_id in dtask.info.data_dependencies.all_ids():
                    data = datamap[data_id]
                    # print("data:", type(data))
                    valid = data.is_valid(device.name, TaskState.MAPPED)
                    local_data += data.size if valid else 0
                    nonlocal_data += data.size if not valid else 0
                    total_data += data.size
                # print("device:", device.name, ", validity:", valid, " local size:", local_data,
                #       " unlocal data:", nonlocal_data)
        local_data = local_data / total_data if total_data > 0 else local_data
        nonlocal_data = nonlocal_data / total_data if total_data > 0 else nonlocal_data
        score = 50 + (30 * local_data - 30 * nonlocal_data - 10 * norm_workload)
        if score > best_score:
            best_score = score
            best_device = device
        # print("device:", device, " score:", score)
    # print("best device:", best_device)
    return (best_device.name,)


def load_task_noise() -> Dict[str, int]:
    loaded_task_noise = dict()
    print("task noise loading..")
    with open("replay.noise", "r") as fp:
        lines = fp.readlines()
        for l in lines:
            l = l.rstrip()
            unpacked = l.split(":")
            key = unpacked[0]
            val = unpacked[1]
            print(">>", key, ", ", val)
            loaded_task_noise[key] = val
    return loaded_task_noise

    
def save_task_noise(task: "SimulatedTask", noise: "Time"):
    with open("replay.noise", "a") as fp:
        noise_value: int = noise.scale_to("us")
        fp.write(str(task.name) + ":" + str(noise_value) + "\n")
