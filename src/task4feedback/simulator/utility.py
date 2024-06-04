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
        num, denom = frac_str.split("/")
        try:
            leading, num = num.split(" ")
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def calculate_heft_upward_rank(tasklist, taskmap, scheduler_state):

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

        # Get task's data dependencies
        if task.info.data_dependencies is not None:
            read = task.info.data_dependencies[AccessType.READ]
            write = task.info.data_dependencies[AccessType.WRITE]
            rw = task.info.data_dependencies[AccessType.READ_WRITE]
            src_data = read + write + rw

        # Get dependents' data and dependencies between them and the task
        # being mapped.
        for dep in task.dependents:
            dependent_instance = taskmap[dep]

            if isinstance(dependent_instance, SimulatedDataTask):
                continue

            # Compute communication overhead
            all_dep_data = dependent_instance.info.data_dependencies.all_ids()
            intersected_data_size = 0
            for sd in src_data:
                for dd in all_dep_data:
                    if sd.id == dd:
                        intersected_data_size += scheduler_state.objects.datamap[
                            sd.id
                        ].size
            # print("intersected data size:", intersected_data_size, " task:", task.name)

            average_comm_time: float = 0
            num_pairs: float = 0
            for d1 in scheduler_state.topology.devices:
                for d2 in scheduler_state.topology.devices:
                    # TODO(hc): Remove CPU condition if we start to consider CPU in the future
                    if (
                        d1.name.architecture == Architecture.CPU
                        or d2.name.architecture == Architecture.CPU
                        or d1 == d2
                    ):
                        continue
                    average_comm_time += float(
                        scheduler_state.topology.get_transfer_time(
                            d1, d2, intersected_data_size
                        ).scale_to("ms")
                    )
                    num_pairs += 1
            average_comm_time /= num_pairs

            # print(task, " vs ", dependent_instance, " src data:", src_data, " all data:", all_dep_data)
            # print(">> size:", intersected_data_size, " average comm time:",
            #      average_comm_time, " data size:", intersected_data_size)

            # Upward calculation
            max_dependent_rank = max(
                dependent_instance.info.heft_rank + average_comm_time,
                max_dependent_rank,
            )

        duration = convert_to_float(
            scheduler_state.get_task_duration(
                task, task.info.runtime.locations[0]
            ).scale_to("ms")
        )

        # Calculate the HEFT rank
        task.info.heft_rank = duration + max_dependent_rank

    # Sort task list by heft rank
    return reversed(sorted(tasklist, key=get_heft_rank))


def map_task_heft(
    tasklist, taskmap, num_devices,
    scheduler_state, update_task_order,
    heft_sorted_tasks, consider_init_placement
):
    agents = {agent: [] for agent in range(0, num_devices)}

    HEFTEvent = namedtuple("HEFTEvent", "task start end")

    def get_start_time(heft_event):
        return heft_event.start

    max_heft = -1
    if update_task_order:
        tasklist[:] = []
        heft_events = []
    # Forward phase to allocate each task to each device
    for task in heft_sorted_tasks:
        duration = convert_to_float(
            scheduler_state.get_task_duration(
                task, task.info.runtime.locations[0]
            ).scale_to("ms")
        )

        # Get task's data dependencies
        if task.info.data_dependencies is not None:
            read = task.info.data_dependencies[AccessType.READ]
            write = task.info.data_dependencies[AccessType.WRITE]
            rw = task.info.data_dependencies[AccessType.READ_WRITE]
            src_data = read + write + rw

        ready_time = 0
        earliest_start = None
        earliest_start_agent = -1

        # Try to insert each task to each agent (device)
        for agent_id, agent in agents.items():

            # Collect unique data to be moved from dependency tasks
            # This is used to identify data from initial placement
            moved_data_from_dependencies = set()

            # Iterate all dependencies and check rank + communication overhead
            # and find task's ready time when the task is assigned to the current
            # agent
            for dep in task.dependencies:
                dependency_instance = taskmap[dep]

                if isinstance(dependency_instance, SimulatedDataTask):
                    continue

                # TODO(hc): We can reuse this information collected in the above later
                # Compute communication overhead
                all_dep_data = dependency_instance.info.data_dependencies.all_ids()
                intersected_data = []
                intersected_data_size = 0
                for sd in src_data:  # SimulatedData
                    for dd in all_dep_data:  # DataID
                        if sd.id == dd:
                            intersected_data.append(sd)
                            intersected_data_size += scheduler_state.objects.datamap[
                                sd.id
                            ].size
                            moved_data_from_dependencies.add(sd.id)
                            print(f"{sd.id} is moved by task {task.name} from {dependency_instance.name}")

                assigned_agent_id = dependency_instance.info.heft_allocation

                comm_time: float = float(
                    scheduler_state.topology.get_transfer_time(
                        Device(Architecture.GPU, assigned_agent_id),
                        Device(Architecture.GPU, agent_id),
                        intersected_data_size,
                    ).scale_to("ms")
                )
                # print("oid: ", assigned_agent_id, ", nid: ", agent_id,
                #     ", toid: ", scheduler_state.topology.connection_pool.get_index(
                #       Device(Architecture.GPU, assigned_agent_id)),
                #     ", tnid: ", scheduler_state.topology.connection_pool.get_index(
                #       Device(Architecture.GPU, agent_id)))
                ready_time = max(
                    taskmap[dep].info.heft_makespan + comm_time, ready_time
                )
                print(f"{task.name}'s ready time: {ready_time}")
                # print(" task:",  task.name, ">> size:", intersected_data_size, " actual comm time:", comm_time)

            if consider_init_placement:
                # Calculate data transfer time from initial placement
                for sd in src_data:
                    if sd.id not in moved_data_from_dependencies:
                        sd_info = scheduler_state.objects.datamap[sd.id].info
                        initial_placement = sd_info.location

                        assert isinstance(initial_placement, Device)

                        comm_time_from_initial_place = float(
                            scheduler_state.topology.get_transfer_time(
                                initial_placement, Device(Architecture.GPU, agent_id),
                                sd_info.size
                            ).scale_to("ms")
                        )

                        print(f"{sd.id} is from initial placement {initial_placement} to {agent_id} and overhead {comm_time_from_initial_place} by task {task.name}")
                        ready_time = max(comm_time_from_initial_place, ready_time)
                        print(f"{task.name}'s ready time: {ready_time}")

            print(task.name, " final ready time:", ready_time)

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
                        print(
                            task.info.id,
                            " earliest start:",
                            tmp_earliest_start,
                            " e2 start:",
                            e2.start,
                            " duration: ",
                            duration,
                            " on device",
                            agent_id,
                        )
                        break

                if not any_slack_found:
                    candidate_earliest_start = max(agent[-1].end, ready_time)
                    print(
                        task.info.id,
                        "(no slack) earliest start:",
                        candidate_earliest_start,
                    )
            else:
                candidate_earliest_start = ready_time

            print(f"task {task.name} candidate earliest finish time: {candidate_earliest_start}, earliest start: {earliest_start}")
            if earliest_start == None or earliest_start > candidate_earliest_start:
                earliest_start_agent = agent_id
                earliest_start = candidate_earliest_start

        heft_event = HEFTEvent(task, earliest_start, earliest_start + duration)
        if update_task_order:
            heft_events.append(heft_event)
        bisect.insort(agents[earliest_start_agent], heft_event, key=lambda x: x.start)
        task.info.heft_makespan = earliest_start + duration
        print("makespan allocation:", task.name, " earliest start:", earliest_start, " duration:", duration, " mkspan:", task.info.heft_makespan, " dependencies:", task.dependencies)
        task.info.heft_allocation = earliest_start_agent
        print(f"heft task {task.name}, allocation: {earliest_start_agent}")
        if task.info.heft_makespan > max_heft:
            max_heft = task.info.heft_makespan

    if update_task_order:
        heft_events = sorted(heft_events, key=get_start_time)
        tasklist[:] = [he.task for he in heft_events]
        order = 0
        for task in tasklist:
            task.info.order = order
            order += 1

    """
    for key, value in agents.items():
       print("Key:", key)
       for vvalue in value:
           print("span:", vvalue.task.info.id, ", ", vvalue.start, " ~ ", vvalue.end)
    """
    return max_heft


def calculate_heft(
    tasklist,
    taskmap,
    num_devices: int,
    scheduler_state: "SystemState",
    update_task_order: bool = False,
    consider_init_placement: bool = True,
) -> float:
    """
    Calculate HEFT (Heterogeneous Earliest Finish Time) for each task.
    This function assumes that the tasklist is already sorted by a topology.
    The below is the equation:

    HEFT_rank = task_duration + max(HEFT_rank(successors))
    """

    heft_sorted_tasks = calculate_heft_upward_rank(tasklist, taskmap, scheduler_state)

    max_heft = map_task_heft(
        tasklist, taskmap, num_devices,
        scheduler_state, update_task_order,
        heft_sorted_tasks, consider_init_placement
    )
    print("HEFTTheory,simtime,", max_heft / 1000)
    """
    for t in tasklist:
        print(t.info.id, "...")
    """
    return max_heft


def load_task_noise(fname: str = "replay.noise") -> Dict[str, int]:
    loaded_task_noise = dict()
    print("task noise loading..")
    try:
        with open(fname, "r") as fp:
            lines = fp.readlines()
            for l in lines:
                l = l.rstrip()
                unpacked = l.split(":")
                key = unpacked[0]
                val = unpacked[1]
                loaded_task_noise[key] = val
    except IOError:
        print(f"Could not read task noise file: {fname}")
        print("Disable task execution time noise")
        return None
    return loaded_task_noise


def save_task_noise(task: "SimulatedTask", noise: int, fname: str = "replay.noise"):
    """
    noise should be 'us' scale.
    """
    with open(fname, "a") as fp:
        fp.write(str(task.name) + ":" + str(noise) + "\n")


def load_task_order(task_objects, fname: str = "replay.order"):
    # Read a stored task order and sort task IDs by it
    loaded_task_key = []
    print("task order loading..")
    with open(fname, "r") as fp:
        lines = fp.readlines()
        for l in lines:
            loaded_task_key.append(l.rstrip())

    def sort_key(task):
        return loaded_task_key.index(str(task.info.id))

    # print("loaded_task_key:", loaded_task_key)
    return sorted(task_objects, key=sort_key)


def save_task_order(task_objects, fname: str = "replay.order"):
    with open(fname, "w") as fp:
        for t in task_objects:
            fp.write(str(t.info.id) + "\n")
