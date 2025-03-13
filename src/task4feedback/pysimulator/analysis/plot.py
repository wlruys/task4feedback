from asyncio import Task
from .recorder import *
import numpy as np
from pandas import *
import matplotlib.pyplot as plt
from ...legacy_types import *
from typing import cast
from task4feedback.pysimulator.simulator import *
import pydot
from mplcursors import cursor  # separate package must be installed


def make_resource_plot(
    recorder: RecorderList,
    resource_type: ResourceType = ResourceType.VCU,
    phase: TaskState = TaskState.LAUNCHED,
    combine_devices: bool = True,
    plot_max: bool = False,
    log_scale: bool = False,
    plot_compute_tasks: bool = False,
    plot_data_tasks: bool = False,
    plot_events: bool = False,
):
    if phase == TaskState.LAUNCHED:
        try:
            r = recorder.get(LaunchedResourceUsageListRecorder)
        except KeyError:
            r = recorder.get(LaunchedResourceUsageRecorder)
    elif phase == TaskState.RESERVED:
        try:
            r = recorder.get(ReservedResourceUsageListRecorder)
        except KeyError:
            r = recorder.get(ReservedResourceUsageRecorder)
    elif phase == TaskState.MAPPED:
        try:
            r = recorder.get(MappedResourceUsageListRecorder)
        except KeyError:
            r = recorder.get(MappedResourceUsageRecorder)
    else:
        raise ValueError("Invalid phase")

    if plot_compute_tasks:
        try:
            compute_task_recorder: Optional[ComputeTaskRecorder] = recorder.get(
                ComputeTaskRecorder
            )
        except KeyError:
            compute_task_recorder = None
    else:
        compute_task_recorder = None

    if plot_data_tasks:
        try:
            data_task_recorder: Optional[DataTaskRecorder] = recorder.get(
                DataTaskRecorder
            )
        except KeyError:
            data_task_recorder = None
    else:
        data_task_recorder = None

    if plot_events:
        try:
            event_recorder: Optional[EventRecorder] = recorder.get(EventRecorder)
        except KeyError:
            event_recorder = None
    else:
        event_recorder = None

    if resource_type == ResourceType.VCU:
        field = "vcu_usage"
        f2 = "vcus"
    elif resource_type == ResourceType.MEMORY:
        field = "memory_usage"
        f2 = "memory"
    elif resource_type == ResourceType.COPY:
        field = "copy_usage"
        f2 = "copy"

    data = getattr(r, field)

    device = list(data.keys())
    colors = plt.cm.tab20.colors
    device_colors = {d: colors[i] for i, d in enumerate(device)}

    if combine_devices:
        ax = plt.gca()

    if not combine_devices:
        fig, axs = plt.subplots(len(device), 1, sharex=True, sharey=True)

    for k, d in enumerate(device):
        ax = axs[k] if not combine_devices else ax

        device_data = data[d]
        time = list(device_data.keys())
        usage = list(device_data.values())

        if isinstance(usage[0], list):
            for i, u in enumerate(usage):
                u: List[Fraction] = u
                f = max(u)
                usage[i] = float(f)

        for i, t in enumerate(time):
            time[i] = t.duration

        df = DataFrame({"time": time, "usage": usage})

        if d == Device(Architecture.GPU, 2):
            df.to_csv("gpu2.csv")

        ax.step(
            df["time"],
            df["usage"],
            label=f"{field} usage for {d}",
            linestyle="-",
            color=device_colors[d],
            where="pre",
        )
        if plot_max:
            ax.axhline(
                getattr(r.max_resources[d], f2),
                color=device_colors[d],
                linestyle="--",
                linewidth=1,
                label=f"Max {field} for {d}",
            )

        if not combine_devices:
            plt.xlabel("Time")
            ax.set_ylabel(f"{d}")
            # plt.title(f"{field} for {d} in {phase} phase")

            if compute_task_recorder is not None:
                _plot_compute_tasks(
                    ax=ax,
                    recorder=compute_task_recorder,
                    data_id=None,
                    device_id=d,
                    device_as_y=False,
                    labels=False,
                )

            if data_task_recorder is not None:
                _plot_data_tasks(
                    ax=ax,
                    recorder=data_task_recorder,
                    data_id=None,
                    device_id=d,
                    device_as_y=False,
                    labels=False,
                )

            if event_recorder is not None:
                _plot_events(
                    ax=ax,
                    recorder=event_recorder,
                    labels=False,
                )

        if log_scale:
            plt.yscale("log")

    if combine_devices:
        plt.xlabel("Time")
        plt.ylabel(f"{field} usage")
        plt.title(f"{field} in {phase} phase")
        plt.legend()

        if compute_task_recorder is not None:
            for d in device:
                _plot_compute_tasks(
                    ax=ax,
                    recorder=compute_task_recorder,
                    data_id=None,
                    device_id=d,
                    device_as_y=False,
                    labels=False,
                )
        if data_task_recorder is not None:
            for d in device:
                _plot_data_tasks(
                    ax=ax,
                    recorder=data_task_recorder,
                    data_id=None,
                    device_id=d,
                    device_as_y=False,
                    labels=False,
                )

    plt.show()


def _plot_events(
    ax: plt.Axes,
    recorder: EventRecorder,
    labels: bool = False,
):
    for time, events in recorder.completed_events.items():
        end_time = time.duration
        for event in events:
            ax.axvline(
                end_time,
                color="black",
                linestyle="--",
                linewidth=1,
            )
            if labels:
                ax.text(
                    end_time,
                    0.5,
                    str(event),
                    fontsize=5,
                    color="black",
                    ha="left",
                    va="bottom",
                )


def _plot_compute_tasks(
    ax: plt.Axes,
    recorder: ComputeTaskRecorder,
    data_id: Optional[DataID] = None,
    device_id: Optional[Device] = None,
    device_as_y: bool = False,
    labels: bool = False,
):
    usage_color_map = {
        AccessType.READ: "green",
        AccessType.WRITE: "red",
        AccessType.READ_WRITE: "yellow",
    }

    # use default matplotlib color map for devices
    device_color_map = plt.cm.tab20.colors

    for taskid, taskrecord in recorder.tasks.items():
        active_read_data = data_id is not None and data_id in taskrecord.read_data
        active_write_data = data_id is not None and data_id in taskrecord.write_data
        active_read_write_data = data_id is not None and (
            data_id in taskrecord.read_write_data
        )
        active_device = device_id is not None and (
            device_id in _get_device_list(taskrecord)
        )

        if data_id is None and active_device:
            ax.axvspan(
                taskrecord.start_time.duration,
                taskrecord.end_time.duration,
                alpha=0.5,
            )

            if labels:
                y_loc = str(taskrecord.devices[0]) if device_as_y else 0.5
                ax.text(
                    taskrecord.start_time.duration,
                    y_loc,
                    str(taskid),
                    fontsize=5,
                    color="black",
                    ha="left",
                    va="bottom",
                )

        if active_read_data:
            ax.axvline(
                taskrecord.start_time.duration,
                color=usage_color_map[AccessType.READ],
                linestyle="--",
                linewidth=1,
            )
            if labels:
                y_loc = str(taskrecord.devices[0]) if device_as_y else 0.5
                ax.text(
                    taskrecord.start_time.duration,
                    y_loc,
                    str(taskid),
                    fontsize=5,
                    color="black",
                    ha="left",
                    va="bottom",
                )
        if active_write_data:
            ax.axvline(
                taskrecord.start_time.duration,
                color=usage_color_map[AccessType.WRITE],
                linestyle="--",
                linewidth=1,
            )
            if labels:
                y_loc = str(taskrecord.devices[0]) if device_as_y else 0.5
                ax.text(
                    taskrecord.start_time.duration,
                    y_loc,
                    str(taskid),
                    fontsize=5,
                    color="black",
                    ha="left",
                    va="bottom",
                )

        if active_read_write_data:
            ax.axvline(
                taskrecord.start_time.duration,
                color=usage_color_map[AccessType.READ_WRITE],
                linestyle="--",
                linewidth=1,
            )
            if labels:
                y_loc = str(taskrecord.devices[0]) if device_as_y else 0.5
                ax.text(
                    taskrecord.start_time.duration,
                    y_loc,
                    str(taskid),
                    fontsize=5,
                    color="black",
                    ha="left",
                    va="bottom",
                )


def _get_device_list(taskrecord: DataTaskRecord | ComputeTaskRecord) -> List[Device]:
    if taskrecord.devices is None:
        return []
    if isinstance(taskrecord.devices, tuple):
        return list(taskrecord.devices)
    if isinstance(taskrecord.devices, list):
        return taskrecord.devices
    return [taskrecord.devices]


def _plot_data_tasks(
    ax: plt.Axes,
    recorder: DataTaskRecorder,
    data_id: Optional[DataID] = None,
    device_id: Optional[Device] = None,
    device_as_y: bool = False,
    labels: bool = False,
):
    usage_color_map = {
        AccessType.READ: "purple",
    }

    type_color_map = {TaskType.DATA: "purple", TaskType.EVICTION: "lightgreen"}

    for taskid, taskrecord in recorder.tasks.items():
        active_data = data_id is not None and data_id in taskrecord.read_data
        active_device = device_id is not None and (
            device_id in _get_device_list(taskrecord)
        )
        active_source = device_id is not None and (device_id == taskrecord.source)

        print(f"Task: {taskid}, Data: {data_id}, Device: {device_id}")

        if active_data or active_device or active_source:
            print(f"Active: {taskid}, color: {type_color_map[taskrecord.type]}")
            print(f"Start: {taskrecord.start_time.duration}")
            print(f"End: {taskrecord.end_time.duration}")

            c = type_color_map[taskrecord.type]
            if not active_data:
                if active_source and not active_device:
                    c = "black"

                if active_device and not active_source:
                    c = "orange"

            ax.axvspan(
                taskrecord.start_time.duration,
                taskrecord.end_time.duration,
                alpha=0.5,
                color=c,
            )
            # ax.axvline(
            #     taskrecord.start_time.duration,
            #     color=c,
            #     linestyle="-",
            #     linewidth=1,
            # )

            # ax.axvline(
            #     taskrecord.end_time.duration,
            #     color=c,
            #     linestyle="--",
            #     linewidth=1,
            # )

            if labels:
                y_loc = str(taskrecord.devices[0]) if device_as_y else 0.5
                ax.text(
                    taskrecord.end_time.duration,
                    y_loc,
                    str(taskid),
                    fontsize=5,
                    color="black",
                    ha="left",
                    va="bottom",
                )


def make_data_plot(
    recorder: RecorderList,
    plot_compute_tasks: bool = True,
    plot_data_tasks: bool = True,
    data_ids: List[DataID] = [],
):
    if plot_compute_tasks:
        try:
            compute_task_recorder: Optional[ComputeTaskRecorder] = recorder.get(
                ComputeTaskRecorder
            )
        except KeyError:
            compute_task_recorder = None
    else:
        compute_task_recorder = None

    if plot_data_tasks:
        try:
            data_task_recorder: Optional[DataTaskRecorder] = recorder.get(
                DataTaskRecorder
            )
        except KeyError:
            data_task_recorder = None
    else:
        data_task_recorder = None

    try:
        data_recorder = recorder.get(FasterDataValidRecorder)
    except KeyError:
        data_recorder = recorder.get(DataValidRecorder)

    intervals = data_recorder.intervals
    for data_id in data_ids:
        fig, ax = plt.subplots()
        import itertools

        colors = plt.cm.tab20.colors

        # Flatten the data structure and collect intervals for plotting
        device_intervals = []

        if data_id not in intervals:
            print(f"Data {data_id} not found in intervals")
            continue

        for device, valid_intervals in intervals[data_id].items():
            for interval in valid_intervals:
                device_intervals.append(
                    (device, interval.start_time, interval.end_time)
                )

        # Sort intervals by device for consistent plotting
        device_intervals.sort(key=lambda x: str(x[0]))

        # Creating a color map for each unique device
        unique_devices = list(set([device for device, _, _ in device_intervals]))
        color_map = {
            device: color
            for device, color in zip(unique_devices, itertools.cycle(colors))
        }

        # Create horizontal bars for each interval
        for i, (device, start, end) in enumerate(device_intervals):
            ax.barh(
                str(device),
                (end - start).duration,
                left=start.duration,
                height=0.4,
                color=color_map[device],
            )

        if compute_task_recorder is not None:
            _plot_compute_tasks(ax=ax, recorder=compute_task_recorder, data_id=data_id)

        if data_task_recorder is not None:
            _plot_data_tasks(ax=ax, recorder=data_task_recorder, data_id=data_id)

        # Formatting the plot
        # ax.xaxis_date()
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xlabel("Time")
        plt.ylabel("Device")
        plt.title("Valid Intervals for Data {}".format(data_id))
        plt.tight_layout()

        plt.show()


def make_dag_and_timeline(
    simulator: SimulatedScheduler,
    plot_dag: bool = True,
    color_dag: bool = True,
    plot_timeline: bool = True,
    plot_data_movement: bool = False,
    file_name: str = None,
    save_file: bool = True,
    show_plot: bool = False,
    timeline_plot_size: Tuple[int, int] = (25, 10),
    timeline_plot_fontsize: str = "xx-large",
):
    if file_name is None:
        file_name = f"""{simulator.scheduler_type}_{simulator.mapper_type}_{simulator.task_order_mode.name}_{str(len(simulator.topology.get_devices(device_type=Architecture.GPU)))}GPUs"""
    recorders = simulator.recorders

    try:
        recorder_instance = recorders.get(ComputeTaskRecorder)
        compute_task_record: Optional[ComputeTaskRecorder] = cast(
            Optional[ComputeTaskRecorder], recorder_instance
        )
    except KeyError as e:  # raise error if not found
        raise KeyError(e)

    # Make color map of upto 32 colors in strings
    colors = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange", "Pink", "Brown"]
    device_colors = {}

    # Make dictionary of task results (Device it ran on, start time, end time)
    task_results = {}
    for taskid, task_record in compute_task_record.tasks.items():
        task_result = {}
        task_result["devices"] = task_record.devices
        if task_record.devices[0].device_id not in device_colors:
            device_colors[task_record.devices[0].device_id] = colors[
                task_record.devices[0].device_id % 8
            ]
        task_result["start_time"] = task_record.start_time
        task_result["end_time"] = task_record.end_time
        task_results[str(taskid)] = task_result

    # Generate pydot graph from tasks
    # Assign same color for same device
    if plot_dag:
        graph = pydot.Dot(graph_type="digraph")
        for name, task_info in simulator.taskmap.items():
            if isinstance(task_info, SimulatedComputeTask):
                task_info = task_info.info
                idx = name.task_idx
                name = str(name)
                if color_dag:
                    node = pydot.Node(
                        name=name,
                        style="filled",
                        fillcolor=device_colors[
                            task_results[name]["devices"][0].device_id
                        ],
                    )
                else:
                    node = pydot.Node(
                        name=name,
                        style="filled",
                        fillcolor="white",
                    )
                graph.add_node(node)
                for dep_id in task_info.dependencies:
                    dep_id = str(dep_id)
                    edge = pydot.Edge(dep_id, name)
                    graph.add_edge(edge)

        if save_file:
            graph.write_png(file_name + "_dag.png")

    if plot_timeline:
        fig, ax = plt.subplots(figsize=timeline_plot_size)
        fig.subplots_adjust(left=0.01, right=0.99)
        for taskid, task_result in task_results.items():
            # Calculate the start time and duration for each task
            start_time = task_result["start_time"].duration
            duration = (task_result["end_time"] - task_result["start_time"]).duration

            # Draw the horizontal bar
            bar = ax.barh(
                task_result["devices"][0].device_id,
                duration,
                left=start_time,
                height=0.4,
                color=device_colors[task_result["devices"][0].device_id],
                edgecolor="black",
            )

            # Add text inside the bar
            text_position = (
                start_time + duration / 2
            )  # Positioning text at the middle of the bar
            ax.text(
                text_position,
                task_result["devices"][0].device_id,
                taskid,  # The text to display (taskid in this case)
                va="center",  # Vertical alignment
                ha="center",  # Horizontal alignment
                color="black",  # Text color
                fontsize=timeline_plot_fontsize,  # Font size
            )

        if plot_data_movement:
            try:
                data_task_record: Optional[DataTaskRecorder] = recorders.get(
                    DataTaskRecorder
                )
            except KeyError as e:  # raise error if not found
                raise KeyError(e)

            for taskid, task_record in data_task_record.tasks.items():
                if task_record.type == TaskType.DATA:
                    start_time = task_record.start_time.duration
                    duration = (task_record.end_time - task_record.start_time).duration

                    if task_record.source.device_id == task_record.devices[0].device_id:
                        continue

                    # Draw the horizontal bar
                    bar = ax.barh(
                        task_record.devices[0].device_id - 0.25,
                        duration,
                        left=start_time,
                        height=0.1,
                        color="black",
                        edgecolor="black",
                    )

                    # Add text inside the bar
                    text_position = start_time + duration / 2
                    ax.text(
                        text_position,
                        task_record.devices[0].device_id - 0.25,
                        str(task_record.source),
                        va="center",
                        ha="center",
                        color="white",
                        fontsize="xx-small",
                    )
                elif task_record.type == TaskType.EVICTION:
                    start_time = task_record.start_time.duration
                    duration = (task_record.end_time - task_record.start_time).duration
                    if task_record.source.device_id == task_record.devices[0].device_id:
                        continue
                    # Draw the horizontal bar
                    bar = ax.barh(
                        task_record.source.device_id + 0.25,
                        duration,
                        left=start_time,
                        height=0.1,
                        color="gray",
                        edgecolor="black",
                    )

                    # Add text inside the bar
                    text_position = start_time + duration / 2
                    ax.text(
                        text_position,
                        task_record.source.device_id + 0.25,
                        "Evict",
                        va="center",
                        ha="center",
                        color="white",
                        fontsize="xx-small",
                    )

        # Add horizontal lines for each device
        for device in device_colors.keys():
            ax.axhline(
                device - 0.5,
                color="black",
                linestyle="--",
                linewidth=1,
            )
            ax.text(
                10,
                device - 0.4,
                str(device),
                va="center",
                ha="left",
                color="black",
                fontsize="small",
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Device")
        ax.set_title("Timeline of Tasks")
        if save_file:
            plt.savefig(file_name + "_timeline.png")
        if show_plot:
            cursor()
            plt.show()


def plot_data_movement_count(
    simulator: SimulatedScheduler,
    threshold: int = 0,  # Threshold for data movement in Bytes. Only plot data movements that are greater than this threshold
    bandwidth: float = 1e10,  # Bandwidth of the network in Bytes/s
    args=None,  # argparse arguments
):
    data_tasks: DataTaskRecorder = simulator.recorders.get(DataTaskRecorder)

    data_movement = []
    for task in data_tasks.tasks.values():
        if task.source.device_id == task.devices[0].device_id:
            continue
        if (task.end_time - task.start_time) < (1e6 * threshold / bandwidth):
            continue
        data_movement.append(task.start_time.duration)
    # Sort the data movement times
    data_movement.sort()

    # Generate the accumulated count of data movements over time
    accumulated_movements = list(range(1, len(data_movement) + 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data_movement, accumulated_movements, drawstyle="steps-post")
    plt.xlabel("Time(us)")
    plt.ylabel("Accumulated Number of Data Movements")
    plt.title(
        f"Accumulated Data Movements Over Time\n{args.mode}_{args.sort}Order_{args.gpus}GPUs_Initial:{args.distribution}_{args.time}us_{args.steps}Steps_Simulated Time: {simulator.time}"
    )
    plt.grid(True)
    plt.savefig(
        f"{args.mode}_{args.sort}_{args.gpus}_{args.distribution}_{args.time}us_{args.steps}Steps.png"
    )  # You can change the file format to .pdf, .svg, etc.


def eviction_event_checker(
    task: DataTaskRecord,
) -> bool:
    if (
        task.devices[0].architecture == Architecture.CPU
        and task.source.architecture != Architecture.CPU
    ):  # Eviction only happens from GPU to CPU
        return True
    return False


def data_size_checker(
    task: DataTaskRecord,
    threshold: int = 0,
) -> bool:
    if (
        task.data_size > threshold
        and task.source.device_id
        != task.devices[0].device_id  # Source and destination should be different
    ):
        return True
    return False


def plot_data_event_count(
    simulator: SimulatedScheduler,
    rule: Callable[
        [DataTaskRecord], bool
    ] = lambda x: False,  # Rule to filter data events. default to false
    name: str = "",  # {args.mode}_{args.sort}Order_{args.gpus}GPUs_Initial:{args.distribution}_{args.time}us_{args.steps}Steps
    title: str = "",  # Title of the plot
    ylabel: str = "Accumulated Number of Data Movements",  # Label of the y-axis
):
    data_tasks: DataTaskRecorder = simulator.recorders.get(DataTaskRecorder)
    if name is "":
        name = f"DataEventCount_{simulator.scheduler_type}_{simulator.mapper_type}_{simulator.task_order_mode.name}_{str(len(simulator.topology.get_devices(device_type=Architecture.GPU)))}GPUs"
    data_movement = []
    for task in data_tasks.tasks.values():
        if rule(task):
            data_movement.append(task.start_time.duration)
    # Sort the data movement times
    data_movement.sort()

    # Generate the accumulated count of data movements over time
    accumulated_movements = list(range(1, len(data_movement) + 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data_movement, accumulated_movements, drawstyle="steps-post")
    plt.xlabel("Time(us)")
    plt.xlim((0, simulator.time.duration))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{name}.png")  # You can change the file format to .pdf, .svg, etc.
