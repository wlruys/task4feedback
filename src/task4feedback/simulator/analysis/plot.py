from .recorder import *
import numpy as np
from pandas import *
import matplotlib.pyplot as plt
from ...types import *


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
    tasks: List[Mapping[TaskID, TaskInfo]],
    recorders: RecorderList,
    plot_dag: bool = True,
    plot_timeline: bool = True,
):
    import pydot

    try:
        compute_task_recorder: Optional[ComputeTaskRecorder] = recorders.get(
            ComputeTaskRecorder
        )
    except KeyError as e:  # raise error if not found
        raise KeyError(e)

    # Make color map of upto 32 colors in strings
    colors = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange", "Pink", "Brown"]
    device_colors = {}

    # Make dictionary of task results (Device it ran on, start time, end time)
    task_results = {}
    for taskid, task_record in compute_task_recorder.tasks.items():
        task_result = {}
        task_result["devices"] = task_record.devices
        if task_record.devices[0] not in device_colors:
            device_colors[task_record.devices[0]] = colors[len(device_colors)]
        task_result["start_time"] = task_record.start_time
        task_result["end_time"] = task_record.end_time
        task_results[str(taskid)] = task_result

    # Generate pydot graph from tasks
    # Assign same color for same device
    if plot_dag:
        graph = pydot.Dot(graph_type="digraph")
        for name, task_info in tasks.items():
            name = str(name)
            node = pydot.Node(
                name=name,
                style="filled",
                fillcolor=device_colors[task_results[name]["devices"][0]],
            )
            graph.add_node(node)
            for dep_id in task_info.dependencies:
                dep_id = str(dep_id)
                edge = pydot.Edge(dep_id, name)
                graph.add_edge(edge)

        graph.write_png("dag.png")

    if plot_timeline:
        fig, ax = plt.subplots()
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
                color=device_colors[task_result["devices"][0]],
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
                fontsize="xx-small",  # Font size
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Device")
        ax.set_title("Timeline of Tasks")
        plt.show()
