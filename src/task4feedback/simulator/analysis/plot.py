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

    for k, d in enumerate(device):
        if not combine_devices:
            ax = plt.gca()

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

        ax.step(
            df["time"],
            df["usage"],
            label=f"{field} usage for {d}",
            linestyle="-",
            color=device_colors[d],
            where="mid",
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
            plt.ylabel(f"{field} usage")
            plt.title(f"{field} for {d} in {phase} phase")

            if compute_task_recorder is not None:
                _plot_compute_tasks(
                    ax, compute_task_recorder, d, device_as_y=False, labels=False
                )

            if data_task_recorder is not None:
                _plot_data_tasks(
                    ax, data_task_recorder, d, device_as_y=False, labels=False
                )

            plt.show()

        if log_scale:
            plt.yscale("log")
    else:
        plt.xlabel("Time")
        plt.ylabel(f"{field} usage")
        plt.title(f"{field} in {phase} phase")
        plt.legend()

        if compute_task_recorder is not None:
            for d in device:
                _plot_compute_tasks(
                    ax, compute_task_recorder, d, device_as_y=False, labels=False
                )
            for d in device:
                _plot_data_tasks(
                    ax, data_task_recorder, d, device_as_y=False, labels=False
                )

        plt.show()


def _plot_compute_tasks(
    ax: plt.Axes,
    recorder: ComputeTaskRecorder,
    data_id: DataID,
    device_as_y: bool = False,
    labels: bool = False,
):
    usage_color_map = {
        AccessType.READ: "green",
        AccessType.WRITE: "red",
        AccessType.READ_WRITE: "yellow",
    }

    for taskid, taskrecord in recorder.tasks.items():
        if data_id in taskrecord.read_data:
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
        if data_id in taskrecord.write_data:
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

        if data_id in taskrecord.read_write_data:
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


def _plot_data_tasks(
    ax: plt.Axes,
    recorder: DataTaskRecorder,
    data_id: DataID,
    device_as_y: bool = False,
    labels: bool = False,
):
    usage_color_map = {
        AccessType.READ: "purple",
    }

    for taskid, taskrecord in recorder.tasks.items():
        if data_id in taskrecord.read_data:
            ax.axvline(
                taskrecord.end_time.duration,
                color=usage_color_map[AccessType.READ],
                linestyle="--",
                linewidth=1,
            )
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
            _plot_compute_tasks(ax, compute_task_recorder, data_id)

        if data_task_recorder is not None:
            _plot_data_tasks(ax, data_task_recorder, data_id)

        # Formatting the plot
        # ax.xaxis_date()
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xlabel("Time")
        plt.ylabel("Device")
        plt.title("Valid Intervals for Data {}".format(data_id))
        plt.tight_layout()

        plt.show()
