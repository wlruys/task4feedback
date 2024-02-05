from .recorder import *
import numpy as np
from pandas import *
import matplotlib.pyplot as plt


def make_plot(
    data_recorder: DataValidRecorder,
    compute_recorder: Optional[ComputeTaskRecorder] = None,
    data_movement_recorder: Optional[DataTaskRecorder] = None,
    data_ids: List[DataID] = [],
):
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

        if compute_recorder is not None:
            for taskid, taskrecord in compute_recorder.tasks.items():
                if data_id in taskrecord.read_data:
                    ax.axvline(
                        taskrecord.start_time.duration,
                        color="green",
                        linestyle="--",
                        linewidth=1,
                    )
                    ax.axvline(
                        taskrecord.end_time.duration,
                        color="green",
                        linestyle="--",
                        linewidth=1,
                    )
                    ax.text(
                        taskrecord.start_time.duration,
                        str(taskrecord.devices[0]),
                        str(taskid),
                        fontsize=5,
                        color="black",
                        ha="left",
                        va="bottom",
                    )
                if data_id in taskrecord.write_data:
                    ax.axvline(
                        taskrecord.start_time.duration,
                        color="red",
                        linestyle="--",
                        linewidth=1,
                    )
                    ax.text(
                        taskrecord.start_time.duration,
                        str(taskrecord.devices[0]),
                        str(taskid),
                        fontsize=5,
                        color="black",
                        ha="left",
                        va="bottom",
                    )

                if data_id in taskrecord.read_write_data:
                    ax.axvline(
                        taskrecord.start_time.duration,
                        color="yellow",
                        linestyle="--",
                        linewidth=1,
                    )
                    ax.text(
                        taskrecord.start_time.duration,
                        str(taskrecord.devices[0]),
                        str(taskid),
                        fontsize=5,
                        color="black",
                        ha="left",
                        va="bottom",
                    )

        if data_movement_recorder is not None:
            for taskid, taskrecord in data_movement_recorder.tasks.items():
                if (
                    data_id == taskrecord.data
                    and (taskrecord.end_time - taskrecord.start_time).duration > 0
                ):
                    ax.axvline(
                        taskrecord.end_time.duration,
                        color="purple",
                        linestyle="--",
                        linewidth=0.5,
                    )
                    ax.text(
                        taskrecord.end_time.duration,
                        str(taskrecord.devices[0]),
                        str(taskid),
                        fontsize=5,
                        color="black",
                        ha="left",
                        va="bottom",
                    )

        # Formatting the plot
        # ax.xaxis_date()
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xlabel("Time")
        plt.ylabel("Device")
        plt.title("Valid Intervals for Data {}".format(data_id))
        plt.tight_layout()

        plt.show()
