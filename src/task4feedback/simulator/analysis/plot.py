from .recorder import *
import numpy as np
from pandas import *
import matplotlib.pyplot as plt


def make_plot(recorder: DataValidRecorder):
    intervals = recorder.intervals
    for data_id in intervals.keys():
        fig, ax = plt.subplots()
        import itertools

        colors = plt.cm.tab20.colors

        # Flatten the data structure and collect intervals for plotting
        device_intervals = []

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

        # Formatting the plot
        # ax.xaxis_date()
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xlabel("Time")
        plt.ylabel("Device")
        plt.title("Valid Intervals for Data {}".format(data_id))
        plt.tight_layout()

        plt.show()
