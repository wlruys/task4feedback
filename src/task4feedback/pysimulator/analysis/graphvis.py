from ...legacy_types import *
from typing import List, Dict, Any, Tuple, Type
from dataclasses import dataclass, field, InitVar
from ..events import *
from ..data import *
from ..device import *
from ..task import *
from ..topology import *

import networkx as nx
import os

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib import animation

import pydot
import io


def build_data_usage_networkx_graph(
    tasks: SimulatedTaskMap, verbose: bool = False
) -> nx.DiGraph:
    G = nx.DiGraph()

    task_color_map = {
        TaskType.COMPUTE: "grey",
        TaskType.DATA: "purple",
        TaskType.EVICTION: "lightgreen",
    }

    data_use_color_map = {
        AccessType.READ: "green",
        AccessType.WRITE: "red",
        AccessType.READ_WRITE: "yellow",
    }

    for name, info in tasks.items():
        color = task_color_map[info.type]
        name = str(name)
        if verbose:
            print(f"Adding node: {name} with color: {color}")

        if isinstance(info, SimulatedDataTask):
            shape = "rectangle"
        else:
            shape = "ellipse"

        G.add_node(name, label=name, color=color, shape=shape)

        read_data = set([d.id for d in info.read_accesses])
        write_data = set([d.id for d in info.write_accesses])
        read_write_data = set([d.id for d in info.read_write_accesses])

        for dep_id in info.dependencies:
            dep = tasks[dep_id]
            if verbose:
                print(f"Dep: {dep}", type(dep))
            if isinstance(dep, SimulatedDataTask):
                data_id = dep.read_accesses[0].id
                # print(f"Data ID: {data_id} from {dep.name}")

                if data_id in read_data:
                    color = data_use_color_map[AccessType.READ]
                elif data_id in read_write_data:
                    color = data_use_color_map[AccessType.READ_WRITE]
                elif data_id in write_data:
                    color = data_use_color_map[AccessType.WRITE]

            G.add_edge(str(dep_id), name, color=color)

    return G


def build_mapping_networkx_graph(
    tasks: SimulatedTaskMap, topology: SimulatedTopology
) -> nx.DiGraph:
    G = nx.DiGraph()

    devices = topology.devices

    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    task_color_map = {device.name: colors[i] for i, device in enumerate(devices)}

    for name, info in tasks.items():
        if isinstance(info, SimulatedDataTask):
            continue

        shape = "ellipse"
        devices = info.assigned_devices
        assert devices is not None
        assert len(devices) == 1

        device = devices[0]

        color = task_color_map[device]
        name = str(name)

        size = (info.duration.duration) / 1e4

        print(f"Adding node: {name} with color: {color}")

        G.add_node(name, label=name, color=color, shape=shape, size=size)

        read_data = set([d.id for d in info.read_accesses])
        write_data = set([d.id for d in info.write_accesses])
        read_write_data = set([d.id for d in info.read_write_accesses])

        for dep_id in info.dependencies:
            dep = tasks[dep_id]
            print(f"Dep: {dep}", type(dep))
            if isinstance(dep, SimulatedDataTask):
                continue

            G.add_edge(str(dep_id), name, color=color)

    return G


def build_state_at_time(
    tasks: SimulatedTaskMap,
    topology: SimulatedTopology,
    time: Time,
    plot_data_tasks: bool = False,
) -> nx.DiGraph:
    G = nx.DiGraph()

    devices = topology.devices

    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    sizes = [100, 200, 300, 400, 100, 100]
    task_color_map = {s: colors[i] for i, s in enumerate(TaskState)}
    task_size_map = {s: sizes[i] for i, s in enumerate(TaskState)}

    for name, info in tasks.items():
        if isinstance(info, SimulatedDataTask) and not plot_data_tasks:
            continue

        if isinstance(info, SimulatedDataTask):
            shape = "rectangle"
        else:
            shape = "ellipse"

        devices = info.assigned_devices
        assert devices is not None
        assert len(devices) == 1

        device = devices[0]

        task_state = info.times.get_state(time)
        print(f"Task {name} has state: {task_state}")
        color = task_color_map[task_state]
        name = str(name)

        size = (info.duration.duration) / 1e4

        print(f"Adding node: {name} with color: {color}")

        G.add_node(name, label=name, color=color, shape=shape, size=size)

        read_data = set([d.id for d in info.read_accesses])
        write_data = set([d.id for d in info.write_accesses])
        read_write_data = set([d.id for d in info.read_write_accesses])

        for dep_id in info.dependencies:
            dep = tasks[dep_id]
            print(f"Dep: {dep}", type(dep))
            if isinstance(dep, SimulatedDataTask) and not plot_data_tasks:
                continue

            G.add_edge(str(dep_id), name, color=color)

    return G


def dot_layout(G):
    # pos = nx.spring_layout(G, seed=5, scale=600)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    for name, (x, y) in pos.items():
        node = G.nodes[name]
        node["x"] = x
        node["y"] = y


def spring_layout(G):
    pos = nx.spring_layout(G, seed=5, scale=600)
    for name, (x, y) in pos.items():
        node = G.nodes[name]
        node["x"] = x
        node["y"] = y


# def draw(G, filename="graph.html"):
#     # spring_layout(G)
#     fig = gv.three(
#         G,
#         layout_algorithm_active=True,
#         graph_height=1000,
#         use_edge_size_normalization=True,
#         use_node_size_normalization=True,
#         node_size_normalization_max=30,
#     )
#     if os.path.exists(filename):
#         os.remove(filename)

#     if filename.endswith(".html"):
#         fig.export_html(filename)
#     elif filename.endswith(".png"):
#         fig.export_png(filename)
#     elif filename.endswith(".jpg"):
#         fig.export_jpg(filename)


# def draw_networkx(G: nx.DiGraph):
#     pos = nx.spring_layout(G, seed=5)
#     nx.draw_networkx_nodes(G, pos=pos, node_size=700)
#     nx.draw_networkx_edges(G, pos=pos)
#     nx.draw_networkx_labels(G, pos=pos, labels=labels)
#     plt.tight_layout()
#     plt.axis("off")
#     plt.show()


def draw_pydot(G: nx.DiGraph):
    pg = nx.drawing.nx_pydot.to_pydot(G)
    png_str = pg.create_png(prog="dot")
    pg.write_png("pydot_graph.png")
    sio = io.BytesIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    implot = plt.imshow(img, aspect="equal")
