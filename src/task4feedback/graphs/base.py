from .mesh.base import Geometry, Cell, Edge
from ..interface import DataBlocks, Graph
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
import gravis as gv
import os


def spring_layout(G):
    pos = nx.spring_layout(G, seed=5, scale=600)
    for name, (x, y) in pos.items():
        node = G.nodes[name]
        node["x"] = x
        node["y"] = y


def draw(G, filename="graph.html"):
    # spring_layout(G)
    fig = gv.d3(
        G,
        layout_algorithm_active=True,
        graph_height=1000,
        use_edge_size_normalization=True,
        use_node_size_normalization=True,
        node_size_normalization_max=30,
    )
    if os.path.exists(filename):
        os.remove(filename)

    if filename.endswith(".html"):
        fig.export_html(filename)
    elif filename.endswith(".png"):
        fig.export_png(filename)
    elif filename.endswith(".jpg"):
        fig.export_jpg(filename)


@dataclass(frozen=True, eq=True, order=True, slots=True)
class DataKey:
    object: Cell | Edge
    id: int | str | tuple


def filter_key_list(
    obj_list: list[DataKey],
) -> tuple[list[Cell], list[tuple[Edge, Cell]]]:
    edge_list = []
    cell_list = []

    for obj in obj_list:
        if isinstance(obj.object, Edge):
            edge_list.append((obj.object, obj.id[0]))
        elif isinstance(obj.object, Cell):
            cell_list.append(obj.object)

    return cell_list, edge_list


tree = lambda: defaultdict(tree)


@dataclass
class NestedKeyDict(dict):
    container: defaultdict = field(default_factory=tree)

    def add(self, keys: DataKey, value):
        d = self.container
        d = d[keys.object]
        keys = keys.id

        if not isinstance(keys, tuple):
            keys = (keys,)

        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value

    def get(self, keys: DataKey | Cell | Edge):
        if isinstance(keys, Cell) or isinstance(keys, Edge):
            return self.container[keys]

        d = self.container
        d = d[keys.object]
        keys = keys.id
        if not isinstance(keys, tuple):
            keys = (keys,)
        for key in keys:
            d = d[key]
        return d

    def _get_leaves(self, d, keys: Cell | Edge):
        leaves = []
        for k, v in d.items():
            if isinstance(v, int):
                leaves.append(v)
            else:
                leaves.extend(self._get_leaves(v, k))
        return leaves

    def get_leaves(self, keys: DataKey | Cell | Edge):
        # Get all leaf int values from the nested dict
        d = self.get(keys)
        leaves = []
        for k, v in d.items():
            if isinstance(v, int):
                leaves.append(v)
            else:
                leaves.extend(self._get_leaves(v, k))
        return leaves

    def __setitem__(self, keys: DataKey | Cell | Edge, value: int):
        self.add(keys, value)

    def __getitem__(self, keys: DataKey):
        return self.get(keys)


@dataclass
class GeometryIDMap:
    object_to_blocks: dict = field(default_factory=lambda: defaultdict(list))
    key_to_block: NestedKeyDict = field(default_factory=NestedKeyDict)
    block_to_object: dict = field(default_factory=dict)
    block_to_key: dict = field(default_factory=dict)

    def add_block(self, key: DataKey, block_id: int):
        self.object_to_blocks[key.object].append(block_id)
        self.key_to_block[key] = block_id
        self.block_to_object[block_id] = key.object
        self.block_to_key[block_id] = key

    def get_block(self, key: DataKey):
        return self.key_to_block[key]

    def get_object(self, block_id: int):
        return self.block_to_object[block_id]

    def get_blocks(self, object: Cell | Edge):
        return self.object_to_blocks[object]


@dataclass
class DataGeometry:
    geometry: Geometry
    blocks: DataBlocks = field(default_factory=DataBlocks)
    map: GeometryIDMap = field(default_factory=GeometryIDMap)

    def add_block(self, key: DataKey, size: int, location: int):
        block = self.blocks.add_block(name=str(key), size=size, location=location)
        self.map.add_block(key, block.id)

    def get_object(self, block_id: int):
        return self.map.get_object(block_id)

    def get_block(self, key: DataKey):
        return self.map.get_block(key)

    def get_blocks(self, object: Cell | Edge):
        return self.map.get_blocks(object)


class ComputeDataGraph(Graph):
    
    def ___init__(self, data: DataGeometry):
        super(ComputeDataGraph, self).__init__()
        self.data = data

    def get_blocks(self):
        return self.data.blocks

    def get_data_geometry(self):
        return self.data
