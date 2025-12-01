from task4feedback.graphs import *
from task4feedback.graphs.mesh import (
    build_geometry,
    generate_quad_mesh,
    generate_tri_mesh,
)
from task4feedback.graphs.mesh.partition import *
from typing import Callable
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


@dataclass
class GraphBuilder:
    config: GraphConfig
    function: Callable[[GraphConfig, DictConfig], Graph]


def make_graph_function(graph_cfg: GraphConfig, cfg: DictConfig) -> Callable[[GraphConfig, DictConfig], Graph]:
    def make_graph(system: System):
        mesh = instantiate(cfg.graph.mesh, L=1, n=graph_cfg.n, domain_ratio=graph_cfg.domain_ratio)

        geom = build_geometry(mesh)
        graph = build_graph(geom, graph_cfg, system=system)


        if cfg.graph.init.partitioner == "metis":
            partitioner = metis_geometry_partition
            partition = partitioner(geom, nparts=cfg.graph.init.nparts)
        elif cfg.graph.init.partitioner == "block_cyclic":
            partitioner = block_cyclic
            partition = block_cyclic(geom)
        elif cfg.graph.init.partitioner == "column_cyclic":
            partitioner = col_cyclic
            partition = col_cyclic(geom)
        elif cfg.graph.init.partitioner == "row_cyclic":
            partitioner = row_cyclic
            partition = row_cyclic(geom)
        elif cfg.graph.init.partitioner == "mincut":
            partitioner = None  # use built-in mincut partitioning
            partition = graph.initial_mincut_partition(
                arch=DeviceType.GPU,
                bandwidth=cfg.system.d2d_bw,
                n_parts=4,
                offset=0,
            )
            partition = graph.maximize_matches(partition)
            print(f"Mincut partition: {partition}")
        else:
            raise NotImplementedError(f"Partitioner {cfg.graph.init.partitioner} is not implemented.")
        # partition = partitioner(geom, nparts=cfg.graph.init.nparts)
        # partition = block_cyclic(geom)
        # partition = graph.initial_mincut_partition(
        #     arch=DeviceType.GPU,
        #     bandwidth=cfg.system.d2d_bw,
        #     n_parts=4,
        #     offset=0,
        # )
        # partition = graph.maximize_matches(partition)

        # print(partition)
        # if isinstance(graph, DynamicJacobiGraph):
        #     for x in range(graph.nx):
        #         for y in range(graph.ny):
        #             print(f"{partition[graph.xy_from_id(x * graph.ny + y)]}", end=" ")
        #         print()

        if cfg.graph.init.gpu_only:
            partition = [x + 1 for x in partition]  # offset by 1 to ignore cpu
            location_list = [i + 1 for i in range(0, cfg.graph.init.nparts)]
        else:
            location_list = [i for i in range(cfg.graph.init.nparts + 1)]  # include cpu as 0

        if isinstance(graph, DynamicJacobiGraph):
            graph.set_cell_locations([-1 for _ in range(len(partition))])
            graph.set_cell_locations(partition, step=0)
        elif isinstance(graph, JacobiGraph):
            print(f"Setting cell locations with partition: {partition}")
            graph.set_cell_locations(partition)
        elif isinstance(graph, CholeskyGraph):
            graph.set_cell_locations(partition)

        if cfg.graph.init.randomize:
            graph.randomize_locations(
                graph_cfg.randomness,
                location_list=location_list,
                step=0 if isinstance(graph, DynamicJacobiGraph) else None,
            )

        return graph

    graph_function = make_graph
    return graph_function


def make_graph_builder(cfg: DictConfig, verbose: bool = True) -> GraphBuilder:
    if verbose:
        print(f"Graph info: {OmegaConf.to_yaml(cfg.graph.config)}")
    graph_config = instantiate(cfg.graph.config)
    graph_function = make_graph_function(graph_config, cfg)
    return GraphBuilder(config=graph_config, function=graph_function)
