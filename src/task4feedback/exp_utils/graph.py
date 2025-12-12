from .definitions import *
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
from dataclasses import dataclass
from .logging_helpers import get_helper_logger

logger = get_helper_logger(__name__)


@dataclass
class GraphBuilder:
    config: GraphConfig
    function: Callable[[GraphConfig, DictConfig], Graph]


def make_graph_function(graph_cfg: GraphConfig, cfg: DictConfig) -> Callable[[GraphConfig, DictConfig], Graph]:
    def make_graph(system: System):
        mesh = instantiate(cfg.graph.mesh, nx=graph_cfg.width, ny=graph_cfg.length)

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
            partitioner = None
            partition = graph.initial_mincut_partition(
                arch=DeviceType.GPU,
                bandwidth=cfg.system.d2d_bw,
                n_parts=4,
                offset=0,
            )
            partition = graph.maximize_matches(partition)
            logger.debug("Mincut partition: %s", partition)
        else:
            raise NotImplementedError(f"Partitioner {cfg.graph.init.partitioner} is not implemented.")

        if cfg.graph.init.gpu_only:
            partition = [x + 1 for x in partition]  # offset by 1 to ignore cpu
            location_list = [i + 1 for i in range(0, cfg.graph.init.nparts)]
        else:
            location_list = [i for i in range(cfg.graph.init.nparts + 1)]  # include cpu as 0

        if isinstance(graph, DynamicJacobiGraph):
            graph.set_cell_locations([-1 for _ in range(len(partition))])
            graph.set_cell_locations(partition, step=0)
        elif isinstance(graph, JacobiGraph):
            logger.debug("Setting cell locations with partition: %s", partition)
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
        logger.info("Graph info:\n%s", OmegaConf.to_yaml(cfg.graph.config))
    graph_config = instantiate(cfg.graph.config)
    graph_function = make_graph_function(graph_config, cfg)
    return GraphBuilder(config=graph_config, function=graph_function)
