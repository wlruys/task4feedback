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

def make_graph_function(
    graph_cfg: GraphConfig, cfg: DictConfig
) -> Callable[[GraphConfig, DictConfig], Graph]:
    def make_graph(system: System):
        mesh = instantiate(cfg.mesh, L=1, n=graph_cfg.n)

        if cfg.init.partitioner == "metis":
            partitioner = metis_geometry_partition
        else:
            raise NotImplementedError(
                f"Partitioner {cfg.init.partitioner} is not implemented."
            )

        geom = build_geometry(mesh)
        graph = build_graph(geom, graph_cfg, system=system)
        partition = partitioner(geom, nparts=cfg.init.nparts)

        if cfg.init.gpu_only:
            partition = [x + 1 for x in partition]  # offset by 1 to ignore cpu
            location_list = [i + 1 for i in range(0, cfg.init.nparts)]
        else:
            location_list = [i for i in range(cfg.init.nparts + 1)]  # include cpu as 0

        if isinstance(graph, DynamicJacobiGraph):
            graph.set_cell_locations([-1 for _ in range(len(partition))])
            graph.set_cell_locations(partition, step=0)
        else:
            graph.set_cell_locations(partition)

        if cfg.init.randomize:
            graph.randomize_locations(
                graph_cfg.randomness,
                location_list=location_list,
                step=0 if isinstance(graph, DynamicJacobiGraph) else None,
            )

        return graph

    graph_function = make_graph
    return graph_function

    return graph_function


def make_graph_builder(cfg: DictConfig) -> GraphBuilder:
    graph_info = cfg.graph
    graph_config = instantiate(graph_info.config)
    graph_function = make_graph_function(graph_config, graph_info)
    return GraphBuilder(config=graph_config, function=graph_function)
