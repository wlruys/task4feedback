from collections.abc import Callable

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from task4feedback.graphs import *
from task4feedback.graphs.mesh import (
    build_geometry,
)
from task4feedback.graphs.mesh.partition import *


@dataclass
class GraphBuilder:
    config: GraphConfig
    function: Callable[[GraphConfig, DictConfig], Graph]


def make_graph_function(
    graph_cfg: GraphConfig, cfg: DictConfig
) -> Callable[[GraphConfig, DictConfig], Graph]:
    def make_graph(system: System):
        mesh = instantiate(
            cfg.graph.mesh, L=1, n=graph_cfg.n, domain_ratio=graph_cfg.domain_ratio
        )

        geom = build_geometry(mesh)
        graph = build_graph(geom, graph_cfg, system=system)

        if isinstance(graph, DynamicJacobiGraph | JacobiGraph):
            # Initial partitioning
            if cfg.graph.init.partitioner == "metis" and cfg.system.n_devices > 2:
                graph.make_partition = graph.initial_mincut_partition
            elif cfg.graph.init.partitioner == "quad":
                graph.make_partition = graph.quadrant_partition
            else:
                # Default to quadrant partition
                graph.make_partition = graph.quadrant_partition

            partition = graph.make_partition(
                arch=DeviceType.GPU,
                bandwidth=cfg.system.d2d_bw,
                n_parts=cfg.system.n_devices - 1,
                offset=0,
            )

            # partition needs minimum number of flips from initial partition
            partition = graph.maximize_matches(partition)

        if cfg.graph.init.gpu_only:
            partition = [x + 1 for x in partition]  # offset by 1 to ignore cpu
            location_list = [i + 1 for i in range(0, cfg.graph.init.nparts)]
        else:
            location_list = list(range(cfg.graph.init.nparts + 1))  # include cpu as 0

        if isinstance(graph, DynamicJacobiGraph):
            graph.set_cell_locations([-1 for _ in range(len(partition))])
            graph.set_cell_locations(partition, step=0)
        elif isinstance(graph, JacobiGraph):
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


def make_graph_builder(cfg: DictConfig, verbose: bool = False) -> GraphBuilder:
    if verbose:
        print(f"Graph info: {OmegaConf.to_yaml(cfg.graph.config)}")
    graph_config = instantiate(cfg.graph.config)
    graph_function = make_graph_function(graph_config, cfg)
    return GraphBuilder(config=graph_config, function=graph_function)
