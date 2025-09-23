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

        if cfg.graph.init.partitioner == "metis":
            partitioner = metis_geometry_partition
        else:
            raise NotImplementedError(f"Partitioner {cfg.graph.init.partitioner} is not implemented.")

        geom = build_geometry(mesh)
        graph = build_graph(geom, graph_cfg, system=system)
        # partition = partitioner(geom, nparts=cfg.graph.init.nparts)
        # partition = block_cyclic(geom)
        partition = graph.initial_mincut_partition(
            arch=DeviceType.GPU,
            bandwidth=cfg.system.d2d_bw,
            n_parts=4,
            offset=0,
        )
        # print(partition)

        """ 
        def initial_mincut_partition(
        self,
        arch: DeviceType = DeviceType.GPU,
        bandwidth: int = 1000,
        n_parts: int = 4,
        offset: int = 1,  # 1 to ignore cpu
    ):"""

        if cfg.graph.init.gpu_only:
            partition = [x + 1 for x in partition]  # offset by 1 to ignore cpu
            location_list = [i + 1 for i in range(0, cfg.graph.init.nparts)]
        else:
            location_list = [i for i in range(cfg.graph.init.nparts + 1)]  # include cpu as 0

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


def make_graph_builder(cfg: DictConfig, verbose: bool = True) -> GraphBuilder:
    if verbose:
        print(f"Graph info: {OmegaConf.to_yaml(cfg.graph.config)}")
    graph_config = instantiate(cfg.graph.config)
    graph_function = make_graph_function(graph_config, cfg)
    return GraphBuilder(config=graph_config, function=graph_function)
