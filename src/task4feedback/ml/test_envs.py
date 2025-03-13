from ..graphs import *
from ..graphs.jacobi import * 
from ..graphs.base import *
from .env import * 
from ..graphs.mesh import *
from .. graphs.mesh.base import *
from ..graphs.mesh.partition import *
from typing import Type 

def build_jacobi_graph(config: JacobiConfig) -> JacobiGraph:
    
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)
    
    jgraph = JacobiGraph(geom, config.steps)
    
    jgraph.apply_variant(JacobiVariant)
    
    partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=4)
    # print(partition)
    jgraph.set_cell_locations(partition)
    
    return jgraph
    
def make_jacobi_env(config: JacobiConfig):
    gmsh.initialize()
    s = uniform_connected_devices(5, 1000000000, 1, 2000)
    jgraph = build_jacobi_graph(config)
    


    d = jgraph.get_blocks()
    m = jgraph
    m.finalize_tasks()
    spec = create_graph_spec()
    input = SimulatorInput(m, d, s)
    env = RuntimeEnv(
        SimulatorFactory(input, spec, DefaultObserverFactory), device="cpu"
    )
    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())
    
    return env

def make_mapper_jacobi_env(config: JacobiConfig, external_mapper: Type[ExternalMapper] | ExternalMapper, internal_mapper: Type[fastsim.Mapper] | fastsim.Mapper):
    gmsh.initialize()
    s = uniform_connected_devices(5, 1000000000, 1, 2000)
    jgraph = build_jacobi_graph(config)
    


    d = jgraph.get_blocks()
    m = jgraph
    m.finalize_tasks()
    spec = create_graph_spec()
    input = SimulatorInput(m, d, s)
    env = MapperRuntimeEnv(
        SimulatorFactory(input, spec, DefaultObserverFactory, internal_mapper=internal_mapper, external_mapper=external_mapper), device="cpu"
    )
    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())
    
    return env