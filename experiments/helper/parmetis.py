from task4feedback.ml.env import RuntimeEnv
from task4feedback.graphs.jacobi import JacobiGraph
from task4feedback.graphs.base import weighted_cell_partition
from task4feedback.interface.wrappers import DeviceType, SimulatorDriver
import task4feedback.fastsim2 as fastsim
from task4feedback.fastsim2 import ParMETIS_wrapper
from mpi4py import MPI
import torch
import numpy as np
from task4feedback.graphs.jacobi import get_length_from_config
def run_parmetis(sim: SimulatorDriver,
                 cfg,
                 verbose=False,
                 offset=1,
                 future_levels=0,
                 itr: float = 1000,
                 unbalance: float = 1.225,
                 target_loads: list[float] = [0.25, 0.25, 0.25, 0.25],
                 n_compute_devices: int = 4) -> int:
    d2d_bandwidth = cfg.system.d2d_bw
    graph_config = hydra.utils.instantiate(cfg.graph.config)
    width = graph_config.n 
    length = get_length_from_config(graph_config)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise ValueError(
            f"Expected 4 ranks, but got {size}. Please run with 4 ranks."
        )
    partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    ParMETIS = ParMETIS_wrapper()
    done = False
    
    if rank == 0:
        graph = sim.input.graph
        assert isinstance(graph, JacobiGraph), "Graph must be a JacobiGraph"
        cell_graph = graph.get_weighted_cell_graph(
                    DeviceType.GPU,
                    bandwidth=d2d_bandwidth,
                    levels=[0, 1],
                )
        edge_cut, partition = weighted_cell_partition(
            cell_graph, nparts=(cfg.system.n_devices - 1)
        )
        cell_to_device = [x + offset for x in partition]
        partition = [-1 for _ in range(sim.observer.graph_spec.max_candidates)]
        sim.enable_external_mapper()
        done = sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE
        candidates = torch.zeros(
            (sim.observer.graph_spec.max_candidates), dtype=torch.int64
        )
        sim.get_mappable_candidates(candidates)
        actions = []
        for i, id in enumerate(candidates):
            mapping_priority = sim.get_mapping_priority(id)
            actions.append(
                fastsim.Action(
                    i,
                    cell_to_device[graph.task_to_cell[id.item()]],
                    mapping_priority,
                    mapping_priority,
                )
            )
        sim.simulator.map_tasks(actions)
        done = sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE
        if verbose:
            for i, loc in enumerate(cell_to_device):
                print(loc, end=" ")
                if (i + 1) % width == 0:
                    print()
    while True:
        done = comm.bcast(done, root=0)
        if done:
            break
        if rank == 0:
            sim.get_mappable_candidates(candidates)
            for i, id in enumerate(candidates):
                partition[i] = (
                    cell_to_device[graph.task_to_cell[id.item()]] - offset
                )
            partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = (
                graph.get_distributed_weighted_graph(
                    bandwidth=d2d_bandwidth,
                    task_ids=candidates.tolist(),
                    partition=partition,
                    future_levels=future_levels,
                    width=width,
                    n_compute_devices=n_compute_devices,
                )
            )
        vtxdist = comm.bcast(vtxdist, root=0)
        xadj = comm.bcast(xadj, root=0)
        adjncy = comm.bcast(adjncy, root=0)
        vwgt = comm.bcast(vwgt, root=0)
        adjwgt = comm.bcast(adjwgt, root=0)
        vsize = comm.bcast(vsize, root=0)

        xadj = xadj[rank]
        adjncy = adjncy[rank]
        vwgt = vwgt[rank]
        adjwgt = adjwgt[rank]
        vsize = vsize[rank]
        wgtflag = 3
        numflag = 0
        ncon = 1
        tpwgts = np.array(target_loads, dtype=np.float32)
        ubvec = np.array([unbalance], dtype=np.float32)
        part = np.array([-1 for _ in range(width**2)], dtype=np.int32)
        comm.Barrier()
        ParMETIS.callParMETIS(
            vtxdist,
            xadj,
            adjncy,
            vwgt,
            vsize,
            adjwgt,
            wgtflag,
            numflag,
            ncon,
            tpwgts,
            ubvec,
            itr,
            part,
        )
        parts = comm.gather(part, root=0)

        if rank == 0:
            for i, p in enumerate(parts):
                for j, dev in enumerate(p):
                    if dev == -1:
                        break
                    task_id = partitioned_tasks[i][j]
                    cell_to_device[graph.task_to_cell[task_id]] = (
                        int(dev) + offset
                    )  # Offset by 1 to ignore CPU
            actions = []
            for i, id in enumerate(candidates):
                mapping_priority = sim.get_mapping_priority(id)
                actions.append(
                    fastsim.Action(
                        i,
                        cell_to_device[graph.task_to_cell[id.item()]],
                        mapping_priority,
                        mapping_priority,
                    )
                )
            sim.simulator.map_tasks(actions)
            done = sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE
            if verbose:
                print()
                for i, loc in enumerate(cell_to_device):
                    print(loc, end=" ")
                    if (i + 1) % 8 == 0:
                        print()