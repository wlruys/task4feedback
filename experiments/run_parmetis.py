import hydra
from omegaconf import DictConfig, OmegaConf
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *
from task4feedback.ml.algorithms import *
from task4feedback.interface.wrappers import start_logger
import wandb
import gmsh
from hydra.utils import instantiate
from hydra.experimental.callbacks import Callback

from helper.graph import make_graph_builder, GraphBuilder
from helper.env import make_env
from helper.model import *

from task4feedback.graphs.jacobi import (
    JacobiGraph,
    LevelPartitionMapper,
    JacobiRoundRobinMapper,
)
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import numpy
from collections import defaultdict
import time
from task4feedback.fastsim2 import ParMETIS_wrapper
import task4feedback.fastsim2 as fastsim
from mpi4py import MPI
import comm
from task4feedback.graphs.base import weighted_cell_partition

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size != 4:
    raise ValueError(
        f"Expected 4 ranks, but got {size}. Please run with 4 ranks."
    )
ParMETIS = ParMETIS_wrapper()


class GitInfo(Callback):
    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        try:
            repo = git.Repo(search_parent_directories=True)
            outdir = Path(config.hydra.runtime.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "git_sha.txt").write_text(repo.head.commit.hexsha)
            (outdir / "git_dirty.txt").write_text(str(repo.is_dirty()))
            diff = repo.git.diff(None)
            (outdir / "git_diff.patch").write_text(diff)

            print(
                "Git SHA:",
                repo.head.commit.hexsha,
                " (dirty)" if repo.is_dirty() else " (clean)",
                flush=True,
            )

        except Exception as e:
            print(f"GitInfo callback failed: {e}")


@hydra.main(config_path="conf", config_name="dynamic_batch", version_base=None)
def main(cfg: DictConfig):
    global ParMETIS
    # Seed for reproducibility
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    done = False
    d2d_bandwidth = cfg.system.d2d_bw
    partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    # Initial setup
    if rank == 0:
        graph_builder = make_graph_builder(cfg)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
    for i in range(1):
        if rank == 0:
            env._reset()
            sim = env.simulator
            eft_sim = sim.copy()
            eft_sim.disable_external_mapper()
            eft_sim.run()
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
            cell_loc = [x + 1 for x in partition]
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
                        cell_loc[graph.task_to_cell[id.item()]],
                        mapping_priority,
                        mapping_priority,
                    )
                )
            sim.simulator.map_tasks(actions)
            done = sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE
            for i, loc in enumerate(cell_loc):
                print(loc, end=" ")
                if (i + 1) % 8 == 0:
                    print()
        while True:
            if rank == 0:
                sim.get_mappable_candidates(candidates)
                if not done:
                    # This step is necessary since the order of candidates differ probably can change this to reordering just before mapping
                    for i, id in enumerate(candidates):
                        partition[i] = (
                            cell_loc[graph.task_to_cell[id.item()]] - 1
                        )  # Offset by 1 to ignore CPU
                    # print(partition)
                    partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = (
                        graph.get_distributed_weighted_graph(
                            bandwidth=d2d_bandwidth,
                            task_ids=candidates.tolist(),
                            partition=partition,
                            future_levels=0,
                        )
                    )
            done = comm.bcast(done, root=0)
            if done:
                break
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
            tpwgts = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
            ubvec = np.array([1.4], dtype=np.float32)
            itr = 10.0
            part = np.array([-1 for _ in range(64)], dtype=np.int32)
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
                        cell_loc[graph.task_to_cell[task_id]] = (
                            int(dev) + 1
                        )  # Offset by 1 to ignore CPU
                actions = []
                for i, id in enumerate(candidates):
                    mapping_priority = sim.get_mapping_priority(id)
                    actions.append(
                        fastsim.Action(
                            i,
                            cell_loc[graph.task_to_cell[id.item()]],
                            mapping_priority,
                            mapping_priority,
                        )
                    )
                sim.simulator.map_tasks(actions)
                done = sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE
                print()
                for i, loc in enumerate(cell_loc):
                    print(loc, end=" ")
                    if (i + 1) % 8 == 0:
                        print()
        if rank == 0:
            print(f"EFT:{eft_sim.time},PARMETIS:{sim.time},{eft_sim.time / sim.time:.2f}")
            config = EvaluationConfig()
            animate_mesh_graph(
                env,
                time_interval=int(env.simulator.time / config.max_frames),
                show=False,
                title="outputs/parmetis_result",
                figsize=config.fig_size,
                dpi=config.dpi,
                bitrate=config.bitrate,
            )


if __name__ == "__main__":
    main()
