from task4feedback.graphs.mesh.base import *
from task4feedback.graphs.mesh.partition import *
from task4feedback.graphs.mesh.plot import *
from task4feedback.graphs.base import *
from task4feedback.graphs.jacobi import (
    JacobiGraph,
    JacobiVariant,
    JacobiVariantGPUOnly,
    JacobiConfig,
    OnlyXYObserverFactory,
    XYDataObserverFactory,
)
from task4feedback.interface import TaskTuple
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *

from dataclasses import dataclass
from task4feedback.ml.ppo import *
import task4feedback as fastsim

# import task4feedback.fastsim2 as fastsim
from task4feedback.interface.wrappers import (
    SimulatorFactory,
    create_graph_spec,
    ExternalObserverFactory,
    FeatureExtractorFactory,
    EdgeFeatureExtractorFactory,
)
from torchrl.envs import StepCounter, TrajCounter, TransformedEnv, Compose
from task4feedback.ml.models import *
import shutil
import os
import sys
from torch import multiprocessing

WIDTH = 4
STEPS = 20
N_DEVICES = 5
WORKERS = 4
GRAPHS_PER_WORKER = 1
GRAPHS_PER_BATCH = WORKERS * GRAPHS_PER_WORKER
MAPPING = [1, 2, 1, 1, 3, 3, 2, 3, 2, 4, 4, 3, 1, 4, 2, 4]
CHANGE_PRIORITY = True
CHANGE_LOCATION = True
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
device = torch.device("cpu")
ENV = EFTIncrementalEnv


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def build_jacobi_graph(config: JacobiConfig) -> JacobiGraph:
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)

    jgraph = JacobiGraph(geom, config)

    jgraph.apply_variant(JacobiVariant)

    partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=4)

    # offset by 1 to ignore cpu
    partition = [x + 1 for x in partition]
    jgraph.set_cell_locations(partition)

    if N_DEVICES == 4:
        partition = [i - 1 for i in MAPPING]

    jgraph.set_cell_locations(partition)

    jgraph.randomize_locations(config.randomness, location_list=[1, 2, 3, 4])

    return jgraph


if __name__ == "__main__":
    rtc = True

    def make_env() -> ENV:
        gmsh.initialize()
        graph = build_jacobi_graph(
            JacobiConfig(
                L=1,
                n=WIDTH,
                steps=STEPS,
                n_part=4,
                randomness=1,
                permute_idx=0,
                interior_size=2000000,
            )
        )
        s = uniform_connected_devices(N_DEVICES, 1000000000, 1, 2000)
        d = graph.get_blocks()
        m = graph
        m.finalize_tasks()
        spec = create_graph_spec(
            max_devices=N_DEVICES,
            max_tasks=30,
            max_data=100,
            max_edges_tasks_data=150,
            max_edges_tasks_tasks=150,
        )
        input = SimulatorInput(
            m,
            d,
            s,
            transition_conditions=fastsim.BatchTransitionConditions(5, 2, 16),
        )
        env = ENV(
            SimulatorFactory(
                input,
                spec,
                OnlyXYObserverFactory(spec),
            ),
            device=device,
            change_priority=CHANGE_PRIORITY,
            change_locations=CHANGE_LOCATION,
            only_gpu=N_DEVICES == 5,
            # bonus_scale=1,
        )
        env = TransformedEnv(env, Compose(StepCounter(), TrajCounter()))

        return env

    env = make_env()

    ppo_config = PPOConfig(
        collect_device=device,
        update_device="cuda",
        workers=WORKERS,
        states_per_collection=WIDTH * WIDTH * STEPS * GRAPHS_PER_BATCH,
        minibatch_size=WIDTH * WIDTH * STEPS * GRAPHS_PER_BATCH // 4,
        gae_gamma=1,
        gae_lmbda=0.1,
        num_collections=1000,
        eval_interval=50,
        normalize_advantage=False,
        clip_vloss=False,
    )
    feature_config = FeatureDimConfig.from_observer(env.observer)
    layer_config = LayerConfig(hidden_channels=64, n_heads=2)
    model = OldSeparateNetwDevice(
        feature_config=feature_config,
        layer_config=layer_config,
        n_devices=5,
    )
    wandb.init(
        project="Jacobi_debug",
        name="TestCombined",
        config=vars(ppo_config),
    )
    # current_file = os.path.splitext(os.path.basename(__file__))[0]
    # PATH = os.path.join(os.path.dirname(__file__), current_file)
    # os.makedirs(PATH, exist_ok=True)
    # shutil.copy(os.path.abspath(sys.argv[0]), PATH)

    run_ppo_torchrl(
        model,
        make_env,
        ppo_config,
    )
