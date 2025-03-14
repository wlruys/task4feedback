from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import RuntimeEnv
from task4feedback.interface.wrappers import (
    DefaultObserverFactory,
    SimulatorFactory,
    create_graph_spec,
)
from torchrl.envs import StepCounter, TrajCounter, TransformedEnv
from task4feedback.ml.ppo import *
import numpy as np
from task4feedback.interface import *
from task4feedback.legacy_graphs import *
import torch
import itertools


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
n_devices = 4  # including CPU
task_duration = 1000
bandwidth = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_simple_env(tasks, data):
    s = uniform_connected_devices(n_devices, 1000000000, 1, bandwidth=bandwidth)
    d = DataBlocks.create_from_legacy_data(data, s)
    m = Graph.create_from_legacy_graph(tasks, data)
    m.finalize_tasks()
    spec = create_graph_spec(max_devices=n_devices)
    input = SimulatorInput(m, d, s)
    env = RuntimeEnv(
        SimulatorFactory(input, spec, DefaultObserverFactory),
        device=device,
        use_eft=True,
    )
    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())
    return env


def make_test_stencil_graph():
    interior_size = task_duration * bandwidth
    boundary_size = task_duration * bandwidth
    all_combinations = [list(c) for c in itertools.permutations(range(4))]

    def task_config(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()
        placement_info.add(
            (Device(Architecture.GPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=1),
        )
        placement_info.add(
            (Device(Architecture.CPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=1),
        )

        return placement_info

    def sizes(data_id: DataID) -> int:
        return boundary_size if data_id.idx[1] == 1 else interior_size

    def blocked_initial_placement(data_id: DataID) -> List[Device]:
        batch = data_config.width // (n_devices // 2)

        device_id_i = int(data_id.idx[-2] // batch)
        device_id_j = int(data_id.idx[-1] // batch)
        idx = device_id_i + (n_devices // 2) * device_id_j

        dev_id = idx % n_devices
        if dev_id == 0:
            return Device(Architecture.CPU, 0)
        else:
            return Device(Architecture.GPU, dev_id - 1)

    def cpu_placement(data_id: DataID) -> List[Device]:
        return Device(Architecture.CPU, 0)

    data_config = StencilDataGraphConfig()
    data_config.n_devices = n_devices
    data_config.dimensions = 2
    data_config.width = 4
    data_config.initial_placement = blocked_initial_placement
    data_config.initial_sizes = sizes
    config = StencilConfig(width=4, steps=14, task_config=task_config)
    tasks, data = make_graph(config, data_config=data_config)
    return tasks, data


if __name__ == "__main__":
    tasks, blocks = make_test_stencil_graph()

    def make_env():
        return make_simple_env(tasks, blocks)

    env = make_env()

    feature_config = FeatureDimConfig.from_observer(env.observer)
    layer_config = LayerConfig(hidden_channels=64, n_heads=2)
    model = OldSeparateNet(
        feature_config=feature_config,
        layer_config=layer_config,
        n_devices=n_devices,
    )
    config = PPOConfig(train_device=device)
    run_ppo_torchrl(
        model,
        make_env,
        config,
        wandb_project="test_stencil",
        wandb_exp_name="test_stencil",
    )
