from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *

from dataclasses import dataclass
from task4feedback.ml.ppo import *
from typing import Callable, List, Self
import numpy as np
from task4feedback import fastsim2 as fastsim
from task4feedback.interface import *
import torch
from typing import Optional, Self

from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import (
    DefaultObserverFactory,
    CompiledDefaultObserverFactory,
    SimulatorDriver,
    SimulatorFactory,
    create_graph_spec,
    start_logger,
    observation_to_heterodata,
    observation_to_heterodata_truncate,
)

import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")  # must be before DataLoader / mp.spawn


# from task4feedback.interface.wrappers import (
#     observation_to_heterodata_truncate as observation_to_heterodata,
# )
from torchrl.data import Composite, TensorSpec, Unbounded, Binary, Bounded
from torchrl.envs.utils import make_composite_from_td
from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs import check_env_specs
from tensordict import TensorDict
from torch_geometric.data import HeteroData, Batch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, HeteroConv
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import StepCounter, TrajCounter, TransformedEnv
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor
import tensordict
from tensordict.nn import (
    TensorDictModule,
    ProbabilisticTensorDictModule,
    TensorDictSequential,
)
import torchrl
import torch_geometric
from task4feedback.graphs.sweep import *
from task4feedback.graphs import *
from task4feedback.graphs.mesh import *
from task4feedback.graphs.mesh.base import *
from task4feedback.graphs.mesh.partition import *
from task4feedback.graphs.mesh.plot import *
from task4feedback.graphs.base import *
from task4feedback.graphs.jacobi import *
from task4feedback.interface import TaskTuple
import time
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *
import wandb
from task4feedback.graphs.dynamic_jacobi import *

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def build_jacobi_graph(config: JacobiConfig, metis_init=False, nparts=4) -> JacobiGraph:
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)

    jgraph = JacobiGraph(geom, config)
    jgraph.apply_variant(JacobiVariant)

    if metis_init:
        partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=nparts)
        # offset by 1 to ignore cpu
        partition = [x + 1 for x in partition]
        jgraph.set_cell_locations(partition)

    jgraph.randomize_locations(config.randomness, location_list=[1, 2, 3, 4])

    return jgraph


def build_dynamic_jacobi_graph(
    config: DynamicJacobiConfig, metis_init=False, nparts=4
) -> DynamicJacobiGraph:
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)

    jgraph = DynamicJacobiGraph(geom, config)

    if metis_init:
        partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=nparts)

        # offset by 1 to ignore cpu
        partition = [x + 1 for x in partition]
        jgraph.set_cell_locations(partition)

    jgraph.randomize_locations(config.randomness, location_list=[1, 2, 3, 4])

    return jgraph


def build_sweep_graph(config: SweepConfig) -> SweepGraph:
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)

    jgraph = SweepGraph(geom, config)
    jgraph.apply_variant(SweepVariant)

    partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=4)

    # offset by 1 to ignore cpu
    partition = [x + 1 for x in partition]
    jgraph.set_cell_locations(partition)

    jgraph.randomize_locations(config.randomness, location_list=[1, 2, 3, 4])

    return jgraph


def make_env(
    graph_builder,
    graph_config,
    system_config,
    feature_config=None,
    runtime_env_t=RuntimeEnv,
    observer_factory_t=XYDataObserverFactory,
    change_priority=True,
    change_locations=True,
    seed=1000,
):
    gmsh.initialize()

    n_devices = system_config["n_devices"]
    bandwidth = system_config["bandwidth"]
    latency = system_config["latency"]
    s = uniform_connected_devices(n_devices, 100000000000000, latency, bandwidth)
    jgraph = graph_builder(graph_config)

    d = jgraph.get_blocks()
    m = jgraph
    m.finalize_tasks()

    if feature_config is None:
        feature_config = {
            "observer_factory": "XYObserverFactory",
            "max_tasks": 100,
            "max_data": 100,
            "max_edges_tasks_tasks": 200,
            "max_edges_tasks_data": 200,
            "max_edges_data_devices": 100,
            "max_edges_tasks_devices": 100,
        }

    max_tasks = feature_config.get("max_tasks", 100)
    max_data = feature_config.get("max_data", 100)
    max_edges_tasks_tasks = feature_config.get("max_edges_tasks_tasks", 200)
    max_edges_tasks_data = feature_config.get("max_edges_tasks_data", 200)
    max_edges_data_devices = feature_config.get("max_edges_data_devices", 100)
    max_edges_tasks_devices = feature_config.get("max_edges_tasks_devices", 100)

    spec = create_graph_spec(
        max_tasks=max_tasks,
        max_data=max_data,
        max_edges_tasks_tasks=max_edges_tasks_tasks,
        max_edges_tasks_data=max_edges_tasks_data,
        max_edges_data_devices=max_edges_data_devices,
        max_edges_tasks_devices=max_edges_tasks_devices,
    )

    input = SimulatorInput(
        m, d, s, transition_conditions=fastsim.BatchTransitionConditions(5, 2, 16)
    )

    env = runtime_env_t(
        SimulatorFactory(input, spec, observer_factory_t),
        device="cpu",
        change_priority=change_priority,
        seed=seed,
        change_locations=change_locations,
    )
    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())

    return env


def layer_init(layer, a=0.01, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.kaiming_uniform_(layer.weight, a=a, nonlinearity="leaky_relu")
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def init_weights(m):
    """
    Initializes LayerNorm layers.
    """
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def loop_through_config(run_config, source_config):
    for key, value in source_config.items():
        if isinstance(value, list):
            run_config[key] = value
        elif isinstance(value, dict):
            run_config[key] = loop_through_config({}, value)
        else:
            run_config[key] = value
    return run_config


def train(wandb_config):
    graph_info = wandb_config["graph_config"]
    graph_class = globals()[graph_info["graph_class"]]

    if graph_class == JacobiGraph:
        graph_config = JacobiConfig(
            interior_size=graph_info["interior_size"],
            boundary_interior_ratio=graph_info["boundary_interior_ratio"],
            randomness=graph_info["randomness"],
            L=graph_info["L"],
            n=graph_info["n"],
            steps=graph_info["steps"],
        )
        graph_function = build_jacobi_graph
    elif graph_class == DynamicJacobiGraph:
        graph_config = DynamicJacobiConfig(
            interior_size=graph_info["interior_size"],
            boundary_interior_ratio=graph_info["boundary_interior_ratio"],
            randomness=graph_info["randomness"],
            L=graph_info["L"],
            n=graph_info["n"],
            steps=graph_info["steps"],
            start_workload=graph_info["start_workload"],
            lower_workload=graph_info["lower_workload"],
            upper_workload=graph_info["upper_workload"],
            step_size=graph_info["step_size"],
            correlation_scale=graph_info["correlation_scale"],
        )
        graph_function = build_dynamic_jacobi_graph
    else:
        raise ValueError(f"Unsupported graph class: {graph_class}")

    # Set up observer factory based on configuration
    feature_config_info = wandb_config["feature_config"]
    reward_config_info = wandb_config["reward_config"]
    observer_factory_type = globals()[feature_config_info["observer_factory"]]
    runtime_env_type = globals()[reward_config_info["runtime_env"]]

    # Get system configuration
    system_config = wandb_config["system_config"]

    # Create environment
    env_config = wandb_config["env_config"]
    env = make_env(
        graph_function,
        graph_config,
        system_config,
        feature_config=feature_config_info,
        runtime_env_t=runtime_env_type,
        observer_factory_t=observer_factory_type,
        change_priority=env_config["change_priority"],
        change_locations=env_config["change_locations"],
        seed=env_config["seed"],
    )

    graph = env.simulator.input.graph
    n_tasks = len(graph)
    print(f"Number of tasks in Env Graph: {n_tasks}")

    # Initialize WandB run
    wandb_run_config = wandb_config["wandb_config"]
    run = wandb.init(
        project=wandb_run_config["project"],
        name=wandb_run_config["name"],
    )

    # Extract feature dimensions from the environment's observer
    feature_config = FeatureDimConfig.from_observer(env.observer)

    # Set up neural network layer configuration
    layer_config = LayerConfig(
        hidden_channels=wandb_config["layer_config"]["hidden_channels"],
        n_heads=wandb_config["layer_config"]["n_heads"],
    )

    # Create model based on configuration
    model_config_info = wandb_config["model_config"]
    model_class = globals()[model_config_info["model_architecture"]]

    model = model_class(
        feature_config=feature_config,
        layer_config=layer_config,
        n_devices=wandb_config["system_config"]["n_devices"],
    )

    # Log model parameters count
    num_params = count_parameters(model)
    # Get the mconfig dictionary safely, defaulting to an empty dict if not found
    mconfig = wandb_config.get("mconfig", {})

    print("Feature config:", feature_config)

    print("Feature config info:", feature_config_info)

    train_config = PPOConfig(
        collect_device="cpu",
        update_device="cpu",
        # collect_device=mconfig.get("collect_device", "cpu"),
        # update_device=mconfig.get("update_device", "cpu"),
        workers=mconfig.get("workers", 1),
        ent_coef=mconfig.get("ent_coef", 0.05),
        gae_lmbda=mconfig.get("gae_lmbda", 1),
        gae_gamma=mconfig.get("gae_gamma", 0.99),
        normalize_advantage=mconfig.get("normalize_advantage", False),
        clip_eps=mconfig.get("clip_eps", 0.2),
        clip_vloss=mconfig.get("clip_vloss", False),
        minibatch_size=mconfig.get("minibatch_size", 250),
        eval_interval=mconfig.get("eval_interval", 50),
        eval_episodes=mconfig.get("eval_episodes", 1),
        states_per_collection=n_tasks * mconfig.get("graphs_per_collection", 10),
        max_grad_norm=mconfig.get("max_grad_norm", 10),
    )

    # Define environment creation function for PPO
    def make_env_fn():
        return make_env(
            graph_function,
            graph_config,
            system_config,
            feature_config=feature_config_info,
            runtime_env_t=runtime_env_type,
            observer_factory_t=observer_factory_type,
            change_priority=env_config["change_priority"],
            change_locations=env_config["change_locations"],
            seed=env_config["seed"],
        )

    # Define a fixed evaluation environment function
    def make_eval_env_fn():
        return make_env(
            graph_function,
            graph_config,
            system_config,
            feature_config=feature_config_info,
            runtime_env_t=runtime_env_type,
            observer_factory_t=observer_factory_type,
            change_priority=False,
            change_locations=False,
            seed=42,
        )

    wandb_params = {}

    wandb_params["graph"] = {
        "class": graph_class.__name__,
        "config": graph_config,
    }

    wandb_params["system"] = system_config

    wandb_params["model"] = {
        "architecture": model_class.__name__,
        "parameters": num_params,
        "hidden_channels": layer_config.hidden_channels,
        "n_heads": layer_config.n_heads,
    }

    print("NUMBER OF PARAMETERS: ", num_params)

    feature_params = env.observer.store_feature_types()
    wandb_params["features"] = {
        "observer_type": feature_config_info["observer_factory"],
    }

    for key, value in feature_params.items():
        clean_key = key.replace("features", "").replace("_types", "")
        if isinstance(value, list):
            wandb_params["features"][clean_key] = value

    wandb_params["training"] = train_config

    wandb_params["environment"] = {
        "change_priority": env_config["change_priority"],
        "change_locations": env_config["change_locations"],
        "seed": env_config["seed"],
    }

    run.config.update(wandb_params)

    model_layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                model_layers.append([name, module.__class__.__name__, params])

    wandb.log(
        {
            "model_architecture": wandb.Table(
                columns=["Layer", "Type", "Parameters"], data=model_layers
            )
        }
    )

    # try:
    #     model = torch.compile(model, fullgraph=True)
    #     print("Using compiled model")
    # except AttributeError:
    #     print("torch.compile not available, using uncompiled model")

    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    # Run PPO training
    run_ppo_cleanrl_no(
        model,
        make_env_fn,
        train_config,
    )
    # run_ppo_cleanrl(
    #     model,
    #     make_env_fn,
    #     train_config,
    # )


if __name__ == "__main__":
    wandb_config = {
        "graph_config": {
            "graph_class": "JacobiGraph",
            "interior_size": 1000,
            "boundary_interior_ratio": 1,
            "randomness": 1,
            "L": 1,
            "n": 4,
            "steps": 5,
            "start_workload": 1000,
            "lower_workload": 500,
            "upper_workload": 2000,
            "step_size": 10000,
            "correlation_scale": 0.1,
        },
        "reward_config": {
            "runtime_env": "GeneralizedIncrementalEFT",
        },
        "system_config": {
            "type": "uniform_connected_devices",
            "n_devices": 5,
            "bandwidth": 1,
            "latency": 1,
        },
        "feature_config": {
            "observer_factory": "XYHeterogeneousObserverFactory",
            "max_tasks": 50,
            "max_data": 80,
            "max_edges_tasks_tasks": 100,
            "max_edges_tasks_data": 200,
            "max_edges_data_devices": 320,
            "max_edges_tasks_devices": 50,
        },
        "layer_config": {
            "hidden_channels": 16,
            "n_heads": 2,
        },
        "mconfig": {
            "graphs_per_collection": 8,
            "collect_device": "cpu",
            "update_device": "cpu",
            "workers": 8,
            "ent_coef": 0,
            "gae_lmbda": 0.99,
            "gae_gamma": 1,
            "normalize_advantage": False,
            "clip_eps": 0.2,
            "clip_vloss": False,
            "minibatch_size": 512,
        },
        "env_config": {
            "change_priority": False,
            "change_locations": False,
            "seed": 1,
        },
        "model_config": {
            "model_architecture": "HeteroConvSeparateNet",
        },
        "wandb_config": {
            "project": "test",
            "name": "Random",
        },
    }

    train(wandb_config)
