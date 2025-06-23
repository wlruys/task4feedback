from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torch import nn
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import IQLLoss, SoftUpdate, DiscreteIQLLoss, HardUpdate, DQNLoss
from ...interface.wrappers import *
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data.replay_buffers import (
    ReplayBuffer,
    TensorDictReplayBuffer,
    ReplayBufferEnsemble,
    LazyMemmapStorage,
    LazyTensorStorage,
)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, Sampler
from typing import Callable, Type, Self, Optional
from ..util import *
from ..models import *
from ..env import *
from torchrl.modules import (
    ProbabilisticActor,
    ValueOperator,
    ActorCriticWrapper,
    QValueActor,
)
import wandb
from task4feedback.graphs.mesh.base import *
from task4feedback.graphs.mesh.partition import *
from task4feedback.graphs.mesh.plot import *
from task4feedback.graphs.base import *
from task4feedback.graphs.jacobi import *
from torchrl.envs.transforms import Reward2GoTransform
from torchrl.data import Categorical
import pickle
from torchrl.modules import EGreedyModule
from tensordict.nn import CudaGraphModule, TensorDictSequential


@torch.no_grad()
def evaluate_policy(
    step,
    graph_func,
    sys_func,
    config_list,
    policy_network,
    animate=False,
    rtc=True,
    title="",
    observer_type=XYObserverFactory,
):
    """
    Evaluate the policy on the environment.
    """

    def env_wrapper(config) -> RuntimeEnv:
        graph = graph_func(config)
        s = sys_func()
        d = graph.get_blocks()
        m = graph
        m.finalize_tasks()
        spec = create_graph_spec()

        if rtc:
            input = SimulatorInput(
                m,
                d,
                s,
                transition_conditions=fastsim.RangeTransitionConditions(5, 5, 16),
            )
        else:
            input = SimulatorInput(
                m, d, s, transition_conditions=fastsim.DefaultTransitionConditions()
            )
        env = RuntimeEnv(
            SimulatorFactory(
                input,
                spec,
                observer_type,
            ),
            device="cpu",
        )
        return env

    episode_rewards = []
    for i, config in enumerate(config_list):
        env = env_wrapper(config)

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            rout = env.rollout(225, policy=policy_network)
            returns = rout["next", "reward"].sum().item()
            episode_rewards.append(returns)

        if animate and i == len(config_list) - 1:
            # Animate the last environment
            title = f"{title}_network_eval_{step}_{rtc}"
            print(title)
            animate_mesh_graph(env, time_interval=250, show=False, title=title)

    return np.mean(episode_rewards)


def run_dqn(
    graph_func,
    sys_func,
    config_list,
    actor_model,
    eval_set,
    steps=10000,
    rtc=True,
):
    if rtc:
        tag = ["rtc"]
    else:
        tag = ["non_rtc"]

    wandb.init(
        project="dqn",
        tags=tag,
        config={
            "steps": steps,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "rtc": rtc,
            "actor_model": actor_model.__class__.__name__,
        },
    )

    def env_wrapper() -> RuntimeEnv:
        config = config_list[0]
        graph = graph_func(config)
        s = sys_func()
        d = graph.get_blocks()
        m = graph
        m.finalize_tasks()
        spec = create_graph_spec()
        input = SimulatorInput(
            m, d, s, transition_conditions=fastsim.DefaultTransitionConditions()
        )
        env = RuntimeEnv(
            SimulatorFactory(
                input,
                spec,
                XYDataObserverFactory,
            ),
            change_priority=False,
            device="cpu",
        )

        env = TransformedEnv(env, StepCounter())

        return env

    action_spec = Categorical(
        n=4,
    )
    composite_action_spec = Composite(action=action_spec)

    graph_func, sys_func, config_list = eval_set
    actor_model_td = HeteroDataWrapper(actor_model)
    r2g = Reward2GoTransform(gamma=1, out_keys=["value_target"])

    qvalue_module = QValueActor(
        module=actor_model_td,
        in_keys=["observation"],
        spec=composite_action_spec,
    )

    # Load values from state_dict save
    # qvalue_module.load_state_dict(
    #     torch.load(f"dqn_model_{291}_2.pth"),
    #     strict=False,
    # )

    greedy_module = EGreedyModule(
        annealing_num_steps=1_000_000,
        eps_init=0.75,
        eps_end=0.05,
        spec=qvalue_module.spec,
    )

    model_explore = TensorDictSequential(
        qvalue_module,
        greedy_module,
    )

    loss_module = DQNLoss(
        value_network=qvalue_module,
        loss_function="l2",
        double_dqn=True,
        delay_value=True,
        action_space=action_spec,
    )

    from torchrl.objectives import ValueEstimators

    loss_module.make_value_estimator(ValueEstimators.TD1, gamma=1)

    target_net_updater = HardUpdate(loss_module, value_network_update_interval=50)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-4)

    batch_size = 256

    from torchrl.data.replay_buffers import (
        PrioritizedSampler,
        SliceSampler,
        PrioritizedSliceSampler,
    )
    from torchrl.envs import MultiStepTransform
    from torchrl.collectors.utils import split_trajectories

    slice_len = 4 * 3
    num_slices = 20

    replay_buffer = TensorDictReplayBuffer(
        # pin_memory=False,
        storage=LazyTensorStorage(max_size=100_000),
        # batch_size=batch_size,
        # transform=MultiStepTransform(n_steps=20, gamma=0.99),
        sampler=PrioritizedSliceSampler(
            max_capacity=100_000,
            slice_len=slice_len,
            strict_length=False,
            alpha=0.6,
            beta=1,
        ),
        # sampler=SliceSampler(slice_len=slice_len, strict_length=False),
        # sampler=Sampler(),
        batch_size=slice_len * num_slices,
        # transform=r2g,
    )

    tasks_per_graph = 4 * 3
    n_graphs = 50

    collector = SyncDataCollector(
        create_env_fn=env_wrapper,
        policy=model_explore,
        frames_per_batch=tasks_per_graph * n_graphs,
        init_random_frames=tasks_per_graph * n_graphs,
        # split_trajs=True,
    )

    c_iter = iter(collector)

    for i in range(148, steps):
        data = next(c_iter)
        replay_buffer.extend(data)

        greedy_module.step(tasks_per_graph * n_graphs)

        for j in range(100):
            batch, info = replay_buffer.sample(return_info=True)
            batch = batch.reshape(num_slices, -1)
            # print(batch["collector", "traj_ids"])
            # print(batch["next", "reward"])
            # print(batch["value_target"])

            # for i in range(5):
            #     print(b[i]["step_count"])
            # import sys

            loss = loss_module(batch)
            # sys.exit()
            q_loss = loss["loss"]
            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()
            target_net_updater.step()

        collector.update_policy_weights_()

        model_path = f"dqn_model_{i}_2.pth"
        torch.save(model_explore.state_dict(), model_path)
        episode_reward = evaluate_policy(
            i,
            graph_func,
            sys_func,
            config_list,
            qvalue_module,
            animate=True,
            rtc=rtc,
            title="2_dqn_model_evaluation",
            observer_type=XYDataObserverFactory,
        )

    collector.shutdown()

    return qvalue_module


def evaluate_loaded_model(
    model,
    eval_set,
    rtc=True,
    animate=False,
    title="model_evaluation",
    observer_type=XYObserverFactory,
):
    graph_func, sys_func, config_list = eval_set
    actor = model[0]

    episode_reward = evaluate_policy(
        0,
        graph_func,
        sys_func,
        config_list,
        actor,
        animate=animate,
        rtc=rtc,
        title=title,
        observer_type=observer_type,
    )
