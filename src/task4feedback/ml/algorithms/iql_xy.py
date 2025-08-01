from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType
from torchrl.objectives import IQLLoss, SoftUpdate
from ...interface.wrappers import *
from torchrl.data.replay_buffers import (
    ReplayBuffer,
    LazyMemmapStorage,
)
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from ..models import *
from ..env import *
from torchrl.modules import ProbabilisticActor, ValueOperator
import wandb

from task4feedback.graphs.mesh.base import *
from task4feedback.graphs.mesh.partition import *
from task4feedback.graphs.mesh.plot import *
from task4feedback.graphs.base import *
from task4feedback.graphs.jacobi import *


def collect_eft_runs(
    graph_func,
    sys_func,
    config_list,
    workers=1,
    samples=1000,
    filename=None,
    rtc=True,
):
    samples_per_batch = min(samples * 224, 1000 * 224)
    replay_buffer = ReplayBuffer(
        storage=LazyMemmapStorage(max_size=224 * samples * len(config_list)),
        sampler=SamplerWithoutReplacement(),
    )
    for k, config in enumerate(config_list):
        print(f"Collecting samples for config {k + 1}/{len(config_list)}")

        def env_wrapper() -> MapperRuntimeEnv:
            graph = graph_func(config)
            s = sys_func()
            d = graph.get_blocks()
            m = graph
            m.finalize_tasks()
            spec = create_graph_spec()
            internal_mapper = fastsim.DequeueEFTMapper
            external_mapper = ExternalMapper

            input = SimulatorInput(
                m,
                d,
                s,
                transition_conditions=fastsim.BatchTransitionConditions(5, 2, 16),
            )

            env = MapperRuntimeEnv(
                SimulatorFactory(
                    input,
                    spec,
                    XYObserverFactory(spec, input.graph),
                    internal_mapper=internal_mapper,
                    external_mapper=external_mapper,
                ),
                device="cpu",
            )
            env = TransformedEnv(env, StepCounter())
            env = TransformedEnv(env, TrajCounter())

            return env

        collector = MultiSyncDataCollector(
            [env_wrapper for _ in range(workers)],
            frames_per_batch=samples_per_batch,
            cat_results=0,
            env_device="cpu",
            policy_device="cpu",
        )
        out_seed = collector.set_seed(0)

        for i, batch in enumerate(collector):
            replay_buffer.extend(batch)
            if samples_per_batch * (i + 1) >= samples * 224:
                break

    collector.shutdown()
    print(
        f"Collected {len(replay_buffer)} samples from {len(config_list)} configs with {workers} workers"
    )

    if filename is not None:
        replay_buffer.save(filename)
    else:
        replay_buffer.dump("eft_replay_buffer")

    return replay_buffer


def evaluate_policy(
    step,
    graph_func,
    sys_func,
    config_list,
    policy_network,
    animate=False,
    rtc=True,
    title="",
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
                XYObserverFactory(spec, input.graph),
            ),
            device="cpu",
        )
        return env

    episode_rewards = []
    for i, config in enumerate(config_list):
        env = env_wrapper(config)
        rout = env.rollout(225, policy=policy_network)
        returns = rout["next", "reward"].sum().item()
        episode_rewards.append(returns)

        if animate and i == len(config_list) - 1:
            # Animate the last environment
            title = f"{title}_network_eval_{step}_{rtc}"
            print(title)
            animate_mesh_graph(env, time_interval=250, show=False, title=title)

    return np.mean(episode_rewards)


def run_iql(
    replay_buffer,
    actor_model,
    value_model,
    action_value_model,
    eval_set,
    steps=10000,
    rtc=True,
):
    if rtc:
        tag = ["rtc"]
    else:
        tag = ["non_rtc"]

    wandb.init(
        project="iql",
        tags=tag,
        config={
            "steps": steps,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "replay_buffer_size": len(replay_buffer),
            "rtc": rtc,
            "actor_model": actor_model.__class__.__name__,
            "value_model": value_model.__class__.__name__,
            "action_value_model": action_value_model.__class__.__name__,
        },
    )
    graph_func, sys_func, config_list = eval_set
    actor_model_td = HeteroDataWrapper(actor_model)
    value_model_td = HeteroDataWrapper(value_model)
    action_value_model_td = HeteroDataWrapper(action_value_model)

    _actor_model = TensorDictModule(
        actor_model_td,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        module=_actor_model,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        default_interaction_type=ExplorationType.DETERMINISTIC,
        cache_dist=True,
        return_log_prob=True,
    )

    value = ValueOperator(
        module=value_model_td,
        in_keys=["observation"],
        out_keys=["state_value"],
    )

    action_value = ValueOperator(
        module=action_value_model_td,
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )

    model = torch.nn.ModuleList([actor, value, action_value])

    loss_module = IQLLoss(
        actor_network=model[0],
        value_network=model[1],
        qvalue_network=model[2],
        loss_function="l2",
        temperature=3,
        expectile=0.7,
    ).to("cpu")

    target_net_updater = SoftUpdate(loss_module, tau=0.005)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-4)

    # model_weights = {name: param.data for name, param in model.named_parameters()}
    # wandb.log({"model_weights": model_weights})

    n_samples = len(replay_buffer)
    print(f"Number of samples in replay buffer: {n_samples}")
    batch_size = 256
    steps_per_epoch = n_samples // batch_size
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {steps}")
    print(f"Total epochs: {steps // steps_per_epoch}")

    for step in range(steps):
        data = replay_buffer.sample(batch_size)

        loss_dict = loss_module(data)
        loss = (
            loss_dict["loss_value"] + loss_dict["loss_actor"] + loss_dict["loss_qvalue"]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        target_net_updater.step()

        if step % 50 == 0:
            if step % 150 == 0:
                animate = True
                model_path = f"model_checkpoint_{step}_{rtc}.pth"
                torch.save(model.state_dict(), model_path)
                # wandb.save(model_path)
            else:
                animate = False
            episode_reward = evaluate_policy(
                step, graph_func, sys_func, config_list, actor, animate=animate, rtc=rtc
            )
            print(f"Epoch {step}: episode reward {episode_reward}")
            wandb.log(
                {
                    "step": step,
                    "loss": loss.item(),
                    "loss_action": loss_dict["loss_actor"].item(),
                    "loss_value": loss_dict["loss_value"].item(),
                    "loss_qvalue": loss_dict["loss_qvalue"].item(),
                    "episode_reward": episode_reward,
                    "fraction_seen": step / steps_per_epoch,
                }
            )

    return model


def evaluate_loaded_model(
    model,
    eval_set,
    rtc=True,
    animate=False,
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
        title="loaded_random_trained_random",
    )
