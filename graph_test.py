from task4feedback import fastsim2 as fastsim
from task4feedback.interface import *
from task4feedback.legacy_graphs import *
import torch
from typing import Optional, Self

from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import (
    DefaultObserverFactory,
    CompiledDefaultObserverFactory,
    SimulatorDriver,
    SimulatorFactory,
    create_graph_spec,
)
from torchrl.data import Composite, TensorSpec, Unbounded, Binary, Bounded
from torchrl.envs.utils import make_composite_from_td
from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs import check_env_specs
from tensordict import TensorDict


def make_test_cholesky_graph():
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

    data_config = CholeskyDataGraphConfig(data_size=100)
    config = CholeskyConfig(blocks=4, task_config=task_config)
    tasks, data = make_graph(config, data_config=data_config)
    return tasks, data


class FastSimEnv(EnvBase):
    def __init__(self, simulator_factory, seed: int = 0, device="cpu"):
        super().__init__(device=device)
        self.batch_size = torch.Size([])

        self.simulator_factory = simulator_factory
        self.simulator = simulator_factory.create(seed)

        self.observation_spec = self._create_observation_spec()
        self.action_spec = self._create_action_spec()
        self.reward_spec = self._create_reward_spec()
        self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

    def _create_observation_spec(self) -> TensorSpec:
        obs = self.simulator.observer.get_observation()
        comp = make_composite_from_td(obs)
        comp = Composite(observation=comp)
        return comp

    def _create_action_spec(self, ndevices: int = 5) -> TensorSpec:
        n_devices = self.simulator_factory.graph_spec.max_devices
        out = Bounded(
            shape=(1,),
            device=self.device,
            dtype=torch.int32,
            low=torch.tensor(0, device=self.device),
            high=torch.tensor(n_devices, device=self.device),
        )
        out = Composite(action=out)
        return out

    def _create_reward_spec(self) -> TensorSpec:
        return Unbounded(shape=(1,), device=self.device, dtype=torch.float32)

    def _get_observation(self) -> TensorDict:
        obs = self.simulator.observer.get_observation()
        td = TensorDict(observation=obs)
        return td

    def _step(self, td: TensorDict) -> TensorDict:
        candidate_tasks = td["observation"]["aux"]["candidates"]["idx"]
        chosen_device = td["action"].item()

        # print(f"candidate_tasks: {candidate_tasks}")
        # print(f"chosen_device: {chosen_device}")

        # print("TD", td)

        candidate = candidate_tasks[0].item()
        local_id = 0
        device = 0
        state = self.simulator.get_state()
        mapping_priority = state.get_mapping_priority(candidate)
        reserving_priority = mapping_priority
        launching_priority = mapping_priority

        actions = [
            fastsim.Action(
                candidate, local_id, device, reserving_priority, launching_priority
            )
        ]
        self.simulator.simulator.map_tasks(actions)
        simulator_status = self.simulator.run_until_external_mapping()

        terminated = torch.tensor((1,), device=self.device, dtype=torch.bool)
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)

        terminated[0] = False
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE
        reward[0] = 0

        if not done:
            assert simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING, (
                f"Unexpected simulator status: {simulator_status}"
            )
            obs = self._get_observation()
        else:
            obs = self._reset()

        out = obs
        out.set("reward", reward)
        out.set("done", done)

        print("OUT")

        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.simulator = self.simulator_factory.create()
        simulator_status = self.simulator.run_until_external_mapping()
        assert simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING, (
            f"Unexpected simulator status: {simulator_status}"
        )

        return self._get_observation()

    def _set_seed(self, seed: Optional[int] = None):
        rng = torch.manual_seed(seed)
        self.rng = rng


def make_env():
    s = uniform_connected_devices(5, 100000, 1, 1000)
    tasks, data = make_test_cholesky_graph()
    d = DataBlocks.create_from_legacy_data(data, s)
    m = Graph.create_from_legacy_graph(tasks, data)
    m.finalize_tasks()
    spec = create_graph_spec()
    input = SimulatorInput(m, d, s)
    print(f"Max devices: {spec.max_devices}")
    return FastSimEnv(SimulatorFactory(input, spec, DefaultObserverFactory))


# s = uniform_connected_devices(5, 100000, 1, 1000)
# tasks, data = make_test_cholesky_graph()
# d = DataBlocks.create_from_legacy_data(data, s)
# # print(d)

# n_tasks = len(tasks)
# m = Graph.create_from_legacy_graph(tasks, data)
# m.finalize_tasks()
# # print(m)

# spec = create_graph_spec()
# input = SimulatorInput(m, d, s)

# env = FastSimEnv(SimulatorFactory(input, spec, DefaultObserverFactory))

# create_env = lambda: FastSimEnv(SimulatorFactory(input, spec, DefaultObserverFactory))


from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
import time

if __name__ == "__main__":
    t = time.perf_counter()
    workers = 4

    # collector = SyncDataCollector(make_env, frames_per_batch=1000)
    collector = MultiSyncDataCollector(
        [make_env for _ in range(workers)], frames_per_batch=1000, total_frames=-1
    )
    for data in collector:
        print(data.shape)
        break
    t = time.perf_counter() - t
    print(f"Time: {t}")

    t = time.perf_counter()
    i = 0
    for data in collector:
        print(data.shape)
        i += 1
        if i == 10:
            break
    t = time.perf_counter() - t
    print(f"Time: {t}")
