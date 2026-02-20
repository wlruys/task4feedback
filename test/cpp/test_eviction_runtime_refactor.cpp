#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "data.hpp"
#include "devices.hpp"
#include "scheduler.hpp"
#include "simulator.hpp"
#include "tasks.hpp"

#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int32_t kNumTasks = 4;
constexpr mem_t kDataBlockSize = 600;
constexpr vcu_t kTaskVCU = MAX_VCUS;
constexpr mem_t kTaskMemory = 1;
constexpr timecount_t kTaskDuration = 1000;

mem_t sum_movement(const std::vector<mem_t> &values) {
  return std::accumulate(values.begin(), values.end(), static_cast<mem_t>(0));
}

struct SimulationResult {
  ExecutionState state = ExecutionState::NONE;
  timecount_t time = 0;
  std::vector<mem_t> total_movement;
  std::vector<mem_t> eviction_movement;
};

struct SimulatorBundle {
  Graph graph;
  Data data;
  Devices devices;
  Topology topology;
  DefaultTransitionConditions conditions;
  StaticMapper mapper;
  std::unique_ptr<StaticTaskInfo> static_info;
  std::unique_ptr<TaskNoise> task_noise;
  std::unique_ptr<SchedulerInput> input;
  std::unique_ptr<Simulator> simulator;

  explicit SimulatorBundle(mem_t gpu_memory)
      : graph(), data(kNumTasks), devices(2), topology(2),
        mapper(DeviceIDList(static_cast<std::size_t>(kNumTasks), 1)) {
    build_graph();
    build_data();
    build_devices(gpu_memory);
    build_topology();

    static_info = std::make_unique<StaticTaskInfo>(graph);
    task_noise = std::make_unique<TaskNoise>(*static_info, 0, 1000);
    input = std::make_unique<SchedulerInput>(graph, *static_info, data, devices, topology,
                                             *task_noise, conditions, 8);
    simulator = std::make_unique<Simulator>(*input, mapper);
    simulator->initialize(false, true);
  }

private:
  void build_graph() {
    for (taskid_t task_id = 0; task_id < kNumTasks; ++task_id) {
      const auto tid = graph.add_task("task_" + std::to_string(task_id));
      if (task_id > 0) {
        graph.add_dependency(tid, task_id - 1);
      }
      std::vector<dataid_t> write_ids{task_id};
      graph.add_write_data(tid, write_ids);
      graph.set_variant(tid, DeviceType::GPU, kTaskVCU, kTaskMemory, kTaskDuration);
    }
    graph.finalize();
  }

  void build_data() {
    for (dataid_t data_id = 0; data_id < kNumTasks; ++data_id) {
      data.create_block(data_id, kDataBlockSize, HOST_ID, "block_" + std::to_string(data_id));
    }
  }

  void build_devices(mem_t gpu_memory) {
    devices.create_device(0, "host", DeviceType::CPU, 2, 2, 100000);
    devices.create_device(1, "gpu0", DeviceType::GPU, 2, 2, gpu_memory);
  }

  void build_topology() {
    for (devid_t src = 0; src < 2; ++src) {
      for (devid_t dst = 0; dst < 2; ++dst) {
        if (src == dst) {
          continue;
        }
        topology.set_bandwidth(src, dst, 1000);
        topology.set_latency(src, dst, 0);
        topology.set_max_connections(src, dst, 1);
      }
    }
  }
};

SimulationResult run_to_completion(Simulator &simulator) {
  const auto state = simulator.run();
  return {
      .state = state,
      .time = simulator.get_current_time(),
      .total_movement = simulator.get_total_data_movement(),
      .eviction_movement = simulator.get_eviction_data_movement(),
  };
}

} // namespace

TEST_CASE("LRU_manager get_max_memory_usage sums all accelerator devices") {
  Devices devices(6);
  devices.create_device(0, "host", DeviceType::CPU, 2, 2, 100000);
  for (devid_t device_id = 1; device_id < 6; ++device_id) {
    devices.create_device(device_id, std::string("gpu") + std::to_string(device_id), DeviceType::GPU, 2, 2,
                          100000);
  }

  LRU_manager lru(devices);
  lru.read(0, 100, 1000); // Host memory should not count.
  lru.read(1, 1, 10);
  lru.read(2, 2, 20);
  lru.read(3, 3, 30);
  lru.read(4, 4, 40);
  lru.read(5, 5, 50);

  CHECK_EQ(lru.get_max_memory_usage(), 150);
}

TEST_CASE("Tight GPU memory increases eviction movement") {
  SimulatorBundle relaxed_memory(/*gpu_memory=*/3000);
  SimulatorBundle tight_memory(/*gpu_memory=*/2000);

  const auto relaxed = run_to_completion(*relaxed_memory.simulator);
  const auto tight = run_to_completion(*tight_memory.simulator);

  CHECK_EQ(relaxed.state, ExecutionState::COMPLETE);
  CHECK_EQ(tight.state, ExecutionState::COMPLETE);
  CHECK_EQ(relaxed.time, 4000);
  CHECK_EQ(tight.time, 4000);

  CHECK_EQ(sum_movement(relaxed.eviction_movement), 0);
  CHECK_GT(sum_movement(tight.eviction_movement), sum_movement(relaxed.eviction_movement));
}

TEST_CASE("Simulator copy keeps parity under eviction pressure") {
  SimulatorBundle pressure(/*gpu_memory=*/1800);

  pressure.simulator->set_steps(1);
  const auto breakpoint_state = pressure.simulator->run();
  CHECK_EQ(breakpoint_state, ExecutionState::BREAKPOINT);

  Simulator branch(*pressure.simulator);

  const auto original = run_to_completion(*pressure.simulator);
  const auto branched = run_to_completion(branch);

  CHECK_EQ(original.state, ExecutionState::COMPLETE);
  CHECK_EQ(branched.state, ExecutionState::COMPLETE);
  CHECK_EQ(original.time, branched.time);
  CHECK_EQ(original.total_movement, branched.total_movement);
  CHECK_EQ(original.eviction_movement, branched.eviction_movement);
}
