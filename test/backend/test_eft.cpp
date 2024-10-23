#include "events.hpp"
#include <iostream>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "communication_manager.hpp"
#include "data_manager.hpp"
#include "devices.hpp"
#include "doctest/doctest.h"
#include "scheduler.hpp"
#include "simulator.hpp"
#include "tasks.hpp"

class SimulatorFixture {
protected:
  Tasks tasks;
  Devices devices;
  Topology topology;
  Data data;
  EFTMapper mapper;
  TaskNoise task_noise;
  CommunicationNoise comm_noise;
  SchedulerInput input;
  constexpr static std::size_t num_data = 3;
  constexpr static std::size_t num_devices = 3;
  constexpr static std::size_t num_tasks = 3;
  constexpr static unsigned int seed = 42;

  SimulatorFixture()
      : tasks(num_tasks), devices(num_devices), topology(num_devices),
        data(num_tasks), mapper(num_tasks, num_devices),
        task_noise(tasks, seed), comm_noise(topology),
        input(tasks, data, devices, topology, mapper, task_noise, comm_noise) {

    // Initialize tasks
    tasks.create_compute_task(0, "Task0", {});
    tasks.create_compute_task(1, "Task1", {});
    tasks.create_compute_task(2, "Task2", {});

    // Initialize data
    data.create_block(0, 10000, 0, "D0");
    data.create_block(1, 10000, 1, "D1");
    data.create_block(2, 10000, 2, "D2");

    for (std::size_t i = 0; i < 3; ++i) {
      tasks.set_read(i, {static_cast<dataid_t>(i)});
      tasks.add_variant(i, DeviceType::CPU, 1, 2001, 7);
      tasks.add_variant(i, DeviceType::GPU, 1, 2002, 5);
    }

    // Initialize devices
    devices.create_device(0, "CPU", DeviceType::CPU, 1, 8000000);
    devices.create_device(1, "GPU", DeviceType::GPU, 1, 8000000);
    devices.create_device(2, "GPU", DeviceType::GPU, 1, 8000000);

    // Initialize topology
    for (std::size_t i = 0; i < num_devices; i++) {
      for (std::size_t j = 0; j < num_devices; j++) {
        topology.set_bandwidth(i, j, 100);
        topology.set_latency(i, j, 0);
        topology.set_max_connections(i, j, 1);
      }
    }
  }
};

TEST_CASE_FIXTURE(SimulatorFixture, "EFTScheduler: run") {
  logger_setup();
  auto simulator = Simulator(input);
  simulator.initialize(true);
  auto state = simulator.run();
  CHECK_EQ(state, ExecutionState::COMPLETE);

  // Check mapping at end
  CHECK_EQ(0, simulator.scheduler.get_state().get_mapping(0));
  CHECK_EQ(1, simulator.scheduler.get_state().get_mapping(1));
  CHECK_EQ(2, simulator.scheduler.get_state().get_mapping(2));
}
