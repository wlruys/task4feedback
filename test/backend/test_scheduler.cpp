#include "events.hpp"
#include <iostream>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "communication_manager.hpp"
#include "data_manager.hpp"
#include "devices.hpp"
#include "scheduler.hpp"
#include "simulator.hpp"
#include "tasks.hpp"

class SimulatorFixture {
protected:
  Tasks tasks;
  Devices devices;
  Topology topology;
  Data data;
  RandomMapper mapper;
  TaskNoise task_noise;
  CommunicationNoise comm_noise;
  SchedulerInput input;
  constexpr static std::size_t num_devices = 2;
  constexpr static std::size_t num_tasks = 3;
  constexpr static unsigned int seed = 42;

  SimulatorFixture()
      : tasks(num_tasks), devices(num_devices), topology(num_devices),
        data(num_tasks), task_noise(tasks, seed), comm_noise(topology),
        input(tasks, data, devices, topology, mapper, task_noise, comm_noise) {

    // Initialize tasks
    tasks.create_compute_task(0, "Task0", {});
    tasks.create_compute_task(1, "Task1", {0});
    tasks.create_compute_task(2, "Task2", {1});

    for (int i = 0; i < 3; ++i) {
      tasks.add_variant(i, DeviceType::CPU, 1, 1024, 100);
      tasks.add_variant(i, DeviceType::GPU, 1, 2048, 50);
    }

    // Initialize devices
    devices.create_device(0, "CPU", DeviceType::CPU, 4, 8192);
    devices.create_device(1, "GPU", DeviceType::GPU, 2, 4096);

    // Initialize topology
    topology.set_bandwidth(0, 1, 64);
    topology.set_bandwidth(1, 0, 64);
    topology.set_latency(0, 1, 0);
    topology.set_latency(1, 0, 0);
    topology.set_max_connections(0, 1, 1);
    topology.set_max_connections(1, 0, 1);
  }
};

TEST_CASE_FIXTURE(SimulatorFixture, "Scheduler initialization") {
  auto simulator = Simulator(input);
  CHECK(!simulator.initialized);
  simulator.initialize(false);
  CHECK(simulator.initialized);
  CHECK_EQ(simulator.scheduler.get_queues().n_mappable(), 1);
}

TEST_CASE_FIXTURE(SimulatorFixture, "Scheduler initialization") {
  auto simulator = Simulator(input);
  simulator.initialize(false);
  auto state = simulator.run();
  CHECK_EQ(state, ExecutionState::COMPLETE);
}

TEST_CASE_FIXTURE(SimulatorFixture, "Breakpoints") {

  auto simulator = Simulator(input);
  simulator.initialize(false);

  // Set a breakpoint for task 0 at the MAPPER event
  simulator.add_task_breakpoint(EventType::MAPPER, 0);

  auto state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Set a breakpoint for task 1 at the MAPPER event
  simulator.add_task_breakpoint(EventType::MAPPER, 1);

  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Continue execution
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::COMPLETE);
}

TEST_CASE_FIXTURE(SimulatorFixture, "Copy") {

  auto simulator = Simulator(input);
  simulator.initialize(false);

  // Set a breakpoint for task 0 at the MAPPER event
  simulator.add_task_breakpoint(EventType::COMPLETER, 0);

  auto state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  Simulator sim2(simulator);
  const timecount_t copy_time = simulator.get_current_time();
  const timecount_t copy_time2 = sim2.get_current_time();
  CHECK_EQ(copy_time, copy_time2);

  state = simulator.run();
  CHECK_EQ(state, ExecutionState::COMPLETE);

  const timecount_t finish_time = simulator.get_current_time();
  const timecount_t frozen_time = sim2.get_current_time();

  CHECK(finish_time != frozen_time);

  state = sim2.run();
  CHECK_EQ(state, ExecutionState::COMPLETE);

  const timecount_t finish_time2 = sim2.get_current_time();

  CHECK_EQ(finish_time, finish_time2);
}

TEST_CASE_FIXTURE(SimulatorFixture, "Data movement and memory costs") {
  // Reset tasks with specific data dependencies
  auto tasks = Tasks(2);
  tasks.create_compute_task(0, "Task0", {});
  tasks.create_compute_task(1, "Task1", {0});

  tasks.add_variant(0, DeviceType::CPU, 1, 1024, 100);
  tasks.add_variant(1, DeviceType::GPU, 1, 2048, 50);

  DataIDList read_data = {0};
  DataIDList write_data = {1};
  tasks.set_read(0, {0});
  tasks.set_write(0, {0});
  tasks.set_read(1, {1});

  // Reset data
  auto data = Data(2);
  data.create_block(0, 1024, 0, "Input");
  data.create_block(1, 2048, 0, "Output");

  // Use StaticMapper instead of RandomMapper
  StaticMapper static_mapper({0, 1});

  // Reinitialize simulator with new configuration
  input = SchedulerInput(tasks, data, devices, topology, static_mapper,
                         task_noise, comm_noise);
  auto simulator = Simulator(input);
  simulator.initialize(true);

  const auto &scheduler_state = simulator.scheduler.get_state();
  const auto &data_manager = scheduler_state.get_data_manager();
  const auto &device_manager = scheduler_state.get_device_manager();

  // Check memory usage at init
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  simulator.add_task_breakpoint(EventType::MAPPER, 0);
  auto state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at mapping task 0

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  simulator.add_task_breakpoint(EventType::MAPPER, 1);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at mapping task 1

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));

  simulator.add_task_breakpoint(EventType::RESERVER, 0);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at reserving task 0

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));

  simulator.add_task_breakpoint(EventType::RESERVER, 1);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at reserving task 1

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));
  CHECK(data_manager.check_valid_reserved({1}, 1));

  // Check memory usage at launching task 0
  simulator.add_task_breakpoint(EventType::LAUNCHER, 0);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 4096);

  // The data movement task 0 -> 1 should have started at this point
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 2048);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));
  CHECK(data_manager.check_valid_reserved({1}, 1));

  simulator.add_task_breakpoint(EventType::LAUNCHER, 1);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at launching task 1
  // Task 0 completes before task 1 starts
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 4096);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));
  CHECK(data_manager.check_valid_reserved({1}, 1));
  CHECK(data_manager.check_valid_launched({1}, 1));

  state = simulator.run();

  CHECK_EQ(state, ExecutionState::COMPLETE);

  // Check memory usage at end
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 2048);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 2048);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 2048);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));
  CHECK(data_manager.check_valid_reserved({1}, 1));
  CHECK(data_manager.check_valid_launched({1}, 1));
}

TEST_CASE_FIXTURE(SimulatorFixture, "Copy: Data Movement and Memory Costs") {
  // Reset tasks with specific data dependencies
  auto tasks = Tasks(2);
  tasks.create_compute_task(0, "Task0", {});
  tasks.create_compute_task(1, "Task1", {0});

  tasks.add_variant(0, DeviceType::CPU, 1, 1024, 100);
  tasks.add_variant(1, DeviceType::GPU, 1, 2048, 50);

  DataIDList read_data = {0};
  DataIDList write_data = {1};
  tasks.set_read(0, {0});
  tasks.set_write(0, {0});
  tasks.set_read(1, {1});

  // Reset data
  auto data = Data(2);
  data.create_block(0, 1024, 0, "Input");
  data.create_block(1, 2048, 0, "Output");

  // Use StaticMapper instead of RandomMapper
  StaticMapper static_mapper({0, 1});

  // Reinitialize simulator with new configuration
  input = SchedulerInput(tasks, data, devices, topology, static_mapper,
                         task_noise, comm_noise);
  auto simulator = Simulator(input);
  simulator.initialize(true);

  const auto &scheduler_state = simulator.scheduler.get_state();
  const auto &data_manager = scheduler_state.get_data_manager();
  const auto &device_manager = scheduler_state.get_device_manager();

  // Check memory usage at init
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  simulator.add_task_breakpoint(EventType::MAPPER, 0);
  auto state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at mapping task 0

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  simulator.add_task_breakpoint(EventType::MAPPER, 1);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at mapping task 1

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));

  simulator.add_task_breakpoint(EventType::RESERVER, 0);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at reserving task 0

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 0);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));

  simulator.add_task_breakpoint(EventType::RESERVER, 1);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at reserving task 1

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));
  CHECK(data_manager.check_valid_reserved({1}, 1));

  // Check memory usage at launching task 0
  simulator.add_task_breakpoint(EventType::LAUNCHER, 0);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);
  std::cout << "COPY POINT" << std::endl;
  auto simulator2 = Simulator(simulator);
  auto state2 = simulator2.run();
  CHECK_EQ(state2, ExecutionState::COMPLETE);

  const auto &device_manager2 =
      simulator2.scheduler.get_state().get_device_manager();
  const auto &data_manager2 =
      simulator2.scheduler.get_state().get_data_manager();

  CHECK_EQ(device_manager2.get_mem<TaskState::MAPPED>(0), 3072);
  CHECK_EQ(device_manager2.get_mem<TaskState::MAPPED>(1), 2048);
  CHECK_EQ(device_manager2.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager2.get_mem<TaskState::RESERVED>(1), 2048);
  CHECK_EQ(device_manager2.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager2.get_mem<TaskState::LAUNCHED>(1), 2048);

  CHECK(data_manager2.check_valid_mapped({0}, 0));
  CHECK(data_manager2.check_valid_reserved({0}, 0));
  CHECK(data_manager2.check_valid_launched({0}, 0));

  CHECK(data_manager2.check_valid_mapped({1}, 0));
  CHECK(data_manager2.check_valid_reserved({1}, 0));
  CHECK(data_manager2.check_valid_launched({1}, 0));

  CHECK(data_manager2.check_valid_mapped({1}, 1));
  CHECK(data_manager2.check_valid_reserved({1}, 1));
  CHECK(data_manager2.check_valid_launched({1}, 1));

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 4096);

  // The data movement task 0 -> 1 should have started at this point
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 2048);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));
  CHECK(data_manager.check_valid_reserved({1}, 1));

  std::cout << "RESUMING ORIGINAL" << std::endl;
  simulator.add_task_breakpoint(EventType::LAUNCHER, 1);
  state = simulator.run();
  CHECK_EQ(state, ExecutionState::BREAKPOINT);

  // Check memory usage at launching task 1
  // Task 0 completes before task 1 starts
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 4096);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 4096);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));
  CHECK(data_manager.check_valid_reserved({1}, 1));
  CHECK(data_manager.check_valid_launched({1}, 1));

  state = simulator.run();

  CHECK_EQ(state, ExecutionState::COMPLETE);

  // Check memory usage at end
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 2048);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 2048);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 3072);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 2048);

  CHECK(data_manager.check_valid_mapped({0}, 0));
  CHECK(data_manager.check_valid_reserved({0}, 0));
  CHECK(data_manager.check_valid_launched({0}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 0));
  CHECK(data_manager.check_valid_reserved({1}, 0));
  CHECK(data_manager.check_valid_launched({1}, 0));

  CHECK(data_manager.check_valid_mapped({1}, 1));
  CHECK(data_manager.check_valid_reserved({1}, 1));
  CHECK(data_manager.check_valid_launched({1}, 1));
}

TEST_CASE_FIXTURE(SimulatorFixture, "Error scenario: Circular dependency") {
  // Reset tasks with circular dependency
  auto tasks = Tasks(2);
  tasks.create_compute_task(0, "Task0", {1});
  tasks.create_compute_task(1, "Task1", {0});

  tasks.add_variant(0, DeviceType::CPU, 1, 1024, 100);
  tasks.add_variant(1, DeviceType::CPU, 1, 1024, 100);

  // Reinitialize simulator with new configuration
  input = SchedulerInput(tasks, data, devices, topology, mapper, task_noise,
                         comm_noise);
  auto simulator = Simulator(input);
  simulator.initialize(false);

  auto state = simulator.run();
  CHECK_EQ(state, ExecutionState::ERROR);
}

TEST_CASE_FIXTURE(SimulatorFixture, "Graph with multiple data dependencies") {
  // Reset tasks with complex dependencies
  const std::size_t n_tasks = 5;
  auto tasks = Tasks(n_tasks);
  tasks.create_compute_task(0, "Task0", {});
  tasks.create_compute_task(1, "Task1", {0});
  tasks.create_compute_task(2, "Task2", {0});
  tasks.create_compute_task(3, "Task3", {1, 2});
  tasks.create_compute_task(4, "Task4", {3});

  for (std::size_t i = 0; i < n_tasks; ++i) {
    tasks.add_variant(i, DeviceType::CPU, 1, 100, 100);
    tasks.add_variant(i, DeviceType::GPU, 1, 200, 50);
  }

  tasks.set_read(0, {0, 1});
  tasks.set_write(0, {2, 3});

  tasks.set_read(1, {2});
  tasks.set_write(1, {2, 4});

  tasks.set_read(2, {3});
  tasks.set_write(2, {3, 5});

  tasks.set_read(3, {4, 5});
  tasks.set_write(3, {6});

  tasks.set_read(4, {6});

  std::cout << "Created tasks" << std::endl;

  // Reset data
  const std::size_t n_data = 7;
  auto data = Data(n_data);
  for (std::size_t i = 0; i < n_data; ++i) {
    data.create_block(i, 1, 0, "Data" + std::to_string(i));
  }

  std::cout << "Created data" << std::endl;

  auto static_mapper = StaticMapper({0, 1, 0, 1, 0});

  std::cout << "Created mapper" << std::endl;
  auto task_noise = TaskNoise(tasks, seed);

  // Reinitialize simulator with new configuration
  input = SchedulerInput(tasks, data, devices, topology, static_mapper,
                         task_noise, comm_noise);
  auto simulator = Simulator(input);

  std::cout << "Created simulator" << std::endl;
  simulator.initialize(true);
  std::cout << "Initialized simulator" << std::endl;

  const auto &scheduler_state = simulator.scheduler.get_state();
  const auto &data_manager = scheduler_state.get_data_manager();
  const auto &device_manager = scheduler_state.get_device_manager();

  // Check initial memory
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 7);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 0);

  auto state = simulator.run();
  CHECK_EQ(state, ExecutionState::COMPLETE);

  CHECK_EQ(scheduler_state.counts.n_completed(), 5);
  CHECK_EQ(scheduler_state.counts.n_data_completed(), 7);

  // Final Locations
  CHECK(data_manager.check_valid_launched({0}, 0));
  CHECK(!data_manager.check_valid_launched({0}, 1));

  CHECK(data_manager.check_valid_launched({1}, 0));
  CHECK(!data_manager.check_valid_launched({1}, 1));

  CHECK(!data_manager.check_valid_launched({2}, 0));
  CHECK(data_manager.check_valid_launched({2}, 1));

  CHECK(data_manager.check_valid_launched({3}, 0));
  CHECK(!data_manager.check_valid_launched({3}, 1));

  CHECK(!data_manager.check_valid_launched({4}, 0));
  CHECK(data_manager.check_valid_launched({4}, 1));

  CHECK(data_manager.check_valid_launched({5}, 0));
  CHECK(data_manager.check_valid_launched({6}, 1));

  CHECK(data_manager.check_valid_launched({6}, 0));
  CHECK(data_manager.check_valid_launched({6}, 1));

  // Check final memory
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(0), 5);
  CHECK_EQ(device_manager.get_mem<TaskState::LAUNCHED>(1), 4);

  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(0), 5);
  CHECK_EQ(device_manager.get_mem<TaskState::RESERVED>(1), 4);

  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(0), 5);
  CHECK_EQ(device_manager.get_mem<TaskState::MAPPED>(1), 4);
}