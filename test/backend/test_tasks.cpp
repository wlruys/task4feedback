#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "include/data_manager.hpp"
#include "include/device_manager.hpp"
#include "include/graph.hpp"
#include "include/scheduler.hpp"
#include "include/simulator.hpp"
#include "include/tasks.hpp"
#include <chrono>

struct TestFixture {
  Tasks tasks;
  Data data;
  Devices devices;
  Topology topology;
  Simulator simulator;
  StaticMapper mapper;

  TestFixture()
      : tasks(3), data(0), devices(2), topology(2),
        simulator(tasks, data, devices, topology, mapper) {
    // Create compute tasks
    tasks.create_compute_task(0, "Task0", {});
    tasks.create_compute_task(1, "Task1", {0});
    tasks.create_compute_task(2, "Task2", {1});

    // Add variants to compute tasks
    tasks.add_variant(0, DeviceType::CPU, 1, 1024, 10);
    tasks.add_variant(1, DeviceType::GPU, 2, 2048, 20);
    tasks.add_variant(2, DeviceType::GPU, 3, 4096, 30);
    tasks.add_variant(2, DeviceType::CPU, 4, 8192, 40);

    // Set read and write data for compute tasks
    tasks.set_read(0, {0});
    tasks.set_write(0, {1});
    tasks.set_read(1, {1});
    tasks.set_write(1, {2});
    tasks.set_read(2, {2});
    tasks.set_write(2, {3});

    // Initialize devices
    devices.create_device(0, "CPU0", DeviceType::CPU, 1000, 102400);
    devices.create_device(1, "GPU0", DeviceType::GPU, 1000, 204800);
  }
};

TEST_CASE_FIXTURE(TestFixture, "Tasks: Create") {
  // Check the size of tasks
  CHECK(tasks.size() == 3);
  CHECK(tasks.compute_size() == 3);
  CHECK(tasks.data_size() == 0);

  // Check dependencies
  CHECK(tasks.get_dependencies(1) == TaskIDList{0});
  CHECK(tasks.get_dependencies(2) == TaskIDList{1});
  CHECK(tasks.get_data_dependencies(1).size() == 0);
  CHECK(tasks.get_data_dependencies(2).size() == 0);
}

TEST_CASE_FIXTURE(TestFixture, "Tasks: Variant") {

  // Check supported architectures
  auto supported_architectures = tasks.get_supported_architectures(0);
  CHECK(supported_architectures.size() == 1);
  CHECK(supported_architectures[0] == DeviceType::CPU);

  supported_architectures = tasks.get_supported_architectures(1);
  CHECK(supported_architectures.size() == 1);
  CHECK(supported_architectures[0] == DeviceType::GPU);

  supported_architectures = tasks.get_supported_architectures(2);
  CHECK(supported_architectures.size() == 2);
  CHECK(supported_architectures[0] == DeviceType::CPU);
  CHECK(supported_architectures[1] == DeviceType::GPU);

  // Check variants
  auto variants = tasks.get_variant_vector(0);
  CHECK(variants.size() == 1);
  CHECK(variants[0].get_arch() == DeviceType::CPU);
  CHECK(variants[0].get_vcus() == 1);
  CHECK(variants[0].get_mem() == 1024);
  CHECK(variants[0].get_execution_time() == 10);

  variants = tasks.get_variant_vector(1);
  CHECK(variants.size() == 1);
  CHECK(variants[0].get_arch() == DeviceType::GPU);
  CHECK(variants[0].get_vcus() == 2);
  CHECK(variants[0].get_mem() == 2048);
  CHECK(variants[0].get_execution_time() == 20);

  variants = tasks.get_variant_vector(2);
  CHECK(variants.size() == 2);
  CHECK(variants[0].get_arch() == DeviceType::CPU);
  CHECK(variants[0].get_vcus() == 4);
  CHECK(variants[0].get_mem() == 8192);
  CHECK(variants[0].get_execution_time() == 40);
  CHECK(variants[1].get_arch() == DeviceType::GPU);
  CHECK(variants[1].get_vcus() == 3);
  CHECK(variants[1].get_mem() == 4096);
  CHECK(variants[1].get_execution_time() == 30);
}

TEST_CASE_FIXTURE(TestFixture, "Tasks: Data Dependencies") {

  // Check read and write data

  auto read_data = tasks.get_read(0);
  CHECK(read_data.size() == 1);
  CHECK(read_data[0] == 0);

  auto write_data = tasks.get_write(0);
  CHECK(write_data.size() == 1);
  CHECK(write_data[0] == 1);

  read_data = tasks.get_read(1);
  CHECK(read_data.size() == 1);
  CHECK(read_data[0] == 1);

  write_data = tasks.get_write(1);
  CHECK(write_data.size() == 1);
  CHECK(write_data[0] == 2);

  read_data = tasks.get_read(2);
  CHECK(read_data.size() == 1);
  CHECK(read_data[0] == 2);

  write_data = tasks.get_write(2);
  CHECK(write_data.size() == 1);
  CHECK(write_data[0] == 3);

  // Initialize data graph
  simulator.initialize(0, true);

  // Check data tasks
  CHECK(tasks.size() == 6);
  CHECK(tasks.compute_size() == 3);
  CHECK(tasks.data_size() == 3);

  // Check data task dependencies and dependents
  CHECK(tasks.get_dependencies(3) == TaskIDList{});
  CHECK(tasks.get_dependents(3) == TaskIDList{0});

  CHECK(tasks.get_dependencies(4) == TaskIDList{0});
  CHECK(tasks.get_dependents(4) == TaskIDList{1});

  CHECK(tasks.get_dependencies(5) == TaskIDList{1});
  CHECK(tasks.get_dependents(5) == TaskIDList{2});

  // Check compute task data dependencies and dependents
  CHECK(tasks.get_data_dependencies(0) == TaskIDList{3});
  CHECK(tasks.get_data_dependents(0) == TaskIDList{4});

  CHECK(tasks.get_data_dependencies(1) == TaskIDList{4});
  CHECK(tasks.get_data_dependents(1) == TaskIDList{5});

  CHECK(tasks.get_data_dependencies(2) == TaskIDList{5});
  CHECK(tasks.get_data_dependents(2) == TaskIDList{});

  // Check depth
  CHECK(tasks.get_depth(0) == 0);
  CHECK(tasks.get_depth(1) == 1);
  CHECK(tasks.get_depth(2) == 2);
}

struct WriteDependencyFixture {
  Tasks tasks;
  Data data;
  Devices devices;
  Topology topology;
  Simulator simulator;
  StaticMapper mapper;

  WriteDependencyFixture()
      : tasks(7), data(0), devices(2), topology(2),
        simulator(tasks, data, devices, topology, mapper) {
    // Create compute tasks
    tasks.create_compute_task(0, "Task0", {});
    tasks.create_compute_task(1, "Task1", {0});
    tasks.create_compute_task(2, "Task2", {0});
    tasks.create_compute_task(3, "Task3", {1});
    tasks.create_compute_task(4, "Task4", {1});
    tasks.create_compute_task(5, "Task5", {2});
    tasks.create_compute_task(6, "Task6", {2});

    // Set read and write data for compute tasks
    tasks.set_read(0, {0});
    tasks.set_write(0, {0});

    tasks.set_read(1, {1});
    tasks.set_write(1, {1});

    tasks.set_read(2, {2});
    tasks.set_write(2, {2});

    tasks.set_read(3, {0, 1});
    tasks.set_read(4, {0, 1});

    tasks.set_read(5, {0, 2});
    tasks.set_read(6, {0, 2});

    // Initialize devices
    devices.create_device(0, "CPU0", DeviceType::CPU, 1000, 1024);
    devices.create_device(1, "GPU0", DeviceType::GPU, 1000, 2048);
  }
};

TEST_CASE_FIXTURE(WriteDependencyFixture, "Tasks: Data Dependencies 2") {

  // Initialize data graph
  simulator.initialize(0, true);

  // Check data tasks
  CHECK(tasks.size() == 18);
  CHECK(tasks.compute_size() == 7);
  CHECK(tasks.data_size() == 11);

  // Check data task dependencies and dependents

  CHECK(tasks.get_dependencies(7) == TaskIDList{});
  CHECK(tasks.get_dependents(7) == TaskIDList{0});
  CHECK(tasks.get_data_id(7) == 0);

  CHECK(tasks.get_dependencies(8) == TaskIDList{});
  CHECK(tasks.get_dependents(8) == TaskIDList{1});
  CHECK(tasks.get_data_id(8) == 1);

  CHECK(tasks.get_dependencies(9) == TaskIDList{});
  CHECK(tasks.get_dependents(9) == TaskIDList{2});
  CHECK(tasks.get_data_id(9) == 2);

  CHECK(tasks.get_dependencies(10) == TaskIDList{0});
  CHECK(tasks.get_dependents(10) == TaskIDList{3});
  CHECK(tasks.get_data_id(10) == 0);

  CHECK(tasks.get_dependencies(11) == TaskIDList{1});
  CHECK(tasks.get_dependents(11) == TaskIDList{3});
  CHECK(tasks.get_data_id(11) == 1);

  CHECK(tasks.get_dependencies(12) == TaskIDList{0});
  CHECK(tasks.get_dependents(12) == TaskIDList{4});
  CHECK(tasks.get_data_id(12) == 0);

  CHECK(tasks.get_dependencies(13) == TaskIDList{1});
  CHECK(tasks.get_dependents(13) == TaskIDList{4});
  CHECK(tasks.get_data_id(13) == 1);

  CHECK(tasks.get_dependencies(14) == TaskIDList{0});
  CHECK(tasks.get_dependents(14) == TaskIDList{5});
  CHECK(tasks.get_data_id(14) == 0);

  CHECK(tasks.get_dependencies(15) == TaskIDList{2});
  CHECK(tasks.get_dependents(15) == TaskIDList{5});
  CHECK(tasks.get_data_id(15) == 2);

  CHECK(tasks.get_dependencies(16) == TaskIDList{0});
  CHECK(tasks.get_dependents(16) == TaskIDList{6});
  CHECK(tasks.get_data_id(16) == 0);

  CHECK(tasks.get_dependencies(17) == TaskIDList{2});
  CHECK(tasks.get_dependents(17) == TaskIDList{6});
  CHECK(tasks.get_data_id(17) == 2);
}