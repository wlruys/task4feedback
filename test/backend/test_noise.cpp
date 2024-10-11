#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "include/noise.hpp"
#include "include/tasks.hpp"

// Fixture for LognormalTaskNoise tests
class LognormalTaskNoiseFixture {
protected:
  const unsigned int seed = 42;
  const std::size_t num_device_types = 2;
  const std::size_t num_tasks = 100;
  Tasks tasks;
  LognormalTaskNoise noise;

public:
  LognormalTaskNoiseFixture() : tasks(num_tasks), noise(tasks, seed) {
    // Create compute tasks
    for (taskid_t i = 0; i < tasks.compute_size(); ++i) {
      tasks.create_compute_task(i, "Task" + std::to_string(i), {});
      tasks.add_variant(i, DeviceType::CPU, 1, 1024, 10);
      tasks.add_variant(i, DeviceType::GPU, 2, 2048, 10);
    }
  }
};

TEST_CASE_FIXTURE(LognormalTaskNoiseFixture,
                  "LognormalTaskNoise generate and get") {
  noise.generate();

  for (taskid_t i = 0; i < tasks.compute_size(); ++i) {
    for (std::size_t j = 0; j < num_device_types; ++j) {
      DeviceType arch = static_cast<DeviceType>(j);
      timecount_t value = noise.get(i, arch);
      CHECK_MESSAGE(value > 0, "Noise value should be greater than zero");
    }
  }

  // sum all noise values to see if it is near the expected mean of 10
  double mean = 0;
  for (taskid_t i = 0; i < tasks.compute_size(); ++i) {
    for (std::size_t j = 0; j < num_device_types; ++j) {
      auto arch = static_cast<DeviceType>(j);
      mean += static_cast<double>(noise.get(i, arch));
    }
  }
  mean /= static_cast<double>(tasks.compute_size() * num_device_types);

  CHECK_MESSAGE(mean == doctest::Approx(10).epsilon(0.1),
                "Mean should be near 10");
}

TEST_CASE_FIXTURE(LognormalTaskNoiseFixture, "LognormalTaskNoise set") {
  for (taskid_t i = 0; i < tasks.compute_size(); ++i) {
    for (std::size_t j = 0; j < num_device_types; ++j) {
      DeviceType arch = static_cast<DeviceType>(j);
      noise.set(i, arch, 20);
      CHECK(noise.get(i, arch) == 20);
    }
  }
}

TEST_CASE_FIXTURE(LognormalTaskNoiseFixture,
                  "LognormalTaskNoise dump and load") {
  noise.generate();
  noise.dump_to_binary("test_noise.bin");

  LognormalTaskNoise loaded_noise(tasks, seed);
  loaded_noise.load_from_binary("test_noise.bin");

  for (taskid_t i = 0; i < tasks.compute_size(); ++i) {
    for (std::size_t j = 0; j < num_device_types; ++j) {
      DeviceType arch = static_cast<DeviceType>(j);
      CHECK(noise.get(i, arch) == loaded_noise.get(i, arch));
    }
  }
}