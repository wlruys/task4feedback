#pragma once
#include "devices.hpp"
#include "tasks.hpp"
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>

class TaskNoise {
protected:
  using noise_t = double;
  Tasks &tasks;
  unsigned int seed = 0;
  std::mt19937 gen;
  std::vector<timecount_t> task_durations;

  [[nodiscard]] virtual timecount_t sample_duration(taskid_t task_id,
                                                    DeviceType arch) const = 0;

public:
  TaskNoise(Tasks &tasks_, unsigned int seed_ = 0)
      : tasks(tasks_), seed(seed_), gen(seed_),
        task_durations(tasks.compute_size() * num_device_types) {}

  [[nodiscard]] timecount_t get(taskid_t task_id, DeviceType arch) const {
    return task_durations[task_id * num_device_types +
                          static_cast<std::size_t>(arch)];
  }

  void set(taskid_t task_id, DeviceType arch, timecount_t value) {
    task_durations[task_id * num_device_types +
                   static_cast<std::size_t>(arch)] = value;
  }

  void set(std::vector<timecount_t> values_) {
    task_durations = std::move(values_);
  }
  virtual void generate() {
    for (taskid_t task_id = 0; task_id < tasks.compute_size(); task_id++) {
      for (std::size_t i = 0; i < num_device_types; i++) {
        set(task_id, static_cast<DeviceType>(i),
            sample_duration(task_id, static_cast<DeviceType>(i)));
      }
    }
  }

  void dump_to_binary(const std::string &filename) const {
    std::ofstream file(filename, std::ios::binary);
    for (const auto &duration : task_durations) {
      file.write(reinterpret_cast<const char *>(&duration),
                 sizeof(timecount_t));
    }
  }

  void load_from_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    const std::size_t size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    task_durations.resize(size / sizeof(timecount_t));
    file.read(reinterpret_cast<char *>(task_durations.data()),
              static_cast<std::streamsize>(size));
  }
};

class LognormalTaskNoise : public TaskNoise {

protected:
  [[nodiscard]] double get_stddev(taskid_t task_id, DeviceType arch) const {
    const double stddev = 0.2;
    return stddev;
  }

  [[nodiscard]] timecount_t sample_duration(taskid_t task_id,
                                            DeviceType arch) const override {
    const timecount_t mean =
        tasks.get_variant(task_id, arch).get_execution_time();
    const noise_t stddev = get_stddev(task_id, arch);
    const noise_t offset = exp(stddev * stddev / 2);
    std::lognormal_distribution<noise_t> dist(0, stddev);
    const noise_t v = dist(gen);
    return static_cast<timecount_t>((v - offset) * static_cast<noise_t>(mean));
  }

public:
  LognormalTaskNoise(Tasks &tasks_, unsigned int seed_ = 0)
      : TaskNoise(tasks_, seed_) {}
};
