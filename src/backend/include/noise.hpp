#pragma once
#include "devices.hpp"
#include "macros.hpp"
#include "tasks.hpp"
#include <fstream>
#include <functional>
#include <random>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

// TODO(wlr): Renable support for sampling different distributions

class TaskNoise {
protected:
  using noise_t = double;
  taskid_t n_tasks{};
  unsigned int seed = 0;
  unsigned int pseed = 1000;
  mutable std::mt19937 gen;
  mutable std::mt19937 pgen;
  std::vector<timecount_t> task_durations;
  std::vector<priority_t> mapping_priority;

  [[nodiscard]] virtual timecount_t sample_priority(taskid_t task_id) const {
    // generate uniform random number between 0 and max tasks - 1
    std::uniform_int_distribution<timecount_t> dist(0, n_tasks - 1);
    return dist(pgen);
  }

  [[nodiscard]] virtual timecount_t sample_duration(timecount_t mean_time) const {
    return mean_time;
  };

public:
  static constexpr uint32_t FILE_VERSION = 1;
  static constexpr size_t BUFFER_SIZE = 8192;
  bool generated = false;

  TaskNoise(StaticTaskInfo &static_graph, unsigned int seed_ = 0, unsigned int pseed = 1000)
      : n_tasks(static_graph.get_n_compute_tasks()), seed(seed_), pseed(pseed), gen(seed_),
        pgen(pseed) {
    try {
      auto n_compute_tasks = static_graph.get_n_compute_tasks();
      size_t duration_size =
          static_cast<size_t>(n_compute_tasks) * static_cast<size_t>(num_device_types);
      size_t priority_size = static_cast<size_t>(n_compute_tasks);

      task_durations.reserve(duration_size);
      task_durations.resize(duration_size, 0);

      mapping_priority.reserve(priority_size);
      mapping_priority.resize(priority_size, 0);
      std::iota(mapping_priority.begin(), mapping_priority.end(), 0);

      SPDLOG_DEBUG("TaskNoise initialized with {} tasks, seed: {}, pseed: {}", n_tasks, seed,
                   pseed);

      generate_duration(static_graph);
      generate_priority(static_graph);

    } catch (const std::bad_alloc &e) {
      throw std::runtime_error("Memory allocation failed in TaskNoise constructor");
    }
  }

  void set_seed(unsigned int seed_) {
    seed = seed_;
    gen.seed(seed);
  }

  void set_pseed(unsigned int pseed_) {
    pseed = pseed_;
    pgen.seed(pseed);
  }

  [[nodiscard]] timecount_t get(taskid_t task_id, DeviceType arch) const {
    const uint8_t arch_type = static_cast<uint8_t>(arch);
    const auto idx = __builtin_ctz(arch_type);
    assert(task_id < n_tasks && "Task ID is out of bounds");
    assert(idx < num_device_types && "Architecture index out of bounds");
    return task_durations[task_id * num_device_types + idx];
  }

  void set(taskid_t task_id, DeviceType arch, timecount_t value) {
    const uint8_t arch_type = static_cast<uint8_t>(arch);
    const auto idx = __builtin_ctz(arch_type);
    assert(task_id < n_tasks && "Task ID is out of bounds");
    assert(idx < num_device_types && "Architecture index out of bounds");
    task_durations[task_id * num_device_types + idx] = value;
  }

  void set(std::vector<timecount_t> values_) {
    task_durations = std::move(values_);
    generated = true;
  }

  void set_priority(taskid_t task_id, priority_t value) {
    mapping_priority[task_id] = value;
  }

  void set_priority(std::vector<priority_t> values_) {
    mapping_priority = std::move(values_);
  }

  [[nodiscard]] priority_t get_priority(taskid_t task_id) const {
    return mapping_priority[task_id];
  }

  std::span<timecount_t> get_durations() {
    return task_durations;
  }

  std::span<priority_t> get_priorities() {
    return mapping_priority;
  }
  const std::span<const timecount_t> get_durations() const {
    return task_durations;
  }
  const std::span<const priority_t> get_priorities() const {
    return mapping_priority;
  }

  [[nodiscard]] timecount_t operator()(taskid_t task_id, DeviceType arch) const {
    return get(task_id, arch);
  }

  void operator()(taskid_t task_id, DeviceType arch, timecount_t value) {
    set(task_id, arch, value);
  }

  virtual void generate_duration(StaticTaskInfo &task_info) {
    for (taskid_t task_id = 0; task_id < n_tasks; task_id++) {
      for (int8_t i = 0; i < num_device_types; i++) {
        auto arch = static_cast<DeviceType>(1 << i);
        bool is_supported = task_info.is_architecture_supported(task_id, arch);
        timecount_t observed_time =
            is_supported ? sample_duration(task_info.get_mean_duration(task_id, arch)) : 0;
        set(task_id, arch, observed_time);
      }
    }
  }

  virtual void generate_priority(StaticTaskInfo &task_info) {
    for (taskid_t task_id = 0; task_id < n_tasks; task_id++) {
      // set_priority(task_id, sample_priority(task_id));
      set_priority(task_id, task_id);
    }
  }

  void generate(StaticTaskInfo &task_info) {
    generate_duration(task_info);
    generate_priority(task_info);
  }

  // Binary dump/load for durations
  void dump_to_binary(const std::string &filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file)
      throw std::runtime_error("Unable to open file for writing: " + filename);

    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    file.write("TASK", 4);
    file.write(reinterpret_cast<const char *>(&FILE_VERSION), sizeof(FILE_VERSION));

    const uint64_t data_size = static_cast<uint64_t>(task_durations.size());
    file.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));

    file.write(reinterpret_cast<const char *>(task_durations.data()),
               data_size * sizeof(timecount_t));

    // Optionally add checksum here if needed for integrity
    if (file.fail())
      throw std::runtime_error("Error writing to file: " + filename);
  }

  void load_from_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
      throw std::runtime_error("Unable to open file for reading: " + filename);

    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    char header[4];
    file.read(header, 4);
    if (std::string(header, 4) != "TASK")
      throw std::runtime_error("Invalid file format");

    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != FILE_VERSION)
      throw std::runtime_error("Unsupported file version");

    uint64_t data_size;
    file.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));

    task_durations.resize(data_size);
    file.read(reinterpret_cast<char *>(task_durations.data()), data_size * sizeof(timecount_t));

    if (file.fail())
      throw std::runtime_error("Error reading from file: " + filename);
    generated = true;
  }

  // Binary dump/load for priorities
  void dump_priorities_to_binary(const std::string &filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file)
      throw std::runtime_error("Unable to open file for writing: " + filename);

    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    file.write("TASK", 4);
    file.write(reinterpret_cast<const char *>(&FILE_VERSION), sizeof(FILE_VERSION));

    const uint64_t data_size = static_cast<uint64_t>(mapping_priority.size());
    file.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));

    file.write(reinterpret_cast<const char *>(mapping_priority.data()),
               data_size * sizeof(priority_t));

    if (file.fail())
      throw std::runtime_error("Error writing to file: " + filename);
  }

  void load_priorities_from_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
      throw std::runtime_error("Unable to open file for reading: " + filename);

    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    char header[4];
    file.read(header, 4);
    if (std::string(header, 4) != "TASK")
      throw std::runtime_error("Invalid file format");

    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != FILE_VERSION)
      throw std::runtime_error("Unsupported file version");

    uint64_t data_size;
    file.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));

    mapping_priority.resize(data_size);
    file.read(reinterpret_cast<char *>(mapping_priority.data()), data_size * sizeof(priority_t));

    if (file.fail())
      throw std::runtime_error("Error reading from file: " + filename);
  }

  void save(const std::string &filename) const {
    dump_to_binary(filename + ".duration");
    dump_priorities_to_binary(filename + ".priority");
  }

  void load(const std::string &filename) {
    load_from_binary(filename + ".duration");
    load_priorities_from_binary(filename + ".priority");
  }
};

// using esf_t = double (*)(uint64_t, uint64_t);

// class ExternalTaskNoise : public TaskNoise {
// protected:
//   // function pointer
//   esf_t extern_function;

//   [[nodiscard]] timecount_t sample_duration(taskid_t task_id, DeviceType arch) const override {
//     return static_cast<timecount_t>(
//         extern_function(static_cast<uint64_t>(task_id), static_cast<uint64_t>(arch)));
//   }

// public:
//   ExternalTaskNoise(Tasks &tasks_, unsigned int seed_ = 0, unsigned int pseed_ = 1000)
//       : TaskNoise(tasks_, seed_, pseed_) {
//   }

//   void set_function(esf_t f) {
//     extern_function = f;
//   }
// };

class LognormalTaskNoise : public TaskNoise {
protected:
  double scale;

  [[nodiscard]] double get_stddev(taskid_t task_id, DeviceType arch) const {
    MONUnusedParameter(task_id);
    MONUnusedParameter(arch);
    const double stddev = scale;
    return stddev;
  }

  [[nodiscard]] timecount_t sample_duration(timecount_t mean_time) const override {
    const double mean = static_cast<double>(mean_time);
    const double stddev = scale * mean;

    if (mean == 0) {
      return 0;
    }

    const double u = std::log((mean * mean) / std::sqrt(mean * mean + stddev * stddev));
    const double s = std::log(1 + ((stddev * stddev) / (mean * mean)));

    std::lognormal_distribution<noise_t> dist(u, s);
    const noise_t duration = dist(gen);
    assert(duration >= 0);

    // std::cout << "LognormalTaskNoise: mean=" << mean << ", stddev=" << stddev
    //           << ", sampled duration=" << duration << std::endl;
    return static_cast<timecount_t>(duration);
  }

public:
  LognormalTaskNoise(StaticTaskInfo &tasks_, unsigned int seed_ = 0, unsigned int pseed_ = 1000,
                     double scale = 500)
      : TaskNoise(tasks_, seed_, pseed_), scale(scale) {
  }
};

class StaticLognormalTaskNoise : public TaskNoise {
protected:
  double stddev;

  [[nodiscard]] double get_stddev(taskid_t task_id, DeviceType arch) const {
    MONUnusedParameter(task_id);
    MONUnusedParameter(arch);
    const double stddev = this->stddev;
    return stddev;
  }

  [[nodiscard]] timecount_t sample_duration(timecount_t mean_time) const override {
    const double mean = static_cast<double>(mean_time);

    if (mean == 0) {
      return 0;
    }

    const double u = std::log((mean * mean) / std::sqrt(mean * mean + stddev * stddev));
    const double s = std::log(1 + ((stddev * stddev) / (mean * mean)));

    std::lognormal_distribution<noise_t> dist(u, s);
    const noise_t duration = dist(gen);
    assert(duration >= 0);

    // std::cout << "StaticLognormalTaskNoise: mean=" << mean << ", stddev=" << stddev
    //           << ", sampled duration=" << duration << std::endl;
    return static_cast<timecount_t>(duration);
  }

public:
  StaticLognormalTaskNoise(StaticTaskInfo &tasks_, unsigned int seed_ = 0,
                           unsigned int pseed_ = 1000, double stddev = 500)
      : TaskNoise(tasks_, seed_, pseed_), stddev(stddev) {
  }
};
