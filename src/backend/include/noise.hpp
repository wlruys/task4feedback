#pragma once
#include "devices.hpp"
#include "macros.hpp"
#include "tasks.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <numeric>
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

  [[nodiscard]] virtual priority_t sample_priority(taskid_t task_id) const {
    (void)task_id;
    if (n_tasks <= 0) {
      return 0;
    }
    // Generate a priority in [0, n_tasks - 1] from the priority RNG.
    std::uniform_int_distribution<priority_t> dist(
        0, static_cast<priority_t>(n_tasks - 1));
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

  virtual std::vector<timecount_t> get_duration_vector() {
    return task_durations;
  }

  virtual void set_duration_vector(const std::vector<timecount_t> &durations) {
    task_durations = durations;
  }

  virtual std::vector<priority_t> get_priority_vector() {
    return mapping_priority;
  }

  virtual void set_priority_vector(const std::vector<priority_t> &priorities) {
    mapping_priority = priorities;
  }

  static int32_t infer_grid_height(int32_t grid_size) {
    if (grid_size <= 0) {
      return 1;
    }
    int32_t h = static_cast<int32_t>(std::floor(std::sqrt(static_cast<double>(grid_size))));
    for (; h > 1; --h) {
      if (grid_size % h == 0) {
        return h;
      }
    }
    return 1;
  }

  static uint64_t morton_encode_2d(uint32_t row, uint32_t col) {
    uint64_t code = 0;
    uint32_t max_coord = std::max(row, col);
    uint32_t bit = 0;
    while (max_coord > 0) {
      const uint64_t r = (static_cast<uint64_t>(row) >> bit) & 1ULL;
      const uint64_t c = (static_cast<uint64_t>(col) >> bit) & 1ULL;
      code |= (r << (2 * bit));
      code |= (c << (2 * bit + 1));
      max_coord >>= 1;
      ++bit;
    }
    return code;
  }

  virtual void generate_priority(StaticTaskInfo &task_info) {
    if (task_info.use_random_priority()) {
      for (taskid_t task_id = 0; task_id < n_tasks; task_id++) {
        set_priority(task_id, sample_priority(task_id));
      }
      return;
    }

    if (!task_info.get_morton_priority_enabled()) {
      for (taskid_t task_id = 0; task_id < n_tasks; task_id++) {
        // TODO(wlr, jae): RESTORE THIS, add external load and save of priorities to override it
        set_priority(task_id, task_id);
      }
      return;
    }

    int32_t max_tag = -1;
    bool has_tags = true;
    for (taskid_t task_id = 0; task_id < n_tasks; task_id++) {
      const auto &info = task_info.get_compute_task_static_info(task_id);
      if (info.tag < 0) {
        has_tags = false;
        break;
      }
      max_tag = std::max(max_tag, info.tag);
    }

    int32_t grid_size = has_tags ? (max_tag + 1) : static_cast<int32_t>(n_tasks);
    if (grid_size <= 0 || grid_size > n_tasks || (has_tags && (n_tasks % grid_size != 0))) {
      grid_size = static_cast<int32_t>(n_tasks);
      has_tags = false;
    }

    int32_t grid_h = task_info.get_grid_h();
    int32_t grid_w = task_info.get_grid_w();

    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != grid_size) {
      grid_h = infer_grid_height(grid_size);
      grid_w = grid_size / grid_h;
    }

    std::vector<uint64_t> morton_code(grid_size);
    for (int32_t local_id = 0; local_id < grid_size; ++local_id) {
      const uint32_t row = static_cast<uint32_t>(local_id % grid_h);
      const uint32_t col = static_cast<uint32_t>(local_id / grid_h);
      morton_code[local_id] = morton_encode_2d(row, col);
    }

    std::vector<int32_t> order(grid_size);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int32_t a, int32_t b) {
                if (morton_code[a] == morton_code[b]) {
                  return a < b;
                }
                return morton_code[a] < morton_code[b];
              });

    std::vector<int32_t> morton_rank(grid_size, 0);
    for (int32_t rank = 0; rank < grid_size; ++rank) {
      morton_rank[order[rank]] = rank;
    }

    for (taskid_t task_id = 0; task_id < n_tasks; task_id++) {
      // TODO(wlr, jae): RESTORE THIS, add external load and save of priorities to override it
      int32_t local_id = has_tags
                             ? task_info.get_compute_task_static_info(task_id).tag
                             : static_cast<int32_t>(task_id);
      if (local_id < 0 || local_id >= grid_size) {
        local_id = static_cast<int32_t>(task_id);
      }

      const int32_t step = has_tags ? (static_cast<int32_t>(task_id) / grid_size) : 0;
      const int32_t priority = step * grid_size + morton_rank[local_id];
      set_priority(task_id, priority);
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
