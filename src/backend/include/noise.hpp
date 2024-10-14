#pragma once
#include "devices.hpp"
#include "macros.hpp"
#include "tasks.hpp"
#include <fstream>
#include <functional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

class TaskNoise {
protected:
  using noise_t = double;
  std::reference_wrapper<Tasks> tasks;
  unsigned int seed = 0;
  unsigned int pseed = 1000;
  mutable std::mt19937 gen;
  mutable std::mt19937 pgen;
  std::vector<timecount_t> task_durations;
  std::vector<priority_t> mapping_priority;

  [[nodiscard]] virtual timecount_t sample_priority(taskid_t task_id) const {
    // generate uniform random number between 0 and max tasks - 1
    std::uniform_int_distribution<std::size_t> dist(0, tasks.get().size() - 1);
    return dist(pgen);
  }

  [[nodiscard]] virtual timecount_t sample_duration(taskid_t task_id,
                                                    DeviceType arch) const {
    const auto &ctasks = tasks.get();
    const auto &task = ctasks.get_compute_task(task_id);
    const auto &variant = task.get_variant(arch);
    return variant.get_observed_time();
  };

  template <typename T>
  static uint64_t calculate_checksum(const std::vector<T> &data) {
    const uint64_t prime = 0x100000001B3ull; // FNV prime
    uint64_t hash = 0xcbf29ce484222325ull;   // FNV offset basis

    const char *byte_data = reinterpret_cast<const char *>(data.data());
    size_t byte_count = data.size() * sizeof(T);

    for (size_t i = 0; i < byte_count; ++i) {
      hash ^= static_cast<uint64_t>(byte_data[i]);
      hash *= prime;
    }

    return hash;
  }

public:
  static constexpr uint32_t FILE_VERSION = 1;
  static constexpr size_t BUFFER_SIZE = 8192;
  bool generated = false;

  TaskNoise(Tasks &tasks_, unsigned int seed_ = 0, unsigned int pseed = 1000)
      : tasks(tasks_), seed(seed_), pseed(seed_), gen(seed_), pgen(pseed),
        task_durations(tasks.get().compute_size() * num_device_types, 0),
        mapping_priority(tasks.get().compute_size(), 0) {
    generate_duration();
    generate_priority();
  }

  [[nodiscard]] timecount_t get(taskid_t task_id, DeviceType arch) const {
    return task_durations.at(task_id * num_device_types +
                             static_cast<std::size_t>(arch));
  }

  void set(taskid_t task_id, DeviceType arch, timecount_t value) {
    task_durations.at(task_id * num_device_types +
                      static_cast<std::size_t>(arch)) = value;
  }

  void set(std::vector<timecount_t> values_) {
    task_durations = std::move(values_);
    generated = true;
  }

  void set_priority(taskid_t task_id, priority_t value) {
    mapping_priority.at(task_id) = value;
  }

  void set_priority(std::vector<priority_t> values_) {
    mapping_priority = std::move(values_);
  }

  [[nodiscard]] priority_t get_priority(taskid_t task_id) const {
    return mapping_priority.at(task_id);
  }

  std::vector<timecount_t> &get_durations() { return task_durations; }
  std::vector<priority_t> &get_priorities() { return mapping_priority; }

  void lock() { generated = true; }

  [[nodiscard]] timecount_t operator()(taskid_t task_id,
                                       DeviceType arch) const {
    return get(task_id, arch);
  }

  void operator()(taskid_t task_id, DeviceType arch, timecount_t value) {
    set(task_id, arch, value);
  }

  virtual void generate_duration() {
    for (taskid_t task_id = 0; task_id < tasks.get().compute_size();
         task_id++) {
      for (std::size_t i = 0; i < num_device_types; i++) {
        set(task_id, static_cast<DeviceType>(i),
            sample_duration(task_id, static_cast<DeviceType>(i)));
      }
    }
    generated = true;
  }

  virtual void generate_priority() {
    for (taskid_t task_id = 0; task_id < tasks.get().compute_size();
         task_id++) {
      set_priority(task_id, sample_priority(task_id));
    }
  }

  void generate() {
    generate_duration();
    generate_priority();
  }

  void dump_to_binary(const std::string &filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    // Set up buffering
    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    // Write header
    file.write("TASK", 4);
    file.write(reinterpret_cast<const char *>(&FILE_VERSION),
               sizeof(FILE_VERSION));

    // Write data size
    uint64_t data_size = task_durations.size();
    file.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));

    // Write data
    const char *data_ptr =
        reinterpret_cast<const char *>(task_durations.data());
    size_t remaining = task_durations.size() * sizeof(timecount_t);
    while (remaining > 0) {
      size_t chunk_size = std::min(remaining, BUFFER_SIZE);
      file.write(data_ptr, static_cast<std::streamsize>(chunk_size));
      data_ptr += chunk_size;
      remaining -= chunk_size;
    }

    // Write checksum
    uint64_t checksum = calculate_checksum(task_durations);
    file.write(reinterpret_cast<const char *>(&checksum), sizeof(checksum));

    if (file.fail()) {
      throw std::runtime_error("Error writing to file: " + filename);
    }
  }

  void load_from_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    // Set up buffering
    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    // Read and verify header
    std::array<char, 4> header;
    file.read(header.data(), header.size());
    if (std::string(header.data(), header.size()) != "TASK") {
      throw std::runtime_error("Invalid file format");
    }

    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != FILE_VERSION) {
      throw std::runtime_error("Unsupported file version");
    }

    // Read data size
    uint64_t data_size;
    file.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));

    // Read data
    task_durations.resize(data_size);
    char *data_ptr = reinterpret_cast<char *>(task_durations.data());
    size_t remaining = data_size * sizeof(timecount_t);
    while (remaining > 0) {
      size_t chunk_size = std::min(remaining, BUFFER_SIZE);
      file.read(data_ptr, static_cast<std::streamsize>(chunk_size));
      data_ptr += chunk_size;
      remaining -= chunk_size;
    }

    // Read and verify checksum
    uint64_t stored_checksum;
    uint64_t calculated_checksum;
    file.read(reinterpret_cast<char *>(&stored_checksum),
              sizeof(stored_checksum));
    calculated_checksum = calculate_checksum(task_durations);

    if (stored_checksum != calculated_checksum) {
      throw std::runtime_error("Checksum mismatch - file may be corrupted");
    }

    if (file.fail()) {
      throw std::runtime_error("Error reading from file: " + filename);
    }
    generated = true;
  }

  void dump_priorities_to_binary(const std::string &filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    // Set up buffering
    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    // Write header
    file.write("TASK", 4);
    file.write(reinterpret_cast<const char *>(&FILE_VERSION),
               sizeof(FILE_VERSION));

    // Write data size
    uint64_t data_size = mapping_priority.size();
    file.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));

    // Write data
    const char *data_ptr =
        reinterpret_cast<const char *>(mapping_priority.data());
    size_t remaining = mapping_priority.size() * sizeof(priority_t);
    while (remaining > 0) {
      size_t chunk_size = std::min(remaining, BUFFER_SIZE);
      file.write(data_ptr, static_cast<std::streamsize>(chunk_size));
      data_ptr += chunk_size;
      remaining -= chunk_size;
    }

    // Write checksum
    uint64_t checksum = calculate_checksum(mapping_priority);
    file.write(reinterpret_cast<const char *>(&checksum), sizeof(checksum));

    if (file.fail()) {
      throw std::runtime_error("Error writing to file: " + filename);
    }
  }

  void load_priorities_from_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    // Set up buffering
    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    // Read and verify header
    std::array<char, 4> header;
    file.read(header.data(), header.size());
    if (std::string(header.data(), header.size()) != "TASK") {
      throw std::runtime_error("Invalid file format");
    }

    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != FILE_VERSION) {
      throw std::runtime_error("Unsupported file version");
    }

    // Read data size
    uint64_t data_size;
    file.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));

    // Read data
    mapping_priority.resize(data_size);
    char *data_ptr = reinterpret_cast<char *>(mapping_priority.data());
    size_t remaining = data_size * sizeof(priority_t);
    while (remaining > 0) {
      size_t chunk_size = std::min(remaining, BUFFER_SIZE);
      file.read(data_ptr, static_cast<std::streamsize>(chunk_size));
      data_ptr += chunk_size;
      remaining -= chunk_size;
    }

    // Read and verify checksum
    uint64_t stored_checksum;
    uint64_t calculated_checksum;
    file.read(reinterpret_cast<char *>(&stored_checksum),
              sizeof(stored_checksum));
    calculated_checksum = calculate_checksum(mapping_priority);

    if (stored_checksum != calculated_checksum) {
      throw std::runtime_error("Checksum mismatch - file may be corrupted");
    }

    if (file.fail()) {
      throw std::runtime_error("Error reading from file: " + filename);
    }
  }
};

using esf_t = double (*)(uint64_t, uint64_t);

class ExternalTaskNoise : public TaskNoise {
protected:
  // function pointer
  esf_t extern_function;

  [[nodiscard]] timecount_t sample_duration(taskid_t task_id,
                                            DeviceType arch) const override {
    return static_cast<timecount_t>(extern_function(
        static_cast<uint64_t>(task_id), static_cast<uint64_t>(arch)));
  }

public:
  ExternalTaskNoise(Tasks &tasks_, unsigned int seed_ = 0,
                    unsigned int pseed_ = 1000)
      : TaskNoise(tasks_, seed_, pseed_) {}

  void set_function(esf_t f) { extern_function = f; }
};

class LognormalTaskNoise : public TaskNoise {

protected:
  [[nodiscard]] double get_stddev(taskid_t task_id, DeviceType arch) const {
    MONUnusedParameter(task_id);
    MONUnusedParameter(arch);
    const double stddev = 2;
    return stddev;
  }

  [[nodiscard]] timecount_t sample_duration(taskid_t task_id,
                                            DeviceType arch) const override {
    const auto mean = static_cast<double>(
        tasks.get().get_variant(task_id, arch).get_observed_time());
    const double stddev = 0.5 * mean;

    if (mean == 0) {
      return 0;
    }

    const double u =
        std::log((mean * mean) / std::sqrt(mean * mean + stddev * stddev));
    const double s = std::log(1 + ((stddev * stddev) / (mean * mean)));

    std::lognormal_distribution<noise_t> dist(u, s);
    const noise_t duration = dist(gen);
    assert(duration >= 0);
    return static_cast<timecount_t>(duration);
  }

public:
  LognormalTaskNoise(Tasks &tasks_, unsigned int seed_ = 0,
                     unsigned int pseed_ = 1000)
      : TaskNoise(tasks_, seed_, pseed_) {}
};
