#pragma once
#include "queues.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <roaring/roaring.h>
#include <set>
#include <span>
#include <stack>
#include <string>
#include <tracy/Tracy.hpp>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

constexpr int32_t TASK_BUFFER_SIZE = 20;
constexpr int32_t EXPECTED_EVICTION_TASKS = 1000;
constexpr int32_t INITIAL_TASKS_SIZE = 1000;

enum class DeviceType : uint8_t {
  CPU = 1,
  GPU = 2
};
constexpr int8_t num_device_types = 2;

inline auto to_string(const DeviceType &arch) {
  switch (arch) {
  case DeviceType::CPU:
    return "CPU";
    break;
  case DeviceType::GPU:
    return "GPU";
    break;
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const DeviceType &arch) {
  os << to_string(arch);
  return os;
}

template <typename T> inline std::vector<T> as_vector(ankerl::unordered_dense::set<T> &set) {
  std::vector<T> vec;
  vec.reserve(set.size());
  for (const auto &item : set) {
    vec.push_back(item);
  }
  return vec;
}

template <typename K, typename V>
inline std::vector<V> as_vector(ankerl::unordered_dense::map<K, V> &map) {
  std::vector<V> vec;
  vec.reserve(map.size());
  for (const auto &item : map) {
    vec.push_back(item.second);
  }
  return vec;
}

template <typename T> inline std::vector<T> as_vector(const ankerl::unordered_dense::set<T> &set) {
  std::vector<T> vec;
  vec.reserve(set.size());
  for (const auto &item : set) {
    vec.push_back(item);
  }
  return vec;
}

template <typename K, typename V>
inline std::vector<V> as_vector(const ankerl::unordered_dense::map<K, V> &map) {
  std::vector<V> vec;
  vec.reserve(map.size());
  for (const auto &item : map) {
    vec.push_back(item.second);
  }
  return vec;
}

enum class TaskType : uint8_t {
  COMPUTE = 1,
  DATA = 2,
  EVICTION = 4,
};
constexpr std::size_t num_task_types = 3;

enum class TaskState : uint8_t {
  SPAWNED = 1,
  MAPPED = 2,
  RESERVED = 4,
  LAUNCHED = 8,
  COMPLETED = 16
};
constexpr std::size_t num_task_states = 5;

inline std::string to_string(const TaskState &state) {
  switch (state) {
  case TaskState::SPAWNED:
    return "SPAWNED";
    break;
  case TaskState::MAPPED:
    return "MAPPED";
    break;
  case TaskState::RESERVED:
    return "RESERVED";
    break;
  case TaskState::LAUNCHED:
    return "LAUNCHED";
    break;
  case TaskState::COMPLETED:
    return "COMPLETED";
    break;
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const TaskState &state) {
  os << to_string(state);
  return os;
}

enum class TaskStatus : int8_t {
  NONE = -1,
  MAPPABLE = 0,
  RESERVABLE = 2,
  LAUNCHABLE = 4,
};
constexpr std::size_t num_task_statuses = 3;

inline std::string to_string(const TaskStatus &status) {
  switch (status) {
  case TaskStatus::MAPPABLE:
    return "MAPPABLE";
    break;
  case TaskStatus::RESERVABLE:
    return "RESERVABLE";
    break;
  case TaskStatus::LAUNCHABLE:
    return "LAUNCHABLE";
    break;
  case TaskStatus::NONE:
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const TaskStatus &state) {
  os << to_string(state);
  return os;
}

class Task {
public:
  taskid_t id{};
  std::string name;

  ankerl::unordered_dense::set<taskid_t> dependencies;
  ankerl::unordered_dense::set<taskid_t> dependents;

  ankerl::unordered_dense::set<taskid_t> data_dependencies;
  ankerl::unordered_dense::set<taskid_t> data_dependents;

  ankerl::unordered_dense::set<taskid_t> temp_dependencies;
  ankerl::unordered_dense::set<taskid_t> temp_dependents;

  ankerl::unordered_dense::set<dataid_t> read;
  ankerl::unordered_dense::set<dataid_t> write;
  ankerl::unordered_dense::set<dataid_t> retire;

  std::vector<dataid_t> unique;
  std::vector<dataid_t> v_read;
  std::vector<taskid_t> most_recent_writers;

  std::vector<uint8_t> arch;
  std::vector<vcu_t> vcu;
  std::vector<mem_t> mem;
  std::vector<timecount_t> time;

  int32_t depth{};
  int32_t type{-1};
  int32_t tag{-1};
};

class DataTask {
public:
  taskid_t id{};
  std::string name;
  taskid_t compute_task{-1}; // ID of the compute task that produces this data
  dataid_t data_id{-1};      // Unique ID for the data
  ankerl::unordered_dense::set<taskid_t> dependencies;
  ankerl::unordered_dense::set<taskid_t> dependents;

  ankerl::unordered_dense::set<taskid_t> temp_dependencies;
  ankerl::unordered_dense::set<taskid_t> temp_dependents;
};

class Graph {
public:
  std::vector<Task> tasks;
  std::vector<DataTask> data_tasks;
  std::vector<taskid_t> sorted;
  std::vector<taskid_t> initial_tasks;
  bool finalized = false;

  Graph() = default;

  ankerl::unordered_dense::map<dataid_t, taskid_t>
      writers; // Maps data IDs to their most recent writer task ID

  taskid_t add_task(const std::string &name) {
    taskid_t id = static_cast<taskid_t>(tasks.size());
    tasks.emplace_back();
    tasks.back().id = id;
    tasks.back().name = name;
    return id;
  }

  const std::span<const taskid_t> get_initial_tasks() const {
    return std::span<const taskid_t>(initial_tasks);
  }

  std::size_t get_n_compute_tasks() const {
    return tasks.size();
  }

  std::size_t get_n_data_tasks() const {
    return data_tasks.size();
  }

  timecount_t get_time(taskid_t task_id, DeviceType arch) const {
    assert(task_id < tasks.size() && "Task ID is out of bounds");
    auto &task = tasks[task_id];
    for (std::size_t i = 0; i < task.arch.size(); i++) {
      if (task.arch[i] == static_cast<uint8_t>(arch)) {
        return task.time[i];
      }
    }
    return -1; // Return -1 if no time is found for the specified architecture
  }

  // Returns a vector of dependency task IDs for the given task
  std::vector<taskid_t> get_task_dependencies(taskid_t task_id) const {
    assert(task_id < tasks.size() && "Task ID is out of bounds");
    const auto &deps = tasks[task_id].dependencies;
    return std::vector<taskid_t>(deps.begin(), deps.end());
  }

  taskid_t add_data_task(const std::string &name, taskid_t compute_task, dataid_t data_id) {
    taskid_t id = static_cast<taskid_t>(data_tasks.size());
    data_tasks.emplace_back();
    data_tasks.back().id = id;
    data_tasks.back().name = name;
    data_tasks.back().compute_task = compute_task;
    data_tasks.back().data_id = data_id;
    return id;
  }

  void add_read_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    assert(task_id < tasks.size() && "Task ID is out of bounds");
    auto &task = tasks[task_id];
    for (const auto &data_id : data_ids) {
      task.read.insert(data_id);
      task.v_read.push_back(data_id);         // Store read data in a vector for ordered access
      task.most_recent_writers.push_back(-1); // Initialize with -1 for no writer
    }
  }

  void add_write_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    assert(task_id < tasks.size() && "Task ID is out of bounds");
    auto &task = tasks[task_id];
    for (const auto &data_id : data_ids) {
      task.write.insert(data_id);
    }
  }

  void add_retire_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    assert(task_id < tasks.size() && "Task ID is out of bounds");
    auto &task = tasks[task_id];
    for (const auto &data_id : data_ids) {
      task.retire.insert(data_id);
    }
  }

  void set_tag(taskid_t task_id, int32_t tag) {
    assert(task_id < tasks.size() && "Task ID is out of bounds");
    tasks[task_id].tag = tag;
  }

  void set_type(taskid_t task_id, int32_t type) {
    assert(task_id < tasks.size() && "Task ID is out of bounds");
    tasks[task_id].type = type;
  }

  void clear_variants(taskid_t task_id) {
    assert(task_id < tasks.size() && "Task ID is out of bounds");
    auto &task = tasks[task_id];
    task.arch.clear();
    task.vcu.clear();
    task.mem.clear();
    task.time.clear();
  }

  void clear_all_variants() {
    for (auto &task : tasks) {
      task.arch.clear();
      task.vcu.clear();
      task.mem.clear();
      task.time.clear();
    }
  }

  void set_variant(taskid_t task_id, DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time) {
    assert(task_id < tasks.size() && "Task ID is out of bounds");
    auto &task = tasks[task_id];

    // std::cout << "[Graph] Setting variant for task " << task_id << ": "
    //           << "Arch=" << to_string(arch) << ", VCU=" << vcu << ", Mem=" << mem
    //           << ", Time=" << time << std::endl;

    task.arch.push_back(static_cast<uint8_t>(arch));
    task.vcu.push_back(vcu);
    task.mem.push_back(mem);
    task.time.push_back(time);
  }

  void add_dependency(taskid_t task_id, taskid_t dependency_id) {
    tasks[task_id].dependencies.insert(dependency_id);
  }

  void add_dependencies(taskid_t task_id, const std::vector<taskid_t> &dependency_ids) {
    auto &task = tasks[task_id];
    for (const auto &dependency_id : dependency_ids) {
      task.dependencies.insert(dependency_id);
    }
  }

  void populate_dependencies_from_dataflow() {
    ankerl::unordered_dense::map<dataid_t, taskid_t> last_writer;
    for (auto &task : tasks) {
      for (const auto &data_id : task.read) {
        auto it = last_writer.find(data_id);
        if (it != last_writer.end()) {
          add_dependency(task.id, it->second);
        }
      }
      for (const auto &data_id : task.write) {
        last_writer[data_id] = task.id;
      }
    }
  }

  void populate_dependents() {
    for (auto &task : tasks) {
      for (const auto &dependency_id : task.dependencies) {
        tasks[dependency_id].dependents.insert(task.id);
      }
    }
  }

  void populate_data_dependents() {
    // Data tasks depend on compute tasks
    // Iterate data tasks
    // Update data dependents of the compute tasks based on data dependencies
    for (auto &data_task : data_tasks) {
      for (const auto &dependency_id : data_task.dependencies) {
        tasks[dependency_id].data_dependents.insert(data_task.id);
      }
    }

    // Compute tasks depend on data tasks
    // Iterate compute tasks
    // Update dependents of the data tasks based on data dependencies
    for (auto &task : tasks) {
      for (const auto &data_task_id : task.data_dependencies) {
        data_tasks[data_task_id].dependents.insert(task.id);
      }
    }
  }

  void populate_unique_data() {
    for (auto &task : tasks) {
      std::set<dataid_t> unique_data_set;
      for (const auto &data_id : task.read) {
        unique_data_set.insert(data_id);
      }
      for (const auto &data_id : task.write) {
        unique_data_set.insert(data_id);
      }
      task.unique.assign(unique_data_set.begin(), unique_data_set.end());
    }
  }

  void populate_initial_tasks() {
    initial_tasks.reserve(INITIAL_TASKS_SIZE);
    for (const auto &task : tasks) {
      if (task.dependencies.empty()) {
        initial_tasks.push_back(task.id);
      }
    }
  }

  void bfs() {
    sorted.clear();
    sorted.reserve(tasks.size());

    std::queue<taskid_t> queue;

    for (auto task_id : initial_tasks) {
      queue.push(task_id);
    }

    for (auto &task : tasks) {
      task.temp_dependencies.clear();
      task.temp_dependents.clear();
      task.temp_dependencies.insert(task.dependencies.begin(), task.dependencies.end());
      task.temp_dependents.insert(task.dependents.begin(), task.dependents.end());
    }

    while (!queue.empty()) {
      taskid_t current = queue.front();
      queue.pop();
      sorted.push_back(current);

      for (const auto &dependent : tasks[current].temp_dependents) {
        tasks[dependent].temp_dependencies.erase(current);
        if (tasks[dependent].temp_dependencies.empty()) {
          queue.push(dependent);
        }
      }
    }
  }

  void dfs() {
    sorted.clear();
    sorted.reserve(tasks.size());

    std::stack<taskid_t> stack;

    for (auto &task : initial_tasks) {
      stack.push(task);
    }

    for (auto &task : tasks) {
      task.temp_dependencies.clear();
      task.temp_dependents.clear();
      task.temp_dependencies.insert(task.dependencies.begin(), task.dependencies.end());
      task.temp_dependents.insert(task.dependents.begin(), task.dependents.end());
    }

    while (!stack.empty()) {
      taskid_t current = stack.top();
      stack.pop();
      sorted.push_back(current);

      for (const auto &dependent : tasks[current].temp_dependents) {
        tasks[dependent].temp_dependencies.erase(current);
        if (tasks[dependent].temp_dependencies.empty()) {
          stack.push(dependent);
        }
      }
    }
  }

  void random_topological_sort(unsigned int seed = 0) {
    sorted.clear();
    sorted.reserve(tasks.size());

    auto r = ContainerQueue<taskid_t, std::priority_queue>(seed);

    for (auto &task : initial_tasks) {
      r.push_random(task);
    }

    for (auto &task : tasks) {
      task.temp_dependencies.clear();
      task.temp_dependents.clear();
      task.temp_dependencies.insert(task.dependencies.begin(), task.dependencies.end());
      task.temp_dependents.insert(task.dependents.begin(), task.dependents.end());
    }

    while (!r.empty()) {
      taskid_t current = r.top();
      r.pop();
      sorted.push_back(current);

      for (const auto &dependent : tasks[current].temp_dependents) {
        tasks[dependent].temp_dependencies.erase(current);
        if (tasks[dependent].temp_dependencies.empty()) {
          r.push_random(dependent);
        }
      }
    }
  }

  void populate_depth() {
    for (auto &task : tasks) {
      task.depth = 0; // Reset depth
    }
    for (const auto &task_id : sorted) {
      auto &task = tasks[task_id];
      for (const auto &dependency_id : task.dependencies) {
        auto &dependency_task = tasks[dependency_id];
        if (dependency_task.depth + 1 > task.depth) {
          task.depth = dependency_task.depth + 1;
        }
      }
    }
  }

  void create_data_task(taskid_t task_id, dataid_t data_id, bool has_writer,
                        taskid_t writer_id = -1) {

    auto &task = tasks[task_id];
    auto data_name = std::to_string(task_id) + "_data_" + std::to_string(data_id);
    auto data_task_id = add_data_task(data_name, task_id, data_id);
    auto &data_task = data_tasks[data_task_id];

    if (has_writer) {
      data_task.dependencies.insert(writer_id);
      auto &writer_task = tasks[writer_id];
      writer_task.data_dependents.insert(data_task_id);
    }

    data_task.dependents.insert(task_id);
    task.data_dependencies.insert(data_task_id);
  }

  void populate_data_dependencies(bool ensure_dependencies = false, bool create_data_tasks = true) {
    writers.clear();

    // Iterate in a valid topological order
    for (auto task_id : sorted) {

      auto &task = tasks[task_id];

      // Create data tasks for all reads from current task
      if (create_data_tasks) {
        for (int32_t i = 0; i < task.v_read.size(); ++i) {
          dataid_t data_id = task.v_read[i];
          auto it = writers.find(data_id);
          taskid_t writer_id = -1;
          bool has_writer = it != writers.end();
          if (has_writer) {

            writer_id = it->second;
            task.most_recent_writers[i] = it->second;
          }
          create_data_task(task_id, data_id, has_writer, writer_id);
        }
      }

      if (ensure_dependencies) {
        // writers are compute tasks
        // Ensure that the compute task depends on all writers of the data it reads
        for (const auto &data_id : task.read) {
          auto it = writers.find(data_id);
          if (it != writers.end()) {
            task.dependencies.insert(it->second);
            tasks[it->second].dependents.insert(task_id);
          }
        }

        // Ensure that the compute task depends on all writers of the data it writes
        for (const auto &data_id : task.write) {
          auto it = writers.find(data_id);
          if (it != writers.end()) {
            task.dependencies.insert(it->second);
            tasks[it->second].dependents.insert(task_id);
          }
        }

        // Ensure that the compute task depends on all writers of the data it retires
        for (const auto &data_id : task.retire) {
          auto it = writers.find(data_id);
          if (it != writers.end()) {
            task.dependencies.insert(it->second);
            tasks[it->second].dependents.insert(task_id);
          }
        }
      }

      // Update writers map with current task's writes
      for (const auto &data_id : task.write) {
        writers[data_id] = task_id;
      }
    }
  }

  void finalize(bool ensure_dependencies = false, bool create_data_tasks = true) {
    if (finalized) {
      std::cerr << "Graph is already finalized. Cannot finalize again." << std::endl;
      std::cerr << "If you want to re-finalize, please create a new Graph instance." << std::endl;
      std::cerr << "Exiting..." << std::endl;
      std::exit(EXIT_FAILURE);
      return;
    }
    finalized = true;
    populate_dependencies_from_dataflow();
    populate_unique_data();
    populate_dependents();
    populate_initial_tasks();
    bfs();
    populate_depth();
    populate_data_dependencies(ensure_dependencies, create_data_tasks);
    // populate_data_dependents();
  }
};

class Variant {
public:
  Resources resources;
  timecount_t time = 0;
  DeviceType arch = DeviceType::GPU;

  Variant() = default;
  Variant(DeviceType arch_, vcu_t vcu_, mem_t mem_, timecount_t time_)
      : resources(vcu_, mem_), time(time_), arch(arch_) {
  }

  [[nodiscard]] DeviceType get_arch() const {
    return arch;
  }

  [[nodiscard]] vcu_t get_vcus() const {
    return resources.vcu;
  }
  [[nodiscard]] mem_t get_mem() const {
    return resources.mem;
  }

  [[nodiscard]] const Resources &get_resources() const {
    return resources;
  }

  [[nodiscard]] timecount_t get_mean_duration() const {
    return time;
  }
};

using VariantList = std::array<Variant, num_device_types>;

struct ComputeTaskStaticInfo {
  int32_t tag{};
  int32_t type{};
  int32_t depth{};
};

struct ComputeTaskVariantInfo {
  uint8_t mask = 0; // bitmask for supported architectures
  std::array<Variant, num_device_types> variants{};
};

struct ComputeTaskDepInfo {
  int32_t s_dependencies;
  int32_t e_dependencies;
  int32_t s_dependents;
  int32_t e_dependents;
  int32_t s_data_dependencies;
  int32_t e_data_dependencies;
  int32_t s_data_dependents;
  int32_t e_data_dependents;
};

struct ComputeTaskDataInfo {
  int32_t s_read{};
  int32_t e_read{};
  int32_t s_write{};
  int32_t e_write{};
  int32_t s_retire{};
  int32_t e_retire{};
  int32_t s_unique{};
  int32_t e_unique{};
};

struct DataTaskStaticInfo {
  int32_t s_dependencies{};
  int32_t e_dependencies{};
  int32_t s_dependents{};
  int32_t e_dependents{};
  int32_t data_id{};
  int32_t compute_task{};
  int64_t pad{}; // padding to align to 32 bytes
};

struct ComputeTaskRuntimeInfo {
  int32_t mapped_device{-1};
  int32_t reserve_priority{};
  int32_t launch_priority{};
  int16_t unmapped{};
  int16_t unreserved{};
  int16_t incomplete{};
  uint8_t state{};
  uint8_t flags{};
};

struct DataTaskRuntimeInfo {
  int32_t source_device{};
  int32_t mapped_device{-1};
  int32_t launch_priority{};
  int16_t incomplete{};
  uint8_t state{};
  uint8_t flags{};
};

struct EvictionTaskRuntimeInfo {
  int32_t data_id{};
  int32_t evicting_on{};
  int32_t compute_task{};
  int32_t source_device{};
  int32_t launch_priority{};
  uint8_t state{};
  uint8_t flags{};
  int16_t pad{};
  int64_t pad2{};
};

struct TaskTimeRecord {
  timecount_t mapped_time{};
  timecount_t reserved_time{};
  timecount_t launched_time{};
  timecount_t completed_time{};
};

struct DataTaskTimeRecord {
  timecount_t launched_time{};
  timecount_t completed_time{};
};

class RoaringBitmap {
private:
  roaring_bitmap_t *bitmap_{nullptr};

public:
  RoaringBitmap() : bitmap_(roaring_bitmap_create()) {
  }

  ~RoaringBitmap() {
    if (bitmap_ != nullptr) {
      roaring_bitmap_free(bitmap_);
      bitmap_ = nullptr;
    }
  }

  RoaringBitmap(const RoaringBitmap &other)
      : bitmap_(other.bitmap_ != nullptr ? roaring_bitmap_copy(other.bitmap_)
                                         : roaring_bitmap_create()) {
  }

  RoaringBitmap &operator=(const RoaringBitmap &other) {
    if (this == &other) {
      return *this;
    }
    roaring_bitmap_t *copied = other.bitmap_ != nullptr ? roaring_bitmap_copy(other.bitmap_)
                                                         : roaring_bitmap_create();
    if (bitmap_ != nullptr) {
      roaring_bitmap_free(bitmap_);
    }
    bitmap_ = copied;
    return *this;
  }

  RoaringBitmap(RoaringBitmap &&other) noexcept : bitmap_(other.bitmap_) {
    other.bitmap_ = nullptr;
  }

  RoaringBitmap &operator=(RoaringBitmap &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (bitmap_ != nullptr) {
      roaring_bitmap_free(bitmap_);
    }
    bitmap_ = other.bitmap_;
    other.bitmap_ = nullptr;
    return *this;
  }

  void add(uint32_t value) {
    roaring_bitmap_add(bitmap_, value);
  }

  void finalize() {
    if (bitmap_ != nullptr) {
      roaring_bitmap_run_optimize(bitmap_);
    }
  }

  [[nodiscard]] bool contains(uint32_t value) const {
    return bitmap_ != nullptr && roaring_bitmap_contains(bitmap_, value);
  }

  [[nodiscard]] std::size_t serialized_size() const {
    return bitmap_ == nullptr ? 0 : roaring_bitmap_portable_size_in_bytes(bitmap_);
  }
};

struct TaskDataMembershipBitmapView {
  const RoaringBitmap *bitmap{nullptr};

  [[nodiscard]] bool valid() const {
    return bitmap != nullptr;
  }

  [[nodiscard]] bool contains(dataid_t data_id) const {
    if (!valid() || data_id < 0) {
      return false;
    }
    return bitmap->contains(static_cast<uint32_t>(data_id));
  }
};

class StaticTaskInfo {

protected:
  std::vector<ComputeTaskDepInfo> compute_task_dep_info;
  std::vector<ComputeTaskDataInfo> compute_task_data_info;
  std::vector<ComputeTaskVariantInfo> compute_task_variant_info;
  std::vector<DataTaskStaticInfo> data_task_static_info;
  std::vector<ComputeTaskStaticInfo> compute_task_static_info;

  std::vector<taskid_t> compute_task_dependencies;
  std::vector<taskid_t> compute_task_dependents;
  std::vector<taskid_t> compute_task_data_dependencies;
  std::vector<taskid_t> compute_task_data_dependents;
  std::vector<dataid_t> compute_task_read;
  std::vector<dataid_t> compute_task_write;
  std::vector<dataid_t> compute_task_retire;
  std::vector<dataid_t> compute_task_recent_writers;
  std::vector<dataid_t> compute_task_unique;

  // CSR cache for read usage: data_id -> compute tasks that read data_id
  std::vector<dataid_t> read_usage_data_ids;
  std::vector<int32_t> read_usage_offsets;
  std::vector<taskid_t> read_usage_tasks;
  ankerl::unordered_dense::map<dataid_t, int32_t> read_usage_row_by_data_id;

  // CSR cache for shared-read topology: compute task -> compute tasks sharing a read data id
  std::vector<int32_t> compute_task_shared_read_offsets;
  std::vector<taskid_t> compute_task_shared_read_neighbors;

  // Task/data membership index for hot read/write/unique checks (Roaring bitmaps).
  std::vector<RoaringBitmap> compute_task_read_membership;
  std::vector<RoaringBitmap> compute_task_write_membership;
  std::vector<RoaringBitmap> compute_task_unique_membership;
  ankerl::unordered_dense::map<uint64_t, int32_t> read_index_by_task_data;
  bool membership_index_enabled{false};
  std::size_t membership_index_bytes{0};

  std::vector<taskid_t> data_task_dependencies;
  std::vector<taskid_t> data_task_dependents;
  // Hot-path lookup caches: built once during static graph construction.
  std::vector<int32_t> compute_task_dependents_offsets;
  std::vector<int32_t> compute_task_data_dependents_offsets;
  std::vector<int32_t> data_task_dependents_offsets;
  std::vector<dataid_t> data_task_data_id_cache;
  std::vector<taskid_t> data_task_compute_task_cache;

  std::vector<std::string> compute_task_names;
  std::vector<std::string> data_task_names;

  int32_t grid_h{-1};
  int32_t grid_w{-1};
  bool morton_priority_enabled{false};
  bool random_priority_enabled{false};

  [[nodiscard]] static uint64_t task_data_key(taskid_t task_id, dataid_t data_id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(task_id)) << 32U) |
           static_cast<uint64_t>(static_cast<uint32_t>(data_id));
  }

  void set_membership_value(std::vector<RoaringBitmap> &bitmaps, taskid_t task_id,
                            dataid_t data_id) {
    if (data_id < 0 || task_id < 0 || task_id >= get_n_compute_tasks()) {
      return;
    }
    bitmaps[static_cast<std::size_t>(task_id)].add(static_cast<uint32_t>(data_id));
  }

  [[nodiscard]] bool contains_membership_value(const std::vector<RoaringBitmap> &bitmaps,
                                               taskid_t task_id, dataid_t data_id) const {
    if (!membership_index_enabled || task_id < 0 || task_id >= get_n_compute_tasks() || data_id < 0) {
      return false;
    }
    return bitmaps[static_cast<std::size_t>(task_id)].contains(static_cast<uint32_t>(data_id));
  }

  [[nodiscard]] TaskDataMembershipBitmapView
  get_membership_view(const std::vector<RoaringBitmap> &bitmaps, taskid_t task_id) const {
    if (!membership_index_enabled || task_id < 0 || task_id >= get_n_compute_tasks()) {
      return {};
    }
    return {.bitmap = &bitmaps[static_cast<std::size_t>(task_id)]};
  }

  void build_task_data_membership_index() {
    compute_task_read_membership.clear();
    compute_task_write_membership.clear();
    compute_task_unique_membership.clear();
    read_index_by_task_data.clear();
    membership_index_enabled = false;
    membership_index_bytes = 0;

    const auto n_compute_tasks = get_n_compute_tasks();
    if (n_compute_tasks <= 0) {
      return;
    }

    compute_task_read_membership.resize(static_cast<std::size_t>(n_compute_tasks));
    compute_task_write_membership.resize(static_cast<std::size_t>(n_compute_tasks));
    compute_task_unique_membership.resize(static_cast<std::size_t>(n_compute_tasks));
    read_index_by_task_data.reserve(compute_task_read.size());

    for (taskid_t task_id = 0; task_id < n_compute_tasks; ++task_id) {
      const auto read_span = get_read(task_id);
      for (int32_t idx = 0; idx < static_cast<int32_t>(read_span.size()); ++idx) {
        const auto data_id = read_span[idx];
        set_membership_value(compute_task_read_membership, task_id, data_id);
        const auto key = task_data_key(task_id, data_id);
        if (read_index_by_task_data.find(key) == read_index_by_task_data.end()) {
          read_index_by_task_data.emplace(key, idx);
        }
      }

      const auto write_span = get_write(task_id);
      for (const auto data_id : write_span) {
        set_membership_value(compute_task_write_membership, task_id, data_id);
      }

      const auto unique_span = get_unique(task_id);
      for (const auto data_id : unique_span) {
        set_membership_value(compute_task_unique_membership, task_id, data_id);
      }
    }

    for (auto &bitmap : compute_task_read_membership) {
      bitmap.finalize();
    }
    for (auto &bitmap : compute_task_write_membership) {
      bitmap.finalize();
    }
    for (auto &bitmap : compute_task_unique_membership) {
      bitmap.finalize();
    }

    std::size_t total_bytes = 0;
    for (const auto &bitmap : compute_task_read_membership) {
      total_bytes += bitmap.serialized_size();
    }
    for (const auto &bitmap : compute_task_write_membership) {
      total_bytes += bitmap.serialized_size();
    }
    for (const auto &bitmap : compute_task_unique_membership) {
      total_bytes += bitmap.serialized_size();
    }

    membership_index_enabled = true;
    membership_index_bytes = total_bytes;
  }

public:
  StaticTaskInfo(int32_t num_compute_tasks, int32_t num_data_tasks) {
    compute_task_dep_info.resize(num_compute_tasks);
    compute_task_data_info.resize(num_compute_tasks);
    compute_task_variant_info.resize(num_compute_tasks);
    compute_task_recent_writers.resize(num_compute_tasks);
    data_task_static_info.resize(num_data_tasks);
    compute_task_static_info.resize(num_compute_tasks);

    compute_task_names.resize(num_compute_tasks);
    data_task_names.resize(num_data_tasks);
    compute_task_dependents_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_data_dependents_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    data_task_dependents_offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);
    data_task_data_id_cache.resize(num_data_tasks, 0);
    data_task_compute_task_cache.resize(num_data_tasks, 0);

    // Keep CSR structures valid even when no precomputation has been run yet.
    read_usage_offsets.resize(1, 0);
    compute_task_shared_read_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    build_task_data_membership_index();
  }

  StaticTaskInfo(Graph &graph) {

    taskid_t num_compute_tasks = graph.get_n_compute_tasks();
    taskid_t num_data_tasks = graph.get_n_data_tasks();

    compute_task_dep_info.resize(num_compute_tasks);
    compute_task_data_info.resize(num_compute_tasks);
    compute_task_variant_info.resize(num_compute_tasks);
    compute_task_recent_writers.resize(num_compute_tasks);
    data_task_static_info.resize(num_data_tasks);
    compute_task_static_info.resize(num_compute_tasks);

    compute_task_names.resize(num_compute_tasks);
    data_task_names.resize(num_data_tasks);
    compute_task_dependents_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_data_dependents_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    data_task_dependents_offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);
    data_task_data_id_cache.resize(num_data_tasks, 0);
    data_task_compute_task_cache.resize(num_data_tasks, 0);

    read_usage_offsets.resize(1, 0);
    compute_task_shared_read_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);

    // std::cout << "Creating static graph..." << std::endl;
    // std::cout << "Number of compute tasks: " << num_compute_tasks << std::endl;
    // std::cout << "Number of data tasks: " << num_data_tasks << std::endl;

    auto &tasks = graph.tasks;
    auto &data_tasks = graph.data_tasks;

    taskid_t compute_dependency_offset = 0;
    taskid_t compute_dependent_offset = 0;
    taskid_t compute_data_dependency_offset = 0;
    taskid_t compute_data_dependent_offset = 0;

    dataid_t read_offset = 0;
    dataid_t write_offset = 0;
    dataid_t retire_offset = 0;
    dataid_t unique_offset = 0;

    taskid_t data_dependency_offset = 0;
    taskid_t data_dependent_offset = 0;

    taskid_t total_compute_dependencies = 0;
    taskid_t total_compute_dependents = 0;
    taskid_t total_compute_data_dependencies = 0;
    taskid_t total_compute_data_dependents = 0;
    taskid_t total_data_dependencies = 0;
    taskid_t total_reads = 0;
    taskid_t total_writes = 0;
    taskid_t total_retire = 0;
    taskid_t total_unique = 0;

    for (const auto &task : tasks) {
      total_compute_dependencies += task.dependencies.size();
      total_compute_dependents += task.dependents.size();
      total_compute_data_dependencies += task.data_dependencies.size();
      total_compute_data_dependents += task.data_dependents.size();
      total_reads += task.read.size();
      total_writes += task.write.size();
      total_retire += task.retire.size();
      total_unique += task.unique.size();
    }

    set_total_compute_task_dependencies(total_compute_dependencies);
    set_total_compute_task_dependents(total_compute_dependents);
    set_total_compute_task_data_dependencies(total_compute_data_dependencies);
    set_total_compute_task_data_dependents(total_compute_data_dependents);
    set_total_reads(total_reads);
    set_total_writes(total_writes);
    set_total_retires(total_retire);
    set_total_unique(total_unique);

    for (const auto &task : tasks) {
      auto compute_dep_info = ComputeTaskDepInfo();
      auto compute_data_info = ComputeTaskDataInfo();
      auto compute_task_info = ComputeTaskStaticInfo();

      compute_dep_info.s_dependencies = compute_dependency_offset;
      compute_dep_info.e_dependencies = compute_dependency_offset + task.dependencies.size();
      compute_dependency_offset += task.dependencies.size();

      compute_dep_info.s_dependents = compute_dependent_offset;
      compute_dep_info.e_dependents = compute_dependent_offset + task.dependents.size();
      compute_dependent_offset += task.dependents.size();

      compute_dep_info.s_data_dependencies = compute_data_dependency_offset;
      compute_dep_info.e_data_dependencies =
          compute_data_dependency_offset + task.data_dependencies.size();
      compute_data_dependency_offset += task.data_dependencies.size();

      compute_dep_info.s_data_dependents = compute_data_dependent_offset;
      compute_dep_info.e_data_dependents =
          compute_data_dependent_offset + task.data_dependents.size();
      compute_data_dependent_offset += task.data_dependents.size();

      compute_data_info.s_read = read_offset;
      compute_data_info.e_read = read_offset + task.read.size();
      read_offset += task.read.size();

      compute_data_info.s_write = write_offset;
      compute_data_info.e_write = write_offset + task.write.size();
      write_offset += task.write.size();

      compute_data_info.s_retire = retire_offset;
      compute_data_info.e_retire = retire_offset + task.retire.size();
      retire_offset += task.retire.size();

      compute_data_info.s_unique = unique_offset;
      compute_data_info.e_unique = unique_offset + task.unique.size();
      unique_offset += task.unique.size();

      compute_task_info.tag = task.tag;
      compute_task_info.type = task.type;

      add_compute_task(task.id, task.name, compute_dep_info, compute_data_info, compute_task_info);

      add_compute_task_dependencies(task.id, as_vector(task.dependencies));
      add_compute_task_dependents(task.id, as_vector(task.dependents));
      add_compute_task_data_dependencies(task.id, as_vector(task.data_dependencies));
      add_compute_task_data_dependents(task.id, as_vector(task.data_dependents));
      add_read(task.id, task.v_read);
      add_most_recent_writers(task.id, task.most_recent_writers);
      add_write(task.id, as_vector(task.write));
      add_retire(task.id, as_vector(task.retire));
      add_unique(task.id, task.unique);
      add_depth(task.id, task.depth);

      for (int i = 0; i < task.arch.size(); ++i) {
        const auto arch = static_cast<DeviceType>(task.arch[i]);
        add_compute_variant(task.id, arch, task.mem[i], task.vcu[i], task.time[i]);
      }
    }

    taskid_t total_data_task_dependencies = 0;
    taskid_t total_data_task_dependents = 0;
    for (const auto &data_task : data_tasks) {
      total_data_task_dependencies += data_task.dependencies.size();
      total_data_task_dependents += data_task.dependents.size();
    }

    set_total_data_task_dependencies(total_data_task_dependencies);
    set_total_data_task_dependents(total_data_task_dependents);

    for (const auto &data_task : data_tasks) {
      auto data_task_info = DataTaskStaticInfo();

      data_task_info.s_dependencies = data_dependency_offset;
      data_task_info.e_dependencies = data_dependency_offset + data_task.dependencies.size();
      data_dependency_offset += data_task.dependencies.size();

      data_task_info.s_dependents = data_dependent_offset;
      data_task_info.e_dependents = data_dependent_offset + data_task.dependents.size();
      data_dependent_offset += data_task.dependents.size();

      data_task_info.data_id = data_task.data_id;
      data_task_info.compute_task = data_task.compute_task;

      add_data_task(data_task.id, data_task.name, data_task_info);

      add_data_task_dependencies(data_task.id, as_vector(data_task.dependencies));
      add_data_task_dependents(data_task.id, as_vector(data_task.dependents));
    }

    build_read_usage_and_shared_read_topology();
    build_task_data_membership_index();
  }

  // Update variants
  void update_variants(Graph &graph) {
    auto &tasks = graph.tasks;

    for (const auto &task : tasks) {
      for (int i = 0; i < task.arch.size(); ++i) {
        const auto arch = static_cast<DeviceType>(task.arch[i]);
        add_compute_variant(task.id, arch, task.mem[i], task.vcu[i], task.time[i]);
      }
    }
  }

  void build_read_usage_and_shared_read_topology() {
    const auto n_compute_tasks = get_n_compute_tasks();

    ankerl::unordered_dense::map<dataid_t, std::vector<taskid_t>> readers_by_data;
    readers_by_data.reserve(compute_task_read.size());

    std::vector<dataid_t> unique_reads_buffer;
    unique_reads_buffer.reserve(32);

    for (taskid_t task_id = 0; task_id < n_compute_tasks; ++task_id) {
      const auto read_span = get_read(task_id);
      if (read_span.empty()) {
        continue;
      }

      unique_reads_buffer.assign(read_span.begin(), read_span.end());
      std::sort(unique_reads_buffer.begin(), unique_reads_buffer.end());
      unique_reads_buffer.erase(
          std::unique(unique_reads_buffer.begin(), unique_reads_buffer.end()),
          unique_reads_buffer.end());

      for (const auto data_id : unique_reads_buffer) {
        readers_by_data[data_id].push_back(task_id);
      }
    }

    read_usage_row_by_data_id.clear();
    read_usage_data_ids.clear();
    read_usage_tasks.clear();
    read_usage_offsets.clear();
    read_usage_offsets.push_back(0);
    read_usage_tasks.reserve(compute_task_read.size());

    std::vector<dataid_t> sorted_data_ids;
    sorted_data_ids.reserve(readers_by_data.size());
    for (const auto &entry : readers_by_data) {
      sorted_data_ids.push_back(entry.first);
    }
    std::sort(sorted_data_ids.begin(), sorted_data_ids.end());
    read_usage_row_by_data_id.reserve(sorted_data_ids.size());
    read_usage_data_ids.reserve(sorted_data_ids.size());

    std::vector<uint64_t> shared_pair_keys;
    for (const auto data_id : sorted_data_ids) {
      auto &readers = readers_by_data[data_id];
      std::sort(readers.begin(), readers.end());
      readers.erase(std::unique(readers.begin(), readers.end()), readers.end());

      read_usage_row_by_data_id[data_id] = static_cast<int32_t>(read_usage_data_ids.size());
      read_usage_data_ids.push_back(data_id);
      read_usage_tasks.insert(read_usage_tasks.end(), readers.begin(), readers.end());
      read_usage_offsets.push_back(static_cast<int32_t>(read_usage_tasks.size()));

      const auto n_readers = readers.size();
      if (n_readers < 2) {
        continue;
      }

      shared_pair_keys.reserve(shared_pair_keys.size() + ((n_readers * (n_readers - 1)) / 2));
      for (std::size_t i = 0; i < n_readers; ++i) {
        const auto lhs = static_cast<uint64_t>(static_cast<uint32_t>(readers[i]));
        for (std::size_t j = i + 1; j < n_readers; ++j) {
          const auto rhs = static_cast<uint64_t>(static_cast<uint32_t>(readers[j]));
          shared_pair_keys.push_back((lhs << 32U) | rhs);
        }
      }
    }

    compute_task_shared_read_offsets.assign(static_cast<std::size_t>(n_compute_tasks) + 1, 0);
    compute_task_shared_read_neighbors.clear();

    if (shared_pair_keys.empty()) {
      return;
    }

    std::sort(shared_pair_keys.begin(), shared_pair_keys.end());
    shared_pair_keys.erase(std::unique(shared_pair_keys.begin(), shared_pair_keys.end()),
                           shared_pair_keys.end());

    for (const auto key : shared_pair_keys) {
      const auto lhs = static_cast<taskid_t>(key >> 32U);
      const auto rhs = static_cast<taskid_t>(key & 0xFFFFFFFFULL);
      compute_task_shared_read_offsets[lhs + 1] += 1;
      compute_task_shared_read_offsets[rhs + 1] += 1;
    }

    for (taskid_t task_id = 0; task_id < n_compute_tasks; ++task_id) {
      compute_task_shared_read_offsets[task_id + 1] += compute_task_shared_read_offsets[task_id];
    }

    compute_task_shared_read_neighbors.resize(
        static_cast<std::size_t>(compute_task_shared_read_offsets.back()), -1);
    auto write_offsets = compute_task_shared_read_offsets;

    for (const auto key : shared_pair_keys) {
      const auto lhs = static_cast<taskid_t>(key >> 32U);
      const auto rhs = static_cast<taskid_t>(key & 0xFFFFFFFFULL);
      compute_task_shared_read_neighbors[write_offsets[lhs]++] = rhs;
      compute_task_shared_read_neighbors[write_offsets[rhs]++] = lhs;
    }

    for (taskid_t task_id = 0; task_id < n_compute_tasks; ++task_id) {
      const auto begin = static_cast<std::size_t>(compute_task_shared_read_offsets[task_id]);
      const auto end = static_cast<std::size_t>(compute_task_shared_read_offsets[task_id + 1]);
      std::sort(compute_task_shared_read_neighbors.begin() + begin,
                compute_task_shared_read_neighbors.begin() + end);
    }
  }

  // Creation and Initialization

  void set_total_compute_task_dependencies(int32_t num_deps) {
    compute_task_dependencies.resize(num_deps, 0);
  }

  void set_total_compute_task_dependents(int32_t num_deps) {
    compute_task_dependents.resize(num_deps, 0);
  }

  void set_total_compute_task_data_dependencies(int32_t num_data_deps) {
    compute_task_data_dependencies.resize(num_data_deps, 0);
  }

  void set_total_compute_task_data_dependents(int32_t num_data_deps) {
    compute_task_data_dependents.resize(num_data_deps, 0);
  }

  void set_total_data_task_dependencies(int32_t num_data_deps) {
    data_task_dependencies.resize(num_data_deps, 0);
  }

  void set_total_data_task_dependents(int32_t num_data_deps) {
    data_task_dependents.resize(num_data_deps, 0);
  }

  void set_total_reads(int32_t num_read) {
    compute_task_read.resize(num_read, 0);
    compute_task_recent_writers.resize(num_read, 0);
  }

  void set_total_writes(int32_t num_write) {
    compute_task_write.resize(num_write, 0);
  }

  void set_total_retires(int32_t num_retire) {
    compute_task_retire.resize(num_retire, 0);
  }

  void set_total_unique(int32_t num_unique) {
    compute_task_unique.resize(num_unique, 0);
  }

  // Optional grid metadata (for priority ordering)
  void set_grid_shape(int32_t h, int32_t w) {
    if (h > 0 && w > 0) {
      grid_h = h;
      grid_w = w;
    } else {
      grid_h = -1;
      grid_w = -1;
    }
  }

  [[nodiscard]] int32_t get_grid_h() const {
    return grid_h;
  }

  [[nodiscard]] int32_t get_grid_w() const {
    return grid_w;
  }

  [[nodiscard]] bool has_grid_shape() const {
    return grid_h > 0 && grid_w > 0;
  }

  void set_morton_priority_enabled(bool enabled) {
    morton_priority_enabled = enabled;
  }

  [[nodiscard]] bool get_morton_priority_enabled() const {
    return morton_priority_enabled;
  }

  void set_use_random_priority(bool enabled) {
    random_priority_enabled = enabled;
  }

  [[nodiscard]] bool use_random_priority() const {
    return random_priority_enabled;
  }

  void add_compute_task(taskid_t id, const std::string &name, const ComputeTaskDepInfo &dep_info,
                        const ComputeTaskDataInfo &data_info,
                        const ComputeTaskStaticInfo &compute_info) {
    compute_task_dep_info[id] = dep_info;
    compute_task_data_info[id] = data_info;
    compute_task_static_info[id] = compute_info;
    compute_task_names[id] = name;

    const auto idx = static_cast<std::size_t>(id);
    compute_task_dependents_offsets[idx] = dep_info.s_dependents;
    compute_task_dependents_offsets[idx + 1] = dep_info.e_dependents;
    compute_task_data_dependents_offsets[idx] = dep_info.s_data_dependents;
    compute_task_data_dependents_offsets[idx + 1] = dep_info.e_data_dependents;
  }

  void add_data_task(taskid_t id, const std::string &name, const DataTaskStaticInfo &static_info) {
    data_task_static_info[id] = static_info;
    data_task_names[id] = name;

    const auto idx = static_cast<std::size_t>(id);
    data_task_dependents_offsets[idx] = static_info.s_dependents;
    data_task_dependents_offsets[idx + 1] = static_info.e_dependents;
    data_task_data_id_cache[idx] = static_info.data_id;
    data_task_compute_task_cache[idx] = static_info.compute_task;
  }

  void add_compute_task_dependencies(taskid_t id, const std::vector<taskid_t> &dependencies) {
    assert(id < compute_task_dep_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_dep_info[id];
    assert(compute_task_dependencies.size() >= info.e_dependencies &&
           "Not enough space in compute_task_dependencies vector");
    std::copy(dependencies.begin(), dependencies.end(),
              compute_task_dependencies.begin() + info.s_dependencies);
  }

  void add_compute_task_dependents(taskid_t id, const std::vector<taskid_t> &dependents) {
    assert(id < compute_task_dep_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_dep_info[id];
    assert(compute_task_dependents.size() >= info.e_dependents &&
           "Not enough space in compute_task_dependents vector");
    std::copy(dependents.begin(), dependents.end(),
              compute_task_dependents.begin() + info.s_dependents);
  }

  void add_compute_task_data_dependencies(taskid_t id, const std::vector<taskid_t> &dependencies) {
    assert(id < compute_task_dep_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_dep_info[id];
    assert(compute_task_data_dependencies.size() >= info.e_data_dependencies &&
           "Not enough space in compute_task_data_dependencies vector");
    std::copy(dependencies.begin(), dependencies.end(),
              compute_task_data_dependencies.begin() + info.s_data_dependencies);
  }

  void add_compute_task_data_dependents(taskid_t id, const std::vector<taskid_t> &dependents) {
    assert(id < compute_task_dep_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_dep_info[id];
    assert(compute_task_data_dependents.size() >= info.e_data_dependents &&
           "Not enough space in compute_task_data_dependents vector");
    std::copy(dependents.begin(), dependents.end(),
              compute_task_data_dependents.begin() + info.s_data_dependents);
  }

  void add_data_task_dependencies(taskid_t id, const std::vector<taskid_t> &dependencies) {
    assert(id < data_task_static_info.size() && "Task ID is out of bounds");
    auto &info = data_task_static_info[id];
    assert(data_task_dependencies.size() >= info.e_dependencies &&
           "Not enough space in data_task_dependencies vector");
    std::copy(dependencies.begin(), dependencies.end(),
              data_task_dependencies.begin() + info.s_dependencies);
  }

  void add_data_task_dependents(taskid_t id, const std::vector<taskid_t> &dependents) {
    assert(id < data_task_static_info.size() && "Task ID is out of bounds");
    auto &info = data_task_static_info[id];
    assert(data_task_dependents.size() >= info.e_dependents &&
           "Not enough space in data_task_dependents vector");
    // copy dependents to corresponding location
    std::copy(dependents.begin(), dependents.end(),
              data_task_dependents.begin() + info.s_dependents);
  }

  void add_read(taskid_t id, const std::vector<dataid_t> &read) {
    assert(id < compute_task_data_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_data_info[id];
    assert(compute_task_read.size() >= info.e_read &&
           "Not enough space in compute_task_read vector");
    // copy read data to corresponding location
    std::copy(read.begin(), read.end(), compute_task_read.begin() + info.s_read);
  }

  void add_most_recent_writers(taskid_t id, const std::vector<taskid_t> &writers) {
    assert(id < compute_task_data_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_data_info[id];
    assert(compute_task_recent_writers.size() >= info.e_read &&
           "Not enough space in compute_task_recent_writers vector");
    std::copy(writers.begin(), writers.end(), compute_task_recent_writers.begin() + info.s_read);
  }

  void add_write(taskid_t id, const std::vector<dataid_t> &write) {
    assert(id < compute_task_data_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_data_info[id];
    assert(compute_task_write.size() >= info.e_write &&
           "Not enough space in compute_task_write vector");
    // copy write data to corresponding location
    std::copy(write.begin(), write.end(), compute_task_write.begin() + info.s_write);
  }

  void add_retire(taskid_t id, const std::vector<dataid_t> &retire) {
    assert(id < compute_task_data_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_data_info[id];
    assert(compute_task_retire.size() >= info.e_retire &&
           "Not enough space in compute_task_retire vector");
    // copy retire data to corresponding location
    std::copy(retire.begin(), retire.end(), compute_task_retire.begin() + info.s_retire);
  }

  void add_unique(taskid_t id, const std::vector<dataid_t> &unique) {
    assert(id < compute_task_data_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_data_info[id];
    assert(compute_task_unique.size() >= info.e_unique &&
           "Not enough space in compute_task_unique vector");
    // copy unique data to corresponding location
    std::copy(unique.begin(), unique.end(), compute_task_unique.begin() + info.s_unique);
  }

  void add_compute_variant(taskid_t id, DeviceType arch, mem_t mem, vcu_t vcu, timecount_t time) {
    assert(id < compute_task_variant_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_variant_info[id];
    uint8_t arch_type = static_cast<uint8_t>(arch);
    info.mask |= arch_type;
    const auto idx = __builtin_ctz(arch_type);
    assert(idx < info.variants.size() && "Architecture index out of bounds");
    info.variants[idx] = Variant(arch, vcu, mem, time);
    // std::cout << "[StaticGraph] Added variant for task " << id << ": "
    //           << "Arch=" << to_string(arch) << ", VCU=" << vcu << ", Mem=" << mem
    //           << ", Time=" << time << std::endl;
  }

  // Getters

  [[nodiscard]] int32_t get_n_compute_tasks() const {
    return static_cast<int32_t>(compute_task_dep_info.size());
  }

  [[nodiscard]] int32_t get_n_data_tasks() const {
    return static_cast<int32_t>(data_task_static_info.size());
  }

  [[nodiscard]] int32_t get_n_tasks() const {
    return get_n_compute_tasks() + get_n_data_tasks();
  }

  [[nodiscard]] bool empty() const {
    return (compute_task_dep_info.empty() && data_task_static_info.empty());
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_dependencies(taskid_t id) const {
    auto &info = compute_task_dep_info[id];
    return {compute_task_dependencies.data() + info.s_dependencies,
            compute_task_dependencies.data() + info.e_dependencies};
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_dependents(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(compute_task_dependents_offsets[idx]);
    const auto end = static_cast<std::size_t>(compute_task_dependents_offsets[idx + 1]);
    return std::span<const taskid_t>(compute_task_dependents).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const taskid_t> get_data_task_dependencies(taskid_t id) const {
    auto &info = data_task_static_info[id];
    return {data_task_dependencies.data() + info.s_dependencies,
            data_task_dependencies.data() + info.e_dependencies};
  }

  [[nodiscard]] std::span<const taskid_t> get_data_task_dependents(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(data_task_dependents_offsets[idx]);
    const auto end = static_cast<std::size_t>(data_task_dependents_offsets[idx + 1]);
    return std::span<const taskid_t>(data_task_dependents).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_data_dependencies(taskid_t id) const {
    auto &info = compute_task_dep_info[id];
    return {compute_task_data_dependencies.data() + info.s_data_dependencies,
            compute_task_data_dependencies.data() + info.e_data_dependencies};
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_data_dependents(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(compute_task_data_dependents_offsets[idx]);
    const auto end = static_cast<std::size_t>(compute_task_data_dependents_offsets[idx + 1]);
    return std::span<const taskid_t>(compute_task_data_dependents).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const dataid_t> get_read_usage_data_ids() const {
    return read_usage_data_ids;
  }

  [[nodiscard]] std::span<const taskid_t> get_tasks_reading_data(dataid_t data_id) const {
    auto it = read_usage_row_by_data_id.find(data_id);
    if (it == read_usage_row_by_data_id.end()) {
      return {};
    }
    const auto row = static_cast<std::size_t>(it->second);
    const auto begin = static_cast<std::size_t>(read_usage_offsets[row]);
    const auto end = static_cast<std::size_t>(read_usage_offsets[row + 1]);
    return std::span<const taskid_t>(read_usage_tasks).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const taskid_t>
  get_compute_task_shared_read_neighbors(taskid_t id) const {
    assert(id >= 0 && id < get_n_compute_tasks() && "Task ID is out of bounds");
    const auto begin = static_cast<std::size_t>(compute_task_shared_read_offsets[id]);
    const auto end = static_cast<std::size_t>(compute_task_shared_read_offsets[id + 1]);
    return std::span<const taskid_t>(compute_task_shared_read_neighbors)
        .subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const dataid_t> get_read(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_read.data() + info.s_read, compute_task_read.data() + info.e_read};
  }

  [[nodiscard]] std::span<const dataid_t> get_write(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_write.data() + info.s_write, compute_task_write.data() + info.e_write};
  }

  [[nodiscard]] std::span<const dataid_t> get_retire(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_retire.data() + info.s_retire, compute_task_retire.data() + info.e_retire};
  }

  [[nodiscard]] std::span<const dataid_t> get_unique(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_unique.data() + info.s_unique, compute_task_unique.data() + info.e_unique};
  }

  [[nodiscard]] bool has_read_data(taskid_t task_id, dataid_t data_id) const {
    if (membership_index_enabled) {
      return contains_membership_value(compute_task_read_membership, task_id, data_id);
    }
    const auto read = get_read(task_id);
    return std::find(read.begin(), read.end(), data_id) != read.end();
  }

  [[nodiscard]] bool has_write_data(taskid_t task_id, dataid_t data_id) const {
    if (membership_index_enabled) {
      return contains_membership_value(compute_task_write_membership, task_id, data_id);
    }
    const auto write = get_write(task_id);
    return std::find(write.begin(), write.end(), data_id) != write.end();
  }

  [[nodiscard]] bool has_unique_data(taskid_t task_id, dataid_t data_id) const {
    if (membership_index_enabled) {
      return contains_membership_value(compute_task_unique_membership, task_id, data_id);
    }
    const auto unique = get_unique(task_id);
    return std::find(unique.begin(), unique.end(), data_id) != unique.end();
  }

  [[nodiscard]] int32_t get_read_data_index(taskid_t task_id, dataid_t data_id) const {
    if (membership_index_enabled) {
      const auto it = read_index_by_task_data.find(task_data_key(task_id, data_id));
      return it == read_index_by_task_data.end() ? -1 : it->second;
    }
    const auto read = get_read(task_id);
    const auto it = std::find(read.begin(), read.end(), data_id);
    return it == read.end() ? -1 : static_cast<int32_t>(it - read.begin());
  }

  [[nodiscard]] TaskDataMembershipBitmapView get_read_membership(taskid_t task_id) const {
    return get_membership_view(compute_task_read_membership, task_id);
  }

  [[nodiscard]] TaskDataMembershipBitmapView get_write_membership(taskid_t task_id) const {
    return get_membership_view(compute_task_write_membership, task_id);
  }

  [[nodiscard]] TaskDataMembershipBitmapView get_unique_membership(taskid_t task_id) const {
    return get_membership_view(compute_task_unique_membership, task_id);
  }

  [[nodiscard]] bool task_data_membership_enabled() const {
    return membership_index_enabled;
  }

  [[nodiscard]] std::size_t get_task_data_membership_bytes() const {
    return membership_index_bytes;
  }

  [[nodiscard]] int32_t get_out_degree(taskid_t compute_task_id) const {
    return get_compute_task_dependencies(compute_task_id).size();
  }

  [[nodiscard]] int32_t get_in_degree(taskid_t compute_task_id) const {
    return get_compute_task_dependents(compute_task_id).size();
  }

  [[nodiscard]] const int32_t get_depth(taskid_t id) const {
    return compute_task_static_info[id].depth;
  }

  void add_depth(taskid_t id, int32_t depth) {
    compute_task_static_info[id].depth = depth;
  }

  [[nodiscard]] std::span<const taskid_t> get_most_recent_writers(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_recent_writers.data() + info.s_read,
            compute_task_recent_writers.data() + info.e_read};
  }

  [[nodiscard]] const VariantList &get_variants(taskid_t id) const {
    return compute_task_variant_info[id].variants;
  }

  [[nodiscard]] const Variant &get_variant(taskid_t id, DeviceType arch) const {
    const auto idx = __builtin_ctz(static_cast<uint8_t>(arch));
    assert(idx < compute_task_variant_info[id].variants.size() &&
           "Architecture index out of bounds for compute task variants");
    return compute_task_variant_info[id].variants[idx];
  }

  [[nodiscard]] const Resources &get_compute_task_resources(taskid_t id, DeviceType arch) const {
    auto &info = compute_task_variant_info[id];
    uint8_t arch_type = static_cast<uint8_t>(arch);
    // assert that mask flag is set for the given architecture
    assert((info.mask & arch_type) != 0 && "Architecture not supported for this compute task");
    const auto idx = __builtin_ctz(arch_type);
    assert(idx < info.variants.size() && "Architecture index out of bounds");
    const auto &variant = info.variants[idx];
    return variant.get_resources();
  }

  [[nodiscard]] const timecount_t get_mean_duration(taskid_t id, DeviceType arch) const {
    auto &info = compute_task_variant_info[id];
    uint8_t arch_type = static_cast<uint8_t>(arch);
    // assert that mask flag is set for the given architecture
    assert((info.mask & arch_type) != 0 && "Architecture not supported for this compute task");
    const auto idx = __builtin_ctz(arch_type);
    assert(idx < info.variants.size() && "Architecture index out of bounds");
    const auto &variant = info.variants[idx];
    return variant.get_mean_duration();
  }

  [[nodiscard]] uint8_t get_compute_task_variant_mask(taskid_t id) const {
    return compute_task_variant_info[id].mask;
  }

  [[nodiscard]] uint8_t get_supported_architecture_mask(taskid_t compute_task_id) const {
    return get_compute_task_variant_mask(compute_task_id);
  }

  [[nodiscard]] bool is_architecture_supported(taskid_t compute_task_id, DeviceType arch) const {
    auto &info = compute_task_variant_info[compute_task_id];
    uint8_t arch_type = static_cast<uint8_t>(arch);
    // assert that mask flag is set for the given architecture
    return (info.mask & arch_type) != 0;
  }

  [[nodiscard]] devicemask_t get_supported_devices_mask(taskid_t compute_task_id) const;

  [[nodiscard]] const std::string &get_compute_task_name(taskid_t id) const {
    return compute_task_names[id];
  }

  [[nodiscard]] const std::string &get_data_task_name(taskid_t id) const {
    return data_task_names[id];
  }

  [[nodiscard]] const dataid_t get_data_id(taskid_t id) const {
    return data_task_data_id_cache[static_cast<std::size_t>(id)];
  }

  [[nodiscard]] const taskid_t get_compute_task(taskid_t id) const {
    return data_task_compute_task_cache[static_cast<std::size_t>(id)];
  }

  // Getters for static task info

  [[nodiscard]] const ComputeTaskDepInfo &get_compute_task_dep_info(taskid_t id) const {
    return compute_task_dep_info[id];
  }

  [[nodiscard]] const ComputeTaskDataInfo &get_compute_task_data_info(taskid_t id) const {
    return compute_task_data_info[id];
  }

  [[nodiscard]] const DataTaskStaticInfo &get_data_task_static_info(taskid_t id) const {
    return data_task_static_info[id];
  }

  [[nodiscard]] const ComputeTaskStaticInfo &get_compute_task_static_info(taskid_t id) const {
    return compute_task_static_info[id];
  }
};

class RuntimeTaskInfo {
protected:
  std::vector<ComputeTaskRuntimeInfo> compute_task_runtime_info;
  std::vector<DataTaskRuntimeInfo> data_task_runtime_info;
  std::vector<EvictionTaskRuntimeInfo> eviction_task_runtime_info;

  std::vector<TaskTimeRecord> compute_task_time_records;
  std::vector<DataTaskTimeRecord> data_task_time_records;
  std::vector<DataTaskTimeRecord> eviction_task_time_records;

  std::vector<std::string> eviction_task_names;

public:
  RuntimeTaskInfo() = default;

  RuntimeTaskInfo(StaticTaskInfo &static_info) {
    int32_t num_compute_tasks = static_cast<int32_t>(static_info.get_n_compute_tasks());
    int32_t num_data_tasks = static_cast<int32_t>(static_info.get_n_data_tasks());
    compute_task_runtime_info.resize(num_compute_tasks);
    data_task_runtime_info.resize(num_data_tasks);
    compute_task_time_records.resize(num_compute_tasks);
    data_task_time_records.resize(num_data_tasks);

    for (int32_t i = 0; i < num_compute_tasks; ++i) {
      initialize_compute_runtime(i, static_info);
    }

    for (int32_t i = 0; i < num_data_tasks; ++i) {
      initialize_data_runtime(i, static_info);
    }

    // eviction_task_runtime_info.reserve(EXPECTED_EVICTION_TASKS);
    // eviction_task_time_records.reserve(EXPECTED_EVICTION_TASKS);
    // eviction_task_names.reserve(EXPECTED_EVICTION_TASKS);
  }

  RuntimeTaskInfo(const RuntimeTaskInfo &other) {

    {
      ZoneScopedN("Copy ComputeTaskRuntimeInfo");
      compute_task_runtime_info = other.compute_task_runtime_info;
    }

    {
      ZoneScopedN("Copy DataTaskRuntimeInfo");
      data_task_runtime_info = other.data_task_runtime_info;
    }

    {
      ZoneScopedN("Copy EvictionTaskRuntimeInfo");
      eviction_task_runtime_info = other.eviction_task_runtime_info;
    }

    {
      ZoneScopedN("Copy ComputeTaskTimeRecords");
      compute_task_time_records = other.compute_task_time_records;
    }

    {
      ZoneScopedN("Copy DataTaskTimeRecords");
      data_task_time_records = other.data_task_time_records;
    }

    {
      ZoneScopedN("Copy EvictionTaskTimeRecords");
      eviction_task_time_records = other.eviction_task_time_records;
    }

    {
      ZoneScopedN("Copy EvictionTaskNames");
      eviction_task_names = other.eviction_task_names;
    }
  }

  // Creation and Initialization

  void initialize_compute_runtime(int32_t compute_task_id, const StaticTaskInfo &static_info) {
    set_compute_task_state(compute_task_id, TaskState::SPAWNED);

    auto &dep_info = static_info.get_compute_task_dep_info(compute_task_id);
    auto n_dependencies = dep_info.e_dependencies - dep_info.s_dependencies;
    auto n_data_dependencies = dep_info.e_data_dependencies - dep_info.s_data_dependencies;
    set_compute_task_unmapped(compute_task_id, n_dependencies);
    set_compute_task_unreserved(compute_task_id, n_dependencies);
    set_compute_task_incomplete(compute_task_id, n_dependencies + n_data_dependencies);
    ;
  }

  void initialize_data_runtime(int32_t data_task_id, const StaticTaskInfo &static_info) {
    set_data_task_state(data_task_id, TaskState::SPAWNED);

    auto &info = static_info.get_data_task_static_info(data_task_id);
    auto n_dependencies = info.e_dependencies - info.s_dependencies;
    set_data_task_incomplete(data_task_id, n_dependencies);
  }

  int32_t add_eviction_task(int32_t compute_task_id, int32_t data_id,
                            int32_t evicting_on_device_id) {
    taskid_t id = static_cast<taskid_t>(eviction_task_runtime_info.size());
    eviction_task_runtime_info.emplace_back();
    eviction_task_time_records.emplace_back();
    std::string name = "EvictionTask_" + std::to_string(compute_task_id) + "_" +
                       std::to_string(data_id) + "_" + std::to_string(evicting_on_device_id);
    eviction_task_names.push_back(name);

    set_eviction_task_state(id, TaskState::RESERVED);
    set_eviction_task_evicting_on(id, evicting_on_device_id);
    set_eviction_task_data_id(id, data_id);
    set_eviction_task_compute_task(id, compute_task_id);
    return id;
  }

  // Getters

  [[nodiscard]] int32_t get_n_compute_tasks() const {
    return static_cast<int32_t>(compute_task_runtime_info.size());
  }

  [[nodiscard]] int32_t get_n_data_tasks() const {
    return static_cast<int32_t>(data_task_runtime_info.size());
  }

  [[nodiscard]] int32_t get_n_eviction_tasks() const {
    return static_cast<int32_t>(eviction_task_runtime_info.size());
  }

  [[nodiscard]] int32_t get_n_tasks() const {
    return get_n_compute_tasks() + get_n_data_tasks() + get_n_eviction_tasks();
  }

  [[nodiscard]] bool empty() const {
    return (compute_task_runtime_info.empty() && data_task_runtime_info.empty() &&
            eviction_task_runtime_info.empty());
  }

  [[nodiscard]] const std::string get_eviction_task_name(taskid_t id) const {
    return eviction_task_names[id];
  }

  [[nodiscard]] const int16_t get_compute_task_unmapped(taskid_t id) const {
    return compute_task_runtime_info[id].unmapped;
  }

  [[nodiscard]] const int16_t get_compute_task_unreserved(taskid_t id) const {
    return compute_task_runtime_info[id].unreserved;
  }

  [[nodiscard]] const int16_t get_compute_task_incomplete(taskid_t id) const {
    return compute_task_runtime_info[id].incomplete;
  }

  [[nodiscard]] const int32_t get_compute_task_mapped_device(taskid_t id) const {
    return compute_task_runtime_info[id].mapped_device;
  }

  [[nodiscard]] const int32_t get_compute_task_reserve_priority(taskid_t id) const {
    return compute_task_runtime_info[id].reserve_priority;
  }

  [[nodiscard]] const int32_t get_compute_task_launch_priority(taskid_t id) const {
    return compute_task_runtime_info[id].launch_priority;
  }

  [[nodiscard]] const TaskState get_compute_task_state(taskid_t id) const {
    return static_cast<TaskState>(compute_task_runtime_info[id].state);
  }

  [[nodiscard]] const uint8_t get_compute_task_flags(taskid_t id) const {
    return compute_task_runtime_info[id].flags;
  }

  [[nodiscard]] const TaskState get_data_task_state(taskid_t id) const {
    return static_cast<TaskState>(data_task_runtime_info[id].state);
  }

  [[nodiscard]] const uint8_t get_data_task_flags(taskid_t id) const {
    return data_task_runtime_info[id].flags;
  }

  [[nodiscard]] const int32_t get_data_task_source_device(taskid_t id) const {
    return data_task_runtime_info[id].source_device;
  }

  [[nodiscard]] const int32_t get_data_task_mapped_device(taskid_t id) const {
    return data_task_runtime_info[id].mapped_device;
  }

  [[nodiscard]] const int32_t get_data_task_launch_priority(taskid_t id) const {
    return data_task_runtime_info[id].launch_priority;
  }

  [[nodiscard]] const TaskState get_eviction_task_state(taskid_t id) const {
    return static_cast<TaskState>(eviction_task_runtime_info[id].state);
  }

  [[nodiscard]] const int32_t get_eviction_task_evicting_on(taskid_t id) const {
    return eviction_task_runtime_info[id].evicting_on;
  }

  [[nodiscard]] const int32_t get_eviction_task_data_id(taskid_t id) const {
    return eviction_task_runtime_info[id].data_id;
  }

  [[nodiscard]] const int32_t get_eviction_task_source_device(taskid_t id) const {
    return eviction_task_runtime_info[id].source_device;
  }

  [[nodiscard]] const timecount_t get_compute_task_mapped_time(taskid_t id) const {
    return compute_task_time_records[id].mapped_time;
  }

  [[nodiscard]] const timecount_t get_compute_task_reserved_time(taskid_t id) const {
    return compute_task_time_records[id].reserved_time;
  }

  [[nodiscard]] const timecount_t get_compute_task_launched_time(taskid_t id) const {
    return compute_task_time_records[id].launched_time;
  }

  [[nodiscard]] const timecount_t get_compute_task_completed_time(taskid_t id) const {
    return compute_task_time_records[id].completed_time;
  }

  [[nodiscard]] const timecount_t get_data_task_launched_time(taskid_t id) const {
    return data_task_time_records[id].launched_time;
  }

  [[nodiscard]] const timecount_t get_data_task_completed_time(taskid_t id) const {
    return data_task_time_records[id].completed_time;
  }

  [[nodiscard]] const timecount_t get_eviction_task_launched_time(taskid_t id) const {
    return eviction_task_time_records[id].launched_time;
  }

  [[nodiscard]] const timecount_t get_eviction_task_completed_time(taskid_t id) const {
    return eviction_task_time_records[id].completed_time;
  }

  [[nodiscard]] TaskState get_compute_task_state_at_time(taskid_t id, timecount_t query) const {
    if (query < compute_task_time_records[id].mapped_time) {
      return TaskState::SPAWNED;
    } else if (query < compute_task_time_records[id].reserved_time) {
      return TaskState::MAPPED;
    } else if (query < compute_task_time_records[id].launched_time) {
      return TaskState::RESERVED;
    } else if (query < compute_task_time_records[id].completed_time) {
      return TaskState::LAUNCHED;
    } else {
      return TaskState::COMPLETED;
    }
  }

  TaskState get_data_task_state_at_time(taskid_t id, timecount_t query) const {
    if (query < data_task_time_records[id].launched_time) {
      return TaskState::RESERVED;
    } else if (query < data_task_time_records[id].completed_time) {
      return TaskState::LAUNCHED;
    } else {
      return TaskState::COMPLETED;
    }
  }

  [[nodiscard]] TaskState get_eviction_task_state_at_time(taskid_t id, timecount_t query) const {
    if (query < eviction_task_time_records[id].launched_time) {
      return TaskState::MAPPED;
    } else if (query < eviction_task_time_records[id].completed_time) {
      return TaskState::RESERVED;
    } else {
      return TaskState::COMPLETED;
    }
  }

  [[nodiscard]] const bool is_data_task_virtual(taskid_t id) const {
    // Virtual tasks have first bit of flags set to 1
    return (data_task_runtime_info[id].flags & 0x01) != 0;
  }

  [[nodiscard]] const bool is_eviction_task_virtual(taskid_t id) const {
    // Virtual tasks have first bit of flags set to 1
    return (eviction_task_runtime_info[id].flags & 0x01) != 0;
  }

  // TODO(wlr): Change status to flag for bit-wise comparison

  bool is_compute_mappable(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.unmapped == 0 && info.state == static_cast<uint8_t>(TaskState::SPAWNED);
  }

  bool is_compute_mapped(taskid_t compute_task_id) const {
    auto &info = compute_task_runtime_info[compute_task_id];
    return info.state >= static_cast<uint8_t>(TaskState::MAPPED);
  }

  bool is_compute_reservable(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.unreserved == 0 && info.state == static_cast<uint8_t>(TaskState::MAPPED);
  }

  bool is_compute_reserved(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.state >= static_cast<uint8_t>(TaskState::RESERVED);
  }

  bool is_compute_launchable(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.incomplete == 0 && info.state == static_cast<uint8_t>(TaskState::RESERVED);
  }

  bool is_compute_launched(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.state >= static_cast<uint8_t>(TaskState::LAUNCHED);
  }

  bool is_compute_completed(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.state >= static_cast<uint8_t>(TaskState::COMPLETED);
  }

  bool is_data_launchable(taskid_t id) const {
    auto &info = data_task_runtime_info[id];
    return info.incomplete == 0 && info.state == static_cast<uint8_t>(TaskState::RESERVED);
  }

  bool is_data_completed(taskid_t id) const {
    auto &info = data_task_runtime_info[id];
    return info.state >= static_cast<uint8_t>(TaskState::COMPLETED);
  }

  bool is_eviction_launchable(taskid_t id) const {
    auto &info = eviction_task_runtime_info[id];
    return info.state >= static_cast<uint8_t>(TaskState::RESERVED);
  }

  bool is_eviction_completed(taskid_t id) const {
    auto &info = eviction_task_runtime_info[id];
    return info.state >= static_cast<uint8_t>(TaskState::COMPLETED);
  }

  TaskStatus get_compute_task_status(taskid_t id) const {
    if (is_compute_mappable(id)) {
      return TaskStatus::MAPPABLE;
    } else if (is_compute_reservable(id)) {
      return TaskStatus::RESERVABLE;
    } else if (is_compute_launchable(id)) {
      return TaskStatus::LAUNCHABLE;
    } else {
      return TaskStatus::NONE;
    }
  }

  TaskStatus get_data_task_status(taskid_t id) const {
    if (is_data_launchable(id)) {
      return TaskStatus::LAUNCHABLE;
    } else if (is_data_completed(id)) {
      return TaskStatus::NONE;
    } else {
      return TaskStatus::NONE;
    }
  }

  TaskStatus get_eviction_task_status(taskid_t id) const {
    if (is_eviction_launchable(id)) {
      return TaskStatus::LAUNCHABLE;
    } else if (is_eviction_completed(id)) {
      return TaskStatus::NONE;
    } else {
      return TaskStatus::NONE;
    }
  }

  // Non const grab fields

  [[nodiscard]] ComputeTaskRuntimeInfo &get_compute_task_runtime_info(taskid_t id) {
    return compute_task_runtime_info[id];
  }
  [[nodiscard]] DataTaskRuntimeInfo &get_data_task_runtime_info(taskid_t id) {
    return data_task_runtime_info[id];
  }
  [[nodiscard]] EvictionTaskRuntimeInfo &get_eviction_task_runtime_info(taskid_t id) {
    return eviction_task_runtime_info[id];
  }

  [[nodiscard]] TaskTimeRecord &get_compute_task_time_record(taskid_t id) {
    return compute_task_time_records[id];
  }
  [[nodiscard]] DataTaskTimeRecord &get_data_task_time_record(taskid_t id) {
    return data_task_time_records[id];
  }
  [[nodiscard]] DataTaskTimeRecord &get_eviction_task_time_record(taskid_t id) {
    return eviction_task_time_records[id];
  }

  // Setters

  void set_compute_task_state(taskid_t id, TaskState state) {
    compute_task_runtime_info[id].state = static_cast<uint8_t>(state);
  }

  void set_data_task_state(taskid_t id, TaskState state) {
    data_task_runtime_info[id].state = static_cast<uint8_t>(state);
  }

  void set_eviction_task_state(taskid_t id, TaskState state) {
    eviction_task_runtime_info[id].state = static_cast<uint8_t>(state);
  }

  void set_compute_task_unmapped(taskid_t id, int16_t unmapped) {
    compute_task_runtime_info[id].unmapped = unmapped;
  }
  void set_compute_task_unreserved(taskid_t id, int16_t unreserved) {
    compute_task_runtime_info[id].unreserved = unreserved;
  }
  void set_compute_task_incomplete(taskid_t id, int16_t incomplete) {
    compute_task_runtime_info[id].incomplete = incomplete;
  }
  void set_compute_task_mapped_device(taskid_t id, int32_t mapped_device) {
    compute_task_runtime_info[id].mapped_device = mapped_device;
  }
  void set_compute_task_reserve_priority(taskid_t id, int32_t reserve_priority) {
    compute_task_runtime_info[id].reserve_priority = reserve_priority;
  }
  void set_compute_task_launch_priority(taskid_t id, int32_t launch_priority) {
    compute_task_runtime_info[id].launch_priority = launch_priority;
  }

  void set_compute_task_state(taskid_t id, uint8_t state) {
    compute_task_runtime_info[id].state = state;
  }
  void set_compute_task_flags(taskid_t id, uint8_t flags) {
    compute_task_runtime_info[id].flags = flags;
  }
  void set_data_task_state(taskid_t id, uint8_t state) {
    data_task_runtime_info[id].state = state;
  }
  void set_data_task_virtual(taskid_t id, bool virtual_task) {
    data_task_runtime_info[id].flags = virtual_task ? (data_task_runtime_info[id].flags | 0x01)
                                                    : (data_task_runtime_info[id].flags & ~0x01);
  }

  void set_data_task_incomplete(taskid_t id, int16_t incomplete) {
    data_task_runtime_info[id].incomplete = incomplete;
  }

  void set_data_task_source_device(taskid_t id, int32_t source_device) {
    data_task_runtime_info[id].source_device = source_device;
  }
  void set_data_task_mapped_device(taskid_t id, int32_t mapped_device) {
    data_task_runtime_info[id].mapped_device = mapped_device;
  }
  void set_data_task_launch_priority(taskid_t id, int32_t launch_priority) {
    data_task_runtime_info[id].launch_priority = launch_priority;
  }
  void set_eviction_task_state(taskid_t id, uint8_t state) {
    eviction_task_runtime_info[id].state = state;
  }
  void set_eviction_task_virtual(taskid_t id, bool virtual_task) {
    eviction_task_runtime_info[id].flags = virtual_task
                                               ? (eviction_task_runtime_info[id].flags | 0x01)
                                               : (eviction_task_runtime_info[id].flags & ~0x01);
  }
  void set_eviction_task_evicting_on(taskid_t id, int32_t evicting_on) {
    eviction_task_runtime_info[id].evicting_on = evicting_on;
  }

  void set_eviction_task_compute_task(taskid_t id, int32_t compute_task_id) {
    eviction_task_runtime_info[id].compute_task = compute_task_id;
  }

  void set_eviction_task_source_device(taskid_t id, int32_t source_device) {
    eviction_task_runtime_info[id].source_device = source_device;
  }

  void set_eviction_task_data_id(taskid_t id, int32_t data_id) {
    eviction_task_runtime_info[id].data_id = data_id;
  }

  void record_mapped(taskid_t id, timecount_t mapped_time) {
    compute_task_time_records[id].mapped_time = mapped_time;
  }
  void record_reserved(taskid_t id, timecount_t reserved_time) {
    compute_task_time_records[id].reserved_time = reserved_time;
  }
  void record_launched(taskid_t id, timecount_t launched_time) {
    compute_task_time_records[id].launched_time = launched_time;
  }
  void record_completed(taskid_t id, timecount_t completed_time) {
    compute_task_time_records[id].completed_time = completed_time;
  }

  timecount_t get_compute_task_duration(taskid_t id) const {
    return compute_task_time_records[id].completed_time -
           compute_task_time_records[id].launched_time;
  }

  timecount_t get_data_task_duration(taskid_t id) const {
    return data_task_time_records[id].completed_time - data_task_time_records[id].launched_time;
  }

  timecount_t get_eviction_task_duration(taskid_t id) const {
    return eviction_task_time_records[id].completed_time -
           eviction_task_time_records[id].launched_time;
  }

  void record_data_launched(taskid_t id, timecount_t launched_time) {
    data_task_time_records[id].launched_time = launched_time;
  }
  void record_data_completed(taskid_t id, timecount_t completed_time) {
    data_task_time_records[id].completed_time = completed_time;
  }

  void record_eviction_launched(taskid_t id, timecount_t launched_time) {
    eviction_task_time_records[id].launched_time = launched_time;
  }

  void record_eviction_completed(taskid_t id, timecount_t completed_time) {
    eviction_task_time_records[id].completed_time = completed_time;
  }

  // Task State modifiers

  bool decrement_compute_task_unmapped(taskid_t id) {
    auto &info = compute_task_runtime_info[id];
    info.unmapped--;
    assert(info.unmapped >= 0 && "Unmapped count cannot be negative");
    SPDLOG_DEBUG("decrement_compute_task_unmapped: id: {}, unmapped: {}, state: {}", id,
                 info.unmapped, info.state);
    return (info.unmapped == 0) && (info.state >= static_cast<uint8_t>(TaskState::SPAWNED));
  }

  bool decrement_compute_task_unreserved(taskid_t id) {
    auto &info = compute_task_runtime_info[id];
    info.unreserved--;
    SPDLOG_DEBUG("decrement_compute_task_unreserved: id: {}, unreserved: {}, state: {}", id,
                 info.unreserved, info.state);
    assert(info.unreserved >= 0 && "Unreserved count cannot be negative");
    return (info.unreserved == 0) && (info.state >= static_cast<uint8_t>(TaskState::MAPPED));
  }

  bool decrement_compute_task_incomplete(taskid_t id) {
    auto &info = compute_task_runtime_info[id];
    info.incomplete--;
    SPDLOG_DEBUG("decrement_compute_task_incomplete: id: {}, incomplete: {}, state: {}", id,
                 info.incomplete, info.state);
    assert(info.incomplete >= 0 && "Incomplete count cannot be negative");
    return (info.incomplete == 0) && (info.state >= static_cast<uint8_t>(TaskState::RESERVED));
  }

  bool decrement_data_task_incomplete(taskid_t id) {
    auto &info = data_task_runtime_info[id];
    info.incomplete--;
    SPDLOG_DEBUG("decrement_data_task_incomplete: id: {}, incomplete: {}, state: {}", id,
                 info.incomplete, info.state);
    assert(info.incomplete >= 0 && "Incomplete count cannot be negative");
    return (info.incomplete == 0) && (info.state >= static_cast<uint8_t>(TaskState::RESERVED));
  }

  taskid_t compute_notify_mapped(taskid_t compute_task_id, devid_t mapped_device,
                                 int32_t reserve_priority, int32_t launch_priority,
                                 timecount_t time, const StaticTaskInfo &static_info,
                                 TaskIDList &compute_task_buffer) {
    auto &my_info = compute_task_runtime_info[compute_task_id];
    auto &my_time_record = compute_task_time_records[compute_task_id];
    my_info.mapped_device = mapped_device;
    my_info.reserve_priority = reserve_priority;
    my_info.launch_priority = launch_priority;
    my_info.state = static_cast<uint8_t>(TaskState::MAPPED);
    my_time_record.mapped_time = time;
    taskid_t write_idx = 0;

    auto my_dependents = static_info.get_compute_task_dependents(compute_task_id);
    compute_task_buffer.resize(my_dependents.size());

    for (const auto &dependent_id : my_dependents) {
      bool is_mappable = decrement_compute_task_unmapped(dependent_id);
      compute_task_buffer[write_idx] = dependent_id;
      write_idx += is_mappable ? 1 : 0;
      SPDLOG_DEBUG("compute_notify_mapped: dependent_id: {}, is_mappable: {}, write_idx: {}",
                   dependent_id, is_mappable, write_idx);
    }
    compute_task_buffer.resize(write_idx);
    return write_idx;
  }

  taskid_t compute_notify_reserved(taskid_t compute_task_id, devid_t mapped_device,
                                   timecount_t time, const StaticTaskInfo &static_info,
                                   TaskIDList &compute_task_buffer) {
    auto &my_info = compute_task_runtime_info[compute_task_id];
    auto &my_time_record = compute_task_time_records[compute_task_id];
    my_info.mapped_device = mapped_device;
    my_info.state = static_cast<uint8_t>(TaskState::RESERVED);
    my_time_record.reserved_time = time;

    taskid_t write_idx = 0;
    auto my_dependents = static_info.get_compute_task_dependents(compute_task_id);
    compute_task_buffer.resize(my_dependents.size());

    for (const auto &dependent_id : my_dependents) {
      bool is_reservable = decrement_compute_task_unreserved(dependent_id);
      compute_task_buffer[write_idx] = dependent_id;
      write_idx += is_reservable ? 1 : 0;
      SPDLOG_DEBUG("compute_notify_reserved: dependent_id: {}, is_reservable: {}, write_idx: {}",
                   dependent_id, is_reservable, write_idx);
    }
    compute_task_buffer.resize(write_idx);
    return write_idx;
  }

  void compute_notify_launched(taskid_t compute_task_id, timecount_t time,
                               const StaticTaskInfo &static_info) {
    auto &my_info = compute_task_runtime_info[compute_task_id];
    auto &my_time_record = compute_task_time_records[compute_task_id];
    my_info.state = static_cast<int8_t>(TaskState::LAUNCHED);
    my_time_record.launched_time = time;
  }

  taskid_t compute_notify_completed(taskid_t compute_task_id, timecount_t time,
                                    const StaticTaskInfo &static_info,
                                    TaskIDList &compute_task_buffer) {
    auto &my_info = compute_task_runtime_info[compute_task_id];
    auto &my_time_record = compute_task_time_records[compute_task_id];
    my_info.state = static_cast<uint8_t>(TaskState::COMPLETED);
    my_time_record.completed_time = time;
    const auto my_dependents = static_info.get_compute_task_dependents(compute_task_id);
    const auto dependent_count = static_cast<taskid_t>(my_dependents.size());
    if (dependent_count == 0) {
      if (!compute_task_buffer.empty()) {
        compute_task_buffer.clear();
      }
      return 0;
    }
    if (dependent_count == 1) {
      const auto dependent_id = my_dependents.front();
      const bool is_launchable = decrement_compute_task_incomplete(dependent_id);
      if (is_launchable) {
        compute_task_buffer.resize(1);
        compute_task_buffer[0] = dependent_id;
        return 1;
      }
      if (!compute_task_buffer.empty()) {
        compute_task_buffer.clear();
      }
      return 0;
    }
    taskid_t write_idx = 0;
    compute_task_buffer.resize(my_dependents.size());

    for (const auto &dependent_id : my_dependents) {
      const bool is_launchable = decrement_compute_task_incomplete(dependent_id);
      if (is_launchable) {
        compute_task_buffer[write_idx] = dependent_id;
        write_idx += 1;
      }
      SPDLOG_DEBUG("compute_notify_completed: dependent_id: {}, is_launchable: {}, write_idx: {}",
                   dependent_id, is_launchable, write_idx);
    }
    if (write_idx != dependent_count) {
      compute_task_buffer.resize(write_idx);
    }

    return write_idx;
  }

  taskid_t compute_notify_data_completed(taskid_t compute_task_id, timecount_t time,
                                         const StaticTaskInfo &static_info,
                                         TaskIDList &data_task_buffer) {
    auto &my_info = compute_task_runtime_info[compute_task_id];

    // state and time assumed to be updated by prior call to notify_completed
    const auto my_data_dependents = static_info.get_compute_task_data_dependents(compute_task_id);
    const auto dependent_count = static_cast<taskid_t>(my_data_dependents.size());
    if (dependent_count == 0) {
      if (!data_task_buffer.empty()) {
        data_task_buffer.clear();
      }
      return 0;
    }
    if (dependent_count == 1) {
      const auto dependent_id = my_data_dependents.front();
      const bool is_launchable = decrement_data_task_incomplete(dependent_id);
      if (is_launchable) {
        data_task_buffer.resize(1);
        data_task_buffer[0] = dependent_id;
        return 1;
      }
      if (!data_task_buffer.empty()) {
        data_task_buffer.clear();
      }
      return 0;
    }
    taskid_t write_idx = 0;
    data_task_buffer.resize(my_data_dependents.size());

    for (const auto &dependent_id : my_data_dependents) {
      const bool is_launchable = decrement_data_task_incomplete(dependent_id);
      if (is_launchable) {
        data_task_buffer[write_idx] = dependent_id;
        write_idx += 1;
      }
      SPDLOG_DEBUG("compute_notify_data_completed: dependent_id: {}, is_launchable: {}, "
                   "write_idx: {}",
                   dependent_id, is_launchable, write_idx);
    }
    if (write_idx != dependent_count) {
      data_task_buffer.resize(write_idx);
    }

    return write_idx;
  }

  void data_notify_reserved(taskid_t data_task_id, devid_t mapped_device, timecount_t time,
                            const StaticTaskInfo &static_info) {
    auto &my_info = data_task_runtime_info[data_task_id];
    auto &my_time_record = data_task_time_records[data_task_id];
    my_info.mapped_device = mapped_device;
    my_info.state = static_cast<uint8_t>(TaskState::RESERVED);
  }

  void data_notify_launched(taskid_t data_task_id, devid_t source_device, timecount_t time,
                            const StaticTaskInfo &static_info) {
    auto &my_info = data_task_runtime_info[data_task_id];
    auto &my_time_record = data_task_time_records[data_task_id];
    my_info.state = static_cast<uint8_t>(TaskState::LAUNCHED);
    my_info.source_device = source_device;
    my_time_record.launched_time = time;
  }

  taskid_t data_notify_completed(taskid_t data_task_id, timecount_t time,
                                 const StaticTaskInfo &static_info,
                                 TaskIDList &compute_task_buffer) {
    auto &my_info = data_task_runtime_info[data_task_id];
    auto &my_time_record = data_task_time_records[data_task_id];

    my_info.state = static_cast<uint8_t>(TaskState::COMPLETED);
    my_time_record.completed_time = time;
    taskid_t write_idx = 0;

    auto my_dependents = static_info.get_data_task_dependents(data_task_id);
    if (my_dependents.empty()) {
      compute_task_buffer.clear();
      return 0;
    }

    if (my_dependents.size() == 1) {
      const auto dependent_id = my_dependents.front();
      const bool is_launchable = decrement_compute_task_incomplete(dependent_id);
      if (is_launchable) {
        compute_task_buffer.resize(1);
        compute_task_buffer[0] = dependent_id;
        return 1;
      }
      compute_task_buffer.clear();
      return 0;
    }

    compute_task_buffer.resize(my_dependents.size());

    for (const auto &dependent_id : my_dependents) {
      const bool is_launchable = decrement_compute_task_incomplete(dependent_id);
      if (is_launchable) {
        compute_task_buffer[write_idx] = dependent_id;
        write_idx += 1;
      }
      SPDLOG_DEBUG("data_notify_completed: dependent_id: {}, is_launchable: {}, write_idx: {}",
                   dependent_id, is_launchable, write_idx);
    }
    if (write_idx != static_cast<taskid_t>(compute_task_buffer.size())) {
      compute_task_buffer.resize(write_idx);
    }

    return write_idx;
  }

  void eviction_notify_reserved(taskid_t eviction_task_id, timecount_t time,
                                const StaticTaskInfo &static_info) {
    auto &my_time_record = eviction_task_time_records[eviction_task_id];
    auto &my_info = eviction_task_runtime_info[eviction_task_id];
    my_info.state = static_cast<uint8_t>(TaskState::RESERVED);
  }

  void eviction_notify_launched(taskid_t eviction_task_id, devid_t source_device_id,
                                timecount_t time, const StaticTaskInfo &static_info) {
    auto &my_time_record = eviction_task_time_records[eviction_task_id];
    auto &my_info = eviction_task_runtime_info[eviction_task_id];
    my_info.source_device = source_device_id;
    my_info.state = static_cast<uint8_t>(TaskState::LAUNCHED);
    my_time_record.launched_time = time;
  }

  void eviction_notify_completed(taskid_t eviction_task_id, timecount_t time) {
    auto &my_time_record = eviction_task_time_records[eviction_task_id];
    auto &my_info = eviction_task_runtime_info[eviction_task_id];
    my_info.state = static_cast<uint8_t>(TaskState::COMPLETED);
    my_time_record.completed_time = time;
  }
};
