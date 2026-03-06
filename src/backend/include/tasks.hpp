#pragma once
#include "queues.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <array>
#include <cassert>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <ostream>
#include <set>
#include <span>
#include <stack>
#include <stdexcept>
#include <string>
#include <tracy/Tracy.hpp>
#include <type_traits>
#include <unordered_map>
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

template <typename T>
inline std::vector<T> as_sorted_vector(const ankerl::unordered_dense::set<T> &set) {
  auto vec = as_vector(set);
  std::sort(vec.begin(), vec.end());
  return vec;
}

template <typename T> struct CsrData {
  std::vector<int32_t> offsets;
  std::vector<T> elements;

  void resize_rows(int32_t num_rows) {
    offsets.resize(static_cast<std::size_t>(num_rows) + 1, 0);
  }

  [[nodiscard]] std::span<const T> operator[](int32_t row) const {
    T4F_INVARIANT(row >= 0);
    const auto idx = static_cast<std::size_t>(row);
    T4F_INVARIANT(idx + 1 < offsets.size());
    const auto start = static_cast<std::size_t>(offsets[idx]);
    const auto len = static_cast<std::size_t>(offsets[idx + 1] - offsets[idx]);
    return std::span<const T>(elements).subspan(start, len);
  }

  [[nodiscard]] int32_t row_size(int32_t row) const {
    T4F_INVARIANT(row >= 0);
    const auto idx = static_cast<std::size_t>(row);
    T4F_INVARIANT(idx + 1 < offsets.size());
    return offsets[idx + 1] - offsets[idx];
  }
};

template <typename T> struct CsrView {
  std::span<const int32_t> offsets;
  std::span<const T> elements;

  [[nodiscard]] std::span<const T> operator[](int32_t row) const {
    T4F_INVARIANT(row >= 0);
    const auto idx = static_cast<std::size_t>(row);
    T4F_INVARIANT(idx + 1 < offsets.size());
    const auto start = static_cast<std::size_t>(offsets[idx]);
    const auto len = static_cast<std::size_t>(offsets[idx + 1] - offsets[idx]);
    return elements.subspan(start, len);
  }
};

template <class DecReadyFn>
static inline taskid_t collect_ready(std::span<const taskid_t> neighbors,
                                     DecReadyFn&& dec_ready,
                                     TaskIDList& out)
{
  const auto n = neighbors.size();
  if (n == 0) {
    out.clear();
    return 0;
  }

  out.resize(n);

  taskid_t w = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const taskid_t tid = neighbors[i];
    const bool ready = dec_ready(tid);
    out[static_cast<std::size_t>(w)] = tid;
    w += static_cast<taskid_t>(ready);
  }

  out.resize(static_cast<std::size_t>(w));
  return w;
}

template <typename Int> static inline void append_decimal(std::string &dst, Int value) {
  static_assert(std::is_integral_v<Int>);
  constexpr std::size_t kMaxDigits = static_cast<std::size_t>(std::numeric_limits<Int>::digits10) + 3;
  char buf[kMaxDigits];
  const auto [ptr, ec] = std::to_chars(buf, buf + kMaxDigits, value);
  T4F_INVARIANT(ec == std::errc{});
  dst.append(buf, ptr);
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

  ankerl::unordered_dense::set<dataid_t> read;
  ankerl::unordered_dense::set<dataid_t> write;
  ankerl::unordered_dense::set<dataid_t> retire;

  std::vector<dataid_t> unique;

  std::vector<dataid_t> sorted_read_cache;
  std::vector<dataid_t> sorted_write_cache;
  std::vector<dataid_t> sorted_retire_cache;
  std::vector<taskid_t> sorted_recent_writer_cache;
  std::vector<uint32_t> sorted_read_gen_cache;
  std::vector<uint32_t> sorted_write_gen_cache;

  std::vector<taskid_t> sorted_dependencies_cache;
  std::vector<taskid_t> sorted_dependents_cache;
  std::vector<taskid_t> sorted_data_dependencies_cache;
  std::vector<taskid_t> sorted_data_dependents_cache;

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

  std::vector<taskid_t> sorted_dependencies_cache;
  std::vector<taskid_t> sorted_dependents_cache;
};

class Graph {
public:
  struct FinalizeOptions {
    bool ensure_dependencies{false};
    bool create_data_tasks{true};
    bool detect_cycles{true};
  };

  std::vector<Task> tasks;
  std::vector<DataTask> data_tasks;
  std::vector<taskid_t> sorted;
  std::vector<taskid_t> initial_tasks;
  bool finalized = false;

  Graph() = default;

  // Dense vector indexed by data_id; value -1 means no writer yet.
  // Sized to max_data_id+1 during finalize().
  std::vector<taskid_t> writers;
  dataid_t max_data_id{-1}; // computed once in finalize() before any data processing

  int32_t total_compute_dependencies_cached{0};
  int32_t total_compute_dependents_cached{0};
  int32_t total_compute_data_dependencies_cached{0};
  int32_t total_compute_data_dependents_cached{0};
  int32_t total_reads_cached{0};
  int32_t total_writes_cached{0};
  int32_t total_retire_cached{0};
  int32_t total_unique_cached{0};
  int32_t total_data_task_dependencies_cached{0};
  int32_t total_data_task_dependents_cached{0};

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
    T4F_INVARIANT(task_id < tasks.size() && "Task ID is out of bounds");
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
    T4F_INVARIANT(task_id < tasks.size() && "Task ID is out of bounds");
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
    T4F_INVARIANT(task_id < tasks.size() && "Task ID is out of bounds");
    auto &task = tasks[task_id];
    for (const auto &data_id : data_ids) {
      task.read.insert(data_id);
    }
  }

  void add_write_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    T4F_INVARIANT(task_id < tasks.size() && "Task ID is out of bounds");
    auto &task = tasks[task_id];
    for (const auto &data_id : data_ids) {
      task.write.insert(data_id);
    }
  }

  void add_retire_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    T4F_INVARIANT(task_id < tasks.size() && "Task ID is out of bounds");
    auto &task = tasks[task_id];
    for (const auto &data_id : data_ids) {
      task.retire.insert(data_id);
    }
  }

  void set_tag(taskid_t task_id, int32_t tag) {
    T4F_INVARIANT(task_id < tasks.size() && "Task ID is out of bounds");
    tasks[task_id].tag = tag;
  }

  void set_type(taskid_t task_id, int32_t type) {
    T4F_INVARIANT(task_id < tasks.size() && "Task ID is out of bounds");
    tasks[task_id].type = type;
  }

  void clear_variants(taskid_t task_id) {
    T4F_INVARIANT(task_id < tasks.size() && "Task ID is out of bounds");
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
    T4F_INVARIANT(task_id < tasks.size() && "Task ID is out of bounds");
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

  void scan_data_domain_and_counts(std::size_t &total_read_count) {
    max_data_id = -1;
    total_read_count = 0;
    for (const auto &task : tasks) {
      for (const auto d : task.read) {
        if (d > max_data_id) max_data_id = d;
      }
      for (const auto d : task.write) {
        if (d > max_data_id) max_data_id = d;
      }
      for (const auto d : task.retire) {
        if (d > max_data_id) max_data_id = d;
      }
      total_read_count += task.read.size();
    }
  }

  void precompute_sorted_data_vectors() {
    for (auto &task : tasks) {
      task.sorted_read_cache = as_sorted_vector(task.read);
      task.sorted_write_cache = as_sorted_vector(task.write);
      task.sorted_retire_cache = as_sorted_vector(task.retire);
    }
  }

  void populate_dependents() {
    for (auto &task : tasks) {
      task.dependents.clear();
    }

    for (auto &task : tasks) {
      for (const auto &dependency_id : task.dependencies) {
        tasks[dependency_id].dependents.insert(task.id);
      }
    }
  }

  void build_compute_dependencies(bool ensure_dependencies = false) {
    populate_dependents();
    if (max_data_id < 0) return;

    std::vector<taskid_t> last_writer(static_cast<std::size_t>(max_data_id) + 1, -1);
    for (auto &task : tasks) {
      auto add_edge_if_writer_exists = [&](dataid_t data_id) {
        const taskid_t writer_id = last_writer[static_cast<std::size_t>(data_id)];
        if (writer_id == taskid_t(-1) || writer_id == task.id) {
          return;
        }

        const auto inserted = task.dependencies.insert(writer_id).second;
        if (inserted) {
          tasks[writer_id].dependents.insert(task.id);
        }
      };

      for (const auto data_id : task.sorted_read_cache) {
        add_edge_if_writer_exists(data_id);
      }

      if (ensure_dependencies) {
        for (const auto data_id : task.sorted_write_cache) {
          add_edge_if_writer_exists(data_id);
        }
        for (const auto data_id : task.sorted_retire_cache) {
          add_edge_if_writer_exists(data_id);
        }
      }

      for (const auto data_id : task.sorted_write_cache) {
        last_writer[static_cast<std::size_t>(data_id)] = task.id;
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
    // sorted_read_cache and sorted_write_cache are built by precompute_sorted_data_vectors().
    for (auto &task : tasks) {
      task.unique.clear();
      if (task.sorted_read_cache.empty()) {
        task.unique = task.sorted_write_cache;
      } else if (task.sorted_write_cache.empty()) {
        task.unique = task.sorted_read_cache;
      } else {
        task.unique.reserve(task.sorted_read_cache.size() + task.sorted_write_cache.size());
        std::set_union(task.sorted_read_cache.begin(), task.sorted_read_cache.end(),
                       task.sorted_write_cache.begin(), task.sorted_write_cache.end(),
                       std::back_inserter(task.unique));
      }
    }
  }

  void populate_initial_tasks() {
    initial_tasks.clear();
    initial_tasks.reserve(tasks.size());
    for (const auto &task : tasks) {
      if (task.dependencies.empty()) {
        initial_tasks.push_back(task.id);
      }
    }
  }

  void build_initial_and_toposort_with_depth(bool detect_cycles = true) {
    populate_initial_tasks();

    sorted.clear();
    sorted.reserve(tasks.size());

    std::vector<int32_t> in_degree(tasks.size(), 0);
    for (auto &task : tasks) {
      task.depth = 0;
      in_degree[task.id] = static_cast<int32_t>(task.dependencies.size());
    }

    std::vector<taskid_t> queue;
    queue.reserve(tasks.size());
    queue.insert(queue.end(), initial_tasks.begin(), initial_tasks.end());

    std::size_t head = 0;
    while (head < queue.size()) {
      const taskid_t current = queue[head++];
      sorted.push_back(current);

      const int32_t next_depth = tasks[current].depth + 1;
      for (const auto dependent : tasks[current].dependents) {
        auto &dependent_task = tasks[dependent];
        if (next_depth > dependent_task.depth) {
          dependent_task.depth = next_depth;
        }
        if (--in_degree[dependent] == 0) {
          queue.push_back(dependent);
        }
      }
    }

    if (detect_cycles && sorted.size() != tasks.size()) {
      throw std::runtime_error(
          "Graph::finalize detected a cycle in compute-task dependencies; topological sort is incomplete.");
    }
  }

  void bfs() {
    sorted.clear();
    sorted.reserve(tasks.size());

    std::vector<int32_t> in_degree(tasks.size(), 0);
    for (const auto &task : tasks) {
      in_degree[task.id] = static_cast<int32_t>(task.dependencies.size());
    }

    std::queue<taskid_t> queue;
    for (auto task_id : initial_tasks) {
      queue.push(task_id);
    }

    while (!queue.empty()) {
      taskid_t current = queue.front();
      queue.pop();
      sorted.push_back(current);

      for (const auto &dependent : tasks[current].dependents) {
        if (--in_degree[dependent] == 0) {
          queue.push(dependent);
        }
      }
    }
  }

  void dfs() {
    sorted.clear();
    sorted.reserve(tasks.size());

    std::vector<int32_t> in_degree(tasks.size(), 0);
    for (const auto &task : tasks) {
      in_degree[task.id] = static_cast<int32_t>(task.dependencies.size());
    }

    std::stack<taskid_t> stack;
    for (auto &task : initial_tasks) {
      stack.push(task);
    }

    while (!stack.empty()) {
      taskid_t current = stack.top();
      stack.pop();
      sorted.push_back(current);

      for (const auto &dependent : tasks[current].dependents) {
        if (--in_degree[dependent] == 0) {
          stack.push(dependent);
        }
      }
    }
  }

  void random_topological_sort(unsigned int seed = 0) {
    sorted.clear();
    sorted.reserve(tasks.size());

    std::vector<int32_t> in_degree(tasks.size(), 0);
    for (const auto &task : tasks) {
      in_degree[task.id] = static_cast<int32_t>(task.dependencies.size());
    }

    auto r = ContainerQueue<taskid_t, std::priority_queue>(seed);
    for (auto &task : initial_tasks) {
      r.push_random(task);
    }

    while (!r.empty()) {
      taskid_t current = r.top();
      r.pop();
      sorted.push_back(current);

      for (const auto &dependent : tasks[current].dependents) {
        if (--in_degree[dependent] == 0) {
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

  void populate_data_dependencies(bool create_data_tasks = true,
                                  std::size_t total_read_count = 0) {
    const std::size_t sz = (max_data_id >= 0) ? static_cast<std::size_t>(max_data_id) + 1 : 0;
    writers.assign(sz, taskid_t(-1));
    // Per-data generation split by writes.
    // Reads observe the current write generation; each write advances it.
    // This makes all reads between two writes share one generation.
    std::vector<uint32_t> write_split_gen_vec(sz, 0);

    for (auto &task : tasks) {
      task.data_dependencies.clear();
      task.data_dependents.clear();
    }

    if (create_data_tasks) {
      data_tasks.clear();
      data_tasks.resize(total_read_count);
    } else {
      data_tasks.clear();
    }

    taskid_t next_data_task_id = 0;
    for (auto task_id : sorted) {

      auto &task = tasks[task_id];
      const auto &sorted_read = task.sorted_read_cache;
      task.sorted_recent_writer_cache.resize(sorted_read.size());
      task.sorted_read_gen_cache.resize(sorted_read.size());

      for (std::size_t i = 0; i < sorted_read.size(); ++i) {
        const dataid_t data_id = sorted_read[i];
        const taskid_t writer_id = writers[static_cast<std::size_t>(data_id)];
        task.sorted_recent_writer_cache[i] = writer_id;
        task.sorted_read_gen_cache[i] = write_split_gen_vec[static_cast<std::size_t>(data_id)];

        if (create_data_tasks) {
          const taskid_t data_task_id = next_data_task_id++;
          auto &data_task = data_tasks[static_cast<std::size_t>(data_task_id)];
          data_task.id = data_task_id;
          data_task.compute_task = task_id;
          data_task.data_id = data_id;

          if (writer_id != taskid_t(-1)) {
            data_task.dependencies.insert(writer_id);
            tasks[writer_id].data_dependents.insert(data_task_id);
          }

          data_task.dependents.insert(task_id);
          task.data_dependencies.insert(data_task_id);
        }
      }
      const auto &sorted_write = task.sorted_write_cache;
      task.sorted_write_gen_cache.resize(sorted_write.size());
      for (std::size_t i = 0; i < sorted_write.size(); ++i) {
        const dataid_t data_id = sorted_write[i];
        const std::size_t idx = static_cast<std::size_t>(data_id);
        task.sorted_write_gen_cache[i] = ++write_split_gen_vec[idx];
        writers[idx] = task_id;
      }
    }

    if (create_data_tasks) {
      T4F_INVARIANT(next_data_task_id == static_cast<taskid_t>(total_read_count));
    }
  }

  void build_sorted_dependency_caches() {
    total_compute_dependencies_cached = 0;
    total_compute_dependents_cached = 0;
    total_compute_data_dependencies_cached = 0;
    total_compute_data_dependents_cached = 0;
    total_reads_cached = 0;
    total_writes_cached = 0;
    total_retire_cached = 0;
    total_unique_cached = 0;

    for (auto &task : tasks) {
      task.sorted_dependencies_cache = as_sorted_vector(task.dependencies);
      task.sorted_dependents_cache = as_sorted_vector(task.dependents);
      task.sorted_data_dependencies_cache = as_sorted_vector(task.data_dependencies);
      task.sorted_data_dependents_cache = as_sorted_vector(task.data_dependents);

      total_compute_dependencies_cached += static_cast<int32_t>(task.sorted_dependencies_cache.size());
      total_compute_dependents_cached += static_cast<int32_t>(task.sorted_dependents_cache.size());
      total_compute_data_dependencies_cached += static_cast<int32_t>(task.sorted_data_dependencies_cache.size());
      total_compute_data_dependents_cached += static_cast<int32_t>(task.sorted_data_dependents_cache.size());
      total_reads_cached += static_cast<int32_t>(task.sorted_read_cache.size());
      total_writes_cached += static_cast<int32_t>(task.sorted_write_cache.size());
      total_retire_cached += static_cast<int32_t>(task.sorted_retire_cache.size());
      total_unique_cached += static_cast<int32_t>(task.unique.size());
    }

    total_data_task_dependencies_cached = 0;
    total_data_task_dependents_cached = 0;

    for (auto &dt : data_tasks) {
      dt.sorted_dependencies_cache = as_sorted_vector(dt.dependencies);
      dt.sorted_dependents_cache = as_sorted_vector(dt.dependents);
      total_data_task_dependencies_cached += static_cast<int32_t>(dt.sorted_dependencies_cache.size());
      total_data_task_dependents_cached += static_cast<int32_t>(dt.sorted_dependents_cache.size());
    }
  }

  void finalize(FinalizeOptions options) {
    if (finalized) {
      std::cerr << "Graph is already finalized. Cannot finalize again." << std::endl;
      std::cerr << "If you want to re-finalize, please create a new Graph instance." << std::endl;
      std::cerr << "Exiting..." << std::endl;
      std::exit(EXIT_FAILURE);
      return;
    }
    finalized = true;

    std::size_t total_read_count = 0;
    scan_data_domain_and_counts(total_read_count);
    precompute_sorted_data_vectors();
    build_compute_dependencies(options.ensure_dependencies);
    build_initial_and_toposort_with_depth(options.detect_cycles);
    populate_data_dependencies(options.create_data_tasks, total_read_count);
    populate_unique_data();
    build_sorted_dependency_caches();
    // populate_data_dependents();
  }

  void finalize() {
    finalize(FinalizeOptions{});
  }

  void finalize(bool ensure_dependencies, bool create_data_tasks_flag) {
    finalize(FinalizeOptions{
        .ensure_dependencies = ensure_dependencies,
        .create_data_tasks = create_data_tasks_flag,
        .detect_cycles = true,
    });
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
  uint8_t mask = 0;
  std::array<Variant, num_device_types> variants{};
};

namespace StateBits {
constexpr uint8_t SPAWNED   = 0x01;
constexpr uint8_t MAPPED    = 0x02;
constexpr uint8_t RESERVED  = 0x04;
constexpr uint8_t LAUNCHED  = 0x08;
constexpr uint8_t COMPLETED = 0x10;
}

static_assert((StateBits::SPAWNED & (StateBits::SPAWNED - 1)) == 0);
static_assert((StateBits::MAPPED & (StateBits::MAPPED - 1)) == 0);
static_assert((StateBits::RESERVED & (StateBits::RESERVED - 1)) == 0);
static_assert((StateBits::LAUNCHED & (StateBits::LAUNCHED - 1)) == 0);
static_assert((StateBits::COMPLETED & (StateBits::COMPLETED - 1)) == 0);
static_assert((StateBits::SPAWNED & StateBits::MAPPED) == 0);
static_assert((StateBits::SPAWNED & StateBits::RESERVED) == 0);
static_assert((StateBits::SPAWNED & StateBits::LAUNCHED) == 0);
static_assert((StateBits::SPAWNED & StateBits::COMPLETED) == 0);
static_assert((StateBits::MAPPED & StateBits::RESERVED) == 0);
static_assert((StateBits::MAPPED & StateBits::LAUNCHED) == 0);
static_assert((StateBits::MAPPED & StateBits::COMPLETED) == 0);
static_assert((StateBits::RESERVED & StateBits::LAUNCHED) == 0);
static_assert((StateBits::RESERVED & StateBits::COMPLETED) == 0);
static_assert((StateBits::LAUNCHED & StateBits::COMPLETED) == 0);


namespace CumulativeState {
constexpr uint8_t SPAWNED   = StateBits::SPAWNED;
constexpr uint8_t MAPPED    = SPAWNED   | StateBits::MAPPED;    // 0x03
constexpr uint8_t RESERVED  = MAPPED    | StateBits::RESERVED;  // 0x07
constexpr uint8_t LAUNCHED  = RESERVED  | StateBits::LAUNCHED;  // 0x0F
constexpr uint8_t COMPLETED = LAUNCHED  | StateBits::COMPLETED; // 0x1F
}

namespace StatusBits {
constexpr uint8_t MAPPABLE   = 0x01; // unmapped == 0 && state == SPAWNED
constexpr uint8_t RESERVABLE = 0x02; // unreserved == 0 && state == MAPPED
constexpr uint8_t LAUNCHABLE = 0x04; // incomplete == 0 && state == RESERVED
}

class StaticTaskInfo {

protected:
  std::vector<ComputeTaskVariantInfo> compute_task_variant_info;
  std::vector<ComputeTaskStaticInfo> compute_task_static_info;

  int32_t num_compute_tasks_{0};
  int32_t num_data_tasks_{0};

  CsrData<taskid_t> compute_task_dependencies;
  CsrData<taskid_t> compute_task_dependents;
  CsrData<taskid_t> compute_task_data_dependencies;
  CsrData<taskid_t> compute_task_data_dependents;
  CsrData<dataid_t> compute_task_read;
  CsrData<dataid_t> compute_task_write;
  CsrData<dataid_t> compute_task_retire;
  std::vector<taskid_t> compute_task_recent_writers;
  std::vector<uint32_t> compute_task_read_generations;
  std::vector<uint32_t> compute_task_write_generations;
  CsrData<dataid_t> compute_task_unique;

  std::vector<dataid_t> read_usage_data_ids;
  CsrData<taskid_t> read_usage;
  std::vector<int32_t> read_usage_row_by_data_id;

  std::vector<dataid_t> write_usage_data_ids;
  CsrData<taskid_t> write_usage;
  std::vector<int32_t> write_usage_row_by_data_id;

  // Shares offsets and row maps with the primary CSR above.
  std::vector<taskid_t> read_usage_by_gen_tasks;
  std::vector<uint32_t> read_usage_by_gen_generations;
  std::vector<taskid_t> write_usage_by_gen_tasks;
  std::vector<uint32_t> write_usage_by_gen_generations;

  CsrData<taskid_t> compute_task_shared_read_neighbors;
  CsrData<taskid_t> data_task_dependencies;
  CsrData<taskid_t> data_task_dependents;

  std::vector<dataid_t> data_task_data_id_cache;
  std::vector<taskid_t> data_task_compute_task_cache;

  std::vector<std::string> compute_task_names;
  mutable std::vector<std::string> data_task_names;

  int32_t grid_h{-1};
  int32_t grid_w{-1};
  bool morton_priority_enabled{false};
  bool random_priority_enabled{false};

  [[nodiscard]] static uint64_t task_data_key(taskid_t task_id, dataid_t data_id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(task_id)) << 32U) |
           static_cast<uint64_t>(static_cast<uint32_t>(data_id));
  }

public:
  StaticTaskInfo(int32_t num_compute_tasks, int32_t num_data_tasks) {
    num_compute_tasks_ = num_compute_tasks;
    num_data_tasks_ = num_data_tasks;
    compute_task_variant_info.resize(num_compute_tasks);
    compute_task_static_info.resize(num_compute_tasks);
    compute_task_dependencies.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_dependents.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_data_dependencies.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_data_dependents.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_read.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_write.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_retire.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_unique.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    data_task_dependencies.offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);
    data_task_dependents.offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);

    compute_task_names.resize(num_compute_tasks);
    data_task_names.resize(num_data_tasks);
    data_task_data_id_cache.resize(num_data_tasks, 0);
    data_task_compute_task_cache.resize(num_data_tasks, 0);

    read_usage.offsets.resize(1, 0);
    write_usage.offsets.resize(1, 0);
    compute_task_shared_read_neighbors.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
  }

  StaticTaskInfo(Graph &graph) {

    taskid_t num_compute_tasks = graph.get_n_compute_tasks();
    taskid_t num_data_tasks = graph.get_n_data_tasks();

    num_compute_tasks_ = num_compute_tasks;
    num_data_tasks_ = num_data_tasks;
    compute_task_variant_info.resize(num_compute_tasks);
    compute_task_static_info.resize(num_compute_tasks);
    compute_task_dependencies.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_dependents.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_data_dependencies.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_data_dependents.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_read.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_write.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_retire.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    compute_task_unique.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    data_task_dependencies.offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);
    data_task_dependents.offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);

    compute_task_names.resize(num_compute_tasks);
    data_task_names.resize(num_data_tasks);
    data_task_data_id_cache.resize(num_data_tasks, 0);
    data_task_compute_task_cache.resize(num_data_tasks, 0);

    read_usage.offsets.resize(1, 0);
    write_usage.offsets.resize(1, 0);
    compute_task_shared_read_neighbors.offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);

    auto &tasks = graph.tasks;
    auto &data_tasks = graph.data_tasks;
    for (int32_t i = 0; i < num_compute_tasks; ++i) {
      const auto &task = tasks[static_cast<std::size_t>(i)];
      compute_task_dependencies.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_dependencies_cache.size());
      compute_task_dependents.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_dependents_cache.size());
      compute_task_data_dependencies.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_data_dependencies_cache.size());
      compute_task_data_dependents.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_data_dependents_cache.size());
      compute_task_read.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_read_cache.size());
      compute_task_write.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_write_cache.size());
      compute_task_retire.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_retire_cache.size());
      compute_task_unique.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.unique.size());
    }

    for (int32_t i = 0; i < num_data_tasks; ++i) {
      const auto &data_task = data_tasks[static_cast<std::size_t>(i)];
      data_task_dependencies.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(data_task.sorted_dependencies_cache.size());
      data_task_dependents.offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(data_task.sorted_dependents_cache.size());
    }

    auto prefix_sum_offsets = [](std::vector<int32_t> &offsets) {
      for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
        offsets[i + 1] += offsets[i];
      }
    };
    prefix_sum_offsets(compute_task_dependencies.offsets);
    prefix_sum_offsets(compute_task_dependents.offsets);
    prefix_sum_offsets(compute_task_data_dependencies.offsets);
    prefix_sum_offsets(compute_task_data_dependents.offsets);
    prefix_sum_offsets(compute_task_read.offsets);
    prefix_sum_offsets(compute_task_write.offsets);
    prefix_sum_offsets(compute_task_retire.offsets);
    prefix_sum_offsets(compute_task_unique.offsets);
    prefix_sum_offsets(data_task_dependencies.offsets);
    prefix_sum_offsets(data_task_dependents.offsets);

    compute_task_dependencies.elements.resize(static_cast<std::size_t>(compute_task_dependencies.offsets.back()));
    compute_task_dependents.elements.resize(static_cast<std::size_t>(compute_task_dependents.offsets.back()));
    compute_task_data_dependencies.elements.resize(static_cast<std::size_t>(compute_task_data_dependencies.offsets.back()));
    compute_task_data_dependents.elements.resize(static_cast<std::size_t>(compute_task_data_dependents.offsets.back()));
    compute_task_read.elements.resize(static_cast<std::size_t>(compute_task_read.offsets.back()));
    compute_task_recent_writers.resize(static_cast<std::size_t>(compute_task_read.offsets.back()));
    compute_task_read_generations.resize(static_cast<std::size_t>(compute_task_read.offsets.back()));
    compute_task_write.elements.resize(static_cast<std::size_t>(compute_task_write.offsets.back()));
    compute_task_write_generations.resize(static_cast<std::size_t>(compute_task_write.offsets.back()));
    compute_task_retire.elements.resize(static_cast<std::size_t>(compute_task_retire.offsets.back()));
    compute_task_unique.elements.resize(static_cast<std::size_t>(compute_task_unique.offsets.back()));
    data_task_dependencies.elements.resize(static_cast<std::size_t>(data_task_dependencies.offsets.back()));
    data_task_dependents.elements.resize(static_cast<std::size_t>(data_task_dependents.offsets.back()));

    for (const auto &task : tasks) {
      const auto idx = static_cast<std::size_t>(task.id);
      compute_task_names[idx] = task.name;
      compute_task_static_info[idx].tag = task.tag;
      compute_task_static_info[idx].type = task.type;
      compute_task_static_info[idx].depth = task.depth;

      std::copy(task.sorted_dependencies_cache.begin(), task.sorted_dependencies_cache.end(),
                compute_task_dependencies.elements.begin() + compute_task_dependencies.offsets[idx]);
      std::copy(task.sorted_dependents_cache.begin(), task.sorted_dependents_cache.end(),
                compute_task_dependents.elements.begin() + compute_task_dependents.offsets[idx]);
      std::copy(task.sorted_data_dependencies_cache.begin(), task.sorted_data_dependencies_cache.end(),
                compute_task_data_dependencies.elements.begin() + compute_task_data_dependencies.offsets[idx]);
      std::copy(task.sorted_data_dependents_cache.begin(), task.sorted_data_dependents_cache.end(),
                compute_task_data_dependents.elements.begin() + compute_task_data_dependents.offsets[idx]);
      std::copy(task.sorted_read_cache.begin(), task.sorted_read_cache.end(),
                compute_task_read.elements.begin() + compute_task_read.offsets[idx]);
      std::copy(task.sorted_recent_writer_cache.begin(), task.sorted_recent_writer_cache.end(),
                compute_task_recent_writers.begin() + compute_task_read.offsets[idx]);
      std::copy(task.sorted_read_gen_cache.begin(), task.sorted_read_gen_cache.end(),
                compute_task_read_generations.begin() + compute_task_read.offsets[idx]);
      std::copy(task.sorted_write_cache.begin(), task.sorted_write_cache.end(),
                compute_task_write.elements.begin() + compute_task_write.offsets[idx]);
      std::copy(task.sorted_write_gen_cache.begin(), task.sorted_write_gen_cache.end(),
                compute_task_write_generations.begin() + compute_task_write.offsets[idx]);
      std::copy(task.sorted_retire_cache.begin(), task.sorted_retire_cache.end(),
                compute_task_retire.elements.begin() + compute_task_retire.offsets[idx]);
      std::copy(task.unique.begin(), task.unique.end(),
                compute_task_unique.elements.begin() + compute_task_unique.offsets[idx]);

      for (int i = 0; i < task.arch.size(); ++i) {
        const auto arch = static_cast<DeviceType>(task.arch[i]);
        add_compute_variant(task.id, arch, task.mem[i], task.vcu[i], task.time[i]);
      }
    }

    for (const auto &data_task : data_tasks) {
      const auto idx = static_cast<std::size_t>(data_task.id);
      data_task_data_id_cache[idx] = data_task.data_id;
      data_task_compute_task_cache[idx] = data_task.compute_task;

      std::copy(data_task.sorted_dependencies_cache.begin(), data_task.sorted_dependencies_cache.end(),
                data_task_dependencies.elements.begin() + data_task_dependencies.offsets[idx]);
      std::copy(data_task.sorted_dependents_cache.begin(), data_task.sorted_dependents_cache.end(),
                data_task_dependents.elements.begin() + data_task_dependents.offsets[idx]);
    }

    build_usage_caches_and_shared_read_topology();
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

  void build_usage_caches_and_shared_read_topology() {
    const auto n_compute_tasks = get_n_compute_tasks();

    struct DataGenTask {
      dataid_t data_id;
      uint32_t gen;
      taskid_t task_id;
    };

    std::vector<DataGenTask> read_triples;
    read_triples.reserve(compute_task_read.elements.size());
    std::vector<DataGenTask> write_triples;
    write_triples.reserve(compute_task_write.elements.size());

    for (taskid_t task_id = 0; task_id < n_compute_tasks; ++task_id) {
      const auto read_span = get_read(task_id);
      const auto read_gen_span = get_read_generations(task_id);
      for (std::size_t i = 0; i < read_span.size(); ++i) {
        read_triples.push_back({read_span[i], read_gen_span[i], task_id});
      }
      const auto write_span = get_write(task_id);
      const auto write_gen_span = get_write_generations(task_id);
      for (std::size_t i = 0; i < write_span.size(); ++i) {
        write_triples.push_back({write_span[i], write_gen_span[i], task_id});
      }
    }

    auto by_data_task = [](const DataGenTask &a, const DataGenTask &b) {
      if (a.data_id != b.data_id) return a.data_id < b.data_id;
      return a.task_id < b.task_id;
    };
    auto by_data_gen_task = [](const DataGenTask &a, const DataGenTask &b) {
      if (a.data_id != b.data_id) return a.data_id < b.data_id;
      if (a.gen != b.gen) return a.gen < b.gen;
      return a.task_id < b.task_id;
    };

    auto max_data_id_in = [](const std::vector<DataGenTask> &triples) -> dataid_t {
      return triples.empty() ? dataid_t(-1) : triples.back().data_id;
    };

    read_usage_data_ids.clear();
    read_usage.offsets.clear();
    read_usage.offsets.push_back(0);
    read_usage.elements.clear();
    read_usage.elements.reserve(read_triples.size());
    read_usage_by_gen_tasks.clear();
    read_usage_by_gen_tasks.reserve(read_triples.size());
    read_usage_by_gen_generations.clear();
    read_usage_by_gen_generations.reserve(read_triples.size());

    std::vector<uint64_t> shared_pair_keys;

    if (!read_triples.empty()) {
      std::sort(read_triples.begin(), read_triples.end(), by_data_task);

      const dataid_t max_read_id = max_data_id_in(read_triples);
      read_usage_row_by_data_id.assign(static_cast<std::size_t>(max_read_id) + 1, -1);
      read_usage_data_ids.reserve(read_triples.size()); // upper bound

      std::size_t group_start = 0;
      while (group_start < read_triples.size()) {
        const dataid_t cur_data = read_triples[group_start].data_id;
        std::size_t group_end = group_start + 1;
        while (group_end < read_triples.size() && read_triples[group_end].data_id == cur_data) {
          ++group_end;
        }
        read_usage_row_by_data_id[static_cast<std::size_t>(cur_data)] =
            static_cast<int32_t>(read_usage_data_ids.size());
        read_usage_data_ids.push_back(cur_data);
        for (std::size_t i = group_start; i < group_end; ++i) {
          read_usage.elements.push_back(read_triples[i].task_id);
        }
        read_usage.offsets.push_back(static_cast<int32_t>(read_usage.elements.size()));

        const auto n_readers = group_end - group_start;
        if (n_readers >= 2) {
          shared_pair_keys.reserve(shared_pair_keys.size() + ((n_readers * (n_readers - 1)) / 2));
          for (std::size_t i = group_start; i < group_end; ++i) {
            const auto lhs = static_cast<uint64_t>(static_cast<uint32_t>(read_triples[i].task_id));
            for (std::size_t j = i + 1; j < group_end; ++j) {
              const auto rhs = static_cast<uint64_t>(static_cast<uint32_t>(read_triples[j].task_id));
              shared_pair_keys.push_back((lhs << 32U) | rhs);
            }
          }
        }
        group_start = group_end;
      }

      std::sort(read_triples.begin(), read_triples.end(), by_data_gen_task);
      for (const auto &t : read_triples) {
        read_usage_by_gen_tasks.push_back(t.task_id);
        read_usage_by_gen_generations.push_back(t.gen);
      }
    }

    write_usage_data_ids.clear();
    write_usage.offsets.clear();
    write_usage.offsets.push_back(0);
    write_usage.elements.clear();
    write_usage.elements.reserve(write_triples.size());
    write_usage_by_gen_tasks.clear();
    write_usage_by_gen_tasks.reserve(write_triples.size());
    write_usage_by_gen_generations.clear();
    write_usage_by_gen_generations.reserve(write_triples.size());

    if (!write_triples.empty()) {
      std::sort(write_triples.begin(), write_triples.end(), by_data_task);

      const dataid_t max_write_id = max_data_id_in(write_triples);
      write_usage_row_by_data_id.assign(static_cast<std::size_t>(max_write_id) + 1, -1);
      write_usage_data_ids.reserve(write_triples.size());

      std::size_t group_start = 0;
      while (group_start < write_triples.size()) {
        const dataid_t cur_data = write_triples[group_start].data_id;
        std::size_t group_end = group_start + 1;
        while (group_end < write_triples.size() && write_triples[group_end].data_id == cur_data) {
          ++group_end;
        }
        write_usage_row_by_data_id[static_cast<std::size_t>(cur_data)] =
            static_cast<int32_t>(write_usage_data_ids.size());
        write_usage_data_ids.push_back(cur_data);
        for (std::size_t i = group_start; i < group_end; ++i) {
          write_usage.elements.push_back(write_triples[i].task_id);
        }
        write_usage.offsets.push_back(static_cast<int32_t>(write_usage.elements.size()));
        group_start = group_end;
      }

      std::sort(write_triples.begin(), write_triples.end(), by_data_gen_task);
      for (const auto &t : write_triples) {
        write_usage_by_gen_tasks.push_back(t.task_id);
        write_usage_by_gen_generations.push_back(t.gen);
      }
    }

    compute_task_shared_read_neighbors.offsets.assign(static_cast<std::size_t>(n_compute_tasks) + 1, 0);
    compute_task_shared_read_neighbors.elements.clear();

    if (shared_pair_keys.empty()) {
      return;
    }

    std::sort(shared_pair_keys.begin(), shared_pair_keys.end());
    shared_pair_keys.erase(std::unique(shared_pair_keys.begin(), shared_pair_keys.end()),
                           shared_pair_keys.end());

    for (const auto key : shared_pair_keys) {
      const auto lhs = static_cast<taskid_t>(key >> 32U);
      const auto rhs = static_cast<taskid_t>(key & 0xFFFFFFFFULL);
      compute_task_shared_read_neighbors.offsets[lhs + 1] += 1;
      compute_task_shared_read_neighbors.offsets[rhs + 1] += 1;
    }

    for (taskid_t task_id = 0; task_id < n_compute_tasks; ++task_id) {
      compute_task_shared_read_neighbors.offsets[task_id + 1] += compute_task_shared_read_neighbors.offsets[task_id];
    }

    compute_task_shared_read_neighbors.elements.resize(
        static_cast<std::size_t>(compute_task_shared_read_neighbors.offsets.back()), -1);
    auto write_offsets_scatter = compute_task_shared_read_neighbors.offsets;

    for (const auto key : shared_pair_keys) {
      const auto lhs = static_cast<taskid_t>(key >> 32U);
      const auto rhs = static_cast<taskid_t>(key & 0xFFFFFFFFULL);
      compute_task_shared_read_neighbors.elements[write_offsets_scatter[lhs]++] = rhs;
      compute_task_shared_read_neighbors.elements[write_offsets_scatter[rhs]++] = lhs;
    }

    for (taskid_t task_id = 0; task_id < n_compute_tasks; ++task_id) {
      const auto begin = static_cast<std::size_t>(compute_task_shared_read_neighbors.offsets[task_id]);
      const auto end = static_cast<std::size_t>(compute_task_shared_read_neighbors.offsets[task_id + 1]);
      std::sort(compute_task_shared_read_neighbors.elements.begin() + begin,
                compute_task_shared_read_neighbors.elements.begin() + end);
    }
  }

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

  void add_compute_variant(taskid_t id, DeviceType arch, mem_t mem, vcu_t vcu, timecount_t time) {
    T4F_INVARIANT(id < compute_task_variant_info.size() && "Task ID is out of bounds");
    auto &info = compute_task_variant_info[id];
    uint8_t arch_type = static_cast<uint8_t>(arch);
    info.mask |= arch_type;
    const auto idx = __builtin_ctz(arch_type);
    T4F_INVARIANT(idx < info.variants.size() && "Architecture index out of bounds");
    info.variants[idx] = Variant(arch, vcu, mem, time);
    // std::cout << "[StaticGraph] Added variant for task " << id << ": "
    //           << "Arch=" << to_string(arch) << ", VCU=" << vcu << ", Mem=" << mem
    //           << ", Time=" << time << std::endl;
  }

  [[nodiscard]] int32_t get_n_compute_tasks() const {
    return num_compute_tasks_;
  }

  [[nodiscard]] int32_t get_n_data_tasks() const {
    return num_data_tasks_;
  }

  [[nodiscard]] int32_t get_n_tasks() const {
    return get_n_compute_tasks() + get_n_data_tasks();
  }

  [[nodiscard]] bool empty() const {
    return num_compute_tasks_ == 0 && num_data_tasks_ == 0;
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_dependencies(taskid_t id) const {
    return compute_task_dependencies[id];
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_dependents(taskid_t id) const {
    return compute_task_dependents[id];
  }

  [[nodiscard]] std::span<const taskid_t> get_data_task_dependencies(taskid_t id) const {
    return data_task_dependencies[id];
  }

  [[nodiscard]] std::span<const taskid_t> get_data_task_dependents(taskid_t id) const {
    return data_task_dependents[id];
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_data_dependencies(taskid_t id) const {
    return compute_task_data_dependencies[id];
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_data_dependents(taskid_t id) const {
    return compute_task_data_dependents[id];
  }

  [[nodiscard]] std::span<const dataid_t> get_read_usage_data_ids() const {
    return read_usage_data_ids;
  }

  [[nodiscard]] std::span<const taskid_t> get_tasks_reading_data(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= read_usage_row_by_data_id.size()) return {};
    const auto row = read_usage_row_by_data_id[idx];
    if (row < 0) return {};
    return read_usage[row];
  }

  [[nodiscard]] std::span<const dataid_t> get_write_usage_data_ids() const {
    return write_usage_data_ids;
  }

  [[nodiscard]] std::span<const taskid_t> get_tasks_writing_data(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= write_usage_row_by_data_id.size()) return {};
    const auto row = write_usage_row_by_data_id[idx];
    if (row < 0) return {};
    return write_usage[row];
  }

  [[nodiscard]] std::span<const taskid_t>
  get_tasks_reading_data_by_gen(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= read_usage_row_by_data_id.size()) return {};
    const auto row = read_usage_row_by_data_id[idx];
    if (row < 0) return {};
    return CsrView<taskid_t>{read_usage.offsets, read_usage_by_gen_tasks}[row];
  }

  [[nodiscard]] std::span<const uint32_t>
  get_read_generations_for_data(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= read_usage_row_by_data_id.size()) return {};
    const auto row = read_usage_row_by_data_id[idx];
    if (row < 0) return {};
    return CsrView<uint32_t>{read_usage.offsets, read_usage_by_gen_generations}[row];
  }

  [[nodiscard]] std::span<const taskid_t>
  get_tasks_writing_data_by_gen(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= write_usage_row_by_data_id.size()) return {};
    const auto row = write_usage_row_by_data_id[idx];
    if (row < 0) return {};
    return CsrView<taskid_t>{write_usage.offsets, write_usage_by_gen_tasks}[row];
  }

  [[nodiscard]] std::span<const uint32_t>
  get_write_generations_for_data(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= write_usage_row_by_data_id.size()) return {};
    const auto row = write_usage_row_by_data_id[idx];
    if (row < 0) return {};
    return CsrView<uint32_t>{write_usage.offsets, write_usage_by_gen_generations}[row];
  }

  [[nodiscard]] std::span<const taskid_t>
  get_compute_task_shared_read_neighbors(taskid_t id) const {
    T4F_INVARIANT(id >= 0 && id < get_n_compute_tasks() && "Task ID is out of bounds");
    return compute_task_shared_read_neighbors[id];
  }

  [[nodiscard]] std::span<const dataid_t> get_read(taskid_t id) const {
    return compute_task_read[id];
  }

  [[nodiscard]] std::span<const dataid_t> get_write(taskid_t id) const {
    return compute_task_write[id];
  }

  [[nodiscard]] std::span<const dataid_t> get_retire(taskid_t id) const {
    return compute_task_retire[id];
  }

  [[nodiscard]] std::span<const dataid_t> get_unique(taskid_t id) const {
    return compute_task_unique[id];
  }

  [[nodiscard]] bool has_read_data(taskid_t task_id, dataid_t data_id) const {
    if (task_id < 0 || task_id >= get_n_compute_tasks() || data_id < 0) return false;
    const auto span = get_read(task_id);
    return std::binary_search(span.begin(), span.end(), data_id);
  }

  [[nodiscard]] bool has_write_data(taskid_t task_id, dataid_t data_id) const {
    if (task_id < 0 || task_id >= get_n_compute_tasks() || data_id < 0) return false;
    const auto span = get_write(task_id);
    return std::binary_search(span.begin(), span.end(), data_id);
  }

  [[nodiscard]] bool has_unique_data(taskid_t task_id, dataid_t data_id) const {
    if (task_id < 0 || task_id >= get_n_compute_tasks() || data_id < 0) return false;
    const auto span = get_unique(task_id);
    return std::binary_search(span.begin(), span.end(), data_id);
  }

  [[nodiscard]] int32_t get_read_data_index(taskid_t task_id, dataid_t data_id) const {
    if (task_id < 0 || task_id >= get_n_compute_tasks() || data_id < 0) return -1;
    const auto span = get_read(task_id);
    auto it = std::lower_bound(span.begin(), span.end(), data_id);
    if (it == span.end() || *it != data_id) return -1;
    return static_cast<int32_t>(it - span.begin());
  }

  [[nodiscard]] int32_t get_write_data_index(taskid_t task_id, dataid_t data_id) const {
    if (task_id < 0 || task_id >= get_n_compute_tasks() || data_id < 0) return -1;
    const auto span = get_write(task_id);
    auto it = std::lower_bound(span.begin(), span.end(), data_id);
    if (it == span.end() || *it != data_id) return -1;
    return static_cast<int32_t>(it - span.begin());
  }

  [[nodiscard]] int32_t get_out_degree(taskid_t compute_task_id) const {
    return get_compute_task_dependents(compute_task_id).size();
  }

  [[nodiscard]] int32_t get_in_degree(taskid_t compute_task_id) const {
    return get_compute_task_dependencies(compute_task_id).size();
  }

  [[nodiscard]] int32_t get_read_generation(taskid_t task_id, dataid_t data_id) const {
    const auto idx = get_read_data_index(task_id, data_id);
    if (idx < 0) return -1;
    const auto span = get_read_generations(task_id);
    return static_cast<int32_t>(span[idx]);
  }

  [[nodiscard]] int32_t get_write_generation(taskid_t task_id, dataid_t data_id) const {
    const auto idx = get_write_data_index(task_id, data_id);
    if (idx < 0) return -1;
    const auto span = get_write_generations(task_id);
    return static_cast<int32_t>(span[idx]);
  }

  [[nodiscard]] const int32_t get_depth(taskid_t id) const {
    return compute_task_static_info[id].depth;
  }

  void add_depth(taskid_t id, int32_t depth) {
    compute_task_static_info[id].depth = depth;
  }

  [[nodiscard]] std::span<const taskid_t> get_most_recent_writers(taskid_t id) const {
    return CsrView<taskid_t>{compute_task_read.offsets, compute_task_recent_writers}[id];
  }

  [[nodiscard]] std::span<const uint32_t> get_read_generations(taskid_t id) const {
    return CsrView<uint32_t>{compute_task_read.offsets, compute_task_read_generations}[id];
  }

  [[nodiscard]] std::span<const uint32_t> get_write_generations(taskid_t id) const {
    return CsrView<uint32_t>{compute_task_write.offsets, compute_task_write_generations}[id];
  }

  [[nodiscard]] const VariantList &get_variants(taskid_t id) const {
    return compute_task_variant_info[id].variants;
  }

  [[nodiscard]] const Variant &get_variant(taskid_t id, DeviceType arch) const {
    const auto idx = __builtin_ctz(static_cast<uint8_t>(arch));
    T4F_INVARIANT(idx < compute_task_variant_info[id].variants.size() &&
           "Architecture index out of bounds for compute task variants");
    return compute_task_variant_info[id].variants[idx];
  }

  [[nodiscard]] const Resources &get_compute_task_resources(taskid_t id, DeviceType arch) const {
    auto &info = compute_task_variant_info[id];
    uint8_t arch_type = static_cast<uint8_t>(arch);
    // assert that mask flag is set for the given architecture
    T4F_INVARIANT((info.mask & arch_type) != 0 && "Architecture not supported for this compute task");
    const auto idx = __builtin_ctz(arch_type);
    T4F_INVARIANT(idx < info.variants.size() && "Architecture index out of bounds");
    const auto &variant = info.variants[idx];
    return variant.get_resources();
  }

  [[nodiscard]] const timecount_t get_mean_duration(taskid_t id, DeviceType arch) const {
    auto &info = compute_task_variant_info[id];
    uint8_t arch_type = static_cast<uint8_t>(arch);
    // assert that mask flag is set for the given architecture
    T4F_INVARIANT((info.mask & arch_type) != 0 && "Architecture not supported for this compute task");
    const auto idx = __builtin_ctz(arch_type);
    T4F_INVARIANT(idx < info.variants.size() && "Architecture index out of bounds");
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
    const auto idx = static_cast<std::size_t>(id);
    auto &name = data_task_names[idx];
    if (name.empty()) {
      constexpr std::size_t kTaskPrefixReserve =
          static_cast<std::size_t>(std::numeric_limits<taskid_t>::digits10) + 8;
      constexpr std::size_t kMaxDataIdChars =
          static_cast<std::size_t>(std::numeric_limits<dataid_t>::digits10) + 2;
      name.reserve(kTaskPrefixReserve + kMaxDataIdChars);
      append_decimal(name, data_task_compute_task_cache[idx]);
      name.append("_data_");
      append_decimal(name, data_task_data_id_cache[idx]);
    }
    return name;
  }

  [[nodiscard]] const dataid_t get_data_id(taskid_t id) const {
    return data_task_data_id_cache[static_cast<std::size_t>(id)];
  }

  [[nodiscard]] const taskid_t get_compute_task(taskid_t id) const {
    return data_task_compute_task_cache[static_cast<std::size_t>(id)];
  }

  [[nodiscard]] int32_t get_compute_task_dependency_count(taskid_t id) const {
    return compute_task_dependencies.row_size(id);
  }

  [[nodiscard]] int32_t get_compute_task_data_dependency_count(taskid_t id) const {
    return compute_task_data_dependencies.row_size(id);
  }

  [[nodiscard]] int32_t get_data_task_dependency_count(taskid_t id) const {
    return data_task_dependencies.row_size(id);
  }

  [[nodiscard]] const ComputeTaskStaticInfo &get_compute_task_static_info(taskid_t id) const {
    return compute_task_static_info[id];
  }
};

class RuntimeTaskInfo {
protected:
  struct ComputeRuntimeSoA {
    std::vector<uint8_t> state;
    std::vector<uint8_t> status;
    std::vector<uint8_t> flags;
    std::vector<int16_t> unmapped;
    std::vector<int16_t> unreserved;
    std::vector<int16_t> incomplete;
    std::vector<int32_t> mapped_device;
    std::vector<int32_t> reserve_priority;
    std::vector<int32_t> launch_priority;
    std::vector<timecount_t> mapped_time;
    std::vector<timecount_t> reserved_time;
    std::vector<timecount_t> launched_time;
    std::vector<timecount_t> completed_time;

    void resize(int32_t n) {
      state.resize(n, 0);
      status.resize(n, 0);
      flags.resize(n, 0);
      unmapped.resize(n, 0);
      unreserved.resize(n, 0);
      incomplete.resize(n, 0);
      mapped_device.resize(n, -1);
      reserve_priority.resize(n, 0);
      launch_priority.resize(n, 0);
      mapped_time.resize(n, 0);
      reserved_time.resize(n, 0);
      launched_time.resize(n, 0);
      completed_time.resize(n, 0);
    }
  };

  struct DataRuntimeSoA {
    std::vector<uint8_t> state;
    std::vector<uint8_t> flags;
    std::vector<int16_t> incomplete;
    std::vector<int32_t> source_device;
    std::vector<int32_t> mapped_device;
    std::vector<int32_t> launch_priority;
    std::vector<timecount_t> launched_time;
    std::vector<timecount_t> completed_time;

    void resize(int32_t n) {
      state.resize(n, 0);
      flags.resize(n, 0);
      incomplete.resize(n, 0);
      source_device.resize(n, 0);
      mapped_device.resize(n, -1);
      launch_priority.resize(n, 0);
      launched_time.resize(n, 0);
      completed_time.resize(n, 0);
    }
  };

  struct EvictionRuntimeSoA {
    std::vector<uint8_t> state;
    std::vector<uint8_t> flags;
    std::vector<int32_t> data_id;
    std::vector<int32_t> evicting_on;
    std::vector<int32_t> compute_task;
    std::vector<int32_t> source_device;
    std::vector<int32_t> launch_priority;
    std::vector<timecount_t> launched_time;
    std::vector<timecount_t> completed_time;
    std::vector<std::string> names;
  };

  ComputeRuntimeSoA compute;
  DataRuntimeSoA data;
  EvictionRuntimeSoA eviction;

  int32_t n_compute{0};
  int32_t n_data{0};

public:
  RuntimeTaskInfo() = default;

  RuntimeTaskInfo(StaticTaskInfo &static_info) {
    n_compute = static_cast<int32_t>(static_info.get_n_compute_tasks());
    n_data = static_cast<int32_t>(static_info.get_n_data_tasks());
    resize_compute(n_compute);
    resize_data(n_data);
    for (int32_t i = 0; i < n_compute; ++i) {
      initialize_compute_runtime(i, static_info);
    }
    for (int32_t i = 0; i < n_data; ++i) {
      initialize_data_runtime(i, static_info);
    }
  }

  RuntimeTaskInfo(const RuntimeTaskInfo &other) {
    {
      ZoneScopedN("Copy ComputeTask SoA");
      compute = other.compute;
    }
    {
      ZoneScopedN("Copy DataTask SoA");
      data = other.data;
    }
    {
      ZoneScopedN("Copy EvictionTask SoA");
      eviction = other.eviction;
    }
    n_compute = other.n_compute;
    n_data = other.n_data;
  }

  void resize_compute(int32_t n) {
    compute.resize(n);
  }

  void resize_data(int32_t n) {
    data.resize(n);
  }

  void initialize_compute_runtime(int32_t id, const StaticTaskInfo &static_info) {
    compute.state[id] = CumulativeState::SPAWNED;
    const auto n_deps =
        static_cast<int16_t>(static_info.get_compute_task_dependency_count(id));
    const auto n_data_deps =
        static_cast<int16_t>(static_info.get_compute_task_data_dependency_count(id));
    compute.unmapped[id] = n_deps;
    compute.unreserved[id] = n_deps;
    compute.incomplete[id] = n_deps + n_data_deps;
    compute.status[id] = (n_deps == 0) ? StatusBits::MAPPABLE : 0;
  }

  void initialize_data_runtime(int32_t id, const StaticTaskInfo &static_info) {
    data.state[id] = CumulativeState::SPAWNED;
    data.incomplete[id] =
        static_cast<int16_t>(static_info.get_data_task_dependency_count(id));
  }

  int32_t add_eviction_task(int32_t compute_task_id, int32_t data_id,
                            int32_t evicting_on_device_id) {
    taskid_t id = static_cast<taskid_t>(eviction.state.size());
    eviction.state.push_back(CumulativeState::RESERVED);
    eviction.flags.push_back(0);
    eviction.data_id.push_back(data_id);
    eviction.evicting_on.push_back(evicting_on_device_id);
    eviction.compute_task.push_back(compute_task_id);
    eviction.source_device.push_back(0);
    eviction.launch_priority.push_back(0);
    eviction.launched_time.push_back(0);
    eviction.completed_time.push_back(0);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "EvictionTask_%d_%d_%d", compute_task_id, data_id,
                  evicting_on_device_id);
    eviction.names.emplace_back(buf);
    return id;
  }

  [[nodiscard]] bool is_compute_mapped(taskid_t id) const {
    return (compute.state[id] & StateBits::MAPPED) != 0;
  }
  [[nodiscard]] bool is_compute_reserved(taskid_t id) const {
    return (compute.state[id] & StateBits::RESERVED) != 0;
  }
  [[nodiscard]] bool is_compute_launched(taskid_t id) const {
    return (compute.state[id] & StateBits::LAUNCHED) != 0;
  }
  [[nodiscard]] bool is_compute_completed(taskid_t id) const {
    return (compute.state[id] & StateBits::COMPLETED) != 0;
  }

  [[nodiscard]] bool is_compute_mappable(taskid_t id) const {
    return (compute.status[id] & StatusBits::MAPPABLE) != 0;
  }
  [[nodiscard]] bool is_compute_reservable(taskid_t id) const {
    return (compute.status[id] & StatusBits::RESERVABLE) != 0;
  }
  [[nodiscard]] bool is_compute_launchable(taskid_t id) const {
    return (compute.status[id] & StatusBits::LAUNCHABLE) != 0;
  }

  [[nodiscard]] bool is_data_launchable(taskid_t id) const {
    return data.incomplete[id] == 0 && data.state[id] == CumulativeState::RESERVED;
  }
  [[nodiscard]] bool is_data_completed(taskid_t id) const {
    return (data.state[id] & StateBits::COMPLETED) != 0;
  }
  [[nodiscard]] bool is_eviction_launchable(taskid_t id) const {
    return (eviction.state[id] & StateBits::RESERVED) != 0;
  }
  [[nodiscard]] bool is_eviction_completed(taskid_t id) const {
    return (eviction.state[id] & StateBits::COMPLETED) != 0;
  }

  [[nodiscard]] bool is_data_task_virtual(taskid_t id) const {
    return (data.flags[id] & 0x01) != 0;
  }
  [[nodiscard]] bool is_eviction_task_virtual(taskid_t id) const {
    return (eviction.flags[id] & 0x01) != 0;
  }

  [[nodiscard]] const uint8_t *compute_state_data() const { return compute.state.data(); }
  [[nodiscard]] const uint8_t *compute_status_data() const { return compute.status.data(); }
  [[nodiscard]] const int32_t *compute_mapped_device_data() const {
    return compute.mapped_device.data();
  }
  [[nodiscard]] uint8_t *compute_state_data() { return compute.state.data(); }
  [[nodiscard]] uint8_t *compute_status_data() { return compute.status.data(); }
  [[nodiscard]] int32_t *compute_mapped_device_data() { return compute.mapped_device.data(); }

  [[nodiscard]] int32_t get_n_compute_tasks() const { return n_compute; }
  [[nodiscard]] int32_t get_n_data_tasks() const { return n_data; }
  [[nodiscard]] int32_t get_n_eviction_tasks() const {
    return static_cast<int32_t>(eviction.state.size());
  }
  [[nodiscard]] int32_t get_n_tasks() const {
    return get_n_compute_tasks() + get_n_data_tasks() + get_n_eviction_tasks();
  }
  [[nodiscard]] bool empty() const {
    return n_compute == 0 && n_data == 0 && eviction.state.empty();
  }

  [[nodiscard]] TaskState get_compute_task_state(taskid_t id) const {
    const auto s = compute.state[id];
    if (s & StateBits::COMPLETED) return TaskState::COMPLETED;
    if (s & StateBits::LAUNCHED)  return TaskState::LAUNCHED;
    if (s & StateBits::RESERVED)  return TaskState::RESERVED;
    if (s & StateBits::MAPPED)    return TaskState::MAPPED;
    return TaskState::SPAWNED;
  }

  [[nodiscard]] TaskState get_data_task_state(taskid_t id) const {
    const auto s = data.state[id];
    if (s & StateBits::COMPLETED) return TaskState::COMPLETED;
    if (s & StateBits::LAUNCHED)  return TaskState::LAUNCHED;
    if (s & StateBits::RESERVED)  return TaskState::RESERVED;
    return TaskState::SPAWNED;
  }

  [[nodiscard]] TaskState get_eviction_task_state(taskid_t id) const {
    const auto s = eviction.state[id];
    if (s & StateBits::COMPLETED) return TaskState::COMPLETED;
    if (s & StateBits::LAUNCHED)  return TaskState::LAUNCHED;
    if (s & StateBits::RESERVED)  return TaskState::RESERVED;
    return TaskState::SPAWNED;
  }

  [[nodiscard]] int16_t get_compute_task_unmapped(taskid_t id) const { return compute.unmapped[id]; }
  [[nodiscard]] int16_t get_compute_task_unreserved(taskid_t id) const {
    return compute.unreserved[id];
  }
  [[nodiscard]] int16_t get_compute_task_incomplete(taskid_t id) const {
    return compute.incomplete[id];
  }
  [[nodiscard]] int32_t get_compute_task_mapped_device(taskid_t id) const {
    return compute.mapped_device[id];
  }
  [[nodiscard]] int32_t get_compute_task_reserve_priority(taskid_t id) const {
    return compute.reserve_priority[id];
  }
  [[nodiscard]] int32_t get_compute_task_launch_priority(taskid_t id) const {
    return compute.launch_priority[id];
  }
  [[nodiscard]] uint8_t get_compute_task_flags(taskid_t id) const { return compute.flags[id]; }

  [[nodiscard]] int32_t get_data_task_source_device(taskid_t id) const {
    return data.source_device[id];
  }
  [[nodiscard]] int32_t get_data_task_mapped_device(taskid_t id) const {
    return data.mapped_device[id];
  }
  [[nodiscard]] int32_t get_data_task_launch_priority(taskid_t id) const {
    return data.launch_priority[id];
  }
  [[nodiscard]] uint8_t get_data_task_flags(taskid_t id) const { return data.flags[id]; }

  [[nodiscard]] const std::string &get_eviction_task_name(taskid_t id) const {
    return eviction.names[id];
  }
  [[nodiscard]] int32_t get_eviction_task_evicting_on(taskid_t id) const {
    return eviction.evicting_on[id];
  }
  [[nodiscard]] int32_t get_eviction_task_data_id(taskid_t id) const { return eviction.data_id[id]; }
  [[nodiscard]] int32_t get_eviction_task_source_device(taskid_t id) const {
    return eviction.source_device[id];
  }

  [[nodiscard]] timecount_t get_compute_task_mapped_time(taskid_t id) const {
    return compute.mapped_time[id];
  }
  [[nodiscard]] timecount_t get_compute_task_reserved_time(taskid_t id) const {
    return compute.reserved_time[id];
  }
  [[nodiscard]] timecount_t get_compute_task_launched_time(taskid_t id) const {
    return compute.launched_time[id];
  }
  [[nodiscard]] timecount_t get_compute_task_completed_time(taskid_t id) const {
    return compute.completed_time[id];
  }
  [[nodiscard]] timecount_t get_data_task_launched_time(taskid_t id) const {
    return data.launched_time[id];
  }
  [[nodiscard]] timecount_t get_data_task_completed_time(taskid_t id) const {
    return data.completed_time[id];
  }
  [[nodiscard]] timecount_t get_eviction_task_launched_time(taskid_t id) const {
    return eviction.launched_time[id];
  }
  [[nodiscard]] timecount_t get_eviction_task_completed_time(taskid_t id) const {
    return eviction.completed_time[id];
  }

  [[nodiscard]] timecount_t get_compute_task_duration(taskid_t id) const {
    return compute.completed_time[id] - compute.launched_time[id];
  }
  [[nodiscard]] timecount_t get_data_task_duration(taskid_t id) const {
    return data.completed_time[id] - data.launched_time[id];
  }
  [[nodiscard]] timecount_t get_eviction_task_duration(taskid_t id) const {
    return eviction.completed_time[id] - eviction.launched_time[id];
  }

  [[nodiscard]] TaskState get_compute_task_state_at_time(taskid_t id,
                                                         timecount_t query) const {
    if (query < compute.mapped_time[id])    return TaskState::SPAWNED;
    if (query < compute.reserved_time[id])  return TaskState::MAPPED;
    if (query < compute.launched_time[id])  return TaskState::RESERVED;
    if (query < compute.completed_time[id]) return TaskState::LAUNCHED;
    return TaskState::COMPLETED;
  }

  [[nodiscard]] TaskState get_data_task_state_at_time(taskid_t id, timecount_t query) const {
    if (query < data.launched_time[id])  return TaskState::RESERVED;
    if (query < data.completed_time[id]) return TaskState::LAUNCHED;
    return TaskState::COMPLETED;
  }

  [[nodiscard]] TaskState get_eviction_task_state_at_time(taskid_t id,
                                                          timecount_t query) const {
    if (query < eviction.launched_time[id])  return TaskState::MAPPED;
    if (query < eviction.completed_time[id]) return TaskState::RESERVED;
    return TaskState::COMPLETED;
  }

  [[nodiscard]] TaskStatus get_compute_task_status(taskid_t id) const {
    const auto s = compute.status[id];
    if (s & StatusBits::LAUNCHABLE) return TaskStatus::LAUNCHABLE;
    if (s & StatusBits::RESERVABLE) return TaskStatus::RESERVABLE;
    if (s & StatusBits::MAPPABLE)   return TaskStatus::MAPPABLE;
    return TaskStatus::NONE;
  }

  // ── Setters ──────────────────────────────────────────────────

  void set_compute_task_state(taskid_t id, TaskState state) {
    switch (state) {
    case TaskState::SPAWNED:   compute.state[id] = CumulativeState::SPAWNED;   break;
    case TaskState::MAPPED:    compute.state[id] = CumulativeState::MAPPED;    break;
    case TaskState::RESERVED:  compute.state[id] = CumulativeState::RESERVED;  break;
    case TaskState::LAUNCHED:  compute.state[id] = CumulativeState::LAUNCHED;  break;
    case TaskState::COMPLETED: compute.state[id] = CumulativeState::COMPLETED; break;
    }
  }

  void set_data_task_state(taskid_t id, TaskState state) {
    switch (state) {
    case TaskState::SPAWNED:   data.state[id] = CumulativeState::SPAWNED;   break;
    case TaskState::MAPPED:    data.state[id] = CumulativeState::MAPPED;    break;
    case TaskState::RESERVED:  data.state[id] = CumulativeState::RESERVED;  break;
    case TaskState::LAUNCHED:  data.state[id] = CumulativeState::LAUNCHED;  break;
    case TaskState::COMPLETED: data.state[id] = CumulativeState::COMPLETED; break;
    }
  }

  void set_eviction_task_state(taskid_t id, TaskState state) {
    switch (state) {
    case TaskState::SPAWNED:   eviction.state[id] = CumulativeState::SPAWNED;   break;
    case TaskState::MAPPED:    eviction.state[id] = CumulativeState::MAPPED;    break;
    case TaskState::RESERVED:  eviction.state[id] = CumulativeState::RESERVED;  break;
    case TaskState::LAUNCHED:  eviction.state[id] = CumulativeState::LAUNCHED;  break;
    case TaskState::COMPLETED: eviction.state[id] = CumulativeState::COMPLETED; break;
    }
  }

  void set_compute_task_unmapped(taskid_t id, int16_t v) { compute.unmapped[id] = v; }
  void set_compute_task_unreserved(taskid_t id, int16_t v) { compute.unreserved[id] = v; }
  void set_compute_task_incomplete(taskid_t id, int16_t v) { compute.incomplete[id] = v; }
  void set_compute_task_mapped_device(taskid_t id, int32_t v) { compute.mapped_device[id] = v; }
  void set_compute_task_reserve_priority(taskid_t id, int32_t v) { compute.reserve_priority[id] = v; }
  void set_compute_task_launch_priority(taskid_t id, int32_t v) { compute.launch_priority[id] = v; }
  void set_compute_task_flags(taskid_t id, uint8_t v) { compute.flags[id] = v; }

  void set_data_task_incomplete(taskid_t id, int16_t v) { data.incomplete[id] = v; }
  void set_data_task_source_device(taskid_t id, int32_t v) { data.source_device[id] = v; }
  void set_data_task_mapped_device(taskid_t id, int32_t v) { data.mapped_device[id] = v; }
  void set_data_task_launch_priority(taskid_t id, int32_t v) { data.launch_priority[id] = v; }
  void set_data_task_virtual(taskid_t id, bool v) {
    data.flags[id] = v ? (data.flags[id] | 0x01) : (data.flags[id] & ~uint8_t{0x01});
  }

  void set_eviction_task_evicting_on(taskid_t id, int32_t v) { eviction.evicting_on[id] = v; }
  void set_eviction_task_compute_task(taskid_t id, int32_t v) { eviction.compute_task[id] = v; }
  void set_eviction_task_source_device(taskid_t id, int32_t v) { eviction.source_device[id] = v; }
  void set_eviction_task_data_id(taskid_t id, int32_t v) { eviction.data_id[id] = v; }
  void set_eviction_task_virtual(taskid_t id, bool v) {
    eviction.flags[id] = v ? (eviction.flags[id] | 0x01) : (eviction.flags[id] & ~uint8_t{0x01});
  }

  void record_mapped(taskid_t id, timecount_t t) { compute.mapped_time[id] = t; }
  void record_reserved(taskid_t id, timecount_t t) { compute.reserved_time[id] = t; }
  void record_launched(taskid_t id, timecount_t t) { compute.launched_time[id] = t; }
  void record_completed(taskid_t id, timecount_t t) { compute.completed_time[id] = t; }
  void record_data_launched(taskid_t id, timecount_t t) { data.launched_time[id] = t; }
  void record_data_completed(taskid_t id, timecount_t t) { data.completed_time[id] = t; }
  void record_eviction_launched(taskid_t id, timecount_t t) { eviction.launched_time[id] = t; }
  void record_eviction_completed(taskid_t id, timecount_t t) { eviction.completed_time[id] = t; }

  bool decrement_compute_task_unmapped(taskid_t id) {
    auto &v = compute.unmapped[id];
    const int16_t nv = --v;
    T4F_INVARIANT(nv >= 0 && "Unmapped count cannot be negative");
    if (nv == 0) {
      if (compute.state[id] == CumulativeState::SPAWNED) {
        compute.status[id] |= StatusBits::MAPPABLE;
        return true;
      }
    }
    return false;
  }

  bool decrement_compute_task_unreserved(taskid_t id) {
    auto &v = compute.unreserved[id];
    const int16_t nv = --v;
    T4F_INVARIANT(nv >= 0 && "Unreserved count cannot be negative");
    if (nv == 0) { // boundary only
      if (compute.state[id] == CumulativeState::MAPPED) {
        compute.status[id] |= StatusBits::RESERVABLE;
        return true;
      }
    }
    return false;
  }

  bool decrement_compute_task_incomplete(taskid_t id) {
    auto &v = compute.incomplete[id];
    const int16_t nv = --v;
    T4F_INVARIANT(nv >= 0 && "Incomplete count cannot be negative");
    if (nv == 0) { // boundary only
      if (compute.state[id] == CumulativeState::RESERVED) {
        compute.status[id] |= StatusBits::LAUNCHABLE;
        return true;
      }
    }
    return false;
  }

  bool decrement_data_task_incomplete(taskid_t id) {
    auto &v = data.incomplete[id];
    const int16_t nv = --v;
    T4F_INVARIANT(nv >= 0 && "Data incomplete count cannot be negative");
    if (nv == 0) { // boundary only
      return data.state[id] == CumulativeState::RESERVED;
    }
    return false;
  }

  taskid_t compute_notify_mapped(taskid_t compute_task_id, devid_t mapped_device,
                                int32_t reserve_priority, int32_t launch_priority,
                                timecount_t time, const StaticTaskInfo &static_info,
                                TaskIDList &compute_task_buffer)
  {
    compute.mapped_device[compute_task_id] = mapped_device;
    compute.reserve_priority[compute_task_id] = reserve_priority;
    compute.launch_priority[compute_task_id]  = launch_priority;
    compute.state[compute_task_id] = CumulativeState::MAPPED;
    compute.mapped_time[compute_task_id] = time;
    compute.status[compute_task_id] &= static_cast<uint8_t>(~StatusBits::MAPPABLE);
    if (compute.unreserved[compute_task_id] == 0) {
      compute.status[compute_task_id] |= StatusBits::RESERVABLE;
    }

    return collect_ready(static_info.get_compute_task_dependents(compute_task_id),
                        [&](taskid_t dep) { return decrement_compute_task_unmapped(dep); },
                        compute_task_buffer);
  }

taskid_t compute_notify_reserved(taskid_t compute_task_id, devid_t mapped_device,
                                 timecount_t time, const StaticTaskInfo &static_info,
                                 TaskIDList &compute_task_buffer)
{
  compute.mapped_device[compute_task_id] = mapped_device;
  compute.state[compute_task_id] = CumulativeState::RESERVED;
  compute.reserved_time[compute_task_id] = time;
  compute.status[compute_task_id] &= static_cast<uint8_t>(~StatusBits::RESERVABLE);
  if (compute.incomplete[compute_task_id] == 0) {
    compute.status[compute_task_id] |= StatusBits::LAUNCHABLE;
  }

  return collect_ready(static_info.get_compute_task_dependents(compute_task_id),
                       [&](taskid_t dep) { return decrement_compute_task_unreserved(dep); },
                       compute_task_buffer);
}

  void compute_notify_launched(taskid_t compute_task_id, timecount_t time,
                               const StaticTaskInfo & /*static_info*/) {
    compute.state[compute_task_id] = CumulativeState::LAUNCHED;
    compute.launched_time[compute_task_id] = time;
    compute.status[compute_task_id] &= ~StatusBits::LAUNCHABLE;
  }

  taskid_t compute_notify_completed(taskid_t compute_task_id, timecount_t time,
                                    const StaticTaskInfo &static_info,
                                    TaskIDList &compute_task_buffer)
  {
    compute.state[compute_task_id] = CumulativeState::COMPLETED;
    compute.completed_time[compute_task_id] = time;

    return collect_ready(static_info.get_compute_task_dependents(compute_task_id),
                        [&](taskid_t dep) { return decrement_compute_task_incomplete(dep); },
                        compute_task_buffer);
  }

  taskid_t compute_notify_data_completed(taskid_t compute_task_id, timecount_t /*time*/,
                                        const StaticTaskInfo &static_info,
                                        TaskIDList &data_task_buffer)
  {
    return collect_ready(static_info.get_compute_task_data_dependents(compute_task_id),
                        [&](taskid_t dt) { return decrement_data_task_incomplete(dt); },
                        data_task_buffer);
  }

  void data_notify_reserved(taskid_t data_task_id, devid_t mapped_device,
                            timecount_t /*time*/, const StaticTaskInfo &) {
    data.mapped_device[data_task_id] = mapped_device;
    data.state[data_task_id] = CumulativeState::RESERVED;
  }

  void data_notify_launched(taskid_t data_task_id, devid_t source_device, timecount_t time,
                            const StaticTaskInfo &) {
    data.state[data_task_id] = CumulativeState::LAUNCHED;
    data.source_device[data_task_id] = source_device;
    data.launched_time[data_task_id] = time;
  }

  taskid_t data_notify_completed(taskid_t data_task_id, timecount_t time,
                                const StaticTaskInfo &static_info,
                                TaskIDList &compute_task_buffer)
  {
    data.state[data_task_id] = CumulativeState::COMPLETED;
    data.completed_time[data_task_id] = time;

    return collect_ready(static_info.get_data_task_dependents(data_task_id),
                        [&](taskid_t ct) { return decrement_compute_task_incomplete(ct); },
                        compute_task_buffer);
  }

  void eviction_notify_reserved(taskid_t eviction_task_id, timecount_t,
                                const StaticTaskInfo &) {
    eviction.state[eviction_task_id] = CumulativeState::RESERVED;
  }

  void eviction_notify_launched(taskid_t eviction_task_id, devid_t source_device_id,
                                timecount_t time, const StaticTaskInfo &) {
    eviction.source_device[eviction_task_id] = source_device_id;
    eviction.state[eviction_task_id] = CumulativeState::LAUNCHED;
    eviction.launched_time[eviction_task_id] = time;
  }

  void eviction_notify_completed(taskid_t eviction_task_id, timecount_t time) {
    eviction.state[eviction_task_id] = CumulativeState::COMPLETED;
    eviction.completed_time[eviction_task_id] = time;
  }
};


namespace task_query {
namespace detail {

[[nodiscard]] inline auto is_exact_mapped(const RuntimeTaskInfo& runtime_info) {
  const uint8_t* states = runtime_info.compute_state_data();
  return [states](taskid_t tid) -> bool {
    return (states[tid] & CumulativeState::COMPLETED) == CumulativeState::MAPPED;
  };
}

[[nodiscard]] inline auto is_exact_mapped_on_device(const RuntimeTaskInfo& runtime_info, devid_t device_id) {
  const uint8_t* states = runtime_info.compute_state_data();
  const int32_t* mapped_devices = runtime_info.compute_mapped_device_data();
  return [states, mapped_devices, device_id](taskid_t tid) -> bool {
    return (states[tid] & CumulativeState::COMPLETED) == CumulativeState::MAPPED &&
           mapped_devices[tid] == device_id;
  };
}

template <typename Predicate>
static inline taskid_t filter_tasks(std::span<const taskid_t> tasks, TaskIDList& out, Predicate&& pred) {
  if (tasks.empty()) {
    out.clear();
    return 0;
  }
  
  out.resize(tasks.size());
  taskid_t w = 0;
  for (const taskid_t tid : tasks) {
    out[w] = tid;
    w += static_cast<taskid_t>(pred(tid));
  }
  
  out.resize(static_cast<std::size_t>(w));
  return w; // Return the count of tasks where predicate is true
}

template <typename Predicate>
static inline taskid_t count_tasks(std::span<const taskid_t> tasks, Predicate&& pred) {
  taskid_t count = 0;
  for (const taskid_t tid : tasks) {
    count += static_cast<taskid_t>(pred(tid));
  }
  return count;
}

template <typename Predicate>
[[nodiscard]] static inline bool any_tasks(std::span<const taskid_t> tasks, Predicate&& pred) {
  for (const taskid_t tid : tasks) {
    if (pred(tid)) {
      return true;
    }
  }
  return false;
}

static inline std::pair<std::span<const taskid_t>, std::span<const taskid_t>>
split_readers_by_gen(const StaticTaskInfo& static_info, dataid_t data_id, uint32_t gen) {
  const auto tasks = static_info.get_tasks_reading_data_by_gen(data_id);
  const auto gens = static_info.get_read_generations_for_data(data_id);
  const auto split = static_cast<std::size_t>(
      std::lower_bound(gens.begin(), gens.end(), gen) - gens.begin());
      
  return {tasks.subspan(0, split), tasks.subspan(split)};
}

} // namespace detail

[[nodiscard]] static inline taskid_t mapped_writers(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info, 
    dataid_t data_id, TaskIDList& out) 
{
  return detail::filter_tasks(static_info.get_tasks_writing_data(data_id), out,
                              detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline taskid_t mapped_writers(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id)
{
  return detail::count_tasks(static_info.get_tasks_writing_data(data_id),
                             detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline taskid_t mapped_writers_count(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id)
{
  return detail::count_tasks(static_info.get_tasks_writing_data(data_id),
                             detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline bool mapped_writers_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id)
{
  return detail::any_tasks(static_info.get_tasks_writing_data(data_id),
                           detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline taskid_t mapped_writers_on_device(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id, TaskIDList& out)
{
  return detail::filter_tasks(static_info.get_tasks_writing_data(data_id), out,
                              detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_writers_on_device(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id)
{
  return detail::count_tasks(static_info.get_tasks_writing_data(data_id),
                             detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_writers_on_device_count(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id)
{
  return detail::count_tasks(static_info.get_tasks_writing_data(data_id),
                             detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline bool mapped_writers_on_device_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id)
{
  return detail::any_tasks(static_info.get_tasks_writing_data(data_id),
                           detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_readers(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info, 
    dataid_t data_id, TaskIDList& out) 
{
  return detail::filter_tasks(static_info.get_tasks_reading_data(data_id), out,
                              detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline taskid_t mapped_readers(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id)
{
  return detail::count_tasks(static_info.get_tasks_reading_data(data_id),
                             detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline taskid_t mapped_readers_count(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id)
{
  return detail::count_tasks(static_info.get_tasks_reading_data(data_id),
                             detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline bool mapped_readers_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id)
{
  return detail::any_tasks(static_info.get_tasks_reading_data(data_id),
                           detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline taskid_t mapped_readers_on_device(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id, TaskIDList& out)
{
  return detail::filter_tasks(static_info.get_tasks_reading_data(data_id), out,
                              detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_readers_on_device(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id)
{
  return detail::count_tasks(static_info.get_tasks_reading_data(data_id),
                             detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_readers_on_device_count(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id)
{
  return detail::count_tasks(static_info.get_tasks_reading_data(data_id),
                             detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline bool mapped_readers_on_device_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id)
{
  return detail::any_tasks(static_info.get_tasks_reading_data(data_id),
                           detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_reads_count(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id)
{
  return mapped_readers_count(static_info, runtime_info, data_id);
}

[[nodiscard]] static inline bool mapped_reads_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id)
{
  return mapped_readers_any(static_info, runtime_info, data_id);
}

[[nodiscard]] static inline taskid_t mapped_reads_on_device_count(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id)
{
  return mapped_readers_on_device_count(static_info, runtime_info, data_id, device_id);
}

[[nodiscard]] static inline bool mapped_reads_on_device_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, devid_t device_id)
{
  return mapped_readers_on_device_any(static_info, runtime_info, data_id, device_id);
}

[[nodiscard]] static inline taskid_t mapped_readers_before(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info, 
    dataid_t data_id, uint32_t gen, TaskIDList& out) 
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::filter_tasks(sub_spans.first, out, detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline bool mapped_readers_before_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::any_tasks(sub_spans.first, detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline taskid_t mapped_readers_before_on_device(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen, devid_t device_id, TaskIDList& out)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::filter_tasks(sub_spans.first, out,
                              detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_readers_before_on_device(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen, devid_t device_id)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::count_tasks(sub_spans.first,
                             detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_readers_before_on_device_count(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen, devid_t device_id)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::count_tasks(sub_spans.first,
                             detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline bool mapped_readers_before_on_device_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen, devid_t device_id)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::any_tasks(sub_spans.first,
                           detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_readers_after(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info, 
    dataid_t data_id, uint32_t gen, TaskIDList& out) 
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::filter_tasks(sub_spans.second, out, detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline bool mapped_readers_after_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::any_tasks(sub_spans.second, detail::is_exact_mapped(runtime_info));
}

[[nodiscard]] static inline taskid_t mapped_readers_after_on_device(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen, devid_t device_id, TaskIDList& out)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::filter_tasks(sub_spans.second, out,
                              detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_readers_after_on_device(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen, devid_t device_id)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::count_tasks(sub_spans.second,
                             detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t mapped_readers_after_on_device_count(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen, devid_t device_id)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::count_tasks(sub_spans.second,
                             detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline bool mapped_readers_after_on_device_any(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info,
    dataid_t data_id, uint32_t gen, devid_t device_id)
{
  const auto sub_spans = detail::split_readers_by_gen(static_info, data_id, gen);
  return detail::any_tasks(sub_spans.second,
                           detail::is_exact_mapped_on_device(runtime_info, device_id));
}

[[nodiscard]] static inline taskid_t oldest_mapped_writer(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info, dataid_t data_id, taskid_t& writer_gen) 
{
  //Smallest generation is oldest writer, so iterate from the front
  const auto tasks = static_info.get_tasks_writing_data_by_gen(data_id);
  const auto gens = static_info.get_write_generations_for_data(data_id);
  const auto is_mapped = detail::is_exact_mapped(runtime_info);
  
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    if (is_mapped(tasks[i])) {
      writer_gen = static_cast<taskid_t>(gens[i]);
      return tasks[i];
    }
  }
  return -1;
}

[[nodiscard]] static inline taskid_t youngest_mapped_writer(
    const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info, dataid_t data_id, taskid_t& writer_gen) 
{
  //Largest generation is youngest writer, so iterate from the back
  const auto tasks = static_info.get_tasks_writing_data_by_gen(data_id);
  const auto gens = static_info.get_write_generations_for_data(data_id);
  const auto is_mapped = detail::is_exact_mapped(runtime_info);
  
  for (std::size_t i = tasks.size(); i-- > 0;) {
    if (is_mapped(tasks[i])){
      writer_gen = static_cast<taskid_t>(gens[i]);
      return tasks[i];
    }
  }
  return -1;
}

[[nodiscard]] static inline bool any_on_device(const StaticTaskInfo& static_info, const RuntimeTaskInfo& runtime_info, TaskIDList& mapped_tasks, devid_t device_id) {
  for (const auto tid : mapped_tasks) {
    if (runtime_info.get_compute_task_mapped_device(tid) == device_id) {
      return true;
    }
  }
  return false;
}

} // namespace task_query
