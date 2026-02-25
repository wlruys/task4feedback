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
#include <cstdio>
#include <iostream>
#include <ostream>
#include <set>
#include <span>
#include <stack>
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

  // Sorted caches built during Graph::finalize() — reused by StaticTaskInfo constructor.
  std::vector<dataid_t> sorted_read_cache;
  std::vector<dataid_t> sorted_write_cache;
  std::vector<dataid_t> sorted_retire_cache;
  std::vector<taskid_t> sorted_recent_writer_cache;
  std::vector<uint32_t> sorted_read_gen_cache;
  std::vector<uint32_t> sorted_write_gen_cache;

  // Sorted caches for deps/dependents built during Graph::finalize().
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

  // Sorted caches built during Graph::finalize() — reused by StaticTaskInfo constructor.
  std::vector<taskid_t> sorted_dependencies_cache;
  std::vector<taskid_t> sorted_dependents_cache;
};

class Graph {
public:
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

  // Cached totals populated by build_sorted_dependency_caches() — used by StaticTaskInfo ctor
  // to avoid a separate counting pass.
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
    // Dense vector indexed by data_id; -1 = no writer yet.
    if (max_data_id < 0) return;
    std::vector<taskid_t> last_writer(static_cast<std::size_t>(max_data_id) + 1, -1);
    for (auto &task : tasks) {
      for (const auto data_id : task.read) {
        const taskid_t w = last_writer[static_cast<std::size_t>(data_id)];
        if (w != taskid_t(-1)) {
          add_dependency(task.id, w);
        }
      }
      for (const auto data_id : task.write) {
        last_writer[static_cast<std::size_t>(data_id)] = task.id;
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
    // sorted_read_cache and sorted_write_cache are built by populate_data_dependencies() above.
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

  void create_data_task(taskid_t task_id, dataid_t data_id, bool has_writer,
                        taskid_t writer_id = -1) {

    auto &task = tasks[task_id];
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%d_data_%d", task_id, data_id);
    auto data_task_id = add_data_task(std::string(buf), task_id, data_id);
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
    // Dense vectors indexed by data_id — O(1) lookup with no hashing overhead.
    // Enables auto-vectorization of the inner read/write loops.
    const std::size_t sz = (max_data_id >= 0) ? static_cast<std::size_t>(max_data_id) + 1 : 0;
    writers.assign(sz, taskid_t(-1));
    std::vector<uint32_t> read_gen_vec(sz, 0);
    std::vector<uint32_t> write_gen_vec(sz, 0);

    // Iterate in a valid topological order
    for (auto task_id : sorted) {

      auto &task = tasks[task_id];

      // Sort read set once; cache for reuse by StaticTaskInfo constructor and populate_unique_data.
      task.sorted_read_cache = as_sorted_vector(task.read);
      const auto &sorted_read = task.sorted_read_cache;

      task.sorted_recent_writer_cache.clear();
      task.sorted_recent_writer_cache.reserve(sorted_read.size());
      task.sorted_read_gen_cache.clear();
      task.sorted_read_gen_cache.reserve(sorted_read.size());

      for (const auto data_id : sorted_read) {
        const taskid_t writer_id = writers[static_cast<std::size_t>(data_id)];
        const bool has_writer = (writer_id != taskid_t(-1));
        task.sorted_recent_writer_cache.push_back(writer_id);
        task.sorted_read_gen_cache.push_back(read_gen_vec[static_cast<std::size_t>(data_id)]++);

        // Create data tasks for all reads from current task.
        if (create_data_tasks) {
          create_data_task(task_id, data_id, has_writer, writer_id);
        }
      }

      if (ensure_dependencies) {
        // Ensure that the compute task depends on all writers of the data it reads/writes/retires.
        for (const auto data_id : task.read) {
          const taskid_t w = writers[static_cast<std::size_t>(data_id)];
          if (w != taskid_t(-1)) {
            task.dependencies.insert(w);
            tasks[w].dependents.insert(task_id);
          }
        }
        for (const auto data_id : task.write) {
          const taskid_t w = writers[static_cast<std::size_t>(data_id)];
          if (w != taskid_t(-1)) {
            task.dependencies.insert(w);
            tasks[w].dependents.insert(task_id);
          }
        }
        for (const auto data_id : task.retire) {
          const taskid_t w = writers[static_cast<std::size_t>(data_id)];
          if (w != taskid_t(-1)) {
            task.dependencies.insert(w);
            tasks[w].dependents.insert(task_id);
          }
        }
      }

      // Sort write set once; cache for reuse by StaticTaskInfo constructor and populate_unique_data.
      task.sorted_write_cache = as_sorted_vector(task.write);
      const auto &sorted_write = task.sorted_write_cache;

      task.sorted_write_gen_cache.clear();
      task.sorted_write_gen_cache.reserve(sorted_write.size());

      // Update write generations and writers with current task's writes.
      for (const auto data_id : sorted_write) {
        const std::size_t idx = static_cast<std::size_t>(data_id);
        task.sorted_write_gen_cache.push_back(write_gen_vec[idx]++);
        writers[idx] = task_id;
      }
    }
  }

  void build_sorted_dependency_caches() {
    // Build sorted caches for compute tasks.
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
      task.sorted_retire_cache = as_sorted_vector(task.retire);

      total_compute_dependencies_cached += static_cast<int32_t>(task.sorted_dependencies_cache.size());
      total_compute_dependents_cached += static_cast<int32_t>(task.sorted_dependents_cache.size());
      total_compute_data_dependencies_cached += static_cast<int32_t>(task.sorted_data_dependencies_cache.size());
      total_compute_data_dependents_cached += static_cast<int32_t>(task.sorted_data_dependents_cache.size());
      total_reads_cached += static_cast<int32_t>(task.sorted_read_cache.size());
      total_writes_cached += static_cast<int32_t>(task.sorted_write_cache.size());
      total_retire_cached += static_cast<int32_t>(task.sorted_retire_cache.size());
      total_unique_cached += static_cast<int32_t>(task.unique.size());
    }

    // Build sorted caches for data tasks.
    total_data_task_dependencies_cached = 0;
    total_data_task_dependents_cached = 0;

    for (auto &dt : data_tasks) {
      dt.sorted_dependencies_cache = as_sorted_vector(dt.dependencies);
      dt.sorted_dependents_cache = as_sorted_vector(dt.dependents);
      total_data_task_dependencies_cached += static_cast<int32_t>(dt.sorted_dependencies_cache.size());
      total_data_task_dependents_cached += static_cast<int32_t>(dt.sorted_dependents_cache.size());
    }
  }

  void finalize(bool ensure_dependencies = false, bool create_data_tasks_flag = true) {
    if (finalized) {
      std::cerr << "Graph is already finalized. Cannot finalize again." << std::endl;
      std::cerr << "If you want to re-finalize, please create a new Graph instance." << std::endl;
      std::cerr << "Exiting..." << std::endl;
      std::exit(EXIT_FAILURE);
      return;
    }
    finalized = true;

    // Compute max_data_id first — required for dense-vector allocation in subsequent steps.
    max_data_id = -1;
    std::size_t total_read_count = 0;
    for (const auto &task : tasks) {
      for (const auto d : task.read)   { if (d > max_data_id) max_data_id = d; }
      for (const auto d : task.write)  { if (d > max_data_id) max_data_id = d; }
      for (const auto d : task.retire) { if (d > max_data_id) max_data_id = d; }
      total_read_count += task.read.size();
    }

    populate_dependencies_from_dataflow();
    populate_dependents();
    populate_initial_tasks();
    bfs();
    populate_depth();

    // Reserve data_tasks to avoid repeated reallocations.
    if (create_data_tasks_flag) {
      data_tasks.reserve(total_read_count);
    }

    populate_data_dependencies(ensure_dependencies, create_data_tasks_flag);
    // populate_unique_data uses sorted caches built by populate_data_dependencies above.
    populate_unique_data();
    // Build sorted caches for deps/dependents/retire after all sets are fully populated.
    // Also accumulates total counts used by StaticTaskInfo constructor.
    build_sorted_dependency_caches();
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

// Individual state bits — each flag is a distinct power-of-two bit.
namespace StateBits {
constexpr uint8_t SPAWNED   = 0x01;
constexpr uint8_t MAPPED    = 0x02;
constexpr uint8_t RESERVED  = 0x04;
constexpr uint8_t LAUNCHED  = 0x08;
constexpr uint8_t COMPLETED = 0x10;
} // namespace StateBits

// Cumulative state values stored in the SoA state arrays.
// Each value ORs in all prior bits so "at least MAPPED" == (state & StateBits::MAPPED) != 0.
namespace CumulativeState {
constexpr uint8_t SPAWNED   = StateBits::SPAWNED;
constexpr uint8_t MAPPED    = SPAWNED   | StateBits::MAPPED;    // 0x03
constexpr uint8_t RESERVED  = MAPPED    | StateBits::RESERVED;  // 0x07
constexpr uint8_t LAUNCHED  = RESERVED  | StateBits::LAUNCHED;  // 0x0F
constexpr uint8_t COMPLETED = LAUNCHED  | StateBits::COMPLETED; // 0x1F
} // namespace CumulativeState

// Readiness flags stored in ct_status — maintained incrementally as counters hit zero.
namespace StatusBits {
constexpr uint8_t MAPPABLE   = 0x01; // unmapped == 0 && state == SPAWNED
constexpr uint8_t RESERVABLE = 0x02; // unreserved == 0 && state == MAPPED
constexpr uint8_t LAUNCHABLE = 0x04; // incomplete == 0 && state == RESERVED
} // namespace StatusBits

class StaticTaskInfo {

protected:
  std::vector<ComputeTaskVariantInfo> compute_task_variant_info;
  std::vector<ComputeTaskStaticInfo> compute_task_static_info;

  int32_t num_compute_tasks_{0};
  int32_t num_data_tasks_{0};

  // CSR offsets for compute-task topology/data spans (size = n_compute_tasks + 1).
  std::vector<int32_t> ct_dependencies_offsets;
  std::vector<int32_t> ct_dependents_offsets;
  std::vector<int32_t> ct_data_dependencies_offsets;
  std::vector<int32_t> ct_data_dependents_offsets;
  std::vector<int32_t> ct_read_offsets;
  std::vector<int32_t> ct_write_offsets;
  std::vector<int32_t> ct_retire_offsets;
  std::vector<int32_t> ct_unique_offsets;

  // CSR offsets for data-task topology (size = n_data_tasks + 1).
  std::vector<int32_t> dt_dependencies_offsets;
  std::vector<int32_t> dt_dependents_offsets;

  std::vector<taskid_t> compute_task_dependencies;
  std::vector<taskid_t> compute_task_dependents;
  std::vector<taskid_t> compute_task_data_dependencies;
  std::vector<taskid_t> compute_task_data_dependents;
  std::vector<dataid_t> compute_task_read;
  std::vector<dataid_t> compute_task_write;
  std::vector<dataid_t> compute_task_retire;
  std::vector<taskid_t> compute_task_recent_writers;
  std::vector<uint32_t> compute_task_read_generations;
  std::vector<uint32_t> compute_task_write_generations;
  std::vector<dataid_t> compute_task_unique;

  // CSR cache for read usage: data_id -> compute tasks that read data_id
  std::vector<dataid_t> read_usage_data_ids;
  std::vector<int32_t> read_usage_offsets;
  std::vector<taskid_t> read_usage_tasks;
  // Dense row-index vector indexed by data_id; -1 means data_id has no readers.
  std::vector<int32_t> read_usage_row_by_data_id;

  // CSR cache for write usage: data_id -> compute tasks that write data_id
  std::vector<dataid_t> write_usage_data_ids;
  std::vector<int32_t> write_usage_offsets;
  std::vector<taskid_t> write_usage_tasks;
  // Dense row-index vector indexed by data_id; -1 means data_id has no writers.
  std::vector<int32_t> write_usage_row_by_data_id;

  // Secondary CSR: same rows as read/write_usage_*, but entries sorted by generation.
  // Shares offsets and row maps with the primary CSR above.
  std::vector<taskid_t> read_usage_by_gen_tasks;
  std::vector<uint32_t> read_usage_by_gen_generations;
  std::vector<taskid_t> write_usage_by_gen_tasks;
  std::vector<uint32_t> write_usage_by_gen_generations;

  // CSR cache for shared-read topology: compute task -> compute tasks sharing a read data id
  std::vector<int32_t> compute_task_shared_read_offsets;
  std::vector<taskid_t> compute_task_shared_read_neighbors;

  // Membership checks use binary search on sorted CSR spans (no extra storage needed).
  std::vector<taskid_t> data_task_dependencies;
  std::vector<taskid_t> data_task_dependents;

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

  // Binary search on sorted CSR spans replaces hash-set membership index.
  // No build_task_data_membership_index() needed.

public:
  StaticTaskInfo(int32_t num_compute_tasks, int32_t num_data_tasks) {
    num_compute_tasks_ = num_compute_tasks;
    num_data_tasks_ = num_data_tasks;
    compute_task_variant_info.resize(num_compute_tasks);
    compute_task_static_info.resize(num_compute_tasks);
    ct_dependencies_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_dependents_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_data_dependencies_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_data_dependents_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_read_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_write_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_retire_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_unique_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    dt_dependencies_offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);
    dt_dependents_offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);

    compute_task_names.resize(num_compute_tasks);
    data_task_names.resize(num_data_tasks);
    data_task_data_id_cache.resize(num_data_tasks, 0);
    data_task_compute_task_cache.resize(num_data_tasks, 0);

    // Keep CSR structures valid even when no precomputation has been run yet.
    read_usage_offsets.resize(1, 0);
    write_usage_offsets.resize(1, 0);
    compute_task_shared_read_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
  }

  StaticTaskInfo(Graph &graph) {

    taskid_t num_compute_tasks = graph.get_n_compute_tasks();
    taskid_t num_data_tasks = graph.get_n_data_tasks();

    num_compute_tasks_ = num_compute_tasks;
    num_data_tasks_ = num_data_tasks;
    compute_task_variant_info.resize(num_compute_tasks);
    compute_task_static_info.resize(num_compute_tasks);
    ct_dependencies_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_dependents_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_data_dependencies_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_data_dependents_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_read_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_write_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_retire_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    ct_unique_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);
    dt_dependencies_offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);
    dt_dependents_offsets.resize(static_cast<std::size_t>(num_data_tasks) + 1, 0);

    compute_task_names.resize(num_compute_tasks);
    data_task_names.resize(num_data_tasks);
    data_task_data_id_cache.resize(num_data_tasks, 0);
    data_task_compute_task_cache.resize(num_data_tasks, 0);

    read_usage_offsets.resize(1, 0);
    write_usage_offsets.resize(1, 0);
    compute_task_shared_read_offsets.resize(static_cast<std::size_t>(num_compute_tasks) + 1, 0);

    // std::cout << "Creating static graph..." << std::endl;
    // std::cout << "Number of compute tasks: " << num_compute_tasks << std::endl;
    // std::cout << "Number of data tasks: " << num_data_tasks << std::endl;

    auto &tasks = graph.tasks;
    auto &data_tasks = graph.data_tasks;
    // Fill slot [id+1] with span sizes, then prefix-sum in place.
    for (int32_t i = 0; i < num_compute_tasks; ++i) {
      const auto &task = tasks[static_cast<std::size_t>(i)];
      ct_dependencies_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_dependencies_cache.size());
      ct_dependents_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_dependents_cache.size());
      ct_data_dependencies_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_data_dependencies_cache.size());
      ct_data_dependents_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_data_dependents_cache.size());
      ct_read_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_read_cache.size());
      ct_write_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_write_cache.size());
      ct_retire_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.sorted_retire_cache.size());
      ct_unique_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(task.unique.size());
    }

    for (int32_t i = 0; i < num_data_tasks; ++i) {
      const auto &data_task = data_tasks[static_cast<std::size_t>(i)];
      dt_dependencies_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(data_task.sorted_dependencies_cache.size());
      dt_dependents_offsets[static_cast<std::size_t>(i) + 1] =
          static_cast<int32_t>(data_task.sorted_dependents_cache.size());
    }

    auto prefix_sum_offsets = [](std::vector<int32_t> &offsets) {
      for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
        offsets[i + 1] += offsets[i];
      }
    };
    prefix_sum_offsets(ct_dependencies_offsets);
    prefix_sum_offsets(ct_dependents_offsets);
    prefix_sum_offsets(ct_data_dependencies_offsets);
    prefix_sum_offsets(ct_data_dependents_offsets);
    prefix_sum_offsets(ct_read_offsets);
    prefix_sum_offsets(ct_write_offsets);
    prefix_sum_offsets(ct_retire_offsets);
    prefix_sum_offsets(ct_unique_offsets);
    prefix_sum_offsets(dt_dependencies_offsets);
    prefix_sum_offsets(dt_dependents_offsets);

    compute_task_dependencies.resize(static_cast<std::size_t>(ct_dependencies_offsets.back()));
    compute_task_dependents.resize(static_cast<std::size_t>(ct_dependents_offsets.back()));
    compute_task_data_dependencies.resize(static_cast<std::size_t>(ct_data_dependencies_offsets.back()));
    compute_task_data_dependents.resize(static_cast<std::size_t>(ct_data_dependents_offsets.back()));
    compute_task_read.resize(static_cast<std::size_t>(ct_read_offsets.back()));
    compute_task_recent_writers.resize(static_cast<std::size_t>(ct_read_offsets.back()));
    compute_task_read_generations.resize(static_cast<std::size_t>(ct_read_offsets.back()));
    compute_task_write.resize(static_cast<std::size_t>(ct_write_offsets.back()));
    compute_task_write_generations.resize(static_cast<std::size_t>(ct_write_offsets.back()));
    compute_task_retire.resize(static_cast<std::size_t>(ct_retire_offsets.back()));
    compute_task_unique.resize(static_cast<std::size_t>(ct_unique_offsets.back()));
    data_task_dependencies.resize(static_cast<std::size_t>(dt_dependencies_offsets.back()));
    data_task_dependents.resize(static_cast<std::size_t>(dt_dependents_offsets.back()));

    for (const auto &task : tasks) {
      const auto idx = static_cast<std::size_t>(task.id);
      compute_task_names[idx] = task.name;
      compute_task_static_info[idx].tag = task.tag;
      compute_task_static_info[idx].type = task.type;
      compute_task_static_info[idx].depth = task.depth;

      std::copy(task.sorted_dependencies_cache.begin(), task.sorted_dependencies_cache.end(),
                compute_task_dependencies.begin() + ct_dependencies_offsets[idx]);
      std::copy(task.sorted_dependents_cache.begin(), task.sorted_dependents_cache.end(),
                compute_task_dependents.begin() + ct_dependents_offsets[idx]);
      std::copy(task.sorted_data_dependencies_cache.begin(), task.sorted_data_dependencies_cache.end(),
                compute_task_data_dependencies.begin() + ct_data_dependencies_offsets[idx]);
      std::copy(task.sorted_data_dependents_cache.begin(), task.sorted_data_dependents_cache.end(),
                compute_task_data_dependents.begin() + ct_data_dependents_offsets[idx]);
      std::copy(task.sorted_read_cache.begin(), task.sorted_read_cache.end(),
                compute_task_read.begin() + ct_read_offsets[idx]);
      std::copy(task.sorted_recent_writer_cache.begin(), task.sorted_recent_writer_cache.end(),
                compute_task_recent_writers.begin() + ct_read_offsets[idx]);
      std::copy(task.sorted_read_gen_cache.begin(), task.sorted_read_gen_cache.end(),
                compute_task_read_generations.begin() + ct_read_offsets[idx]);
      std::copy(task.sorted_write_cache.begin(), task.sorted_write_cache.end(),
                compute_task_write.begin() + ct_write_offsets[idx]);
      std::copy(task.sorted_write_gen_cache.begin(), task.sorted_write_gen_cache.end(),
                compute_task_write_generations.begin() + ct_write_offsets[idx]);
      std::copy(task.sorted_retire_cache.begin(), task.sorted_retire_cache.end(),
                compute_task_retire.begin() + ct_retire_offsets[idx]);
      std::copy(task.unique.begin(), task.unique.end(),
                compute_task_unique.begin() + ct_unique_offsets[idx]);

      for (int i = 0; i < task.arch.size(); ++i) {
        const auto arch = static_cast<DeviceType>(task.arch[i]);
        add_compute_variant(task.id, arch, task.mem[i], task.vcu[i], task.time[i]);
      }
    }

    for (const auto &data_task : data_tasks) {
      const auto idx = static_cast<std::size_t>(data_task.id);
      data_task_names[idx] = data_task.name;
      data_task_data_id_cache[idx] = data_task.data_id;
      data_task_compute_task_cache[idx] = data_task.compute_task;

      std::copy(data_task.sorted_dependencies_cache.begin(), data_task.sorted_dependencies_cache.end(),
                data_task_dependencies.begin() + dt_dependencies_offsets[idx]);
      std::copy(data_task.sorted_dependents_cache.begin(), data_task.sorted_dependents_cache.end(),
                data_task_dependents.begin() + dt_dependents_offsets[idx]);
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

    // Flat triple: (data_id, gen, task_id).
    struct DataGenTask {
      dataid_t data_id;
      uint32_t gen;
      taskid_t task_id;
    };

    // Build flat arrays of triples from all tasks in one pass.
    std::vector<DataGenTask> read_triples;
    read_triples.reserve(compute_task_read.size());
    std::vector<DataGenTask> write_triples;
    write_triples.reserve(compute_task_write.size());

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

    // Comparators for the two sort passes.
    auto by_data_task = [](const DataGenTask &a, const DataGenTask &b) {
      if (a.data_id != b.data_id) return a.data_id < b.data_id;
      return a.task_id < b.task_id;
    };
    auto by_data_gen_task = [](const DataGenTask &a, const DataGenTask &b) {
      if (a.data_id != b.data_id) return a.data_id < b.data_id;
      if (a.gen != b.gen) return a.gen < b.gen;
      return a.task_id < b.task_id;
    };

    // Helper: compute max data_id across a sorted triples array (last element's data_id).
    auto max_data_id_in = [](const std::vector<DataGenTask> &triples) -> dataid_t {
      return triples.empty() ? dataid_t(-1) : triples.back().data_id;
    };

    // ---- Build read usage CSRs (two-sort, no per-group allocations) ----

    read_usage_data_ids.clear();
    read_usage_offsets.clear();
    read_usage_offsets.push_back(0);
    read_usage_tasks.clear();
    read_usage_tasks.reserve(read_triples.size());
    read_usage_by_gen_tasks.clear();
    read_usage_by_gen_tasks.reserve(read_triples.size());
    read_usage_by_gen_generations.clear();
    read_usage_by_gen_generations.reserve(read_triples.size());

    std::vector<uint64_t> shared_pair_keys;

    if (!read_triples.empty()) {
      // Pass 1: sort by (data_id, task_id) → primary CSR + shared pairs.
      std::sort(read_triples.begin(), read_triples.end(), by_data_task);

      // Compute max read data_id and size dense row map accordingly.
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
          read_usage_tasks.push_back(read_triples[i].task_id);
        }
        read_usage_offsets.push_back(static_cast<int32_t>(read_usage_tasks.size()));

        // Collect shared-read pairs while triples are task_id-sorted.
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

      // Pass 2: re-sort by (data_id, gen, task_id) → gen CSR.
      // Group structure (data_id groups and their sizes) is unchanged; only internal order differs.
      // No per-group allocation: just stream all re-sorted triples.
      std::sort(read_triples.begin(), read_triples.end(), by_data_gen_task);
      for (const auto &t : read_triples) {
        read_usage_by_gen_tasks.push_back(t.task_id);
        read_usage_by_gen_generations.push_back(t.gen);
      }
    }

    write_usage_data_ids.clear();
    write_usage_offsets.clear();
    write_usage_offsets.push_back(0);
    write_usage_tasks.clear();
    write_usage_tasks.reserve(write_triples.size());
    write_usage_by_gen_tasks.clear();
    write_usage_by_gen_tasks.reserve(write_triples.size());
    write_usage_by_gen_generations.clear();
    write_usage_by_gen_generations.reserve(write_triples.size());

    if (!write_triples.empty()) {
      // Pass 1: sort by (data_id, task_id) → primary CSR.
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
          write_usage_tasks.push_back(write_triples[i].task_id);
        }
        write_usage_offsets.push_back(static_cast<int32_t>(write_usage_tasks.size()));
        group_start = group_end;
      }

      // Pass 2: re-sort by (data_id, gen, task_id) → gen CSR.
      std::sort(write_triples.begin(), write_triples.end(), by_data_gen_task);
      for (const auto &t : write_triples) {
        write_usage_by_gen_tasks.push_back(t.task_id);
        write_usage_by_gen_generations.push_back(t.gen);
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
    auto write_offsets_scatter = compute_task_shared_read_offsets;

    for (const auto key : shared_pair_keys) {
      const auto lhs = static_cast<taskid_t>(key >> 32U);
      const auto rhs = static_cast<taskid_t>(key & 0xFFFFFFFFULL);
      compute_task_shared_read_neighbors[write_offsets_scatter[lhs]++] = rhs;
      compute_task_shared_read_neighbors[write_offsets_scatter[rhs]++] = lhs;
    }

    for (taskid_t task_id = 0; task_id < n_compute_tasks; ++task_id) {
      const auto begin = static_cast<std::size_t>(compute_task_shared_read_offsets[task_id]);
      const auto end = static_cast<std::size_t>(compute_task_shared_read_offsets[task_id + 1]);
      std::sort(compute_task_shared_read_neighbors.begin() + begin,
                compute_task_shared_read_neighbors.begin() + end);
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
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_dependencies_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_dependencies_offsets[idx + 1]);
    return std::span<const taskid_t>(compute_task_dependencies).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_dependents(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_dependents_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_dependents_offsets[idx + 1]);
    return std::span<const taskid_t>(compute_task_dependents).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const taskid_t> get_data_task_dependencies(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(dt_dependencies_offsets[idx]);
    const auto end = static_cast<std::size_t>(dt_dependencies_offsets[idx + 1]);
    return std::span<const taskid_t>(data_task_dependencies).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const taskid_t> get_data_task_dependents(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(dt_dependents_offsets[idx]);
    const auto end = static_cast<std::size_t>(dt_dependents_offsets[idx + 1]);
    return std::span<const taskid_t>(data_task_dependents).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_data_dependencies(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_data_dependencies_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_data_dependencies_offsets[idx + 1]);
    return std::span<const taskid_t>(compute_task_data_dependencies).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_data_dependents(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_data_dependents_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_data_dependents_offsets[idx + 1]);
    return std::span<const taskid_t>(compute_task_data_dependents).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const dataid_t> get_read_usage_data_ids() const {
    return read_usage_data_ids;
  }

  [[nodiscard]] std::span<const taskid_t> get_tasks_reading_data(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= read_usage_row_by_data_id.size()) return {};
    const auto row = read_usage_row_by_data_id[idx];
    if (row < 0) return {};
    const auto begin = static_cast<std::size_t>(read_usage_offsets[static_cast<std::size_t>(row)]);
    const auto end = static_cast<std::size_t>(read_usage_offsets[static_cast<std::size_t>(row) + 1]);
    return std::span<const taskid_t>(read_usage_tasks).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const dataid_t> get_write_usage_data_ids() const {
    return write_usage_data_ids;
  }

  [[nodiscard]] std::span<const taskid_t> get_tasks_writing_data(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= write_usage_row_by_data_id.size()) return {};
    const auto row = write_usage_row_by_data_id[idx];
    if (row < 0) return {};
    const auto begin = static_cast<std::size_t>(write_usage_offsets[static_cast<std::size_t>(row)]);
    const auto end = static_cast<std::size_t>(write_usage_offsets[static_cast<std::size_t>(row) + 1]);
    return std::span<const taskid_t>(write_usage_tasks).subspan(begin, end - begin);
  }

  // Secondary CSRs: same rows as the primary task-id-sorted CSRs above, but entries are
  // sorted ascending by generation.  The parallel generations span is always co-indexed
  // with the tasks span so callers can zip-iterate or binary-search on generation.

  [[nodiscard]] std::span<const taskid_t>
  get_tasks_reading_data_by_gen(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= read_usage_row_by_data_id.size()) return {};
    const auto row = read_usage_row_by_data_id[idx];
    if (row < 0) return {};
    const auto begin = static_cast<std::size_t>(read_usage_offsets[static_cast<std::size_t>(row)]);
    const auto end = static_cast<std::size_t>(read_usage_offsets[static_cast<std::size_t>(row) + 1]);
    return std::span<const taskid_t>(read_usage_by_gen_tasks).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const uint32_t>
  get_read_generations_for_data(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= read_usage_row_by_data_id.size()) return {};
    const auto row = read_usage_row_by_data_id[idx];
    if (row < 0) return {};
    const auto begin = static_cast<std::size_t>(read_usage_offsets[static_cast<std::size_t>(row)]);
    const auto end = static_cast<std::size_t>(read_usage_offsets[static_cast<std::size_t>(row) + 1]);
    return std::span<const uint32_t>(read_usage_by_gen_generations).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const taskid_t>
  get_tasks_writing_data_by_gen(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= write_usage_row_by_data_id.size()) return {};
    const auto row = write_usage_row_by_data_id[idx];
    if (row < 0) return {};
    const auto begin = static_cast<std::size_t>(write_usage_offsets[static_cast<std::size_t>(row)]);
    const auto end = static_cast<std::size_t>(write_usage_offsets[static_cast<std::size_t>(row) + 1]);
    return std::span<const taskid_t>(write_usage_by_gen_tasks).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const uint32_t>
  get_write_generations_for_data(dataid_t data_id) const {
    const auto idx = static_cast<std::size_t>(data_id);
    if (data_id < 0 || idx >= write_usage_row_by_data_id.size()) return {};
    const auto row = write_usage_row_by_data_id[idx];
    if (row < 0) return {};
    const auto begin = static_cast<std::size_t>(write_usage_offsets[static_cast<std::size_t>(row)]);
    const auto end = static_cast<std::size_t>(write_usage_offsets[static_cast<std::size_t>(row) + 1]);
    return std::span<const uint32_t>(write_usage_by_gen_generations).subspan(begin, end - begin);
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
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_read_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_read_offsets[idx + 1]);
    return std::span<const dataid_t>(compute_task_read).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const dataid_t> get_write(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_write_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_write_offsets[idx + 1]);
    return std::span<const dataid_t>(compute_task_write).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const dataid_t> get_retire(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_retire_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_retire_offsets[idx + 1]);
    return std::span<const dataid_t>(compute_task_retire).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const dataid_t> get_unique(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_unique_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_unique_offsets[idx + 1]);
    return std::span<const dataid_t>(compute_task_unique).subspan(begin, end - begin);
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
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_read_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_read_offsets[idx + 1]);
    return std::span<const taskid_t>(compute_task_recent_writers).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const uint32_t> get_read_generations(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_read_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_read_offsets[idx + 1]);
    return std::span<const uint32_t>(compute_task_read_generations).subspan(begin, end - begin);
  }

  [[nodiscard]] std::span<const uint32_t> get_write_generations(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    const auto begin = static_cast<std::size_t>(ct_write_offsets[idx]);
    const auto end = static_cast<std::size_t>(ct_write_offsets[idx + 1]);
    return std::span<const uint32_t>(compute_task_write_generations).subspan(begin, end - begin);
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

  [[nodiscard]] int32_t get_compute_task_dependency_count(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    return ct_dependencies_offsets[idx + 1] - ct_dependencies_offsets[idx];
  }

  [[nodiscard]] int32_t get_compute_task_data_dependency_count(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    return ct_data_dependencies_offsets[idx + 1] - ct_data_dependencies_offsets[idx];
  }

  [[nodiscard]] int32_t get_data_task_dependency_count(taskid_t id) const {
    const auto idx = static_cast<std::size_t>(id);
    return dt_dependencies_offsets[idx + 1] - dt_dependencies_offsets[idx];
  }

  [[nodiscard]] const ComputeTaskStaticInfo &get_compute_task_static_info(taskid_t id) const {
    return compute_task_static_info[id];
  }
};

class RuntimeTaskInfo {
protected:
  // ── Compute task SoA arrays ──────────────────────────────────
  // Hot: scanned in bulk for intersection / filtering
  std::vector<uint8_t> ct_state;        // cumulative state bits (CumulativeState::*)
  std::vector<uint8_t> ct_status;       // readiness flags (StatusBits::*)
  std::vector<uint8_t> ct_flags;        // user flags

  // Warm: touched during map/reserve/launch notifications
  std::vector<int16_t> ct_unmapped;
  std::vector<int16_t> ct_unreserved;
  std::vector<int16_t> ct_incomplete;

  // Cold: written once per phase transition
  std::vector<int32_t> ct_mapped_device;
  std::vector<int32_t> ct_reserve_priority;
  std::vector<int32_t> ct_launch_priority;

  // Time records (cold — written once, read during analysis)
  std::vector<timecount_t> ct_mapped_time;
  std::vector<timecount_t> ct_reserved_time;
  std::vector<timecount_t> ct_launched_time;
  std::vector<timecount_t> ct_completed_time;

  // ── Data task SoA arrays ─────────────────────────────────────
  std::vector<uint8_t> dt_state;
  std::vector<uint8_t> dt_flags;
  std::vector<int16_t> dt_incomplete;
  std::vector<int32_t> dt_source_device;
  std::vector<int32_t> dt_mapped_device;
  std::vector<int32_t> dt_launch_priority;
  std::vector<timecount_t> dt_launched_time;
  std::vector<timecount_t> dt_completed_time;

  // ── Eviction task SoA arrays (dynamically grown) ─────────────
  std::vector<uint8_t> et_state;
  std::vector<uint8_t> et_flags;
  std::vector<int32_t> et_data_id;
  std::vector<int32_t> et_evicting_on;
  std::vector<int32_t> et_compute_task;
  std::vector<int32_t> et_source_device;
  std::vector<int32_t> et_launch_priority;
  std::vector<timecount_t> et_launched_time;
  std::vector<timecount_t> et_completed_time;
  std::vector<std::string> et_names;

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
      ct_state = other.ct_state;
      ct_status = other.ct_status;
      ct_flags = other.ct_flags;
      ct_unmapped = other.ct_unmapped;
      ct_unreserved = other.ct_unreserved;
      ct_incomplete = other.ct_incomplete;
      ct_mapped_device = other.ct_mapped_device;
      ct_reserve_priority = other.ct_reserve_priority;
      ct_launch_priority = other.ct_launch_priority;
      ct_mapped_time = other.ct_mapped_time;
      ct_reserved_time = other.ct_reserved_time;
      ct_launched_time = other.ct_launched_time;
      ct_completed_time = other.ct_completed_time;
    }
    {
      ZoneScopedN("Copy DataTask SoA");
      dt_state = other.dt_state;
      dt_flags = other.dt_flags;
      dt_incomplete = other.dt_incomplete;
      dt_source_device = other.dt_source_device;
      dt_mapped_device = other.dt_mapped_device;
      dt_launch_priority = other.dt_launch_priority;
      dt_launched_time = other.dt_launched_time;
      dt_completed_time = other.dt_completed_time;
    }
    {
      ZoneScopedN("Copy EvictionTask SoA");
      et_state = other.et_state;
      et_flags = other.et_flags;
      et_data_id = other.et_data_id;
      et_evicting_on = other.et_evicting_on;
      et_compute_task = other.et_compute_task;
      et_source_device = other.et_source_device;
      et_launch_priority = other.et_launch_priority;
      et_launched_time = other.et_launched_time;
      et_completed_time = other.et_completed_time;
      et_names = other.et_names;
    }
    n_compute = other.n_compute;
    n_data = other.n_data;
  }

  // ── Bulk allocation ──────────────────────────────────────────

  void resize_compute(int32_t n) {
    ct_state.resize(n, 0);
    ct_status.resize(n, 0);
    ct_flags.resize(n, 0);
    ct_unmapped.resize(n, 0);
    ct_unreserved.resize(n, 0);
    ct_incomplete.resize(n, 0);
    ct_mapped_device.resize(n, -1);
    ct_reserve_priority.resize(n, 0);
    ct_launch_priority.resize(n, 0);
    ct_mapped_time.resize(n, 0);
    ct_reserved_time.resize(n, 0);
    ct_launched_time.resize(n, 0);
    ct_completed_time.resize(n, 0);
  }

  void resize_data(int32_t n) {
    dt_state.resize(n, 0);
    dt_flags.resize(n, 0);
    dt_incomplete.resize(n, 0);
    dt_source_device.resize(n, 0);
    dt_mapped_device.resize(n, -1);
    dt_launch_priority.resize(n, 0);
    dt_launched_time.resize(n, 0);
    dt_completed_time.resize(n, 0);
  }

  // ── Initialization ───────────────────────────────────────────

  void initialize_compute_runtime(int32_t id, const StaticTaskInfo &static_info) {
    ct_state[id] = CumulativeState::SPAWNED;
    const auto n_deps =
        static_cast<int16_t>(static_info.get_compute_task_dependency_count(id));
    const auto n_data_deps =
        static_cast<int16_t>(static_info.get_compute_task_data_dependency_count(id));
    ct_unmapped[id] = n_deps;
    ct_unreserved[id] = n_deps;
    ct_incomplete[id] = n_deps + n_data_deps;
    ct_status[id] = (n_deps == 0) ? StatusBits::MAPPABLE : 0;
  }

  void initialize_data_runtime(int32_t id, const StaticTaskInfo &static_info) {
    dt_state[id] = CumulativeState::SPAWNED;
    dt_incomplete[id] =
        static_cast<int16_t>(static_info.get_data_task_dependency_count(id));
  }

  int32_t add_eviction_task(int32_t compute_task_id, int32_t data_id,
                            int32_t evicting_on_device_id) {
    taskid_t id = static_cast<taskid_t>(et_state.size());
    et_state.push_back(CumulativeState::RESERVED);
    et_flags.push_back(0);
    et_data_id.push_back(data_id);
    et_evicting_on.push_back(evicting_on_device_id);
    et_compute_task.push_back(compute_task_id);
    et_source_device.push_back(0);
    et_launch_priority.push_back(0);
    et_launched_time.push_back(0);
    et_completed_time.push_back(0);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "EvictionTask_%d_%d_%d", compute_task_id, data_id,
                  evicting_on_device_id);
    et_names.emplace_back(buf);
    return id;
  }

  // ── State checks (single AND — SIMD-friendly) ───────────────

  [[nodiscard]] bool is_compute_mapped(taskid_t id) const {
    return (ct_state[id] & StateBits::MAPPED) != 0;
  }
  [[nodiscard]] bool is_compute_reserved(taskid_t id) const {
    return (ct_state[id] & StateBits::RESERVED) != 0;
  }
  [[nodiscard]] bool is_compute_launched(taskid_t id) const {
    return (ct_state[id] & StateBits::LAUNCHED) != 0;
  }
  [[nodiscard]] bool is_compute_completed(taskid_t id) const {
    return (ct_state[id] & StateBits::COMPLETED) != 0;
  }

  // ── Status checks (precomputed — no counter reads) ──────────

  [[nodiscard]] bool is_compute_mappable(taskid_t id) const {
    return (ct_status[id] & StatusBits::MAPPABLE) != 0;
  }
  [[nodiscard]] bool is_compute_reservable(taskid_t id) const {
    return (ct_status[id] & StatusBits::RESERVABLE) != 0;
  }
  [[nodiscard]] bool is_compute_launchable(taskid_t id) const {
    return (ct_status[id] & StatusBits::LAUNCHABLE) != 0;
  }

  [[nodiscard]] bool is_data_launchable(taskid_t id) const {
    return dt_incomplete[id] == 0 && dt_state[id] == CumulativeState::RESERVED;
  }
  [[nodiscard]] bool is_data_completed(taskid_t id) const {
    return (dt_state[id] & StateBits::COMPLETED) != 0;
  }
  [[nodiscard]] bool is_eviction_launchable(taskid_t id) const {
    return (et_state[id] & StateBits::RESERVED) != 0;
  }
  [[nodiscard]] bool is_eviction_completed(taskid_t id) const {
    return (et_state[id] & StateBits::COMPLETED) != 0;
  }

  [[nodiscard]] bool is_data_task_virtual(taskid_t id) const {
    return (dt_flags[id] & 0x01) != 0;
  }
  [[nodiscard]] bool is_eviction_task_virtual(taskid_t id) const {
    return (et_flags[id] & 0x01) != 0;
  }

  // ── Raw array access (for SIMD bulk operations) ─────────────

  [[nodiscard]] const uint8_t *compute_state_data() const { return ct_state.data(); }
  [[nodiscard]] const uint8_t *compute_status_data() const { return ct_status.data(); }
  [[nodiscard]] uint8_t *compute_state_data() { return ct_state.data(); }
  [[nodiscard]] uint8_t *compute_status_data() { return ct_status.data(); }

  // ── Getters ──────────────────────────────────────────────────

  [[nodiscard]] int32_t get_n_compute_tasks() const { return n_compute; }
  [[nodiscard]] int32_t get_n_data_tasks() const { return n_data; }
  [[nodiscard]] int32_t get_n_eviction_tasks() const {
    return static_cast<int32_t>(et_state.size());
  }
  [[nodiscard]] int32_t get_n_tasks() const {
    return get_n_compute_tasks() + get_n_data_tasks() + get_n_eviction_tasks();
  }
  [[nodiscard]] bool empty() const {
    return n_compute == 0 && n_data == 0 && et_state.empty();
  }

  [[nodiscard]] TaskState get_compute_task_state(taskid_t id) const {
    const auto s = ct_state[id];
    if (s & StateBits::COMPLETED) return TaskState::COMPLETED;
    if (s & StateBits::LAUNCHED)  return TaskState::LAUNCHED;
    if (s & StateBits::RESERVED)  return TaskState::RESERVED;
    if (s & StateBits::MAPPED)    return TaskState::MAPPED;
    return TaskState::SPAWNED;
  }

  [[nodiscard]] TaskState get_data_task_state(taskid_t id) const {
    const auto s = dt_state[id];
    if (s & StateBits::COMPLETED) return TaskState::COMPLETED;
    if (s & StateBits::LAUNCHED)  return TaskState::LAUNCHED;
    if (s & StateBits::RESERVED)  return TaskState::RESERVED;
    return TaskState::SPAWNED;
  }

  [[nodiscard]] TaskState get_eviction_task_state(taskid_t id) const {
    const auto s = et_state[id];
    if (s & StateBits::COMPLETED) return TaskState::COMPLETED;
    if (s & StateBits::LAUNCHED)  return TaskState::LAUNCHED;
    if (s & StateBits::RESERVED)  return TaskState::RESERVED;
    return TaskState::SPAWNED;
  }

  [[nodiscard]] int16_t get_compute_task_unmapped(taskid_t id) const { return ct_unmapped[id]; }
  [[nodiscard]] int16_t get_compute_task_unreserved(taskid_t id) const {
    return ct_unreserved[id];
  }
  [[nodiscard]] int16_t get_compute_task_incomplete(taskid_t id) const {
    return ct_incomplete[id];
  }
  [[nodiscard]] int32_t get_compute_task_mapped_device(taskid_t id) const {
    return ct_mapped_device[id];
  }
  [[nodiscard]] int32_t get_compute_task_reserve_priority(taskid_t id) const {
    return ct_reserve_priority[id];
  }
  [[nodiscard]] int32_t get_compute_task_launch_priority(taskid_t id) const {
    return ct_launch_priority[id];
  }
  [[nodiscard]] uint8_t get_compute_task_flags(taskid_t id) const { return ct_flags[id]; }

  [[nodiscard]] int32_t get_data_task_source_device(taskid_t id) const {
    return dt_source_device[id];
  }
  [[nodiscard]] int32_t get_data_task_mapped_device(taskid_t id) const {
    return dt_mapped_device[id];
  }
  [[nodiscard]] int32_t get_data_task_launch_priority(taskid_t id) const {
    return dt_launch_priority[id];
  }
  [[nodiscard]] uint8_t get_data_task_flags(taskid_t id) const { return dt_flags[id]; }

  [[nodiscard]] const std::string &get_eviction_task_name(taskid_t id) const {
    return et_names[id];
  }
  [[nodiscard]] int32_t get_eviction_task_evicting_on(taskid_t id) const {
    return et_evicting_on[id];
  }
  [[nodiscard]] int32_t get_eviction_task_data_id(taskid_t id) const { return et_data_id[id]; }
  [[nodiscard]] int32_t get_eviction_task_source_device(taskid_t id) const {
    return et_source_device[id];
  }

  // Time record getters
  [[nodiscard]] timecount_t get_compute_task_mapped_time(taskid_t id) const {
    return ct_mapped_time[id];
  }
  [[nodiscard]] timecount_t get_compute_task_reserved_time(taskid_t id) const {
    return ct_reserved_time[id];
  }
  [[nodiscard]] timecount_t get_compute_task_launched_time(taskid_t id) const {
    return ct_launched_time[id];
  }
  [[nodiscard]] timecount_t get_compute_task_completed_time(taskid_t id) const {
    return ct_completed_time[id];
  }
  [[nodiscard]] timecount_t get_data_task_launched_time(taskid_t id) const {
    return dt_launched_time[id];
  }
  [[nodiscard]] timecount_t get_data_task_completed_time(taskid_t id) const {
    return dt_completed_time[id];
  }
  [[nodiscard]] timecount_t get_eviction_task_launched_time(taskid_t id) const {
    return et_launched_time[id];
  }
  [[nodiscard]] timecount_t get_eviction_task_completed_time(taskid_t id) const {
    return et_completed_time[id];
  }

  [[nodiscard]] timecount_t get_compute_task_duration(taskid_t id) const {
    return ct_completed_time[id] - ct_launched_time[id];
  }
  [[nodiscard]] timecount_t get_data_task_duration(taskid_t id) const {
    return dt_completed_time[id] - dt_launched_time[id];
  }
  [[nodiscard]] timecount_t get_eviction_task_duration(taskid_t id) const {
    return et_completed_time[id] - et_launched_time[id];
  }

  [[nodiscard]] TaskState get_compute_task_state_at_time(taskid_t id,
                                                         timecount_t query) const {
    if (query < ct_mapped_time[id])    return TaskState::SPAWNED;
    if (query < ct_reserved_time[id])  return TaskState::MAPPED;
    if (query < ct_launched_time[id])  return TaskState::RESERVED;
    if (query < ct_completed_time[id]) return TaskState::LAUNCHED;
    return TaskState::COMPLETED;
  }

  [[nodiscard]] TaskState get_data_task_state_at_time(taskid_t id, timecount_t query) const {
    if (query < dt_launched_time[id])  return TaskState::RESERVED;
    if (query < dt_completed_time[id]) return TaskState::LAUNCHED;
    return TaskState::COMPLETED;
  }

  [[nodiscard]] TaskState get_eviction_task_state_at_time(taskid_t id,
                                                          timecount_t query) const {
    if (query < et_launched_time[id])  return TaskState::MAPPED;
    if (query < et_completed_time[id]) return TaskState::RESERVED;
    return TaskState::COMPLETED;
  }

  [[nodiscard]] TaskStatus get_compute_task_status(taskid_t id) const {
    const auto s = ct_status[id];
    if (s & StatusBits::LAUNCHABLE) return TaskStatus::LAUNCHABLE;
    if (s & StatusBits::RESERVABLE) return TaskStatus::RESERVABLE;
    if (s & StatusBits::MAPPABLE)   return TaskStatus::MAPPABLE;
    return TaskStatus::NONE;
  }

  // ── Setters ──────────────────────────────────────────────────

  void set_compute_task_state(taskid_t id, TaskState state) {
    switch (state) {
    case TaskState::SPAWNED:   ct_state[id] = CumulativeState::SPAWNED;   break;
    case TaskState::MAPPED:    ct_state[id] = CumulativeState::MAPPED;    break;
    case TaskState::RESERVED:  ct_state[id] = CumulativeState::RESERVED;  break;
    case TaskState::LAUNCHED:  ct_state[id] = CumulativeState::LAUNCHED;  break;
    case TaskState::COMPLETED: ct_state[id] = CumulativeState::COMPLETED; break;
    }
  }

  void set_data_task_state(taskid_t id, TaskState state) {
    switch (state) {
    case TaskState::SPAWNED:   dt_state[id] = CumulativeState::SPAWNED;   break;
    case TaskState::MAPPED:    dt_state[id] = CumulativeState::MAPPED;    break;
    case TaskState::RESERVED:  dt_state[id] = CumulativeState::RESERVED;  break;
    case TaskState::LAUNCHED:  dt_state[id] = CumulativeState::LAUNCHED;  break;
    case TaskState::COMPLETED: dt_state[id] = CumulativeState::COMPLETED; break;
    }
  }

  void set_eviction_task_state(taskid_t id, TaskState state) {
    switch (state) {
    case TaskState::SPAWNED:   et_state[id] = CumulativeState::SPAWNED;   break;
    case TaskState::MAPPED:    et_state[id] = CumulativeState::MAPPED;    break;
    case TaskState::RESERVED:  et_state[id] = CumulativeState::RESERVED;  break;
    case TaskState::LAUNCHED:  et_state[id] = CumulativeState::LAUNCHED;  break;
    case TaskState::COMPLETED: et_state[id] = CumulativeState::COMPLETED; break;
    }
  }

  void set_compute_task_unmapped(taskid_t id, int16_t v) { ct_unmapped[id] = v; }
  void set_compute_task_unreserved(taskid_t id, int16_t v) { ct_unreserved[id] = v; }
  void set_compute_task_incomplete(taskid_t id, int16_t v) { ct_incomplete[id] = v; }
  void set_compute_task_mapped_device(taskid_t id, int32_t v) { ct_mapped_device[id] = v; }
  void set_compute_task_reserve_priority(taskid_t id, int32_t v) { ct_reserve_priority[id] = v; }
  void set_compute_task_launch_priority(taskid_t id, int32_t v) { ct_launch_priority[id] = v; }
  void set_compute_task_flags(taskid_t id, uint8_t v) { ct_flags[id] = v; }

  void set_data_task_incomplete(taskid_t id, int16_t v) { dt_incomplete[id] = v; }
  void set_data_task_source_device(taskid_t id, int32_t v) { dt_source_device[id] = v; }
  void set_data_task_mapped_device(taskid_t id, int32_t v) { dt_mapped_device[id] = v; }
  void set_data_task_launch_priority(taskid_t id, int32_t v) { dt_launch_priority[id] = v; }
  void set_data_task_virtual(taskid_t id, bool v) {
    dt_flags[id] = v ? (dt_flags[id] | 0x01) : (dt_flags[id] & ~uint8_t{0x01});
  }

  void set_eviction_task_evicting_on(taskid_t id, int32_t v) { et_evicting_on[id] = v; }
  void set_eviction_task_compute_task(taskid_t id, int32_t v) { et_compute_task[id] = v; }
  void set_eviction_task_source_device(taskid_t id, int32_t v) { et_source_device[id] = v; }
  void set_eviction_task_data_id(taskid_t id, int32_t v) { et_data_id[id] = v; }
  void set_eviction_task_virtual(taskid_t id, bool v) {
    et_flags[id] = v ? (et_flags[id] | 0x01) : (et_flags[id] & ~uint8_t{0x01});
  }

  // Time recording
  void record_mapped(taskid_t id, timecount_t t) { ct_mapped_time[id] = t; }
  void record_reserved(taskid_t id, timecount_t t) { ct_reserved_time[id] = t; }
  void record_launched(taskid_t id, timecount_t t) { ct_launched_time[id] = t; }
  void record_completed(taskid_t id, timecount_t t) { ct_completed_time[id] = t; }
  void record_data_launched(taskid_t id, timecount_t t) { dt_launched_time[id] = t; }
  void record_data_completed(taskid_t id, timecount_t t) { dt_completed_time[id] = t; }
  void record_eviction_launched(taskid_t id, timecount_t t) { et_launched_time[id] = t; }
  void record_eviction_completed(taskid_t id, timecount_t t) { et_completed_time[id] = t; }

  // ── Counter decrements with status maintenance ───────────────
  // These update ct_status bits incrementally so callers never need
  // to recompute readiness from scratch.

  bool decrement_compute_task_unmapped(taskid_t id) {
    auto &v = ct_unmapped[id];
    const int16_t nv = --v;
    assert(nv >= 0 && "Unmapped count cannot be negative");
    if (nv == 0) {
      if (ct_state[id] == CumulativeState::SPAWNED) {
        ct_status[id] |= StatusBits::MAPPABLE;
        return true;
      }
    }
    return false;
  }

  bool decrement_compute_task_unreserved(taskid_t id) {
    auto &v = ct_unreserved[id];
    const int16_t nv = --v;
    assert(nv >= 0 && "Unreserved count cannot be negative");
    if (nv == 0) { // boundary only
      if (ct_state[id] == CumulativeState::MAPPED) {
        ct_status[id] |= StatusBits::RESERVABLE;
        return true;
      }
    }
    return false;
  }

  bool decrement_compute_task_incomplete(taskid_t id) {
    auto &v = ct_incomplete[id];
    const int16_t nv = --v;
    assert(nv >= 0 && "Incomplete count cannot be negative");
    if (nv == 0) { // boundary only
      if (ct_state[id] == CumulativeState::RESERVED) {
        ct_status[id] |= StatusBits::LAUNCHABLE;
        return true;
      }
    }
    return false;
  }

  bool decrement_data_task_incomplete(taskid_t id) {
    auto &v = dt_incomplete[id];
    const int16_t nv = --v;
    assert(nv >= 0 && "Data incomplete count cannot be negative");
    if (nv == 0) { // boundary only
      return dt_state[id] == CumulativeState::RESERVED;
    }
    return false;
  }

  // ── Notification methods ─────────────────────────────────────

  taskid_t compute_notify_mapped(taskid_t compute_task_id, devid_t mapped_device,
                                int32_t reserve_priority, int32_t launch_priority,
                                timecount_t time, const StaticTaskInfo &static_info,
                                TaskIDList &compute_task_buffer)
  {
    ct_mapped_device[compute_task_id] = mapped_device;
    ct_reserve_priority[compute_task_id] = reserve_priority;
    ct_launch_priority[compute_task_id]  = launch_priority;
    ct_state[compute_task_id] = CumulativeState::MAPPED;
    ct_mapped_time[compute_task_id] = time;
    ct_status[compute_task_id] &= static_cast<uint8_t>(~StatusBits::MAPPABLE);

    return collect_ready(static_info.get_compute_task_dependents(compute_task_id),
                        [&](taskid_t dep) { return decrement_compute_task_unmapped(dep); },
                        compute_task_buffer);
  }

taskid_t compute_notify_reserved(taskid_t compute_task_id, devid_t mapped_device,
                                 timecount_t time, const StaticTaskInfo &static_info,
                                 TaskIDList &compute_task_buffer)
{
  ct_mapped_device[compute_task_id] = mapped_device;
  ct_state[compute_task_id] = CumulativeState::RESERVED;
  ct_reserved_time[compute_task_id] = time;
  ct_status[compute_task_id] &= static_cast<uint8_t>(~StatusBits::RESERVABLE);

  return collect_ready(static_info.get_compute_task_dependents(compute_task_id),
                       [&](taskid_t dep) { return decrement_compute_task_unreserved(dep); },
                       compute_task_buffer);
}

  void compute_notify_launched(taskid_t compute_task_id, timecount_t time,
                               const StaticTaskInfo & /*static_info*/) {
    ct_state[compute_task_id] = CumulativeState::LAUNCHED;
    ct_launched_time[compute_task_id] = time;
    ct_status[compute_task_id] &= ~StatusBits::LAUNCHABLE;
  }

  taskid_t compute_notify_completed(taskid_t compute_task_id, timecount_t time,
                                    const StaticTaskInfo &static_info,
                                    TaskIDList &compute_task_buffer)
  {
    ct_state[compute_task_id] = CumulativeState::COMPLETED;
    ct_completed_time[compute_task_id] = time;

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
    dt_mapped_device[data_task_id] = mapped_device;
    dt_state[data_task_id] = CumulativeState::RESERVED;
  }

  void data_notify_launched(taskid_t data_task_id, devid_t source_device, timecount_t time,
                            const StaticTaskInfo &) {
    dt_state[data_task_id] = CumulativeState::LAUNCHED;
    dt_source_device[data_task_id] = source_device;
    dt_launched_time[data_task_id] = time;
  }

  taskid_t data_notify_completed(taskid_t data_task_id, timecount_t time,
                                const StaticTaskInfo &static_info,
                                TaskIDList &compute_task_buffer)
  {
    dt_state[data_task_id] = CumulativeState::COMPLETED;
    dt_completed_time[data_task_id] = time;

    return collect_ready(static_info.get_data_task_dependents(data_task_id),
                        [&](taskid_t ct) { return decrement_compute_task_incomplete(ct); },
                        compute_task_buffer);
  }

  void eviction_notify_reserved(taskid_t eviction_task_id, timecount_t,
                                const StaticTaskInfo &) {
    et_state[eviction_task_id] = CumulativeState::RESERVED;
  }

  void eviction_notify_launched(taskid_t eviction_task_id, devid_t source_device_id,
                                timecount_t time, const StaticTaskInfo &) {
    et_source_device[eviction_task_id] = source_device_id;
    et_state[eviction_task_id] = CumulativeState::LAUNCHED;
    et_launched_time[eviction_task_id] = time;
  }

  void eviction_notify_completed(taskid_t eviction_task_id, timecount_t time) {
    et_state[eviction_task_id] = CumulativeState::COMPLETED;
    et_completed_time[eviction_task_id] = time;
  }
};
