#pragma once
#include "resources.hpp"
#include "settings.hpp"

#include "devices.hpp"
#include "spdlog/spdlog.h"
#include <array>
#include <cassert>
#include <iostream>
#include <ostream>
#include <set>
#include <span>
#include <string>
#include <tabulate/table.hpp>
#include <tabulate/tabulate.hpp>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace tabulate;

enum class TaskType {
  COMPUTE = 0,
  DATA = 1
};
constexpr std::size_t num_task_types = 2;

enum class TaskState {
  SPAWNED = 0,
  MAPPED = 1,
  RESERVED = 2,
  LAUNCHED = 3,
  COMPLETED = 4
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

enum class TaskStatus {
  NONE = -1,
  MAPPABLE = 0,
  RESERVABLE = 1,
  LAUNCHABLE = 2,
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

class Variant {
public:
  Resources resources;
  timecount_t time = 0;
  DeviceType arch = DeviceType::NONE;

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

  // TODO: This REALLY needs to be renamed. This is the mean time.
  [[nodiscard]] timecount_t get_observed_time() const {
    return time;
  }
};

using VariantList = std::array<Variant, num_device_types>;

class DepCount {
public:
  depcount_t unmapped = 0;
  depcount_t unreserved = 0;
  depcount_t incomplete = 0;
};

class Task {
protected:
  TaskIDList dependencies;
  TaskIDList dependents;

public:
  taskid_t id = 0;
  uint64_t depth = 0;

  Task() = default;
  Task(taskid_t id) : id(id) {
  }

  void set_depth(uint64_t depth_) {
    this->depth = depth_;
  }

  [[nodiscard]] taskid_t get_id() const {
    return id;
  }
  [[nodiscard]] uint64_t get_depth() const {
    return depth;
  }

  void set_dependencies(TaskIDList _dependencies) {
    this->dependencies = std::move(_dependencies);
  }

  void set_dependents(TaskIDList _dependents) {
    this->dependents = std::move(_dependents);
  }

  [[nodiscard]] const TaskIDList &get_dependencies() const {
    return dependencies;
  }
  [[nodiscard]] const TaskIDList &get_dependents() const {
    return dependents;
  }

  void add_dependency(taskid_t dependency) {
    dependencies.push_back(dependency);
  }

  void add_dependent(taskid_t dependent) {
    dependents.push_back(dependent);
  }
};

class ComputeTask : public Task {
protected:
  TaskIDList data_dependencies;
  TaskIDList data_dependents;

  DataIDList read;
  DataIDList write;
  DataIDList unique;

  VariantList variants;

  int type = -1;
  int tag = -1;

public:
  static constexpr TaskType task_type = TaskType::COMPUTE;

  ComputeTask() = default;
  ComputeTask(taskid_t id) {
    this->id = id;
  }

  void add_variant(DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time) {
    variants.at(static_cast<std::size_t>(arch)) = Variant(arch, vcu, mem, time);
  }

  void add_variant(DeviceType arch, Variant variant) {
    variants.at(static_cast<std::size_t>(arch)) = variant;
  }

  Variant &get_variant(DeviceType arch) {
    return variants.at(static_cast<std::size_t>(arch));
  }

  [[nodiscard]] const Variant &get_variant(DeviceType arch) const {
    return variants.at(static_cast<std::size_t>(arch));
  }

  [[nodiscard]] const VariantList &get_variants() const {
    return variants;
  }
  [[nodiscard]] std::vector<Variant> get_variant_vector() const;

  void set_read(DataIDList _read) {
    this->read = std::move(_read);
  }
  void set_write(DataIDList _write) {
    this->write = std::move(_write);
  }

  void set_type(int type_) {
    this->type = type_;
  }

  void set_tag(int tag_) {
    this->tag = tag_;
  }

  [[nodiscard]] int get_type() const {
    return type;
  }

  [[nodiscard]] int get_tag() const {
    return tag;
  }

  [[nodiscard]] std::vector<DeviceType> get_supported_architectures() const;

  [[nodiscard]] const DataIDList &get_read() const {
    return read;
  }
  [[nodiscard]] const DataIDList &get_write() const {
    return write;
  }

  void add_data_dependency(taskid_t dependency) {
    data_dependencies.push_back(dependency);
  }

  void add_data_dependent(taskid_t dependent) {
    data_dependents.push_back(dependent);
  }

  [[nodiscard]] const TaskIDList &get_data_dependencies() const {
    return data_dependencies;
  }

  [[nodiscard]] const TaskIDList &get_data_dependents() const {
    return data_dependents;
  }

  void find_unique_data() {
    std::set<dataid_t> unique_set;
    for (auto data_id : read) {
      unique_set.insert(data_id);
    }
    for (auto data_id : write) {
      unique_set.insert(data_id);
    }
    unique.assign(unique_set.begin(), unique_set.end());
  }

  [[nodiscard]] const DataIDList &get_unique() const {
    return unique;
  }
};

class DataTask : public Task {
private:
  dataid_t data_id;
  taskid_t compute_task;

public:
  static constexpr TaskType task_type = TaskType::DATA;

  DataTask() = default;
  DataTask(taskid_t id_) {
    this->id = id_;
  }

  void set_data_id(dataid_t data_id_) {
    this->data_id = data_id_;
  }
  void set_compute_task(taskid_t compute_task_) {
    this->compute_task = compute_task_;
  }

  [[nodiscard]] taskid_t get_compute_task() const {
    return compute_task;
  }

  [[nodiscard]] dataid_t get_data_id() const {
    return data_id;
  }
};

class MinimalTask {
public:
  taskid_t id;
  std::unordered_map<taskid_t, taskid_t> dependencies;
  std::vector<taskid_t> dependents;

  MinimalTask() = default;
  MinimalTask(taskid_t id) : id(id) {
  }

  MinimalTask(const MinimalTask &other) = default;

  MinimalTask(MinimalTask &&other) noexcept
      : id(std::exchange(other.id, 0)), dependencies(std::move(other.dependencies)),
        dependents(std::move(other.dependents)) {
  }

  MinimalTask(const Task &task) : id(task.id) {
    const auto &task_dependencies = task.get_dependencies();
    const auto &task_dependents = task.get_dependents();

    for (auto dep : task_dependencies) {
      dependencies[dep] = dep;
    }
    dependents.assign(task_dependents.begin(), task_dependents.end());
  }

  // Copy assignment operator
  MinimalTask &operator=(const MinimalTask &other) {
    if (this != &other) {
      id = other.id;
      dependencies = other.dependencies;
      dependents = other.dependents;
    }
    return *this;
  }

  // Move assignment operator
  MinimalTask &operator=(MinimalTask &&other) noexcept {
    if (this != &other) {
      id = std::exchange(other.id, 0);
      dependencies = std::move(other.dependencies);
      dependents = std::move(other.dependents);
    }
    return *this;
  }
};

struct TaskTypeBundle {
  TaskType type;
  taskid_t id;
};

using ComputeTaskList = std::vector<ComputeTask>;
using DataTaskList = std::vector<DataTask>;
using TaskList = std::vector<Task>;
using MixedTaskIDList = std::vector<TaskTypeBundle>;

class GraphManager;

class Tasks {
protected:
  const taskid_t num_compute_tasks;
  taskid_t current_task_id = 0;
  ComputeTaskList compute_tasks;
  DataTaskList data_tasks;
  std::vector<std::string> task_names;
  mutable bool initialized = false;

  ComputeTaskList &get_compute_tasks() {
    return compute_tasks;
  }
  DataTaskList &get_data_tasks() {
    return data_tasks;
  }

  ComputeTask &get_compute_task(taskid_t id) {
    return compute_tasks[id];
  }
  DataTask &get_data_task(taskid_t id) {
    return data_tasks[id - num_compute_tasks];
  }
  Task &get_task(taskid_t id);

  void create_data_task(ComputeTask &task, bool has_writer, taskid_t writer_id, dataid_t data_id);

public:
  Tasks(taskid_t num_compute_tasks);

  [[nodiscard]] bool is_initialized() const {
    return initialized;
  }

  void set_initalized() const {
    assert(!initialized);
    initialized = true;
  }

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] std::size_t compute_size() const;
  [[nodiscard]] std::size_t data_size() const;
  [[nodiscard]] bool empty() const;
  [[nodiscard]] bool is_compute(taskid_t id) const;
  [[nodiscard]] bool is_data(taskid_t id) const;

  void add_compute_task(ComputeTask task);
  void add_data_task(DataTask task);

  void create_compute_task(taskid_t tid, std::string name, TaskIDList dependencies);
  void add_variant(taskid_t id, DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time);
  void set_read(taskid_t id, DataIDList read);
  void set_write(taskid_t id, DataIDList write);
  void set_type(taskid_t id, int type);
  void set_tag(taskid_t id, int tag);

  [[nodiscard]] int get_type(taskid_t id) const;
  [[nodiscard]] int get_tag(taskid_t id) const;

  [[nodiscard]] const ComputeTaskList &get_compute_tasks() const {
    return compute_tasks;
  }
  [[nodiscard]] const DataTaskList &get_data_tasks() const {
    return data_tasks;
  }

  [[nodiscard]] const ComputeTask &get_compute_task(taskid_t id) const {
    assert(id < num_compute_tasks);
    return compute_tasks.at(id);
  }
  [[nodiscard]] const DataTask &get_data_task(taskid_t id) const {
    assert(id >= num_compute_tasks);
    return data_tasks.at(id - num_compute_tasks);
  }

  [[nodiscard]] const TaskIDList &get_dependencies(taskid_t id) const;
  [[nodiscard]] const TaskIDList &get_dependents(taskid_t id) const;
  [[nodiscard]] const VariantList &get_variants(taskid_t id) const;
  [[nodiscard]] const Variant &get_variant(taskid_t id, DeviceType arch) const;
  [[nodiscard]] const DataIDList &get_read(taskid_t id) const;
  [[nodiscard]] const DataIDList &get_write(taskid_t id) const;

  [[nodiscard]] const Resources &get_task_resources(taskid_t id) const;
  [[nodiscard]] const Resources &get_task_resources(taskid_t id, DeviceType arch) const;

  [[nodiscard]] const TaskIDList &get_data_dependencies(taskid_t id) const;
  [[nodiscard]] const TaskIDList &get_data_dependents(taskid_t id) const;

  [[nodiscard]] std::size_t get_depth(taskid_t id) const;
  [[nodiscard]] dataid_t get_data_id(taskid_t id) const;

  [[nodiscard]] std::string const &get_name(taskid_t id) const {
    return task_names.at(id);
  }

  [[nodiscard]] std::vector<DeviceType> get_supported_architectures(taskid_t id) const;

  [[nodiscard]] std::vector<Variant> get_variant_vector(taskid_t id) const {
    return get_compute_task(id).get_variant_vector();
  }

  [[nodiscard]] const Task &get_task(taskid_t id) const;

  friend class GraphManager;
};