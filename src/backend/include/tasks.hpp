#pragma once
#include "resources.hpp"
#include "settings.hpp"

#include "devices.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <ostream>
#include <string>
#include <tabulate/table.hpp>
#include <tabulate/tabulate.hpp>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace tabulate;

enum class TaskType { COMPUTE = 0, DATA = 1 };
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
  DeviceType arch = DeviceType::NONE;
  Resources resources;
  timecount_t time;

  Variant() = default;
  Variant(DeviceType arch_, vcu_t vcu_, mem_t mem_, timecount_t time_)
      : arch(arch_), resources(vcu_, mem_), time(time_) {}

  [[nodiscard]] vcu_t get_vcus() const { return resources.vcu; }
  [[nodiscard]] mem_t get_mem() const { return resources.mem; }

  [[nodiscard]] const Resources &get_resources() const { return resources; }

  [[nodiscard]] timecount_t get_execution_time() const { return time; }
};

using VariantList = std::array<Variant, num_device_types>;

class DepCount {
public:
  depcount_t unmapped;
  depcount_t unreserved;
  depcount_t incomplete;
};

class Task {
protected:
  TaskIDList dependencies;
  TaskIDList dependents;

public:
  taskid_t id;
  uint64_t depth;

  Task() = default;
  Task(taskid_t id) : id(id) {}

  void set_dependencies(TaskIDList _dependencies) {
    this->dependencies = std::move(_dependencies);
  }

  void set_dependents(TaskIDList _dependents) {
    this->dependents = std::move(_dependents);
  }

  [[nodiscard]] const TaskIDList &get_dependencies() const {
    return dependencies;
  }
  [[nodiscard]] const TaskIDList &get_dependents() const { return dependents; }

  void add_dependency(taskid_t dependency) {
    dependencies.push_back(dependency);
  }

  void add_dependent(taskid_t dependent) { dependents.push_back(dependent); }
};

class ComputeTask : public Task {
protected:
  TaskIDList data_dependencies;
  TaskIDList data_dependents;

  DataIDList read;
  DataIDList write;

  VariantList variants;

public:
  static constexpr TaskType task_type = TaskType::COMPUTE;

  ComputeTask() = default;
  ComputeTask(taskid_t id) { this->id = id; }

  void add_variant(DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time) {
    variants[static_cast<std::size_t>(arch)] = Variant(arch, vcu, mem, time);
  }

  Variant &get_variant(DeviceType arch) {
    return variants[static_cast<std::size_t>(arch)];
  }

  [[nodiscard]] const Variant &get_variant(DeviceType arch) const {
    return variants[static_cast<std::size_t>(arch)];
  }

  [[nodiscard]] const VariantList &get_variants() const { return variants; }

  void set_read(DataIDList _read) { this->read = std::move(_read); }
  void set_write(DataIDList _write) { this->write = std::move(_write); }

  [[nodiscard]] std::vector<DeviceType> get_supported_architectures() const;

  [[nodiscard]] const DataIDList &get_read() const { return read; }
  [[nodiscard]] const DataIDList &get_write() const { return write; }

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
};

class DataTask : public Task {
public:
  static constexpr TaskType task_type = TaskType::DATA;
  dataid_t data_id;

  DataTask() = default;
  DataTask(taskid_t id) { this->id = id; }
};

class MinimalTask {
public:
  taskid_t id;
  std::unordered_map<taskid_t, taskid_t> dependencies;
  std::vector<taskid_t> dependents;

  MinimalTask() = default;
  MinimalTask(taskid_t id) : id(id) {}

  MinimalTask(const MinimalTask &other) = default;

  MinimalTask(MinimalTask &&other) noexcept
      : id(std::exchange(other.id, 0)),
        dependencies(std::move(other.dependencies)),
        dependents(std::move(other.dependents)) {}

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