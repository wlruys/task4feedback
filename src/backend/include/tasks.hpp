#pragma once
#include "resources.hpp"
#include "settings.hpp"

#include <array>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cassert>

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

enum class TaskStatus {
  MAPPABLE = 0,
  RESERVABLE = 1,
  LAUNCHABLE = 2,
};
constexpr std::size_t num_task_statuses = 3;

class DepCount {
public:
  depcount_t unmapped;
  depcount_t unreserved;
  depcount_t incomplete;
};

class Task {
public:
  taskid_t id;
  TaskIDList dependencies;
  TaskIDList dependents;

  Task() = default;
  Task(taskid_t id) : id(id) {}
};

class ComputeTask : public Task {
public:
  static constexpr TaskType task_type = TaskType::COMPUTE;
  taskid_t id;
  TaskIDList dependencies;
  TaskIDList dependents;

  TaskIDList data_dependencies;
  TaskIDList data_dependents;

  VariantList variants;

  DataIDList read;
  DataIDList write;

  ComputeTask() = default;
  ComputeTask(taskid_t id) : id(id) {}
};

class DataTask : public Task {
public:
  static constexpr TaskType task_type = TaskType::DATA;
  taskid_t id;
  TaskIDList dependencies;
  TaskIDList dependents;

  dataid_t data_id;

  DataTask() = default;
  DataTask(taskid_t id) : id(id) {}
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

  MinimalTask(const ComputeTask &task) : id(task.id) {
    for (auto dep : task.dependencies) {
      dependencies[dep] = dep;
    }
    dependents.assign(task.dependents.begin(), task.dependents.end());
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

class TaskManager {
public:
  // Store the task objects
  std::vector<ComputeTask> tasks;
  std::vector<DataTask> data_tasks;

  // Store the task state
  std::vector<TaskState> state;
  std::vector<TaskState> data_state;

  std::vector<DepCount> counts;
  std::vector<DepCount> data_counts;

  std::vector<devid_t> mapping;

  TaskManager(std::size_t n) {
    tasks.resize(n);
    state.resize(n, TaskState::SPAWNED);
    counts.resize(n, DepCount());
    mapping.resize(n, 0);
  };

  TaskManager(std::vector<ComputeTask> &tasks) : tasks(tasks) {
    state.resize(tasks.size(), TaskState::SPAWNED);
    counts.resize(tasks.size(), DepCount());
    mapping.resize(tasks.size(), 0);
  }

  TaskManager(std::vector<ComputeTask> tasks, std::vector<TaskState> state,
              std::vector<DepCount> counts, std::vector<devid_t> mapping)
      : tasks(std::move(tasks)) {
    std::copy(state.begin(), state.end(), this->state.begin());
    std::copy(counts.begin(), counts.end(), this->counts.begin());
    std::copy(mapping.begin(), mapping.end(), this->mapping.begin());
  }

  TaskManager(TaskManager &tm) {
    tasks = tm.tasks;
    std::copy(tm.state.begin(), tm.state.end(), state.begin());
    std::copy(tm.counts.begin(), tm.counts.end(), counts.begin());
    std::copy(tm.mapping.begin(), tm.mapping.end(), mapping.begin());
  }

  [[nodiscard]] std::size_t size() const { return tasks.size(); }

  void add_task(taskid_t id, TaskIDList dependencies) {
    tasks[id] = ComputeTask(id);
    tasks[id].dependencies = std::move(dependencies);
  }

  void set_read(taskid_t id, DataIDList read) { tasks[id].read = read; }

  void set_write(taskid_t id, DataIDList write) { tasks[id].write = write; }

  void add_variant(taskid_t id, DeviceType arch, mem_t mem, vcu_t vcu,
                   timecount_t time) {
    auto arch_idx = static_cast<std::size_t>(arch);
    tasks[id].variants[arch_idx] = Variant{arch, vcu, mem, time};
  }

  TaskState set_state(taskid_t id, TaskState _state) {
    auto old_state = this->state[id];
    this->state[id] = _state;
    return old_state;
  }

  TaskState get_state(taskid_t id) { return this->state[id]; }

  void set_mapping(taskid_t id, devid_t devid) { mapping[id] = devid; }

  bool is_mappable(taskid_t id) {
    return counts[id].unmapped == 0 &&
           this->get_state(id) == TaskState::SPAWNED;
  }

  bool is_reservable(taskid_t id) {
    return counts[id].unreserved == 0 &&
           this->get_state(id) == TaskState::MAPPED;
  }

  bool is_launchable(taskid_t id) {
    return counts[id].incomplete == 0 &&
           this->get_state(id) == TaskState::RESERVED;
  }

  void decrement_unmapped(taskid_t id) {
    counts[id].unmapped--;
    assert(counts[id].unmapped >= 0);
  }

  void decrement_unreserved(taskid_t id) {
    counts[id].unreserved--;
    assert(counts[id].unreserved >= 0);
  }

  void decrement_incomplete(taskid_t id) {
    counts[id].incomplete--;
    assert(counts[id].incomplete >= 0);
  }

  void print_task(taskid_t id) {
    std::cout << "Task ID: " << tasks[id].id << std::endl;
    std::cout << "Dependencies: ";
    for (auto dep : tasks[id].dependencies) {
      std::cout << dep << " ";
    }
    std::cout << std::endl;

    std::cout << "Dependents: ";
    for (auto dep : tasks[id].dependents) {
      std::cout << dep << " ";
    }
    std::cout << std::endl;

    std::cout << "Data Dependencies: ";
    for (auto dep : tasks[id].data_dependencies) {
      std::cout << dep << " ";
    }
    std::cout << std::endl;

    std::cout << "Data Dependents: ";
    for (auto dep : tasks[id].data_dependents) {
      std::cout << dep << " ";
    }
    std::cout << std::endl;

    std::cout << "Variants: ";
    for (auto variant : tasks[id].variants) {
      if (variant.arch == DeviceType::NONE) {
        continue;
      }
      std::cout << "Device Type: " << variant.arch << " VCU: " << variant.vcu
                << " MEM: " << variant.mem << " TIME: " << variant.time
                << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Read: ";
    for (auto data : tasks[id].read) {
      std::cout << data << " ";
    }
    std::cout << std::endl;

    std::cout << "Write: ";
    for (auto data : tasks[id].write) {
      std::cout << data << " ";
    }
    std::cout << std::endl;

    std::cout << "State: " << static_cast<int>(state[id]) << std::endl;

    std::cout << "Mapping: " << mapping[id] << std::endl;

    std::cout << "Counts: " << counts[id].unmapped << " "
              << counts[id].unreserved << " " << counts[id].incomplete
              << std::endl;

    std::cout << std::endl;
  }
};