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

struct Mapping {
  bool mapped = false;
  devid_t device = 0;
};

class Variant : public Resources {
public:
  DeviceType arch = DeviceType::NONE;

  Variant() = default;
  Variant(DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time)
      : arch(arch), vcu(vcu), mem(mem), time(time) {}
};

using VariantList = std::array<Variant, num_device_types>;

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

  // Store the task names
  std::vector<std::string> task_names;
  std::vector<std::string> data_task_names;

  // Store the task state
  std::vector<TaskState> state;
  std::vector<TaskState> data_state;

  std::vector<DepCount> counts;
  std::vector<DepCount> data_counts;

  std::vector<Mapping> mapping;

  TaskManager(std::size_t n) {
    tasks.resize(n);
    task_names.resize(n);
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

  void initialize_counts() {
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      counts[i].unmapped =
          static_cast<depcount_t>(tasks[i].dependencies.size());
      counts[i].unreserved =
          static_cast<depcount_t>(tasks[i].dependencies.size());
      counts[i].incomplete =
          static_cast<depcount_t>(tasks[i].dependencies.size());
    }
  }

  void add_task(taskid_t id, std::string name, TaskIDList dependencies) {
    tasks[id] = ComputeTask(id);
    tasks[id].dependencies = std::move(dependencies);
    task_names[id] = std::move(name);
  }

  Task &get_task(taskid_t id) { return tasks[id]; }

  auto &get_name(taskid_t id) { return task_names[id]; }

  void set_read(taskid_t id, DataIDList read) {
    tasks[id].read = std::move(read);
  }

  void set_write(taskid_t id, DataIDList write) {
    tasks[id].write = std::move(write);
  }

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

  depcount_t get_unmapped(taskid_t id) { return counts[id].unmapped; }
  depcount_t get_unreserved(taskid_t id) { return counts[id].unreserved; }
  depcount_t get_incomplete(taskid_t id) { return counts[id].incomplete; }

  bool decrement_unmapped(taskid_t id) {
    counts[id].unmapped--;
    assert(counts[id].unmapped >= 0);
    return (counts[id].unmapped == 0);
  }

  bool decrement_unreserved(taskid_t id) {
    counts[id].unreserved--;
    assert(counts[id].unreserved >= 0);
    return (counts[id].unreserved == 0);
  }

  bool decrement_incomplete(taskid_t id) {
    counts[id].incomplete--;
    assert(counts[id].incomplete >= 0);
    return (counts[id].incomplete == 0);
  }

  TaskStatus get_status(taskid_t id) {
    if (is_launchable(id)) {
      return TaskStatus::LAUNCHABLE;
    }
    if (is_reservable(id)) {
      return TaskStatus::RESERVABLE;
    }
    if (is_mappable(id)) {
      return TaskStatus::MAPPABLE;
    }
    return TaskStatus::NONE;
  }

private:
  class TaskPrinter {
  private:
    TaskManager &tm;

  public:
    TaskPrinter(TaskManager &tm) : tm(tm) {}

    Color get_task_color(taskid_t id) {
      auto task_state = tm.get_state(id);
      if (task_state == TaskState::SPAWNED && !tm.is_mappable(id)) {
        return Color::white;
      }
      if (task_state == TaskState::SPAWNED) {
        return Color::blue;
      }

      if (task_state == TaskState::MAPPED && !tm.is_reservable(id)) {
        return Color::cyan;
      }
      if (task_state == TaskState::MAPPED) {
        return Color::magenta;
      }

      if (task_state == TaskState::RESERVED && !tm.is_launchable(id)) {
        return Color::yellow;
      }
      if (task_state == TaskState::RESERVED) {
        return Color::green;
      }

      if (task_state == TaskState::LAUNCHED) {
        return Color::red;
      }

      if (task_state == TaskState::COMPLETED) {
        return Color::grey;
      }
      return Color::white;
    }

    template <typename DependencyList>
    auto make_list_table(DependencyList &dependencies) {
      Table dep_table;
      Table::Row_t deps;
      for (auto dep : dependencies) {
        std::cout << dep << std::endl;
        deps.emplace_back(std::to_string(dep));
      }
      dep_table.add_row(deps);

      std::cout << dep_table << std::endl;
      return dep_table;
    }
    template <typename DependencyList>
    auto make_list_table(DependencyList &dependencies, std::string name) {
      Table dep_table;
      Table::Row_t deps;
      std::vector<Color> colors;
      deps.emplace_back(name);
      for (auto dep : dependencies) {
        deps.emplace_back(std::to_string(dep));
        colors.push_back(get_task_color(dep));
      }
      dep_table.add_row(deps);

      for (std::size_t i = 0; i < colors.size(); ++i) {
        dep_table[0][i + 1].format().font_color(colors[i]);
      }

      return dep_table;
    }

    template <typename DependencyList>
    auto make_list_table_named(DependencyList &dependencies, std::string name) {
      Table dep_table;
      Table::Row_t deps;
      std::vector<Color> colors;
      deps.emplace_back(name);
      for (auto dep : dependencies) {
        std::string dep_name = tm.get_name(dep);
        deps.emplace_back(dep_name);
        colors.push_back(get_task_color(dep));
      }
      dep_table.add_row(deps);

      for (std::size_t i = 0; i < colors.size(); ++i) {
        dep_table[0][i + 1].format().font_color(colors[i]);
      }

      return dep_table;
    }

    template <typename DataList>
    auto make_data_table(DataList &read, DataList &write) {
      Table data_table;
      Table::Row_t read_row;
      Table::Row_t write_row;

      read_row.emplace_back("Read");
      write_row.emplace_back("Write");

      for (auto data : read) {
        read_row.emplace_back(std::to_string(data));
      }
      for (auto data : write) {
        write_row.emplace_back(std::to_string(data));
      }
      data_table.add_row(read_row);
      data_table.add_row(write_row);
      return data_table;
    }

    auto make_variant_table(Variant v) {

      Table device_table;
      Table::Row_t device_row;
      Table::Row_t variant_row;

      device_row.emplace_back(to_string(v.arch));

      Table variant_table;
      Table::Row_t header_row;
      Table::Row_t value_row;

      header_row.emplace_back("VCU");
      header_row.emplace_back("MEM");
      header_row.emplace_back("TIME");

      value_row.emplace_back(std::to_string(v.vcu));
      value_row.emplace_back(std::to_string(v.mem));
      value_row.emplace_back(std::to_string(v.time));

      variant_table.add_row(header_row);
      variant_table.add_row(value_row);

      variant_row.emplace_back(variant_table);

      device_table.add_row(device_row);
      device_table.add_row(variant_row);

      return device_table;
    }

    template <typename VariantList>
    auto make_variant_tables(VariantList vlist) {
      Table variant_table;
      Table::Row_t variants;

      for (auto &v : vlist) {
        if (v.arch == DeviceType::NONE) {
          continue;
        }
        auto table = make_variant_table(v);
        variants.push_back(table);
      }
      variant_table.add_row(variants);

      return variant_table;
    }

    auto make_status_table(taskid_t id) {
      Table status_table;

      Table::Row_t header_row = {"ID",        "Name",     "State",
                                 "Status",    "Unmapped", "Unreserved",
                                 "Incomplete"};

      Table::Row_t value_row = {std::to_string(id),
                                tm.get_name(id),
                                to_string(tm.get_state(id)),
                                to_string(tm.get_status(id)),
                                std::to_string(tm.get_unmapped(id)),
                                std::to_string(tm.get_unreserved(id)),
                                std::to_string(tm.get_incomplete(id))};

      status_table.add_row(header_row);
      status_table.add_row(value_row);

      return status_table;
    }

    auto wrap_tables(
        const std::vector<std::function<tabulate::Table(taskid_t)>> &generators,
        taskid_t id) {
      Table table;

      Table::Row_t task_name;
      task_name.emplace_back("Task " + tm.get_name(id));
      table.add_row(task_name);

      for (const auto &generator : generators) {
        Table::Row_t inner_row;
        auto inner_table = generator(id);
        inner_row.emplace_back(inner_table);
        table.add_row(inner_row);
      }

      return table;
    }

    auto print_tables(
        const std::vector<std::function<tabulate::Table(taskid_t)>> &generators,
        taskid_t id) {
      Table table;

      Table::Row_t task_name;
      task_name.emplace_back("Task " + tm.get_name(id));
      table.add_row(task_name);
      table.format()
          .font_color(get_task_color(id))
          .border_top("+")
          .border_bottom("+")
          .border_left("+")
          .border_right("+");

      std::cout << table << std::endl;

      for (const auto &generator : generators) {
        Table::Row_t inner_row;
        auto inner_table = generator(id);
        if (inner_table.size() > 0) {
          if (inner_table[0].size() == 1) {
            continue;
          }
          std::cout << inner_table << std::endl;
        }
      }
    }

    static auto wrap_in_task_table(taskid_t id, tabulate::Table table) {
      Table task_table;
      Table::Row_t task_row;
      Table::Row_t table_row;
      task_row.emplace_back("Task " + std::to_string(id));
      table_row.emplace_back(table);

      task_table.add_row(task_row);
      task_table.add_row(table_row);

      return task_table;
    }
  };

public:
  void print_task(taskid_t id) {
    TaskPrinter printer(*this);

    auto status_table_generator = [&](taskid_t id) {
      return printer.make_status_table(id);
    };

    auto dependency_table_generator = [&](taskid_t id) {
      return printer.make_list_table_named(tasks[id].dependencies,
                                           "Dependencies");
    };

    auto dependent_table_generator = [&](taskid_t id) {
      return printer.make_list_table_named(tasks[id].dependents, "Dependents");
    };

    auto data_table_generator = [&](taskid_t id) {
      return printer.make_data_table(tasks[id].read, tasks[id].write);
    };

    auto variant_table_generator = [&](taskid_t id) {
      return printer.make_variant_tables(tasks[id].variants);
    };

    printer.print_tables({status_table_generator, dependency_table_generator,
                          dependent_table_generator, data_table_generator,
                          variant_table_generator},
                         id);
  }
};
