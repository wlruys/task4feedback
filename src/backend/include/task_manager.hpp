#pragma once
#include "resources.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <ctime>

using ComputeTaskList = std::vector<ComputeTask>;
using DataTaskList = std::vector<DataTask>;
using TaskList = std::vector<Task>;

class GraphManager;
class TaskManager;

class Tasks {
protected:
  const taskid_t num_compute_tasks;
  taskid_t current_task_id = 0;
  ComputeTaskList compute_tasks;
  DataTaskList data_tasks;
  std::vector<std::string> task_names;

  ComputeTaskList &get_compute_tasks() { return compute_tasks; }
  DataTaskList &get_data_tasks() { return data_tasks; }

  ComputeTask &get_compute_task(taskid_t id) { return compute_tasks[id]; }
  DataTask &get_data_task(taskid_t id) {
    return data_tasks[id - num_compute_tasks];
  }
  Task &get_task(taskid_t id);

public:
  Tasks(taskid_t num_compute_tasks);

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] std::size_t compute_size() const;
  [[nodiscard]] std::size_t data_size() const;
  [[nodiscard]] bool empty() const;
  [[nodiscard]] bool is_compute(taskid_t id) const;
  [[nodiscard]] bool is_data(taskid_t id) const;

  void add_compute_task(ComputeTask task);
  void add_data_task(DataTask task);

  void create_compute_task(taskid_t tid, std::string name,
                           TaskIDList dependencies);
  void add_variant(taskid_t id, DeviceType arch, vcu_t vcu, mem_t mem,
                   timecount_t time);
  void set_read(taskid_t id, DataIDList read);
  void set_write(taskid_t id, DataIDList write);

  [[nodiscard]] const ComputeTaskList &get_compute_tasks() const {
    return compute_tasks;
  }
  [[nodiscard]] const DataTaskList &get_data_tasks() const {
    return data_tasks;
  }

  [[nodiscard]] const ComputeTask &get_compute_task(taskid_t id) const {
    return compute_tasks[id];
  }
  [[nodiscard]] const DataTask &get_data_task(taskid_t id) const {
    return data_tasks[id - num_compute_tasks];
  }

  [[nodiscard]] const TaskIDList &get_dependencies(taskid_t id) const;
  [[nodiscard]] const TaskIDList &get_dependents(taskid_t id) const;
  [[nodiscard]] const VariantList &get_variants(taskid_t id) const;
  [[nodiscard]] const Variant &get_variant(taskid_t id, DeviceType arch) const;
  [[nodiscard]] const DataIDList &get_read(taskid_t id) const;
  [[nodiscard]] const DataIDList &get_write(taskid_t id) const;

  [[nodiscard]] const Resources &get_task_resources(taskid_t id) const;
  [[nodiscard]] const Resources &get_task_resources(taskid_t id,
                                                    DeviceType arch) const;

  [[nodiscard]] std::string const &get_name(taskid_t id) const {
    return task_names[id];
  }

  [[nodiscard]] std::vector<DeviceType>
  get_supported_architectures(taskid_t id) const;

  [[nodiscard]] const Task &get_task(taskid_t id) const;

  friend class GraphManager;
};

class TaskStateInfo {
protected:
  std::vector<TaskState> state;
  std::vector<DepCount> counts;

  void set_state(taskid_t id, TaskState _state) { state[id] = _state; }
  void set_unmapped(taskid_t id, depcount_t count);
  void set_unreserved(taskid_t id, depcount_t count);
  void set_incomplete(taskid_t id, depcount_t count);

  bool decrement_unmapped(taskid_t id);
  bool decrement_unreserved(taskid_t id);
  bool decrement_incomplete(taskid_t id);

public:
  // Store the task state
  std::vector<devid_t> mapping;

  PriorityList mapping_priority;
  PriorityList reserving_priority;
  PriorityList launching_priority;

  TaskStateInfo() = default;
  TaskStateInfo(std::size_t n_tasks);

  [[nodiscard]] TaskState get_state(taskid_t id) const { return state[id]; }

  [[nodiscard]] TaskStatus get_status(taskid_t id) const;

  [[nodiscard]] bool is_mappable(taskid_t id) const;
  [[nodiscard]] bool is_reservable(taskid_t id) const;
  [[nodiscard]] bool is_launchable(taskid_t id) const;

  bool is_mapped(taskid_t id) const;
  bool is_reserved(taskid_t id) const;
  bool is_launched(taskid_t id) const;

  [[nodiscard]] depcount_t get_unmapped(taskid_t id) const {
    return counts[id].unmapped;
  }
  [[nodiscard]] depcount_t get_unreserved(taskid_t id) const {
    return counts[id].unreserved;
  }
  [[nodiscard]] depcount_t get_incomplete(taskid_t id) const {
    return counts[id].incomplete;
  }

  void set_mapping(taskid_t id, devid_t devid) { mapping[id] = devid; }
  void set_mapping_priority(taskid_t id, priority_t p);
  void set_mapping_priority(PriorityList &ps) {
    mapping_priority = std::move(ps);
  }

  void set_reserving_priority(taskid_t id, priority_t p);
  void set_reserving_priority(PriorityList &ps) {
    reserving_priority = std::move(ps);
  }
  void set_launching_priority(taskid_t id, priority_t p);
  void set_launching_priority(PriorityList &ps) {
    launching_priority = std::move(ps);
  }

  [[nodiscard]] devid_t get_mapping(taskid_t id) const { return mapping[id]; };
  [[nodiscard]] priority_t get_mapping_priority(taskid_t id) const {
    return mapping_priority[id];
  };
  [[nodiscard]] const PriorityList &get_mapping_priorities() const {
    return mapping_priority;
  }
  [[nodiscard]] priority_t get_reserving_priority(taskid_t id) const {
    return reserving_priority[id];
  };
  [[nodiscard]] const PriorityList &get_reserving_priorities() const {
    return reserving_priority;
  }
  [[nodiscard]] priority_t get_launching_priority(taskid_t id) const {
    return launching_priority[id];
  };
  [[nodiscard]] const PriorityList &get_launching_priorities() const {
    return launching_priority;
  }

  friend class TaskManager;
};

class TaskRecords {

private:
  static std::size_t task_to_index(taskid_t id, std::size_t state_index) {
    return id * n_tracked_states + state_index;
  }

public:
  static constexpr std::size_t mapped_idx = 0;
  static constexpr std::size_t reserved_idx = 1;
  static constexpr std::size_t launched_idx = 2;
  static constexpr std::size_t completed_idx = 3;
  static constexpr std::size_t n_tracked_states = 4;

  std::vector<timecount_t> state_times;

  TaskRecords() = default;
  TaskRecords(std::size_t n_tasks) {
    state_times.resize(n_tasks * n_tracked_states);
  }

  void record_mapped(taskid_t id, timecount_t time);
  void record_reserved(taskid_t id, timecount_t time);
  void record_launched(taskid_t id, timecount_t time);
  void record_completed(taskid_t id, timecount_t time);

  [[nodiscard]] timecount_t get_mapped_time(taskid_t id) const;
  [[nodiscard]] timecount_t get_reserved_time(taskid_t id) const;
  [[nodiscard]] timecount_t get_launched_time(taskid_t id) const;
  [[nodiscard]] timecount_t get_completed_time(taskid_t id) const;
};

#define TASK_MANAGER_TASK_BUFFER_SIZE 10

class TaskManager {
private:
  void initialize_state();

public:
  Tasks &tasks;
  TaskStateInfo state;
  TaskRecords records;

  TaskIDList task_buffer;

  bool initialized = false;

  TaskManager(Tasks &tasks)
      : tasks(tasks), state(tasks.size()), records(tasks.size()){};
  [[nodiscard]] std::size_t size() const { return tasks.size(); }

  void initialize() {
    task_buffer.reserve(TASK_MANAGER_TASK_BUFFER_SIZE);
    initialize_state();
    initialized = true;
  }

  void set_mapping_priority(PriorityList &ps) {
    state.set_mapping_priority(ps);
  }

  [[nodiscard]] const TaskStateInfo &get_state() const { return state; }
  [[nodiscard]] const TaskRecords &get_records() const { return records; }
  [[nodiscard]] const Tasks &get_tasks() const { return tasks; }

  void set_mapping(taskid_t id, devid_t devid);
  const TaskIDList &notify_mapped(taskid_t id, timecount_t time);
  const TaskIDList &notify_reserved(taskid_t id, timecount_t time);
  void notify_launched(taskid_t id, timecount_t time);
  const TaskIDList &notify_completed(taskid_t id, timecount_t time);

  [[nodiscard]] const Variant &get_task_variant(taskid_t id,
                                                DeviceType arch) const {
    return tasks.get_variant(id, arch);
  }
  [[nodiscard]] const Resources &get_task_resources(taskid_t id,
                                                    DeviceType arch) const {
    return get_task_variant(id, arch).resources;
  }

  [[nodiscard]] Resources copy_task_resources(taskid_t id,
                                              DeviceType arch) const {
    return get_task_resources(id, arch);
  }

  void print_task(taskid_t id);
};

class TaskPrinter {
private:
  TaskManager &tm;

public:
  TaskPrinter(TaskManager &tm) : tm(tm) {}

  Color get_task_color(taskid_t id);

  template <typename DependencyList>
  Table make_list_table(DependencyList &dependencies);

  template <typename DependencyList>
  Table make_list_table(DependencyList &dependencies, std::string name);

  template <typename DependencyList>
  Table make_list_table_named(DependencyList &dependencies, std::string name);

  template <typename DataList>
  Table make_data_table(DataList &read, DataList &write);

  static Table make_variant_table(Variant v);

  template <typename VariantList> Table make_variant_tables(VariantList vlist);

  Table make_status_table(taskid_t id);

  [[nodiscard]] Table wrap_tables(
      const std::vector<std::function<tabulate::Table(taskid_t)>> &generators,
      taskid_t id) const;

  void print_tables(
      const std::vector<std::function<tabulate::Table(taskid_t)>> &generators,
      taskid_t id);

  static Table wrap_in_task_table(taskid_t id, tabulate::Table table);
};
