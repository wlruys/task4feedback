#pragma once
#include "graph.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <ctime>

class TaskManager;

class TaskStateInfo {
protected:
  std::vector<TaskState> state;
  std::vector<DepCount> counts;
  std::vector<bool> existed;

  void set_state(taskid_t id, TaskState _state) { state[id] = _state; }
  void set_unmapped(taskid_t id, depcount_t count);
  void set_unreserved(taskid_t id, depcount_t count);
  void set_incomplete(taskid_t id, depcount_t count);

  bool decrement_unmapped(taskid_t id);
  bool decrement_unreserved(taskid_t id);
  bool decrement_incomplete(taskid_t id);

  void set_data_task_existed(taskid_t id) { existed[id] = true; }

public:
  // Store the task state
  std::vector<devid_t> mapping;

  PriorityList mapping_priority;
  PriorityList reserving_priority;
  PriorityList launching_priority;

  std::vector<TaskIDList> eviction_dependencies;
  std::vector<TaskIDList> eviction_dependents;

  TaskStateInfo() = default;
  TaskStateInfo(std::size_t n_tasks);

  [[nodiscard]] TaskState get_state(taskid_t id) const { return state[id]; }

  [[nodiscard]] TaskStatus get_status(taskid_t id) const;

  [[nodiscard]] bool is_mappable(taskid_t id) const;
  [[nodiscard]] bool is_reservable(taskid_t id) const;
  [[nodiscard]] bool is_launchable(taskid_t id) const;

  [[nodiscard]] bool is_mapped(taskid_t id) const;
  [[nodiscard]] bool is_reserved(taskid_t id) const;
  [[nodiscard]] bool is_launched(taskid_t id) const;

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

  [[nodiscard]] bool data_task_existed(taskid_t id) const {
    return existed[id];
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

  void initialize(bool create_data_tasks = false) {
    task_buffer.reserve(TASK_MANAGER_TASK_BUFFER_SIZE);
    GraphManager::finalize(tasks, create_data_tasks);
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
