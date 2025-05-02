#pragma once
#include "graph.hpp"
#include "macros.hpp"
#include "noise.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <ctime>
#include <functional>

class TaskManager;

class TaskStateInfo {
protected:
  std::vector<TaskState> state;
  std::vector<DepCount> counts;
  std::vector<bool> is_virtual;
  std::vector<devid_t> sources;
  std::size_t n_compute_tasks;
  std::size_t n_data_tasks;

  void set_state(taskid_t id, TaskState _state) {
    state[id] = _state;
  }
  void set_unmapped(taskid_t id, depcount_t count);
  void set_unreserved(taskid_t id, depcount_t count);
  void set_incomplete(taskid_t id, depcount_t count);

  bool decrement_unmapped(taskid_t id);
  bool decrement_unreserved(taskid_t id);
  bool decrement_incomplete(taskid_t id);

  void set_data_task_virtual(taskid_t id) {
    assert(id >= n_compute_tasks);
    is_virtual.at(id - n_compute_tasks) = true;
  }
  void set_data_task_source(taskid_t id, devid_t source) {
    assert(id >= n_compute_tasks);
    sources.at(id - n_compute_tasks) = source;
  }

public:
  // Store the task state
  std::vector<devid_t> mapping;

  PriorityList mapping_priority;
  PriorityList reserving_priority;
  PriorityList launching_priority;

  std::vector<TaskIDList> eviction_dependencies;
  std::vector<TaskIDList> eviction_dependents;

  TaskStateInfo() = default;
  TaskStateInfo(const Tasks &tasks);

  TaskStateInfo(const TaskStateInfo &other) = default;
  TaskStateInfo &operator=(const TaskStateInfo &other) = default;

  [[nodiscard]] TaskState get_state(taskid_t id) const {
    return state.at(id);
  }

  [[nodiscard]] TaskStatus get_status(taskid_t id) const;

  [[nodiscard]] bool is_mappable(taskid_t id) const;
  [[nodiscard]] bool is_reservable(taskid_t id) const;
  [[nodiscard]] bool is_launchable(taskid_t id) const;

  [[nodiscard]] bool is_mapped(taskid_t id) const;
  [[nodiscard]] bool is_reserved(taskid_t id) const;
  [[nodiscard]] bool is_launched(taskid_t id) const;
  [[nodiscard]] bool is_completed(taskid_t id) const;

  [[nodiscard]] depcount_t get_unmapped(taskid_t id) const {
    return counts.at(id).unmapped;
  }
  [[nodiscard]] depcount_t get_unreserved(taskid_t id) const {
    return counts.at(id).unreserved;
  }
  [[nodiscard]] depcount_t get_incomplete(taskid_t id) const {
    return counts.at(id).incomplete;
  }

  void set_mapping(taskid_t id, devid_t devid) {
    assert(id < n_compute_tasks);
    mapping.at(id) = devid;
  }

  void set_reserving_priority(taskid_t id, priority_t p);
  void set_reserving_priority(PriorityList &ps) {
    reserving_priority = std::move(ps);
  }
  void set_launching_priority(taskid_t id, priority_t p);
  void set_launching_priority(PriorityList &ps) {
    launching_priority = std::move(ps);
  }

  [[nodiscard]] devid_t get_mapping(taskid_t id) const {
    return mapping[id];
  };

  [[nodiscard]] std::vector<devid_t> &get_mappings() {
    return mapping;
  }

  [[nodiscard]] priority_t get_reserving_priority(taskid_t id) const {
    assert(id < reserving_priority.size());
    return reserving_priority.at(id);
  };
  [[nodiscard]] const PriorityList &get_reserving_priorities() const {
    return reserving_priority;
  }
  [[nodiscard]] priority_t get_launching_priority(taskid_t id) const {
    assert(id < launching_priority.size());
    return launching_priority.at(id);
  };
  [[nodiscard]] const PriorityList &get_launching_priorities() const {
    return launching_priority;
  }

  [[nodiscard]] bool get_data_task_virtual(taskid_t id) const {
    assert(id >= n_compute_tasks);
    const auto offset = id - n_compute_tasks;
    return is_virtual.at(offset);
  }

  [[nodiscard]] devid_t get_data_task_source(taskid_t id) const {
    assert(id >= n_compute_tasks);
    const auto offset = id - n_compute_tasks;
    return sources.at(offset);
  }

  [[nodiscard]] std::size_t size() const {
    return state.size();
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
  TaskRecords(const Tasks &tasks) {
    state_times.resize(tasks.size() * n_tracked_states, 0);
  }

  TaskRecords(const TaskRecords &other) = default;
  TaskRecords &operator=(const TaskRecords &other) = default;

  void record_mapped(taskid_t id, timecount_t time);
  void record_reserved(taskid_t id, timecount_t time);
  void record_launched(taskid_t id, timecount_t time);
  void record_completed(taskid_t id, timecount_t time);

  [[nodiscard]] timecount_t get_mapped_time(taskid_t id) const;
  [[nodiscard]] timecount_t get_reserved_time(taskid_t id) const;
  [[nodiscard]] timecount_t get_launched_time(taskid_t id) const;
  [[nodiscard]] timecount_t get_completed_time(taskid_t id) const;
  [[nodiscard]] TaskState get_state_at_time(taskid_t id, timecount_t time) const;

  std::vector<timecount_t> &get_time_record() {
    return state_times;
  }
};

#define TASK_MANAGER_TASK_BUFFER_SIZE 10

class TaskManager {
private:
  void initialize_state();

public:
  std::reference_wrapper<Tasks> tasks;
  std::reference_wrapper<TaskNoise> noise;
  TaskStateInfo state;
  TaskRecords records;

  TaskIDList task_buffer;

  bool initialized = false;

  TaskManager(Tasks &tasks, TaskNoise &noise) : tasks(tasks), noise(noise) {};
  [[nodiscard]] std::size_t size() const {
    return tasks.get().size();
  }

  TaskManager(const TaskManager &other) = default;

  void initialize(bool create_data_tasks = false) {
    task_buffer.reserve(TASK_MANAGER_TASK_BUFFER_SIZE);
    // GraphManager::finalize(tasks, create_data_tasks);
    assert(tasks.get().is_initialized());
    initialize_state();
    initialized = true;
  }

  [[nodiscard]] priority_t get_mapping_priority(taskid_t id) const {
    return noise.get().get_priority(id);
  }

  [[nodiscard]] devid_t get_mapping(taskid_t id) const {
    return state.get_mapping(id);
  }

  [[nodiscard]] priority_t get_reserving_priority(taskid_t id) const {
    return state.get_reserving_priority(id);
  }
  [[nodiscard]] priority_t get_launching_priority(taskid_t id) const {
    return state.get_launching_priority(id);
  }

  [[nodiscard]] const PriorityList &get_mapping_priorities() const {
    return noise.get().get_priorities();
  }

  [[nodiscard]] const PriorityList &get_reserving_priorities() const {
    return state.get_reserving_priorities();
  }
  [[nodiscard]] const PriorityList &get_launching_priorities() const {
    return state.get_launching_priorities();
  }

  void set_mapping_priority(taskid_t id, priority_t p) {
    noise.get().set_priority(id, p);
  }

  void set_reserving_priority(taskid_t id, priority_t p) {
    state.set_reserving_priority(id, p);
  }

  void set_launching_priority(taskid_t id, priority_t p) {
    state.set_launching_priority(id, p);
  }

  [[nodiscard]] const TaskStateInfo &get_state() const {
    return state;
  }
  [[nodiscard]] const TaskRecords &get_records() const {
    return records;
  }
  [[nodiscard]] const Tasks &get_tasks() const {
    return tasks;
  }

  bool is_mapped(taskid_t id) const {
    return state.is_mapped(id);
  }

  bool is_reserved(taskid_t id) const {
    return state.is_reserved(id);
  }

  bool is_launched(taskid_t id) const {
    return state.is_launched(id);
  }

  bool is_completed(taskid_t id) const {
    return state.is_completed(id);
  }

  void set_state(taskid_t id, TaskState _state) {
    state.set_state(id, _state);
  }

  void set_mapping(taskid_t id, devid_t devid);
  void set_source(taskid_t id, devid_t source) {
    state.set_data_task_source(id, source);
  }
  void set_virtual(taskid_t id) {
    state.set_data_task_virtual(id);
  }

  [[nodiscard]] devid_t get_source(taskid_t id) const {
    return state.get_data_task_source(id);
  }

  [[nodiscard]] bool is_virtual(taskid_t id) const {
    return state.get_data_task_virtual(id);
  }

  const TaskIDList &notify_mapped(taskid_t id, timecount_t time);
  const TaskIDList &notify_reserved(taskid_t id, timecount_t time);
  void notify_launched(taskid_t id, timecount_t time);
  const TaskIDList &notify_completed(taskid_t id, timecount_t time);
  const TaskIDList &notify_data_completed(taskid_t id, timecount_t time);

  [[nodiscard]] const Variant &get_task_variant(taskid_t id, DeviceType arch) const {
    return tasks.get().get_variant(id, arch);
  }
  [[nodiscard]] const Resources &get_task_resources(taskid_t id, DeviceType arch) const {
    return get_task_variant(id, arch).resources;
  }

  [[nodiscard]] Resources copy_task_resources(taskid_t id, DeviceType arch) const {
    return get_task_resources(id, arch);
  }

  [[nodiscard]] timecount_t get_execution_time(taskid_t task_id, DeviceType arch) const {
    const auto &ctasks = get_tasks();
    assert(ctasks.is_compute(task_id));
    return noise.get().get(task_id, arch);
  }

  void print_task(taskid_t id);

  [[nodiscard]] bool is_data_task_virtual(taskid_t task_id) const {
    return state.get_data_task_virtual(task_id);
  }

  [[nodiscard]] devid_t get_data_task_source(taskid_t task_id) const {
    return state.get_data_task_source(task_id);
  }

  friend class SchedulerState;
};

class TaskPrinter {
private:
  std::reference_wrapper<TaskManager> tm;

public:
  TaskPrinter(TaskManager &tm) : tm(tm) {
  }

  [[nodiscard]] Color get_task_color(taskid_t id) const;

  template <typename DependencyList> Table make_list_table(DependencyList &dependencies);

  template <typename DependencyList>
  Table make_list_table(DependencyList &dependencies, std::string name);

  template <typename DependencyList>
  Table make_list_table_named(DependencyList &dependencies, std::string name);

  template <typename DataList> Table make_data_table(DataList &read, DataList &write);

  static Table make_variant_table(Variant v);

  template <typename VariantList> Table make_variant_tables(VariantList vlist);

  Table make_status_table(taskid_t id);

  [[nodiscard]] Table
  wrap_tables(const std::vector<std::function<tabulate::Table(taskid_t)>> &generators,
              taskid_t id) const;

  void print_tables(const std::vector<std::function<tabulate::Table(taskid_t)>> &generators,
                    taskid_t id);

  static Table wrap_in_task_table(taskid_t id, tabulate::Table table);
};
