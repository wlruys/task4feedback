#include "include/task_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"

// TaskStateInfo

TaskStateInfo::TaskStateInfo(const Tasks &tasks) {
  auto n = tasks.size();
  n_compute_tasks = tasks.compute_size();
  n_data_tasks = tasks.data_size();
  state.resize(n, TaskState::SPAWNED);
  counts.resize(n, DepCount());
  mapping.resize(n_compute_tasks, -1);
  // mapping_priority.resize(n, 0);
  reserving_priority.resize(n, 0);
  launching_priority.resize(n, 0);
  is_virtual.resize(n, false);
  sources.resize(n, 0);
}

TaskStatus TaskStateInfo::get_status(taskid_t id) const {
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

bool TaskStateInfo::is_mappable(taskid_t id) const {
  return counts[id].unmapped == 0 && this->get_state(id) == TaskState::SPAWNED;
}

bool TaskStateInfo::is_reservable(taskid_t id) const {
  return counts[id].unreserved == 0 && this->get_state(id) == TaskState::MAPPED;
}

bool TaskStateInfo::is_launchable(taskid_t id) const {
  return counts[id].incomplete == 0 && this->get_state(id) == TaskState::RESERVED;
}

bool TaskStateInfo::is_mapped(taskid_t id) const {
  return this->get_state(id) >= TaskState::MAPPED;
}

bool TaskStateInfo::is_reserved(taskid_t id) const {
  return this->get_state(id) >= TaskState::RESERVED;
}

bool TaskStateInfo::is_launched(taskid_t id) const {
  return this->get_state(id) >= TaskState::LAUNCHED;
}

bool TaskStateInfo::is_completed(taskid_t id) const {
  return this->get_state(id) >= TaskState::COMPLETED;
}

bool TaskStateInfo::decrement_unmapped(taskid_t id) {
  counts[id].unmapped--;
  assert(counts[id].unmapped >= 0);
  // return if the task just became mappable
  return (counts[id].unmapped == 0) && (state[id] == TaskState::SPAWNED);
}

bool TaskStateInfo::decrement_unreserved(taskid_t id) {
  counts[id].unreserved--;
  assert(counts[id].unreserved >= 0);
  // return if the task just became reservable
  return (counts[id].unreserved == 0) && (state[id] == TaskState::MAPPED);
}

bool TaskStateInfo::decrement_incomplete(taskid_t id) {
  counts[id].incomplete--;
  assert(counts[id].incomplete >= 0);
  // return if the task just became launchable
  return (counts[id].incomplete == 0) && (state[id] == TaskState::RESERVED);
}

// void TaskStateInfo::set_mapping_priority(taskid_t id, priority_t priority) {
//   mapping_priority[id] = priority;
// }

void TaskStateInfo::set_reserving_priority(taskid_t id, priority_t priority) {
  reserving_priority[id] = priority;
}

void TaskStateInfo::set_launching_priority(taskid_t id, priority_t priority) {
  launching_priority[id] = priority;
}

void TaskStateInfo::set_unmapped(taskid_t id, depcount_t count) {
  counts[id].unmapped = count;
}

void TaskStateInfo::set_unreserved(taskid_t id, depcount_t count) {
  counts[id].unreserved = count;
}

void TaskStateInfo::set_incomplete(taskid_t id, depcount_t count) {
  counts[id].incomplete = count;
}

// TaskRecords

void TaskRecords::record_mapped(taskid_t id, timecount_t time) {
  auto index = task_to_index(id, mapped_idx);
  state_times[index] = time;
}

void TaskRecords::record_reserved(taskid_t id, timecount_t time) {
  auto index = task_to_index(id, reserved_idx);
  state_times[index] = time;
}

void TaskRecords::record_launched(taskid_t id, timecount_t time) {
  auto index = task_to_index(id, launched_idx);
  state_times[index] = time;
}

void TaskRecords::record_completed(taskid_t id, timecount_t time) {
  auto index = task_to_index(id, completed_idx);
  state_times[index] = time;
}

timecount_t TaskRecords::get_mapped_time(taskid_t id) const {
  auto index = task_to_index(id, mapped_idx);
  return state_times[index];
}

timecount_t TaskRecords::get_reserved_time(taskid_t id) const {
  auto index = task_to_index(id, reserved_idx);
  return state_times[index];
}

timecount_t TaskRecords::get_launched_time(taskid_t id) const {
  auto index = task_to_index(id, launched_idx);
  return state_times[index];
}

timecount_t TaskRecords::get_completed_time(taskid_t id) const {
  auto index = task_to_index(id, completed_idx);
  return state_times[index];
}

TaskState TaskRecords::get_state_at_time(taskid_t id, timecount_t query_time) const {
  if (query_time < get_mapped_time(id)) {
    return TaskState::SPAWNED;
  }
  if (query_time < get_reserved_time(id)) {
    return TaskState::MAPPED;
  }
  if (query_time < get_launched_time(id)) {
    return TaskState::RESERVED;
  }
  if (query_time < get_completed_time(id)) {
    return TaskState::LAUNCHED;
  }
  return TaskState::COMPLETED;
}

// Task Manager
void TaskManager::initialize_state() {
  ZoneScoped;
  const auto &const_tasks = this->tasks.get();
  state = TaskStateInfo(tasks);
  records = TaskRecords(tasks);

  const auto &compute_tasks = const_tasks.get_compute_tasks();
  for (const auto &task : compute_tasks) {
    taskid_t id = task.id;
    auto n_deps = static_cast<depcount_t>(task.get_dependencies().size());
    state.set_state(id, TaskState::SPAWNED);
    state.set_unmapped(id, n_deps);
    state.set_unreserved(id, n_deps);

    auto n_data_deps = static_cast<depcount_t>(task.get_data_dependencies().size());
    auto n_total_deps = n_deps + n_data_deps;

    state.set_incomplete(id, n_total_deps);
  }

  for (const auto &task : const_tasks.get_data_tasks()) {
    taskid_t id = task.id;
    auto n_deps = static_cast<depcount_t>(task.get_dependencies().size());
    state.set_state(id, TaskState::SPAWNED);
    state.set_unmapped(id, 0);
    state.set_unreserved(id, 0);
    state.set_incomplete(id, n_deps);
  }
}

void TaskManager::set_mapping(taskid_t id, devid_t devid) {
  state.set_mapping(id, devid);
}

const TaskIDList &TaskManager::notify_mapped(taskid_t id, timecount_t time) {
  ZoneScoped;
  const auto &task_objects = get_tasks();

  records.record_mapped(id, time);
  state.set_state(id, TaskState::MAPPED);

  // clear exising task buffer
  task_buffer.clear();

  for (auto dependent_id : task_objects.get_dependents(id)) {
    if (state.decrement_unmapped(dependent_id)) {
      task_buffer.push_back(dependent_id);
      assert(state.get_state(dependent_id) == TaskState::SPAWNED);
      assert(task_objects.is_compute(dependent_id));
    }
  }

  return task_buffer;
}

const TaskIDList &TaskManager::notify_reserved(taskid_t id, timecount_t time) {
  ZoneScoped;
  const auto &task_objects = get_tasks();

  records.record_reserved(id, time);
  state.set_state(id, TaskState::RESERVED);

  // clear existing task buffer
  task_buffer.clear();

  for (auto dependent_id : task_objects.get_dependents(id)) {
    if (state.decrement_unreserved(dependent_id)) {
      task_buffer.push_back(dependent_id);
      assert(state.get_state(dependent_id) == TaskState::MAPPED);
      assert(task_objects.is_compute(dependent_id));
    }
  }

  return task_buffer;
}

void TaskManager::notify_launched(taskid_t id, timecount_t time) {
  state.set_state(id, TaskState::LAUNCHED);
  records.record_launched(id, time);
}

const TaskIDList &TaskManager::notify_completed(taskid_t id, timecount_t time) {
  ZoneScoped;
  const auto &task_objects = get_tasks();

  records.record_completed(id, time);
  state.set_state(id, TaskState::COMPLETED);
  // clear task_buffer
  task_buffer.clear();
  if (!task_objects.is_eviction(id)) {
    for (auto dependent_id : task_objects.get_dependents(id)) {
      if (state.decrement_incomplete(dependent_id)) {
        task_buffer.push_back(dependent_id);
        assert(state.get_state(dependent_id) == TaskState::RESERVED);
        assert(task_objects.is_compute(dependent_id));
      }
    }
  }

  return task_buffer;
}

const TaskIDList &TaskManager::notify_data_completed(taskid_t id, timecount_t time) {
  MONUnusedParameter(time);
  ZoneScoped;
  const auto &task_objects = get_tasks();
  // clear task_buffer
  task_buffer.clear();

  for (auto dependent_id : task_objects.get_data_dependents(id)) {
    if (state.decrement_incomplete(dependent_id)) {
      task_buffer.push_back(dependent_id);
      assert(state.get_state(dependent_id) == TaskState::RESERVED);
      assert(task_objects.is_data(dependent_id));
    }
  }

  return task_buffer;
}
