#include "scheduler.hpp"
#include "data_manager.hpp"
#include "devices.hpp"
#include "event_manager.hpp"
#include "include/events.hpp"
#include "include/settings.hpp"
#include "include/tasks.hpp"
#include "macros.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include "task_manager.hpp"
#include "tasks.hpp"
#include <cstdint>
#include <iostream>

// SchedulerState

[[nodiscard]] timecount_t SchedulerState::get_mapped_time(taskid_t task_id) const {
  /*Get the time a task was mapped */
  const auto &records = task_manager.get_records();
  return records.get_mapped_time(task_id);
}

[[nodiscard]] timecount_t SchedulerState::get_reserved_time(taskid_t task_id) const {
  /*Get the time a task was reserved */
  const auto &records = task_manager.get_records();
  return records.get_reserved_time(task_id);
}

[[nodiscard]] timecount_t SchedulerState::get_launched_time(taskid_t task_id) const {
  /*Get the time a task was launched */
  const auto &records = task_manager.get_records();
  return records.get_launched_time(task_id);
}

[[nodiscard]] timecount_t SchedulerState::get_completed_time(taskid_t task_id) const {
  /*Get the time a task was completed */
  const auto &records = task_manager.get_records();
  return records.get_completed_time(task_id);
}

bool SchedulerState::track_resource_guard() const {
/* Compilation guard for when resource tracking (memory and vcu usage over time) is disabled */
#ifndef SIM_TRACK_RESOURCES
  spdlog::warn("SIM_TRACK_RESOURCES not defined. Resource tracking is disabled.");
  return true;
#else
  return false;
#endif
}

bool SchedulerState::track_location_guard() const {
/* Compilation guard for when location tracking (data location over time) is disabled */
#ifndef SIM_TRACK_LOCATION
  spdlog::warn("SIM_TRACK_LOCATION not defined. Location tracking is disabled.");
  return true;
#else
  return false;
#endif
}

[[nodiscard]] vcu_t SchedulerState::get_mapped_vcu_at(devid_t device_id, timecount_t time) const {
  /* Get the VCU mapped to a device at a given time */
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_vcu_at_time<TaskState::MAPPED>(device_id, time);
}
[[nodiscard]] vcu_t SchedulerState::get_reserved_vcu_at(devid_t device_id, timecount_t time) const {
  /* Get the VCU reserved to a device at a given time */
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_vcu_at_time<TaskState::RESERVED>(device_id, time);
}
[[nodiscard]] vcu_t SchedulerState::get_launched_vcu_at(devid_t device_id, timecount_t time) const {
  /* Get the VCU launched to a device at a given time */
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_vcu_at_time<TaskState::LAUNCHED>(device_id, time);
}

[[nodiscard]] mem_t SchedulerState::get_mapped_mem_at(devid_t device_id, timecount_t time) const {
  /* Get the memory mapped to a device at a given time */
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_mem_at_time<TaskState::MAPPED>(device_id, time);
}
[[nodiscard]] mem_t SchedulerState::get_reserved_mem_at(devid_t device_id, timecount_t time) const {
  /* Get the memory reserved to a device at a given time */
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_mem_at_time<TaskState::RESERVED>(device_id, time);
}
[[nodiscard]] mem_t SchedulerState::get_launched_mem_at(devid_t device_id, timecount_t time) const {
  /* Get the memory launched to a device at a given time */
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_mem_at_time<TaskState::LAUNCHED>(device_id, time);
}

[[nodiscard]] ResourceEventArray<vcu_t>
SchedulerState::get_mapped_vcu_events(devid_t device_id) const {
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_vcu_events<TaskState::MAPPED>(device_id);
}
[[nodiscard]] ResourceEventArray<vcu_t>
SchedulerState::get_reserved_vcu_events(devid_t device_id) const {
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_vcu_events<TaskState::RESERVED>(device_id);
}
[[nodiscard]] ResourceEventArray<vcu_t>
SchedulerState::get_launched_vcu_events(devid_t device_id) const {
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_vcu_events<TaskState::LAUNCHED>(device_id);
}

[[nodiscard]] ResourceEventArray<mem_t>
SchedulerState::get_mapped_mem_events(devid_t device_id) const {
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_mem_events<TaskState::MAPPED>(device_id);
}
[[nodiscard]] ResourceEventArray<mem_t>
SchedulerState::get_reserved_mem_events(devid_t device_id) const {
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_mem_events<TaskState::RESERVED>(device_id);
}
[[nodiscard]] ResourceEventArray<mem_t>
SchedulerState::get_launched_mem_events(devid_t device_id) const {
  if (track_resource_guard()) {
    return {};
  }
  return device_manager.get_mem_events<TaskState::LAUNCHED>(device_id);
}

[[nodiscard]] TaskState SchedulerState::get_state_at(taskid_t task_id, timecount_t time) const {
  /* Get the state of a task at a given time */
  if (track_resource_guard()) {
    return {};
  }
  const auto &records = task_manager.get_records();
  return records.get_state_at_time(task_id, time);
}

ValidEventArray SchedulerState::get_valid_intervals_mapped(dataid_t data_id,
                                                           devid_t device_id) const {
  /* Get the valid intervals for a data mapped to a device */
  if (track_location_guard()) {
    return {};
  }
  return data_manager.get_valid_intervals_mapped(data_id, device_id);
}

ValidEventArray SchedulerState::get_valid_intervals_reserved(dataid_t data_id,
                                                             devid_t device_id) const {
  /* Get the valid intervals for a data reserved to a device*/
  if (track_location_guard()) {
    return {};
  }

  return data_manager.get_valid_intervals_reserved(data_id, device_id);
}

ValidEventArray SchedulerState::get_valid_intervals_launched(dataid_t data_id,
                                                             devid_t device_id) const {
  /* Get the valid intervals for a data launched to a device */
  if (track_location_guard()) {
    return {};
  }

  return data_manager.get_valid_intervals_launched(data_id, device_id);
}

bool SchedulerState::check_valid_mapped_at(dataid_t data_id, devid_t device_id,
                                           timecount_t query_time) const {
  /* Check if a data is valid at a given time (in the mapping location table) */
  if (track_location_guard()) {
    return false;
  }

  return data_manager.check_valid_at_time_mapped(data_id, device_id, query_time);
}

bool SchedulerState::check_valid_reserved_at(dataid_t data_id, devid_t device_id,
                                             timecount_t query_time) const {
  /* Check if a data is valid at a given time (in the reservation location table) */
  if (track_location_guard()) {
    return false;
  }

  return data_manager.check_valid_at_time_reserved(data_id, device_id, query_time);
}

bool SchedulerState::check_valid_launched_at(dataid_t data_id, devid_t device_id,
                                             timecount_t query_time) const {
  /* Check if a data is valid at a given time (in the launch location table) */
  if (track_location_guard()) {
    return false;
  }

  return data_manager.check_valid_at_time_launched(data_id, device_id, query_time);
}

timecount_t SchedulerState::get_execution_time(taskid_t task_id) const {
  auto device_id = task_manager.state.get_mapping(task_id);
  auto arch = device_manager.devices.get().get_type(device_id);
  return task_manager.get_execution_time(task_id, arch);
}

const Resources &SchedulerState::get_task_resources(taskid_t task_id, devid_t device_id) const {
  const Resources &task_resources =
      task_manager.get_task_resources(task_id, device_manager.devices.get().get_type(device_id));
  return task_resources;
}

const Resources &SchedulerState::get_task_resources(taskid_t task_id) const {
  devid_t device_id = task_manager.state.get_mapping(task_id);
  const Resources &task_resources =
      task_manager.get_task_resources(task_id, device_manager.devices.get().get_type(device_id));
  return task_resources;
}

ResourceRequest SchedulerState::request_map_resources(taskid_t task_id, devid_t device_id) const {

  const Resources &task_resources = get_task_resources(task_id, device_id);
  const auto &task = task_manager.get_tasks().get_compute_task(task_id);
  mem_t non_local_mem = data_manager.non_local_size_mapped(task.get_unique(), device_id);
  Resources requested = {task_resources.vcu, task_resources.mem + non_local_mem};
  Resources missing;
  return {requested, missing};
}

bool SchedulerState::is_data_task(taskid_t task_id) const {
  const auto &tasks = task_manager.get_tasks();
  return tasks.is_data(task_id);
}

bool SchedulerState::is_compute_task(taskid_t task_id) const {
  const auto &tasks = task_manager.get_tasks();
  return tasks.is_compute(task_id);
}

ResourceRequest SchedulerState::request_reserve_resources(taskid_t task_id,
                                                          devid_t device_id) const {
  const auto &task = task_manager.get_tasks().get_compute_task(task_id);

  const Resources &task_resources = get_task_resources(task_id, device_id);
  mem_t non_local_mem = data_manager.non_local_size_reserved(task.get_unique(), device_id);
  Resources requested = {task_resources.vcu, task_resources.mem + non_local_mem};
  auto missing_mem = device_manager.overflow_mem<TaskState::RESERVED>(device_id, requested.mem);
  return {requested, Resources(0, missing_mem)};
}

ResourceRequest SchedulerState::request_launch_resources(taskid_t task_id,
                                                         devid_t device_id) const {
  const Resources &task_resources = get_task_resources(task_id, device_id);
  // mem_t non_local_mem =
  //     data_manager.non_local_size_launched(task.get_read(), device_id);
  // non_local_mem +=
  //     data_manager.non_local_size_launched(task.get_write(), device_id);
  // assert(non_local_mem == 0);
  SPDLOG_DEBUG("Requesting launch resources for task {} on device {}", task_id, device_id);
  SPDLOG_DEBUG("Task resources: VCU: {}, MEM: {}", task_resources.vcu, task_resources.mem);
  Resources requested = {task_resources.vcu, task_resources.mem};
  auto missing_vcu = device_manager.overflow_vcu<TaskState::LAUNCHED>(device_id, requested.vcu);
  return {requested, Resources(missing_vcu, 0)};
}

void SchedulerState::map_resources(taskid_t task_id, devid_t device_id,
                                   const Resources &requested) {
  MONUnusedParameter(task_id);
  device_manager.add_resources<TaskState::MAPPED>(device_id, requested, global_time);
}

void SchedulerState::reserve_resources(taskid_t task_id, devid_t device_id,
                                       const Resources &requested) {
  MONUnusedParameter(task_id);
  device_manager.add_resources<TaskState::RESERVED>(device_id, requested, global_time);
}

void SchedulerState::launch_resources(taskid_t task_id, devid_t device_id,
                                      const Resources &requested) {
  MONUnusedParameter(task_id);
  device_manager.add_resources<TaskState::LAUNCHED>(device_id, requested, global_time);
}

void SchedulerState::free_resources(taskid_t task_id) {
  devid_t device_id = task_manager.state.get_mapping(task_id);
  const auto &task_resources = get_task_resources(task_id);
  device_manager.remove_resources<TaskState::MAPPED>(device_id, task_resources, global_time);
  device_manager.remove_resources<TaskState::RESERVED>(device_id, task_resources, global_time);
  device_manager.remove_resources<TaskState::LAUNCHED>(device_id, task_resources, global_time);
}

const TaskIDList &SchedulerState::notify_mapped(taskid_t task_id) {
  return task_manager.notify_mapped(task_id, global_time);
}

const TaskIDList &SchedulerState::notify_reserved(taskid_t task_id) {
  return task_manager.notify_reserved(task_id, global_time);
}

void SchedulerState::notify_launched(taskid_t task_id) {
  task_manager.notify_launched(task_id, global_time);
}

const TaskIDList &SchedulerState::notify_completed(taskid_t task_id) {
  return task_manager.notify_completed(task_id, global_time);
}

const TaskIDList &SchedulerState::notify_data_completed(taskid_t task_id) {
  // This notifies data tasks that WAIT on the completion of this (task_id) compute task
  return task_manager.notify_data_completed(task_id, global_time);
}

bool SchedulerState::is_mapped(taskid_t task_id) const {
  return task_manager.state.is_mapped(task_id);
}

bool SchedulerState::is_reserved(taskid_t task_id) const {
  return task_manager.state.is_reserved(task_id);
}

bool SchedulerState::is_launched(taskid_t task_id) const {
  return task_manager.state.is_launched(task_id);
}

bool SchedulerState::is_mappable(taskid_t task_id) const {
  return task_manager.state.is_mappable(task_id);
}

bool SchedulerState::is_reservable(taskid_t task_id) const {
  return task_manager.state.is_reservable(task_id);
}

bool SchedulerState::is_launchable(taskid_t task_id) const {
  return task_manager.state.is_launchable(task_id);
}

void SchedulerState::set_mapping(taskid_t task_id, devid_t device_id) {
  task_manager.state.set_mapping(task_id, device_id);
}

devid_t SchedulerState::get_mapping(taskid_t task_id) const {
  return task_manager.state.get_mapping(task_id);
}

std::vector<devid_t> &SchedulerState::get_mappings() {
  return task_manager.state.get_mappings();
}

const PriorityList &SchedulerState::get_mapping_priorities() const {
  return task_manager.get_mapping_priorities();
}

const PriorityList &SchedulerState::get_reserving_priorities() const {
  return task_manager.get_reserving_priorities();
}

const PriorityList &SchedulerState::get_launching_priorities() const {
  return task_manager.get_launching_priorities();
}

priority_t SchedulerState::get_mapping_priority(taskid_t task_id) const {
  return task_manager.get_mapping_priority(task_id);
}

priority_t SchedulerState::get_reserving_priority(taskid_t task_id) const {
  return task_manager.get_reserving_priority(task_id);
}

priority_t SchedulerState::get_launching_priority(taskid_t task_id) const {
  return task_manager.get_launching_priority(task_id);
}

void SchedulerState::set_mapping_priority(taskid_t task_id, priority_t priority) {
  task_manager.set_mapping_priority(task_id, priority);
}

void SchedulerState::set_reserving_priority(taskid_t task_id, priority_t priority) {
  task_manager.set_reserving_priority(task_id, priority);
}

void SchedulerState::set_launching_priority(taskid_t task_id, priority_t priority) {
  task_manager.set_launching_priority(task_id, priority);
}

void SchedulerState::update_mapped_cost(taskid_t task_id, devid_t device_id) {
  DeviceType arch = device_manager.devices.get().get_type(device_id);
  timecount_t time = task_manager.tasks.get().get_variant(task_id, arch).get_observed_time();

  costs.count_mapped(device_id, time);
}
void SchedulerState::update_reserved_cost(taskid_t task_id, devid_t device_id) {
  DeviceType arch = device_manager.devices.get().get_type(device_id);
  timecount_t time = task_manager.tasks.get().get_variant(task_id, arch).get_observed_time();

  costs.count_reserved(device_id, time);
}
void SchedulerState::update_launched_cost(taskid_t task_id, devid_t device_id) {
  DeviceType arch = device_manager.devices.get().get_type(device_id);
  timecount_t time = task_manager.tasks.get().get_variant(task_id, arch).get_observed_time();

  costs.count_launched(device_id, time);
}
void SchedulerState::update_completed_cost(taskid_t task_id, devid_t device_id) {
  DeviceType arch = device_manager.devices.get().get_type(device_id);
  timecount_t time = task_manager.tasks.get().get_variant(task_id, arch).get_observed_time();
  costs.count_completed(device_id, time);
}

bool SchedulerState::is_data_task_virtual(taskid_t task_id) const {
  return task_manager.is_data_task_virtual(task_id);
}

devid_t SchedulerState::get_data_task_source(taskid_t task_id) const {
  return task_manager.get_data_task_source(task_id);
}

// Scheduler Queues
void SchedulerQueues::push_mappable(taskid_t id, priority_t p) {
  mappable.push(id, p);
}
void SchedulerQueues::push_mappable(const TaskIDList &ids, const PriorityList &ps) {
  assert(ps.size() >= ids.size());
  for (auto id : ids) {
    assert(ps.size() > id);
    push_mappable(id, ps.at(id));
  }
}

void SchedulerQueues::push_reservable(taskid_t id, priority_t p, devid_t device) {
  reservable.at(device).push(id, p);
}

void SchedulerQueues::push_reservable(const TaskIDList &ids, const PriorityList &ps,
                                      devid_t device) {
  assert(ps.size() >= ids.size());
  for (auto id : ids) {
    push_reservable(id, ps.at(id), device);
  }
}

void SchedulerQueues::push_launchable(taskid_t id, priority_t p, devid_t device) {
  launchable.at(device).push(id, p);
}

void SchedulerQueues::push_launchable(const TaskIDList &ids, const PriorityList &ps,
                                      devid_t device) {
  assert(ps.size() >= ids.size());
  for (auto id : ids) {
    push_launchable(id, ps.at(id), device);
  }
}

void SchedulerQueues::push_launchable_data(taskid_t id, priority_t p, devid_t device) {
  data_launchable.at(device).push(id, p);
}

void SchedulerQueues::push_launchable_data(const TaskIDList &ids, const PriorityList &ps,
                                           devid_t device) {
  assert(ps.size() >= ids.size());
  for (auto id : ids) {
    push_launchable_data(id, ps.at(id), device);
  }
}

// void SchedulerQueues::id_to_queue(taskid_t id, const TaskStateInfo &state) {
//   if (state.is_mappable(id)) {
//     push_mappable(id, state.get_mapping_priority(id));
//   } else if (state.is_reservable(id)) {
//     push_reservable(id, state.get_reserving_priority(id),
//                     state.get_mapping(id));
//   } else if (state.is_launchable(id)) {
//     push_launchable(id, state.get_launching_priority(id),
//                     state.get_mapping(id));
//   }
// }

// TaskType SchedulerQueues::id_to_type(taskid_t id, const Tasks &tasks) {
//   if (tasks.is_compute(id)) {
//     return TaskType::COMPUTE;
//   }
//   return TaskType::DATA;
// }

// TODO(wlr): Deal with data tasks
// void SchedulerQueues::populate(const TaskManager &task_manager) {
//   const auto &state = task_manager.get_state();
//   const auto &compute_tasks = task_manager.get_tasks().get_compute_tasks();

//   for (const auto &compute_task : compute_tasks) {
//     id_to_queue(compute_task.id, state);
//   }
// }

// void SchedulerQueues::populate(const TaskIDList &ids,
//                                const TaskManager &task_manager) {
//   const auto &state = task_manager.get_state();

//   for (auto id : ids) {
//     id_to_queue(id, state);
//   }
// }

// TaskCountInfo

TaskCountInfo::TaskCountInfo(std::size_t n_devices)
    : per_device_mapped_tasks(n_devices), per_device_reserved_tasks(n_devices),
      per_device_launched_tasks(n_devices), per_device_completed_tasks(n_devices),
      per_device_data_completed_tasks(n_devices) {
}

void TaskCountInfo::count_mapped(taskid_t task_id, devid_t device_id) {
  n_active_tasks += 1;
  n_mapped_tasks += 1;
  per_device_mapped_tasks.at(device_id) += 1;

  // add to active_tasks set
  active_tasks.insert(task_id);
}

void TaskCountInfo::count_reserved(taskid_t task_id, devid_t device_id) {
  n_reserved_tasks += 1;
  per_device_reserved_tasks.at(device_id) += 1;
}

void TaskCountInfo::count_launched(taskid_t task_id, devid_t device_id) {
  n_launched_tasks += 1;
  per_device_launched_tasks.at(device_id) += 1;
}

void TaskCountInfo::count_completed(taskid_t task_id, devid_t device_id) {
  assert(n_active_tasks >= 1);
  n_active_tasks -= 1;

  assert(n_mapped_tasks >= 1);
  n_mapped_tasks -= 1;
  assert(per_device_mapped_tasks.at(device_id) >= 1);
  per_device_mapped_tasks.at(device_id) -= 1;

  assert(n_reserved_tasks >= 1);
  n_reserved_tasks -= 1;
  assert(per_device_reserved_tasks.at(device_id) >= 1);
  per_device_reserved_tasks.at(device_id) -= 1;

  assert(n_launched_tasks >= 1);
  n_launched_tasks -= 1;
  assert(per_device_launched_tasks.at(device_id) >= 1);
  per_device_launched_tasks.at(device_id) -= 1;

  n_completed_tasks += 1;
  per_device_completed_tasks.at(device_id) += 1;

  // remove from active_tasks set
  active_tasks.erase(task_id);
}

void TaskCountInfo::count_data_completed(taskid_t task_id, devid_t device_id) {
  n_data_completed_tasks += 1;
  per_device_data_completed_tasks[device_id] += 1;
}

// TaskCostInfo

TaskCostInfo::TaskCostInfo(std::size_t n_tasks, std::size_t n_devices)
    : per_device_mapped_time(n_devices), per_device_reserved_time(n_devices),
      per_device_launched_time(n_devices), per_device_completed_time(n_devices),
      per_device_data_completed_time(n_devices) {
}

void TaskCostInfo::count_mapped(devid_t device_id, timecount_t time) {
  per_device_mapped_time.at(device_id) += time;
}

void TaskCostInfo::count_reserved(devid_t device_id, timecount_t time) {
  per_device_reserved_time.at(device_id) += time;
}

void TaskCostInfo::count_launched(devid_t device_id, timecount_t time) {
  per_device_launched_time.at(device_id) += time;
}

void TaskCostInfo::count_completed(devid_t device_id, timecount_t time) {

  assert(per_device_mapped_time.at(device_id) >= time);
  per_device_mapped_time.at(device_id) -= time;
  assert(per_device_reserved_time.at(device_id) >= time);
  per_device_reserved_time.at(device_id) -= time;
  assert(per_device_launched_time.at(device_id) >= time);
  per_device_launched_time.at(device_id) -= time;

  per_device_completed_time.at(device_id) += time;
}

void TaskCostInfo::count_data_completed(devid_t device_id, timecount_t time) {
  per_device_data_completed_time.at(device_id) += time;
}

// TransitionConstraints

// Scheduler

size_t Scheduler::get_mappable_candidates(TorchInt64Arr1D &output_tensor) {

  auto v = output_tensor.view();
  auto &s = this->state;
  bool condition = queues.has_mappable() && conditions.get().should_map(s, queues);

  if (!condition) {
    return 0;
  }

  auto &mappable = queues.mappable;
  auto top_k_tasks = mappable.get_top_k();

  const auto copy_size = std::min(output_tensor.size(), top_k_tasks.size());

  for (size_t i = 0; i < copy_size; i++) {
    v(i) = top_k_tasks[i];
  }
  return copy_size;
}

// Original function kept for backward compatibility
TaskIDList &Scheduler::get_mappable_candidates() {
  auto &s = this->state;
  bool condition = queues.has_mappable() && conditions.get().should_map(s, queues);
  clear_task_buffer();

  if (!condition) {
    return task_buffer;
  }

  auto &mappable = queues.mappable;
  auto top_k_tasks = mappable.get_top_k();

  task_buffer.insert(task_buffer.end(), top_k_tasks.begin(), top_k_tasks.end());
  return task_buffer;
}

const TaskIDList &Scheduler::map_task(taskid_t task_id, Action &action) {
  auto &s = this->state;
  auto current_time = s.global_time;

  devid_t chosen_device = action.device;

  SPDLOG_DEBUG("Mapping task {} to device {}", s.get_task_name(task_id), chosen_device);
  // std::cout << "Mapping task " << s.get_task_name(task_id) << " to device " << chosen_device
  //           << std::endl;

  assert(s.is_mappable(task_id));
  assert(s.is_compute_task(task_id));

  priority_t rp = action.reservable_priority;
  priority_t lp = action.launchable_priority;

  s.set_mapping(task_id, chosen_device);
  s.set_reserving_priority(task_id, rp);
  s.set_launching_priority(task_id, lp);

  // Update mapped resources
  auto [requested, missing] = s.request_map_resources(task_id, chosen_device);
  s.map_resources(task_id, chosen_device, requested);

  // Update data locations
  const ComputeTask &task = s.task_manager.get_tasks().get_compute_task(task_id);
  s.data_manager.read_update_mapped(task.get_read(), chosen_device, current_time);
  s.data_manager.read_update_mapped(task.get_write(), chosen_device, current_time);
  s.data_manager.write_update_mapped(task.get_write(), chosen_device, current_time);

  // Notify dependents and enqueue newly mappable tasks
  const auto &newly_mappable_tasks = s.notify_mapped(task_id);
  success_count += 1;
  state.counts.count_mapped(task_id, chosen_device);
  state.update_mapped_cost(task_id, chosen_device);

  breakpoints.check_task_breakpoint(EventType::MAPPER, task_id);

  // Check if the mapped task is reservable, and if so, enqueue it
  if (s.is_reservable(task_id)) {
    SPDLOG_DEBUG("Task {} is reservable at time {}", s.get_task_name(task_id), s.global_time);
    push_reservable(task_id, chosen_device);
  }

  return newly_mappable_tasks;
}

void Scheduler::remove_mapped_tasks(ActionList &action_list) {
  std::vector<std::size_t> positions;

  for (auto &action : action_list) {
    positions.push_back(action.pos);
  }

  queues.mappable.remove(positions);
}

ExecutionState Scheduler::map_tasks_from_python(ActionList &action_list,
                                                EventManager &event_manager) {
  success_count = 0;
  auto &mappable = queues.mappable;
  auto top_k_tasks = mappable.get_top_k();

  // std::cout << "Mapping tasks from python" << std::endl;
  // std::cout << "Len action list" << action_list.size() << std::endl;
  // std::cout << "Len top k tasks" << top_k_tasks.size() << std::endl;

  if (!action_list.empty()) {
    TaskIDList nmt;
    for (auto &action : action_list) {
      const auto task_id = top_k_tasks[action.pos];
      const auto &tasks = map_task(task_id, action);
      nmt.insert(nmt.end(), tasks.begin(), tasks.end());
    }
    remove_mapped_tasks(action_list);
    push_mappable(nmt);
  }

  /*If we still should be mapping, continue making calls to the mapper */

  if (queues.has_mappable() && conditions.get().should_map(state, queues)) {
    // timecount_t mapper_time = state.global_time;
    // event_manager.create_event(EventType::MAPPER, mapper_time, TaskIDList());
    return ExecutionState::EXTERNAL_MAPPING;
  } else {

    if (is_breakpoint()) {
      // TODO(wlr): This needs to be tested.
      SPDLOG_DEBUG("Breaking from mapper at time {}", state.global_time);
      timecount_t mapper_time = state.global_time;
      event_manager.create_event(EventType::MAPPER, mapper_time);
      return ExecutionState::BREAKPOINT;
    }

    SPDLOG_DEBUG("Ending mapper at time {}", state.global_time);
    timecount_t reserver_time = state.global_time + TIME_TO_RESERVE;
    event_manager.create_event(EventType::RESERVER, reserver_time);
    return ExecutionState::RUNNING;
  }
}

void Scheduler::map_tasks(MapperEvent &map_event, EventManager &event_manager, Mapper &mapper) {
  success_count = 0;

  auto &s = this->state;
  auto &task_states = state.task_manager.state;

  SPDLOG_DEBUG("Starting mapper at time {}", s.global_time);
  SPDLOG_DEBUG("Mappable Queue Size: {}", queues.mappable.size());
  bool break_flag = false;

  while (queues.has_mappable() && conditions.get().should_map(s, queues)) {

    if (is_breakpoint()) {
      break_flag = true;
      SPDLOG_DEBUG("Breaking from mapper at time {}", s.global_time);
      break;
    }

    taskid_t task_id = queues.mappable.top();
    queues.mappable.pop();
    assert(task_states.is_mappable(task_id));
    Action action = mapper.map_task(task_id, s);
    // spdlog::info("Mapping task {} at time {} to device {}",
    //              s.get_task_name(task_id), s.global_time, action.device);
    const auto &newly_mappable_tasks = map_task(task_id, action);
    // spdlog::debug("Newly mappable tasks: {}", newly_mappable_tasks.size());
    push_mappable(newly_mappable_tasks);
  }

  if (break_flag) {
    timecount_t mapper_time = s.global_time;
    event_manager.create_event(EventType::MAPPER, mapper_time);
    return;
  }

  // The next event is a reserving event
  timecount_t reserver_time = s.global_time + SCHEDULER_TIME_GAP;
  event_manager.create_event(EventType::RESERVER, reserver_time);
}

void Scheduler::enqueue_data_tasks(taskid_t id) {
  auto &s = this->state;
  auto &task_manager = state.task_manager;
  const auto &tasks = state.task_manager.get_tasks();

  assert(tasks.is_compute(id));

  const auto &data_dependencies = tasks.get_data_dependencies(id);

  SPDLOG_DEBUG("Enqueueing {} data tasks for task {}", data_dependencies.size(),
               s.get_task_name(id));

  for (auto data_task_id : data_dependencies) {
    assert(tasks.is_data(data_task_id));

    task_manager.set_state(data_task_id, TaskState::RESERVED);
    if (s.is_launchable(data_task_id)) {
      SPDLOG_DEBUG("Data task {} is launchable at time {}", s.get_task_name(data_task_id),
                   s.global_time);
      push_launchable_data(data_task_id);
    }
  }
}

SuccessPair Scheduler::reserve_task(taskid_t task_id, devid_t device_id,
                                    TaskDeviceList &tasks_requesting_eviction) {
  auto &s = this->state;
  auto current_time = s.global_time;

  assert(s.is_compute_task(task_id));
  assert(s.is_reservable(task_id));
  assert(s.get_mapping(task_id) == device_id);

  SPDLOG_DEBUG("Attempting to reserve task {} at time {} on device {}", s.get_task_name(task_id),
               current_time, device_id);

  // Get total required task memory
  const auto [requested, missing] = s.request_reserve_resources(task_id, device_id);

  if (missing.mem > 0) {
    SPDLOG_DEBUG("Task {} will evict memory on device {} since requested {} memory but missing {} "
                 "memory at time {}",
                 s.get_task_name(task_id), device_id, requested.mem, missing.mem, current_time);
    tasks_requesting_eviction.push_back(std::make_tuple(task_id, device_id, missing.mem));
    return {false, nullptr};
  }

  SPDLOG_DEBUG("Reserving task {} at time {} on device {}", s.get_task_name(task_id), current_time,
               device_id);

  // Update reserved resources
  s.reserve_resources(task_id, device_id, requested);

  // Update data locations
  const ComputeTask &task = s.task_manager.get_tasks().get_compute_task(task_id);
  s.data_manager.read_update_reserved(task.get_read(), device_id, current_time);
  s.data_manager.read_update_reserved(task.get_write(), device_id, current_time);
  s.data_manager.write_update_reserved(task.get_write(), device_id, current_time);

  const auto &newly_reservable_tasks = s.notify_reserved(task_id);

  success_count += 1;
  enqueue_data_tasks(task_id);
  s.counts.count_reserved(task_id, device_id);
  s.update_reserved_cost(task_id, device_id);
  breakpoints.check_task_breakpoint(EventType::RESERVER, task_id);

  // Check if the reserved task is launchable, and if so, enqueue it
  if (s.is_launchable(task_id)) {
    SPDLOG_DEBUG("Task {} is launchable at time {}", s.get_task_name(task_id), current_time);
    push_launchable(task_id, device_id);
  }

  return {true, &newly_reservable_tasks};
}

void Scheduler::reserve_tasks(ReserverEvent &reserve_event, EventManager &event_manager) {
  auto &s = this->state;

  auto &reservable = queues.reservable;
  reservable.reset();
  reservable.current_or_next_active();

  SPDLOG_DEBUG("Reserving tasks at time {}", s.global_time);
  SPDLOG_DEBUG("Reservable Queue Size: {}", queues.reservable.total_active_size());
  bool break_flag = false;

  TaskDeviceList tasks_requesting_eviction;
  tasks_requesting_eviction.reserve(reservable.size());

  while (queues.has_active_reservable() && conditions.get().should_reserve(s, queues)) {

    if (is_breakpoint()) {
      break_flag = true;
      SPDLOG_DEBUG("Breaking from reserver at time {}", s.global_time);
      break;
    }

    if (reservable.get_active().empty()) {
      reservable.next();
      continue;
    }

    auto device_id = static_cast<devid_t>(reservable.get_active_index());
    taskid_t task_id = reservable.top();
    auto [success, newly_reservable_tasks] =
        reserve_task(task_id, device_id, tasks_requesting_eviction);
    if (!success) {
      reservable.deactivate();
      reservable.next();
      continue;
    }

    reservable.pop();

    // Notify dependents and enqueue newly reservable tasks

    // spdlog::debug("Newly reservable tasks: {}",
    // newly_reservable_tasks.size());
    if (newly_reservable_tasks != nullptr) {
      if (!newly_reservable_tasks->empty()) {
        push_reservable(*newly_reservable_tasks);
      }
    }

    // Cycle to the next active device queue
    reservable.next();
  }

  if (break_flag) {
    timecount_t reserver_time = s.global_time;
    event_manager.create_event(EventType::RESERVER, reserver_time);
    return;
  }
  if (!tasks_requesting_eviction.empty()) {
    SPDLOG_DEBUG("Eviction is not implemented and called");
    assert(0);
    // The next event is a eviction event
    timecount_t launcher_time = s.global_time + TIME_TO_LAUNCH;
    event_manager.create_event(EventType::EVICTOR, launcher_time);
  } else {
    // The next event is a launching event
    timecount_t launcher_time = s.global_time + TIME_TO_LAUNCH;
    event_manager.create_event(EventType::LAUNCHER, launcher_time);
  }
}

bool Scheduler::launch_compute_task(taskid_t task_id, devid_t device_id,
                                    EventManager &event_manager) {
  auto &s = this->state;
  auto current_time = s.global_time;

  SPDLOG_DEBUG("Attempting to launch compute task {} at time {} on device {}",
               s.get_task_name(task_id), s.global_time, device_id);

  assert(s.is_launchable(task_id));
  assert(s.get_mapping(task_id) == device_id);
  assert(s.is_compute_task(task_id));

  const auto [requested, missing] = s.request_launch_resources(task_id, device_id);

  if (missing.vcu > 0) {
    SPDLOG_DEBUG("Task {} requested {} VCU but missing {} VCU at time {}", s.get_task_name(task_id),
                 requested.vcu, missing.vcu, s.global_time);
    return false;
  }

  const auto &task = s.task_manager.get_tasks().get_compute_task(task_id);

  SPDLOG_DEBUG("Launching compute task {} at time {} on device {}", s.get_task_name(task_id),
               current_time, device_id);

  // Update data locations for WRITE data (create them here)
  s.data_manager.read_update_launched(task.get_write(), device_id, current_time);
  s.data_manager.write_update_launched(task.get_write(), device_id, current_time);

  // All READ data should already be here (prefetched by data tasks)
  s.data_manager.check_valid_launched(task.get_read(), device_id);

  // Update launched resources
  s.launch_resources(task_id, device_id, requested);

  // Record launching time
  s.notify_launched(task_id);
  success_count += 1;
  s.counts.count_launched(task_id, device_id);
  s.update_launched_cost(task_id, device_id);

  breakpoints.check_task_breakpoint(EventType::LAUNCHER, task_id);

  // Create completion event
  timecount_t completion_time = s.global_time + s.get_execution_time(task_id);
  event_manager.create_event(EventType::COMPLETER, completion_time, TaskIDList({task_id}));

  return true;
}

bool Scheduler::launch_data_task(taskid_t task_id, devid_t destination_id,
                                 EventManager &event_manager) {
  auto &s = this->state;
  auto current_time = s.global_time;

  SPDLOG_DEBUG("Attempting to launch data task {} at time {} on device {}",
               s.get_task_name(task_id), s.global_time, destination_id);

  assert(s.is_launchable(task_id));

  const auto &task = s.task_manager.get_tasks().get_data_task(task_id);
  const dataid_t data_id = task.get_data_id();

  auto [found, source_id] = s.data_manager.request_source(data_id, destination_id);

  if (!found) {
    SPDLOG_DEBUG("Data task {} missing available source at time {}", s.get_task_name(task_id),
                 s.global_time);
    return false;
  }
  s.task_manager.set_source(task_id, source_id);
  auto duration = s.data_manager.start_move(data_id, source_id, destination_id, current_time);

  if (duration.is_virtual) {
    SPDLOG_DEBUG("Data task {} is virtual at time {}", s.get_task_name(task_id), s.global_time);
    s.task_manager.set_virtual(task_id);
  } else {
    SPDLOG_DEBUG("Data task {} moving from {} to {} at time {}", s.get_task_name(task_id),
                 source_id, destination_id, s.global_time);
  }

  // Record launching time
  s.notify_launched(task_id);
  success_count += 1;
  breakpoints.check_task_breakpoint(EventType::LAUNCHER, task_id);

  // Create completion event
  timecount_t completion_time = s.global_time + duration.duration;
  event_manager.create_event(EventType::COMPLETER, completion_time, TaskIDList({task_id}));

  return true;
}

void Scheduler::launch_tasks(LauncherEvent &launch_event, EventManager &event_manager) {

  auto &s = this->state;

  auto &launchable = queues.launchable;
  launchable.reset();
  launchable.current_or_next_active();

  SPDLOG_DEBUG("Launching tasks at time {}", s.global_time);
  SPDLOG_DEBUG("Launchable Queue Size: {}", queues.launchable.total_active_size());
  bool break_flag = false;

  while (queues.has_active_launchable() && conditions.get().should_launch(s, queues)) {

    if (is_breakpoint()) {
      SPDLOG_DEBUG("Breaking from launcher at time {}", s.global_time);
      break_flag = true;
      break;
    }

    if (launchable.get_active().empty()) {
      launchable.next();
      continue;
    }

    taskid_t task_id = launchable.top();
    auto device_id = static_cast<devid_t>(launchable.get_active_index());

    bool success = launch_compute_task(task_id, device_id, event_manager);
    if (!success) {
      launchable.deactivate();
      launchable.next();
      continue;
    }
    launchable.pop();
    launchable.next();
  }

  if (break_flag) {
    timecount_t launcher_time = s.global_time;
    event_manager.create_event(EventType::LAUNCHER, launcher_time);
    return;
  }

  SPDLOG_DEBUG("Launching data tasks at time {}", s.global_time);
  SPDLOG_DEBUG("Data Launchable Queue Size: {}", queues.data_launchable.total_active_size());

  auto &data_launchable = queues.data_launchable;
  data_launchable.reset();
  data_launchable.current_or_next_active();

  while (queues.has_active_data_launchable() && conditions.get().should_launch_data(s, queues)) {

    if (is_breakpoint()) {
      SPDLOG_DEBUG("Breaking from data launcher at time {}", s.global_time);
      break_flag = true;
      break;
    }

    if (data_launchable.get_active().empty()) {
      data_launchable.next();
      continue;
    }

    taskid_t task_id = data_launchable.top();
    assert(s.is_data_task(task_id));
    auto device_id = static_cast<devid_t>(data_launchable.get_active_index());

    bool success = launch_data_task(task_id, device_id, event_manager);
    if (!success) {
      data_launchable.deactivate();
      data_launchable.next();
      continue;
    }
    data_launchable.pop();
    data_launchable.next();
  }

  if (break_flag) {
    timecount_t launcher_time = s.global_time;
    event_manager.create_event(EventType::LAUNCHER, launcher_time);
    return;
  }

  scheduler_event_count -= 1;

  if (scheduler_event_count == 0 and success_count > 0) {
    event_manager.create_event(EventType::MAPPER, s.global_time + SCHEDULER_TIME_GAP + TIME_TO_MAP);
    scheduler_event_count += 1;
  }
}

// TODO(wlr): implement eviction event
void Scheduler::evict(EvictorEvent &eviction_event, EventManager &event_manager) {
  MONUnusedParameter(eviction_event);
  MONUnusedParameter(event_manager);
}

void Scheduler::complete_compute_task(taskid_t task_id, devid_t device_id) {
  auto &s = this->state;

  SPDLOG_DEBUG("Completing compute task {} at time {} on device {}", s.get_task_name(task_id),
               s.global_time, device_id);

  // Free mapped, reserved, and launched resources
  s.free_resources(task_id);
  s.counts.count_completed(task_id, device_id);
  s.update_completed_cost(task_id, device_id);

  const auto &newly_launchable_data_tasks = s.notify_data_completed(task_id);
  push_launchable_data(newly_launchable_data_tasks);
  // spdlog::debug("Newly launchable data tasks: {}",
  //               newly_launchable_data_tasks.size());
}

void Scheduler::complete_data_task(taskid_t task_id, devid_t destination_id) {
  auto &s = this->state;
  auto current_time = s.global_time;
  s.counts.count_data_completed(task_id, destination_id);

  SPDLOG_DEBUG("Completing data task {} at time {} on device {}", s.get_task_name(task_id),
               s.global_time, destination_id);

  auto source_id = s.task_manager.get_source(task_id);
  auto is_virtual = s.task_manager.is_virtual(task_id);
  auto data_id = s.task_manager.get_tasks().get_data_task(task_id).get_data_id();
  s.data_manager.complete_move(data_id, source_id, destination_id, is_virtual, current_time);
}

void Scheduler::complete_task(CompleterEvent &complete_event, EventManager &event_manager) {
  assert(complete_event.tasks.size() == 1);
  auto &s = this->state;
  taskid_t task_id = complete_event.tasks.front();

  if (s.is_compute_task(task_id)) {
    devid_t device_id = s.get_mapping(task_id);
    complete_compute_task(task_id, device_id);
  } else {
    const auto &data_task = s.task_manager.get_tasks().get_data_task(task_id);
    devid_t device_id = s.get_mapping(data_task.get_compute_task());
    complete_data_task(task_id, device_id);
  }

  // Notify dependents and enqueue newly launchable tasks
  const auto &newly_launchable_tasks = s.notify_completed(task_id);
  push_launchable(newly_launchable_tasks);
  // spdlog::debug("Newly launchable compute tasks: {}",
  //               newly_launchable_tasks.size());

  success_count += 1;
  breakpoints.check_task_breakpoint(EventType::COMPLETER, task_id);

  if (scheduler_event_count == 0) {
    event_manager.create_event(EventType::MAPPER, s.global_time + TIME_TO_MAP + SCHEDULER_TIME_GAP);
    scheduler_event_count += 1;
  }
}