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
  // Even though write only blocks are generated, we need to reserve memory space
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

bool SchedulerState::is_eviction_task(taskid_t task_id) const {
  const auto &tasks = task_manager.get_tasks();
  return tasks.is_eviction(task_id);
}

ResourceRequest SchedulerState::request_reserve_resources(taskid_t task_id,
                                                          devid_t device_id) const {
  const auto &task = task_manager.get_tasks().get_compute_task(task_id);

  const Resources &task_resources = get_task_resources(task_id, device_id);
  mem_t non_local_mem = data_manager.non_local_size_reserved(task.get_unique(), device_id);
  Resources requested = {task_resources.vcu, task_resources.mem + non_local_mem};
  auto missing_mem = device_manager.overflow_mem<TaskState::RESERVED>(device_id, requested.mem);
  return {.requested = requested, .missing = Resources(0, missing_mem)};
}

ResourceRequest SchedulerState::request_launch_resources(taskid_t task_id,
                                                         devid_t device_id) const {
  const Resources &task_resources = get_task_resources(task_id, device_id);
  SPDLOG_DEBUG("Requesting launch resources for task {} on device {}", get_task_name(task_id),
               device_id);
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

void SchedulerQueues::push_launchable_eviction(taskid_t id, priority_t p, devid_t device) {
  eviction_launchable.at(device).push(id, p);
}

void SchedulerQueues::push_launchable_eviction(const TaskIDList &ids, const PriorityList &ps,
                                               devid_t device) {
  assert(ps.size() >= ids.size());
  for (auto id : ids) {
    push_launchable_eviction(id, ps.at(id), device);
  }
}

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

size_t Scheduler::get_mappable_candidates(std::span<int64_t> v) {

  auto &s = this->state;
  bool condition = queues.has_mappable() && conditions.get().should_map(s, queues);

  if (!condition) {
    return 0;
  }

  auto &mappable = queues.mappable;
  auto top_k_tasks = mappable.get_top_k();

  const auto copy_size = std::min(v.size(), top_k_tasks.size());

  for (size_t i = 0; i < copy_size; i++) {
    v[i] = top_k_tasks[i];
  }
  return copy_size;
}

// Original function kept for backward compatibility
TaskIDList &Scheduler::get_mappable_candidates() {
  ZoneScoped;
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
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;

  devid_t chosen_device = action.device;

  SPDLOG_DEBUG("Time:{} Mapping task {} to device {}", current_time, s.get_task_name(task_id),
               chosen_device);
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
  s.data_manager.read_update_mapped(task.get_unique(), chosen_device, current_time);
  s.data_manager.write_update_mapped(task.get_write(), chosen_device, current_time);
  // Ground truth
  // Consider this scenario:
  //   GPU0   |   GPU1
  // ---------|----------
  // read B0  |
  //          |  Write B0
  // read B0  |
  //          |  Read B0
  // ---------|----------
  //   2*B0   |    B0
  // Since GPU1 invalidated B0 once before the last read, total mapped memory size is 2*B0
  // However, during execution, GPU1 will remove mapped memory from GPU0 in task completion
  // Thus, every duplicate memory is removed in the end.
  // IT SHOULD BE launched_location == reserved_location == mapped_location """in the end""".
  // This becomes tricky when eviction happens.
  // Currently, we keep track of mapped but not reserved tasks.
  // We check if there are any usage of the data by the victim device.
  // If there are, it means that

  s.mapped_but_not_reserved_tasks.insert(task_id);

  // Notify dependents and enqueue newly mappable tasks
  const auto &newly_mappable_tasks = s.notify_mapped(task_id);
  success_count += 1;
  state.counts.count_mapped(task_id, chosen_device);
  state.update_mapped_cost(task_id, chosen_device);

  // Print most recent writers
  // const auto &most_recent_writers = task.get_most_recent_writers();
  // const auto &read = task.get_read();

  // std::cout << "Most recent writers for task " << s.get_task_name(task_id) << std::endl;
  // for (int i = 0; i < most_recent_writers.size(); i++) {
  //   std::cout << read[i] << " : " << most_recent_writers[i] << std::endl;
  // }

  breakpoints.check_task_breakpoint(EventType::MAPPER, task_id);

  // Check if the mapped task is reservable, and if so, enqueue it
  if (s.is_reservable(task_id)) {
    SPDLOG_DEBUG("Time:{} Task {} is reservable", current_time, s.get_task_name(task_id));
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
  ZoneScoped;
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
      SPDLOG_DEBUG("Time:{} Breaking from mapper", state.global_time);
      timecount_t mapper_time = state.global_time;
      event_manager.create_event(EventType::MAPPER, mapper_time);
      return ExecutionState::BREAKPOINT;
    }

    SPDLOG_DEBUG("Time:{} Ending mapper", state.global_time);
    timecount_t reserver_time = state.global_time + TIME_TO_RESERVE;
    event_manager.create_event(EventType::RESERVER, reserver_time);
    return ExecutionState::RUNNING;
  }
}

void Scheduler::map_tasks(MapperEvent &map_event, EventManager &event_manager, Mapper &mapper) {
  ZoneScoped;
  success_count = 0;

  auto &s = this->state;
  auto &task_states = state.task_manager.state;
  auto current_time = s.global_time;

  SPDLOG_DEBUG("Time:{} Starting mapper", current_time);
  SPDLOG_DEBUG("Time:{} Mappable Queue Size: {}", current_time, queues.mappable.size());
  bool break_flag = false;

  while (queues.has_mappable() && conditions.get().should_map(s, queues)) {

    if (is_breakpoint()) {
      break_flag = true;
      SPDLOG_DEBUG("Time:{} Breaking from mapper", current_time);
      break;
    }

    taskid_t task_id = queues.mappable.top();
    queues.mappable.pop();
    assert(task_states.is_mappable(task_id));
    Action action = mapper.map_task(task_id, s);
    // spdlog::info("Mapping task {} at time {} to device {}",
    //              s.get_task_name(task_id), current_time, action.device);
    const auto &newly_mappable_tasks = map_task(task_id, action);
    // spdlog::debug("Newly mappable tasks: {}", newly_mappable_tasks.size());
    push_mappable(newly_mappable_tasks);
  }

  if (break_flag) {
    timecount_t mapper_time = current_time;
    event_manager.create_event(EventType::MAPPER, mapper_time);
    return;
  }

  // The next event is a reserving event
  timecount_t reserver_time = current_time + SCHEDULER_TIME_GAP;
  event_manager.create_event(EventType::RESERVER, reserver_time);
}

void Scheduler::enqueue_data_tasks(taskid_t id) {
  auto &s = this->state;
  auto &task_manager = state.task_manager;
  auto current_time = s.global_time;
  const auto &tasks = state.task_manager.get_tasks();

  assert(tasks.is_compute(id));

  const auto &data_dependencies = tasks.get_data_dependencies(id);

  SPDLOG_DEBUG("Time:{} Enqueueing {} data tasks for task {}", current_time,
               data_dependencies.size(), s.get_task_name(id));

  for (auto data_task_id : data_dependencies) {
    assert(tasks.is_data(data_task_id));

    task_manager.set_state(data_task_id, TaskState::RESERVED);
    if (s.is_launchable(data_task_id)) {
      SPDLOG_DEBUG("Time:{} Data task {} is launchable", current_time,
                   s.get_task_name(data_task_id));
      push_launchable_data(data_task_id);
    }
  }
}

SuccessPair Scheduler::reserve_task(taskid_t task_id, devid_t device_id,
                                    TaskDeviceList &tasks_requesting_eviction) {
  ZoneScoped;

  auto &s = this->state;
  auto current_time = s.global_time;
  auto &mapped = s.mapped_but_not_reserved_tasks;

  assert(s.is_compute_task(task_id));
  assert(s.is_reservable(task_id));
  assert(s.get_mapping(task_id) == device_id);

  SPDLOG_DEBUG("Time:{} Attempting to reserve task {} on device {}", current_time,
               s.get_task_name(task_id), device_id);

  // Get total required task memory
  const auto [requested, missing] = s.request_reserve_resources(task_id, device_id);

  if (missing.mem > 0) {
    SPDLOG_DEBUG(
        "Time:{} Task {} will evict memory on device {} since requested {} memory but missing {} "
        "memory",
        current_time, s.get_task_name(task_id), device_id, requested.mem, missing.mem);
    tasks_requesting_eviction.push_back(std::make_tuple(task_id, device_id));
    return {false, nullptr};
  }

  SPDLOG_DEBUG("Time:{} Reserving task {} on device {}", current_time, s.get_task_name(task_id),
               device_id);

  // Update reserved resources
  s.reserve_resources(task_id, device_id, requested);
  SPDLOG_DEBUG("Time:{} Task {} requested memsize {} resulting in reserved size of {} at device {}",
               current_time, s.get_task_name(task_id), requested.mem,
               s.get_device_manager().get_mem<TaskState::RESERVED>(device_id), device_id);

  // Update data locations
  const ComputeTask &task = s.task_manager.get_tasks().get_compute_task(task_id);
  s.data_manager.read_update_reserved(task.get_unique(), device_id, current_time);
  s.data_manager.write_update_reserved(task.get_write(), device_id, current_time);

  // erase task_id from s.mapped_but_not_reserved_tasks
  mapped.erase(mapped.find(task_id));

  const auto &newly_reservable_tasks = s.notify_reserved(task_id);

  success_count += 1;
  enqueue_data_tasks(task_id);
  s.counts.count_reserved(task_id, device_id);
  s.update_reserved_cost(task_id, device_id);
  breakpoints.check_task_breakpoint(EventType::RESERVER, task_id);

  // Check if the reserved task is launchable, and if so, enqueue it
  if (s.is_launchable(task_id)) {
    SPDLOG_DEBUG("Time:{} Task {} is launchable", current_time, s.get_task_name(task_id));
    push_launchable(task_id, device_id);
  } else {
    s.unlaunched_compute_tasks.push_back(task_id);
  }

  return {true, &newly_reservable_tasks};
}

void Scheduler::reserve_tasks(ReserverEvent &reserve_event, EventManager &event_manager) {
  ZoneScoped;
  // Can't reserve tasks if we are in the middle of an eviction
  auto current_time = this->state.global_time;
  assert(this->eviction_state == EvictionState::NONE);

  auto &s = this->state;

  auto &reservable = queues.reservable;
  reservable.reset();
  reservable.current_or_next_active();

  SPDLOG_DEBUG("Time:{} Reserving tasks", current_time);
  SPDLOG_DEBUG("Time:{} Reservable Queue Size: {}", current_time,
               queues.reservable.total_active_size());
  bool break_flag = false;
  bool success_flag = false;
  tasks_requesting_eviction.clear();
  while (queues.has_active_reservable() && conditions.get().should_reserve(s, queues)) {

    if (is_breakpoint()) {
      break_flag = true;
      SPDLOG_DEBUG("Time:{} Breaking from reserver", current_time);
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
    success_flag = true;

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
    timecount_t reserver_time = current_time;
    event_manager.create_event(EventType::RESERVER, reserver_time);
    return;
  }
  if (!tasks_requesting_eviction.empty()) {
    // Should not start eviction if there are tasks not launchable due to data movement.
    if (s.unlaunched_compute_tasks.empty()) {
      SPDLOG_DEBUG("Time:{} Eviction will start for {} tasks", current_time,
                   tasks_requesting_eviction.size());
      this->eviction_state = EvictionState::WAITING_FOR_COMPLETION; // This should be set to false
                                                                    // after the eviction is over
    } else {
      SPDLOG_DEBUG("Time:{} Eviction will start after launching {} tasks", current_time,
                   s.unlaunched_compute_tasks.size());
    }
    // timecount_t launcher_time = s.global_time + TIME_TO_LAUNCH;
    // // The next event is a eviction event
    // event_manager.create_event(EventType::EVICTOR, launcher_time);
  }
  // else {
  // The next event is a launching event
  timecount_t launcher_time = current_time + TIME_TO_LAUNCH;
  event_manager.create_event(EventType::LAUNCHER, launcher_time);
  // }
}

bool Scheduler::launch_compute_task(taskid_t task_id, devid_t device_id,
                                    EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;

  SPDLOG_DEBUG("Time:{} Attempting to launch compute task {} on device {}", current_time,
               s.get_task_name(task_id), device_id);

  assert(s.is_launchable(task_id));
  assert(s.get_mapping(task_id) == device_id);
  assert(s.is_compute_task(task_id));

  const auto [requested, missing] = s.request_launch_resources(task_id, device_id);

  if (missing.vcu > 0) {
    SPDLOG_DEBUG("Time:{} Task {} requested {} VCU but missing {} VCU", s.get_task_name(task_id),
                 current_time, requested.vcu, missing.vcu);
    return false;
  }

  const auto &task = s.task_manager.get_tasks().get_compute_task(task_id);

  SPDLOG_DEBUG("Time:{} Launching compute task {} on device {}", current_time,
               s.get_task_name(task_id), device_id);

  // Update data locations for WRITE data (create them here)
  s.data_manager.read_update_launched(task.get_write(), device_id,
                                      current_time); // This adds memory
  s.data_manager.write_update_launched(task.get_write(), device_id,
                                       current_time); // This invalidates other devices

  // All READ data should already be here (prefetched by data tasks)
  assert(s.data_manager.check_valid_launched(task.get_read(), device_id));

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
  event_manager.create_event(EventType::COMPLETER, completion_time, task_id);

  return true;
}

bool Scheduler::launch_data_task(taskid_t task_id, devid_t destination_id,
                                 EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;

  SPDLOG_DEBUG("Time:{} Attempting to launch data task {} on device {}", s.global_time,
               s.get_task_name(task_id), destination_id);

  assert(s.is_launchable(task_id));

  const auto &task = s.task_manager.get_tasks().get_data_task(task_id);
  const dataid_t data_id = task.get_data_id();

  auto [found, source_id] = s.data_manager.request_source(data_id, destination_id);

  if (!found) {
    SPDLOG_DEBUG("Time:{} Data task {} missing available source", s.global_time,
                 s.get_task_name(task_id));
    return false;
  }
  s.task_manager.set_source(task_id, source_id);
  auto duration = s.data_manager.start_move(data_id, source_id, destination_id, current_time);

  if (duration.is_virtual) {
    SPDLOG_DEBUG("Time:{} Data task {} is virtual", s.global_time, s.get_task_name(task_id));
    s.task_manager.set_virtual(task_id);
  } else {
    SPDLOG_DEBUG("Time:{} Data task {} moving from {} to {}", s.global_time,
                 s.get_task_name(task_id), source_id, destination_id);
  }

  // Record launching time
  s.notify_launched(task_id);
  success_count += 1;
  breakpoints.check_task_breakpoint(EventType::LAUNCHER, task_id);

  // Create completion event
  timecount_t completion_time = s.global_time + duration.duration;
  event_manager.create_event(EventType::COMPLETER, completion_time, task_id);

  return true;
}

bool Scheduler::launch_eviction_task(taskid_t task_id, devid_t destination_id,
                                     EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;

  SPDLOG_DEBUG("Time:{} Attempting to launch eviction task {} on device {}", current_time, task_id,
               destination_id);

  assert(s.is_launchable(task_id));

  const auto &task = s.task_manager.get_eviction_task(task_id);
  const dataid_t data_id = task.get_data_id();

  auto [found, source_id] = s.data_manager.request_source(data_id, destination_id);

  if (!found) {
    SPDLOG_DEBUG("Time:{} Eviction task {} missing available source for block {}", s.global_time,
                 s.get_task_name(task_id), data_id);
    return false;
  }
  SPDLOG_DEBUG("Time:{} Eviction task {} found source {} for block {}", current_time, task_id,
               source_id, data_id);

  s.task_manager.set_source(task_id, source_id);
  auto duration = s.data_manager.start_move(data_id, source_id, destination_id, current_time);

  if (duration.is_virtual) {
    SPDLOG_DEBUG("Time:{} Eviction task {} is virtual", current_time, s.get_task_name(task_id));
    s.task_manager.set_virtual(task_id);
  } else {
    SPDLOG_DEBUG("Time:{} Eviction task {} moving from {} to {}", current_time,
                 s.get_task_name(task_id), source_id, destination_id);
  }

  // Record launching time
  s.notify_launched(task_id);
  success_count += 1;

  // Create completion event
  timecount_t completion_time = s.global_time + duration.duration;
  event_manager.create_event(EventType::COMPLETER, completion_time, task_id);

  return true;
}

void Scheduler::launch_tasks(LauncherEvent &launch_event, EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;

  auto &launchable = queues.launchable;
  launchable.reset();
  launchable.current_or_next_active();

  SPDLOG_DEBUG("Time:{} Launching task", current_time);
  SPDLOG_DEBUG("Time:{} Launchable Queue Size: {}", current_time,
               queues.launchable.total_active_size());
  bool break_flag = false;

  while (queues.has_active_launchable() && conditions.get().should_launch(s, queues)) {

    if (is_breakpoint()) {
      SPDLOG_DEBUG("Time:{} Breaking from launcher", current_time);
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
    compute_on_fly += 1;
    launchable.pop();
    launchable.next();
  }

  if (break_flag) {
    event_manager.create_event(EventType::LAUNCHER, current_time);
    return;
  }

  auto &eviction_launchable = queues.eviction_launchable;
  eviction_launchable.reset();
  eviction_launchable.current_or_next_active();

  SPDLOG_DEBUG("Time:{} Eviction Launchable Queue Size: {}", current_time,
               queues.eviction_launchable.total_active_size());

  while (queues.has_active_eviction_launchable() &&
         conditions.get().should_launch_data(s, queues)) {

    SPDLOG_DEBUG("Time:{} Inner eviction launch loop, queue Size: {}", current_time,
                 eviction_launchable.total_active_size());

    if (is_breakpoint()) {
      SPDLOG_DEBUG("Time:{} Breaking from eviction launcher", current_time);
      break_flag = true;
      break;
    }

    if (eviction_launchable.get_active().empty()) {
      SPDLOG_DEBUG("Time:{} Empty eviction launchable queue", current_time);
      eviction_launchable.next();
      continue;
    }

    taskid_t task_id = eviction_launchable.top();
    assert(s.is_eviction_task(task_id));
    auto device_id = static_cast<devid_t>(eviction_launchable.get_active_index());
    // This should always be the host device
    assert(device_id == HOST_ID);

    bool success = launch_eviction_task(task_id, device_id, event_manager);
    if (!success) {
      eviction_launchable.deactivate();
      eviction_launchable.next();
      continue;
    }
    eviction_launchable.pop();
    eviction_launchable.next();
  }

  if (break_flag) {
    event_manager.create_event(EventType::LAUNCHER, current_time);
    return;
  }

  SPDLOG_DEBUG("Time:{} Launching data tasks", current_time);

  auto &data_launchable = queues.data_launchable;
  data_launchable.reset();
  data_launchable.current_or_next_active();
  SPDLOG_DEBUG("Time:{} Data Launchable Queue Size: {}", current_time,
               queues.data_launchable.total_active_size());

  while (queues.has_active_data_launchable() && conditions.get().should_launch_data(s, queues)) {

    if (is_breakpoint()) {
      SPDLOG_DEBUG("Time:{} Breaking from data launcher", current_time);
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
    event_manager.create_event(EventType::LAUNCHER, current_time);
    return;
  }

  scheduler_event_count -= 1;
  if (scheduler_event_count == 0 and success_count > 0) {
    if (this->eviction_state != EvictionState::NONE) {
      if (this->eviction_state == EvictionState::RUNNING &&
          eviction_count == 0) { // Sometimes eviction completes before launcher
        SPDLOG_DEBUG("Time:{} Evictor finished", current_time);
        event_manager.create_event(EventType::RESERVER, current_time);
        this->eviction_state = EvictionState::NONE;
      } else
        return;
    } else
      event_manager.create_event(EventType::MAPPER,
                                 current_time + SCHEDULER_TIME_GAP + TIME_TO_MAP);
    scheduler_event_count += 1;
  }
}

void Scheduler::evict(EvictorEvent &eviction_event, EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto &launchable = queues.launchable;
  auto &task_manager = s.task_manager;
  auto &data_launchable = queues.data_launchable;
  const auto &data_manager = s.data_manager;
  const auto &lru_manager = s.data_manager.get_lru_manager();
  const auto &mapped = s.mapped_but_not_reserved_tasks;
  const auto &tasks = task_manager.get_tasks();

  auto current_time = s.global_time;

  if (eviction_state == EvictionState::WAITING_FOR_COMPLETION) {
    if (launchable.total_size() + data_launchable.total_size() + compute_on_fly) {
      SPDLOG_DEBUG("Time:{} Evictor waiting for all {} compute and {} data task to finish",
                   current_time, launchable.total_size(), data_launchable.total_size());
      event_manager.create_event(EventType::LAUNCHER, current_time);
      return;
    } else {
      SPDLOG_DEBUG("Starting evictor at {}", current_time);
      eviction_count = 0;

      for (auto &taskdevice : tasks_requesting_eviction) {
        auto [task_id, device_id] = taskdevice;
        const auto [requested, missing] = s.request_reserve_resources(task_id, device_id);
        if (missing.mem) { // There is still memory to evict
          const auto &task = tasks.get_compute_task(task_id);
          auto &data_ids = lru_manager.getLRUids(device_id, missing.mem, task.get_unique());
          for (auto data_id : data_ids) {
            auto sources = data_manager.get_valid_launched_locations(data_id);
            assert(!sources.empty());
            if (sources.size() == 1) {
              eviction_count += 1;
              auto eviction_task_id = task_manager.create_eviction_task(task_id, data_id, HOST_ID,
                                                                        device_id, current_time);

              SPDLOG_DEBUG(
                  "Time:{} Launching eviction task {} to evict block {} for task {} on device {} ",
                  current_time, s.get_task_name(eviction_task_id), data_id,
                  s.get_task_name(task_id), device_id);
              push_launchable_eviction(eviction_task_id);
            } else { // There are multiple sources for this data
              // We need to invalidate the data on the device
              // Invalidate the data
              bool future_usage = false;
              for (auto unreserved_task_id : mapped) {
                if (task_manager.get_state().get_mapping(unreserved_task_id) == device_id) {
                  auto &uniq = tasks.get_compute_task(unreserved_task_id).get_unique();
                  if (std::find(uniq.begin(), uniq.end(), data_id) != uniq.end()) {
                    future_usage = true;
                    break;
                  }
                }
              }

              bool write_after_read = false;
              std::unordered_set<taskid_t> buffer_set;
              for (auto unreserved_task_id : mapped) {
                auto &write = tasks.get_compute_task(unreserved_task_id).get_write();
                if (std::find(write.begin(), write.end(), data_id) != write.end()) {
                  buffer_set.insert(unreserved_task_id);
                }
              }

              if (!buffer_set.empty()) {
                // 1) Find the unique “top” task in buffer_set
                std::vector<taskid_t> top_tasks;

                for (auto tid : buffer_set) {
                  bool has_buffer_predecessor = false;
                  std::stack<taskid_t> stk;
                  std::unordered_set<taskid_t> visited;

                  stk.push(tid);
                  visited.insert(tid);

                  while (!stk.empty() && !has_buffer_predecessor) {
                    auto curr = stk.top();
                    stk.pop();

                    for (auto dep : tasks.get_compute_task(curr).get_dependencies()) {
                      if (!mapped.count(dep))
                        continue;

                      if (buffer_set.count(dep) && dep != tid) {
                        has_buffer_predecessor = true;
                        break;
                      }
                      if (visited.insert(dep).second) {
                        stk.push(dep);
                      }
                    }
                  }

                  if (!has_buffer_predecessor) {
                    top_tasks.push_back(tid);
                  }
                }

                assert(top_tasks.size() == 1);
                taskid_t top = top_tasks.front();

                // 2) Now do a full DFS from top down its .get_dependencies(),
                //    collecting all reachable deps that are in mapped
                std::vector<taskid_t> top_task_dependencies;
                std::stack<taskid_t> stk;
                std::unordered_set<taskid_t> visited;

                // start from each direct dependency
                for (auto dep0 : tasks.get_compute_task(top).get_dependencies()) {
                  if (mapped.count(dep0) && visited.insert(dep0).second) {
                    top_task_dependencies.push_back(dep0);
                    stk.push(dep0);
                  }
                }

                while (!stk.empty()) {
                  auto curr = stk.top();
                  stk.pop();

                  for (auto dep : tasks.get_compute_task(curr).get_dependencies()) {
                    if (!mapped.count(dep))
                      continue;
                    if (visited.insert(dep).second) {
                      top_task_dependencies.push_back(dep);
                      stk.push(dep);
                    }
                  }
                }

                // 'top' is your head write-task
                // 'top_task_dependencies' now contains *all* of its dependencies
                // (direct and indirect), in the order first discovered.
                if (s.get_mapping(top) != device_id) {
                  write_after_read = true;
                  for (auto dep : top_task_dependencies) {
                    if (s.get_mapping(dep) == device_id) {
                      bool is_read = false;
                      for (auto read_data_id : tasks.get_compute_task(dep).get_read()) {
                        if (read_data_id == data_id)
                          is_read = true;
                      }
                      if (!is_read)
                        continue;
                      write_after_read = false;
                      break;
                    }
                  }
                }
              }

              SPDLOG_DEBUG("Time:{} Invalidating block {} for task {} on device {}", current_time,
                           data_id, s.get_task_name(task_id), device_id);
              s.data_manager.evict_on_update_launched(DataIDList{data_id}, device_id, current_time,
                                                      future_usage, write_after_read);
            }
          }
        } else {
          SPDLOG_DEBUG("Time:{} No need to evict for task {} on device {}", current_time,
                       s.get_task_name(task_id), device_id);
        }
      }

      SPDLOG_DEBUG("Time:{} Evictor launched {} eviction tasks", current_time, eviction_count);
      eviction_state = EvictionState::RUNNING;
    }
  }
  if (eviction_state == EvictionState::RUNNING) {
    if (eviction_count) {
      SPDLOG_DEBUG("Time:{} Evictor waiting for all eviction tasks to finish", current_time);
      event_manager.create_event(EventType::LAUNCHER, current_time);
      return;
    } else {
      SPDLOG_DEBUG("Time:{} Evictor finished", current_time);
      event_manager.create_event(EventType::RESERVER, current_time);
      this->eviction_state = EvictionState::NONE;
    }
  }
}

void Scheduler::complete_compute_task(taskid_t task_id, devid_t device_id) {
  ZoneScoped;
  auto &s = this->state;
  auto &task = s.task_manager.get_tasks().get_compute_task(task_id);
  SPDLOG_DEBUG("Time:{} Completing compute task {} on device {}", s.global_time,
               s.get_task_name(task_id), device_id);

  // Free mapped, reserved, and launched resources
  s.free_resources(task_id);
  s.counts.count_completed(task_id, device_id);
  s.update_completed_cost(task_id, device_id);
  // Remove retired data
  for (auto data_id : task.get_retire()) {
    s.data_manager.retire_data(data_id, device_id, s.global_time);
  }

  const auto &newly_launchable_data_tasks = s.notify_data_completed(task_id);
  push_launchable_data(newly_launchable_data_tasks);
  // spdlog::debug("Newly launchable data tasks: {}",
  //               newly_launchable_data_tasks.size());
  compute_on_fly -= 1;
}

void Scheduler::complete_data_task(taskid_t task_id, devid_t destination_id) {
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;
  s.counts.count_data_completed(task_id, destination_id);

  SPDLOG_DEBUG("Time:{} Completing data task {} on device {}", s.global_time,
               s.get_task_name(task_id), destination_id);

  auto source_id = s.task_manager.get_source(task_id);
  auto is_virtual = s.task_manager.is_virtual(task_id);
  auto data_id = s.task_manager.get_tasks().get_data_task(task_id).get_data_id();
  s.data_manager.complete_move(data_id, source_id, destination_id, is_virtual, current_time);
}

void Scheduler::complete_eviction_task(taskid_t eviction_task_id, devid_t destination_id) {
  auto &s = this->state;
  auto current_time = s.global_time;
  s.counts.count_data_completed(eviction_task_id, destination_id);

  SPDLOG_DEBUG("Time:{} Completing eviction task {} on device {}", current_time,
               s.get_task_name(eviction_task_id), destination_id);

  auto source_id = s.task_manager.get_source(eviction_task_id);
  auto is_virtual = s.task_manager.is_virtual(eviction_task_id);
  const auto &eviction_task = s.task_manager.get_eviction_task(eviction_task_id);
  auto data_id = eviction_task.get_data_id();
  s.data_manager.complete_eviction_move(data_id, source_id, destination_id, is_virtual,
                                        current_time);
  s.device_manager.add_mem<TaskState::MAPPED>(
      destination_id, s.data_manager.get_data().get_size(data_id), current_time);
  s.device_manager.add_mem<TaskState::RESERVED>(
      destination_id, s.data_manager.get_data().get_size(data_id), current_time);

  auto invalidate_device_id = eviction_task.get_invalidate_device();

  bool future_usage = false;
  for (auto unreserved_task_id : s.mapped_but_not_reserved_tasks) {
    if (s.task_manager.get_state().get_mapping(unreserved_task_id) == invalidate_device_id) {
      auto &uniq = s.task_manager.get_tasks().get_compute_task(unreserved_task_id).get_unique();
      if (std::find(uniq.begin(), uniq.end(), data_id) != uniq.end()) {
        future_usage = true;
        break;
      }
    }
  }

  bool write_after_read = false;
  const auto &tasks = s.task_manager.get_tasks();
  const auto &mapped = s.mapped_but_not_reserved_tasks;
  std::unordered_set<taskid_t> buffer_set;
  for (auto unreserved_task_id : mapped) {
    auto &write = tasks.get_compute_task(unreserved_task_id).get_write();
    if (std::find(write.begin(), write.end(), data_id) != write.end()) {
      buffer_set.insert(unreserved_task_id);
    }
  }

  if (!buffer_set.empty()) {
    // 1) Find the unique “top” task in buffer_set
    std::vector<taskid_t> top_tasks;

    for (auto tid : buffer_set) {
      bool has_buffer_predecessor = false;
      std::stack<taskid_t> stk;
      std::unordered_set<taskid_t> visited;

      stk.push(tid);
      visited.insert(tid);

      while (!stk.empty() && !has_buffer_predecessor) {
        auto curr = stk.top();
        stk.pop();

        for (auto dep : tasks.get_compute_task(curr).get_dependencies()) {
          if (!mapped.count(dep))
            continue;

          if (buffer_set.count(dep) && dep != tid) {
            has_buffer_predecessor = true;
            break;
          }
          if (visited.insert(dep).second) {
            stk.push(dep);
          }
        }
      }

      if (!has_buffer_predecessor) {
        top_tasks.push_back(tid);
      }
    }

    assert(top_tasks.size() == 1);
    taskid_t top = top_tasks.front();

    // 2) Now do a full DFS from top down its .get_dependencies(),
    //    collecting all reachable deps that are in mapped
    std::vector<taskid_t> top_task_dependencies;
    std::stack<taskid_t> stk;
    std::unordered_set<taskid_t> visited;

    // start from each direct dependency
    for (auto dep0 : tasks.get_compute_task(top).get_dependencies()) {
      if (mapped.count(dep0) && visited.insert(dep0).second) {
        top_task_dependencies.push_back(dep0);
        stk.push(dep0);
      }
    }

    while (!stk.empty()) {
      auto curr = stk.top();
      stk.pop();

      for (auto dep : tasks.get_compute_task(curr).get_dependencies()) {
        if (!mapped.count(dep))
          continue;
        if (visited.insert(dep).second) {
          top_task_dependencies.push_back(dep);
          stk.push(dep);
        }
      }
    }

    // 'top' is your head write-task
    // 'top_task_dependencies' now contains *all* of its dependencies
    // (direct and indirect), in the order first discovered.
    if (s.get_mapping(top) != invalidate_device_id) {
      write_after_read = true;
      for (auto dep : top_task_dependencies) {
        if (s.get_mapping(dep) == invalidate_device_id) {
          bool is_read = false;
          for (auto read_data_id : tasks.get_compute_task(dep).get_read()) {
            if (read_data_id == data_id)
              is_read = true;
          }
          if (!is_read)
            continue;
          write_after_read = false;
          break;
        }
      }
    }
  }

  s.data_manager.evict_on_update_launched(DataIDList{data_id}, invalidate_device_id, current_time,
                                          future_usage, write_after_read);

  eviction_count -= 1;
  SPDLOG_DEBUG("Time:{} Eviction task {} completed {} left", current_time,
               s.get_task_name(eviction_task_id), eviction_count);
}

void Scheduler::complete_task(CompleterEvent &complete_event, EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  taskid_t task_id = complete_event.task;

  if (s.is_eviction_task(task_id)) {
    const auto &eviction_task = s.task_manager.get_eviction_task(task_id);
    devid_t device_id = eviction_task.get_device_id();
    complete_eviction_task(task_id, device_id);
  } else if (s.is_compute_task(task_id)) {
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
  auto it = s.unlaunched_compute_tasks.begin();
  while (it != s.unlaunched_compute_tasks.end()) {
    if (s.is_launchable(*it)) {
      // erase returns the iterator following the erased element
      it = s.unlaunched_compute_tasks.erase(it);
    } else {
      ++it;
    }
  }
  // spdlog::debug("Newly launchable compute tasks: {}",
  //               newly_launchable_tasks.size());

  success_count += 1;
  breakpoints.check_task_breakpoint(EventType::COMPLETER, task_id);
  if (scheduler_event_count == 0) {
    if (this->eviction_state == EvictionState::WAITING_FOR_COMPLETION) {
      event_manager.create_event(EventType::EVICTOR, s.global_time + SCHEDULER_TIME_GAP);
    } else if (this->eviction_state == EvictionState::RUNNING) {
      if (eviction_count) {
        event_manager.create_event(EventType::LAUNCHER, s.global_time + SCHEDULER_TIME_GAP);
      } else {
        event_manager.create_event(EventType::RESERVER, s.global_time + SCHEDULER_TIME_GAP);
        this->eviction_state = EvictionState::NONE;
      }
    } else {
      event_manager.create_event(EventType::MAPPER,
                                 s.global_time + SCHEDULER_TIME_GAP + TIME_TO_MAP);
    }
    scheduler_event_count += 1;
  }
  if (spdlog::get_level() == spdlog::level::debug) {
    auto &device_manager = s.get_device_manager();
    auto &lru_manager = s.get_data_manager().get_lru_manager();
    // check memory state, whether it is consistent for debugging purpose
    bool flag = false;
    for (devid_t i = 0; i < device_manager.get_devices().size(); i++) {
      mem_t launched_mem = device_manager.get_mem<TaskState::LAUNCHED>(i);
      mem_t reserved_mem = device_manager.get_mem<TaskState::RESERVED>(i);
      mem_t mapped_mem = device_manager.get_mem<TaskState::MAPPED>(i);
      mem_t lru_mem = lru_manager.get_mem(i);
      SPDLOG_DEBUG("Device {}: launched {}, lru {}, reserved {}, mapped {}", i, launched_mem,
                   lru_mem, reserved_mem, mapped_mem);
      if (i > 0 && mapped_mem < launched_mem)
        flag = true;
      assert(launched_mem == lru_mem);
    }
    if (flag) {
      SPDLOG_DEBUG("Memory state is inconsistent");
    }
  }
}