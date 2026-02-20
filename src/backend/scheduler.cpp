#include "scheduler.hpp"
#include "data.hpp"
#include "devices.hpp"
#include "events.hpp"
#include "macros.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include "tasks.hpp"
#include <cstdint>
#include <iostream>

// Scheduler

namespace {
[[nodiscard]] inline uint64_t pack_eviction_key(dataid_t data_id, devid_t device_id) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(device_id)) << 32) |
         static_cast<uint32_t>(data_id);
}
} // namespace

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

taskid_t Scheduler::map_task(taskid_t compute_task_id, Action &action) {
  ZoneScoped;
  auto &s = state;
  auto &task_runtime = s.task_runtime;
  auto &static_graph = s.get_tasks();
  auto &data_manager = s.data_manager;
  auto current_time = s.global_time;
  const auto &data = s.get_data();
  taskid_t mappable_write_idx = 0;

  devid_t chosen_device = action.device;

  SPDLOG_DEBUG("Time:{} Mapping task {}:{} to device {}", current_time,
               static_graph.get_compute_task_name(compute_task_id), compute_task_id, chosen_device);

  assert(task_runtime.is_compute_mappable(compute_task_id));

  priority_t rp = action.reservable_priority;
  priority_t lp = action.launchable_priority;
  SPDLOG_DEBUG("Time:{} Reservable priority: {}, Launchable priority: {}", current_time, rp, lp);

  // Update mapped resources
  auto [requested, missing] = s.request_map_resources(compute_task_id, chosen_device);
  s.map_resources(compute_task_id, chosen_device, requested);

  // Update data locations
  auto &device_manager = s.get_device_manager();
  data_manager.read_update_mapped(data, device_manager, static_graph.get_unique(compute_task_id),
                                  chosen_device, current_time);
  data_manager.write_update_mapped(data, device_manager, static_graph.get_write(compute_task_id),
                                   chosen_device, current_time);
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

  s.mapped_but_not_reserved_tasks.insert(compute_task_id);
  s.is_mapped_not_reserved[static_cast<std::size_t>(compute_task_id)] = 1;

  // Maintain incremental future_usage counters for eviction
  auto &unique_count_map = s.mapped_unique_count[chosen_device];
  for (auto d : static_graph.get_unique(compute_task_id)) {
    unique_count_map[d]++;
  }

  // Maintain write index: data_id → set of mapped-not-reserved writers (avoids roaring bitmap scan)
  for (auto d : static_graph.get_write(compute_task_id)) {
    s.mapped_write_by_data[d].insert(compute_task_id);
  }

  // Maintain read-count index: enables O(1) precheck to skip dependency DFS
  auto &read_count_map = s.mapped_read_count_by_data_device[chosen_device];
  for (auto d : static_graph.get_read(compute_task_id)) {
    read_count_map[d]++;
  }

  // Notify dependents and enqueue newly mappable tasks
  task_runtime.compute_notify_mapped(compute_task_id, chosen_device, rp, lp, current_time,
                                     static_graph, compute_task_buffer);
  s.update_mapped_cost(compute_task_id, chosen_device);
  success_count += 1;
  breakpoints.decrement_steps();

  // Check if the mapped task is reservable, and if so, enqueue it
  if (task_runtime.is_compute_reservable(compute_task_id)) {
    SPDLOG_DEBUG("Time:{} Task {} is reservable", current_time,
                 static_graph.get_compute_task_name(compute_task_id));
    // TODO(wlr): Check if delayed enqueue is faster
    push_reservable(compute_task_id, chosen_device);
  }

  return compute_task_buffer.size();
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

  python_mapper_buffer.clear();

  if (!action_list.empty()) {
    for (auto &action : action_list) {
      const auto task_id = top_k_tasks[action.pos];
      map_task(task_id, action);

      python_mapper_buffer.reserve(python_mapper_buffer.size() + compute_task_buffer.size());
      std::copy(compute_task_buffer.begin(), compute_task_buffer.end(),
                std::back_inserter(python_mapper_buffer));
    }

    remove_mapped_tasks(action_list);
    SPDLOG_DEBUG("Time:{} Newly mappable tasks: {}", state.global_time,
                 python_mapper_buffer.size());
    push_mappable(python_mapper_buffer);
  }

  /*If we still should be mapping, continue making calls to the mapper */

  if (queues.has_mappable() && conditions.get().should_map(state, queues)) {
    return ExecutionState::EXTERNAL_MAPPING;
  } else {

    if (is_breakpoint()) {
      // TODO(wlr): Currently breakpoints of Python mappers are broken.
      // TODO(wlr): Not sure if this is still true. Need to test.
      SPDLOG_DEBUG("Time:{} Breaking from mapper", state.global_time);
      timecount_t mapper_time = state.global_time;
      event_manager.create_event(EventType::MAPPER, mapper_time);
      return ExecutionState::BREAKPOINT;
    } else {
      SPDLOG_DEBUG("Time:{} Ending mapper", state.global_time);
      timecount_t reserver_time = state.global_time + TIME_TO_RESERVE;
      event_manager.create_event(EventType::RESERVER, reserver_time);
      return ExecutionState::RUNNING;
    }
  }
}

void Scheduler::skip_map_tasks(MapperEvent &map_event, EventManager &event_manager) {
  success_count = 0;
  SPDLOG_DEBUG("Time:{} Skipping mapper", state.global_time);
  timecount_t reserver_time = state.global_time + SCHEDULER_TIME_GAP;
  event_manager.create_event(EventType::RESERVER, reserver_time);
}

void Scheduler::skip_reserve_tasks(ReserverEvent &reserve_event, EventManager &event_manager) {
  SPDLOG_DEBUG("Time:{} Skipping reserver", state.global_time);
  timecount_t launcher_time = state.global_time + SCHEDULER_TIME_GAP;
  event_manager.create_event(EventType::LAUNCHER, launcher_time);
}

void Scheduler::map_tasks(MapperEvent &map_event, EventManager &event_manager, Mapper &mapper) {
  ZoneScoped;

  success_count = 0;
  auto &s = this->state;
  auto &task_runtime = s.task_runtime;
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
    assert(task_runtime.is_compute_mappable(task_id));
    Action action = mapper.map_task(task_id, s);
    map_task(task_id, action);

    push_mappable(compute_task_buffer);
  }

  if (break_flag) {
    timecount_t mapper_time = current_time;
    event_manager.create_event(EventType::MAPPER, mapper_time);
  } else {
    timecount_t reserver_time = current_time + SCHEDULER_TIME_GAP;
    event_manager.create_event(EventType::RESERVER, reserver_time);
  }
}

void Scheduler::enqueue_data_tasks(taskid_t compute_task_id) {
  auto &s = this->state;
  auto &task_runtime = s.task_runtime;
  auto &static_graph = s.get_tasks();
  auto current_time = s.global_time;

  devid_t mapped_device = task_runtime.get_compute_task_mapped_device(compute_task_id);
  const auto data_dependencies = static_graph.get_compute_task_data_dependencies(compute_task_id);

  SPDLOG_DEBUG("Time:{} Enqueueing {} data tasks for task {}", current_time,
               data_dependencies.size(), static_graph.get_compute_task_name(compute_task_id));

  for (auto data_task_id : data_dependencies) {
    task_runtime.data_notify_reserved(data_task_id, mapped_device, current_time, static_graph);
    task_runtime.set_data_task_launch_priority(
        data_task_id, task_runtime.get_compute_task_launch_priority(compute_task_id));
    s.update_data_reserved_cost(data_task_id, mapped_device);
    if (task_runtime.is_data_launchable(data_task_id)) {
      SPDLOG_DEBUG("Time:{} Data task {}:{} is launchable", current_time,
                   static_graph.get_data_task_name(data_task_id), data_task_id);
      push_launchable_data(data_task_id);
    }
  }
}

bool Scheduler::reserve_task(taskid_t compute_task_id, devid_t device_id) {
  ZoneScoped;
  auto &s = this->state;
  auto &task_runtime = s.task_runtime;
  auto &static_graph = s.get_tasks();
  auto current_time = s.global_time;
  auto &mapped = s.mapped_but_not_reserved_tasks;
  auto &device_manager = s.get_device_manager();
  const auto &data = s.get_data();

  assert(task_runtime.is_compute_reservable(compute_task_id));
  assert(task_runtime.get_compute_task_mapped_device(compute_task_id) == device_id);

  SPDLOG_DEBUG("Time:{} Attempting to reserve task {} on device {}", current_time,
               static_graph.get_compute_task_name(compute_task_id), device_id);

  // Get total required task memory
  const auto [requested, missing] = s.request_reserve_resources(compute_task_id, device_id);

  if (missing.mem > 0) {
    SPDLOG_DEBUG(
        "Time:{} Task {} will evict memory on device {} since requested {} memory but missing {} "
        "memory",
        current_time, static_graph.get_compute_task_name(compute_task_id), device_id, requested.mem,
        missing.mem);
    tasks_requesting_eviction.push_back(std::make_tuple(compute_task_id, device_id));
    return false;
  }

  // Update reserved resources
  s.reserve_resources(compute_task_id, device_id, requested);
  SPDLOG_DEBUG("Time:{} Task {} requested memsize {} resulting in reserved size of {} at device {}",
               current_time, static_graph.get_compute_task_name(compute_task_id), requested.mem,
               device_manager.get_mem<TaskState::RESERVED>(device_id), device_id);

  // Update data locations
  s.data_manager.read_update_reserved(
      data, device_manager, static_graph.get_unique(compute_task_id), device_id, current_time);
  s.data_manager.write_update_reserved(
      data, device_manager, static_graph.get_write(compute_task_id), device_id, current_time);

  // erase task_id from s.mapped_but_not_reserved_tasks
  mapped.erase(mapped.find(compute_task_id));
  s.is_mapped_not_reserved[static_cast<std::size_t>(compute_task_id)] = 0;

  // Maintain incremental future_usage counters for eviction
  auto &unique_count_map = s.mapped_unique_count[device_id];
  for (auto d : static_graph.get_unique(compute_task_id)) {
    auto &cnt = unique_count_map[d];
    if (--cnt == 0) {
      unique_count_map.erase(d);
    }
  }

  // Remove from write index
  for (auto d : static_graph.get_write(compute_task_id)) {
    auto &writers = s.mapped_write_by_data[d];
    writers.erase(compute_task_id);
    if (writers.empty()) {
      s.mapped_write_by_data.erase(d);
    }
  }

  // Remove from read-count index
  auto &read_count_map = s.mapped_read_count_by_data_device[device_id];
  for (auto d : static_graph.get_read(compute_task_id)) {
    auto &cnt = read_count_map[d];
    if (--cnt == 0) {
      read_count_map.erase(d);
    }
  }

  task_runtime.compute_notify_reserved(compute_task_id, device_id, current_time, static_graph,
                                       compute_task_buffer);

  success_count += 1;
  enqueue_data_tasks(compute_task_id);
  s.update_reserved_cost(compute_task_id, device_id);

  // Check if the reserved task is launchable, and if so, enqueue it
  if (task_runtime.is_compute_launchable(compute_task_id)) {
    SPDLOG_DEBUG("Time:{} Task {} is launchable", current_time,
                 static_graph.get_compute_task_name(compute_task_id));
    push_launchable(compute_task_id, device_id);
  }

  return true;
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
    bool success = reserve_task(task_id, device_id);
    if (!success) {
      reservable.deactivate();
      reservable.next();
      continue;
    }
    success_flag = true;

    reservable.pop();

    push_reservable(compute_task_buffer);

    // Cycle to the next active device queue
    reservable.next();
  }

  if (break_flag) [[unlikely]] {
    timecount_t reserver_time = current_time;
    event_manager.create_event(EventType::RESERVER, reserver_time);
    return;
  }

  // NOTE(wlr): Shouldn't eviction pause new tasks from being reserved? We can just wait for current
  // tasks to complete?

  if (!tasks_requesting_eviction.empty()) {
    // Should not start eviction if there are tasks not launchable due to data movement.
    if (s.counts.n_unlaunched_reserved() == 0) {
      SPDLOG_DEBUG("Time:{} Eviction will start for {} tasks", current_time,
                   tasks_requesting_eviction.size());
      this->eviction_state = EvictionState::WAITING_FOR_COMPLETION; // This should be set to false
                                                                    // after the eviction is over
      // Create an event to start the eviction process
      event_manager.create_event(EventType::EVICTOR, current_time + SCHEDULER_TIME_GAP);
      return;
    } else {
      SPDLOG_DEBUG("Time:{} Eviction will start after launching {} tasks", current_time,
                   s.counts.n_unlaunched_reserved());
    }
  }

  timecount_t launcher_time = current_time + TIME_TO_LAUNCH;
  event_manager.create_event(EventType::LAUNCHER, launcher_time);
}

bool Scheduler::launch_compute_task(taskid_t compute_task_id, devid_t device_id,
                                    EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;
  auto &task_runtime = s.task_runtime;
  auto &data_manager = s.data_manager;
  const auto &static_graph = s.get_tasks();
  auto &device_manager = s.get_device_manager();
  const auto &data = s.get_data();

  SPDLOG_DEBUG("Time:{} Attempting to launch compute task {}:{} on device {}", current_time,
               static_graph.get_compute_task_name(compute_task_id), compute_task_id, device_id);

  assert(task_runtime.is_compute_launchable(compute_task_id));
  assert(task_runtime.get_compute_task_mapped_device(compute_task_id) == device_id);

  const auto [requested, missing] = s.request_launch_resources(compute_task_id, device_id);

  if (missing.vcu > 0) {
    SPDLOG_DEBUG("Time:{} Task {}:{} requested {} VCU but missing {} VCU",
                 static_graph.get_compute_task_name(compute_task_id), compute_task_id, current_time,
                 requested.vcu, missing.vcu);
    return false;
  }

  // Update data locations for WRITE data (create them here)
  auto write_data = static_graph.get_write(compute_task_id);
  data_manager.read_update_launched(data, device_manager, write_data, device_id,
                                    current_time); // This adds memory
  data_manager.write_update_launched(data, device_manager, write_data, device_id,
                                     current_time); // This invalidates other devices

  // All READ data should already be here (prefetched by data tasks)
  assert(data_manager.check_valid_launched(static_graph.get_read(compute_task_id), device_id));

  // Update launched resources
  s.launch_resources(compute_task_id, device_id, requested);

  // Record launching time
  task_runtime.compute_notify_launched(compute_task_id, current_time, static_graph);
  success_count += 1;
  s.update_launched_cost(compute_task_id, device_id);

  // Create completion event
  timecount_t execution_time = s.get_execution_time(compute_task_id);
  SPDLOG_DEBUG("Time:{} Launching compute task {}:{} with execution time {}", current_time,
               static_graph.get_compute_task_name(compute_task_id), compute_task_id,
               execution_time);
  timecount_t completion_time = s.global_time + execution_time;
  event_manager.create_event(EventType::COMPUTE_COMPLETER, completion_time, compute_task_id,
                             device_id);

  return true;
}

bool Scheduler::launch_data_task(taskid_t data_task_id, devid_t destination_device_id,
                                 EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;
  auto &task_runtime = s.task_runtime;
  auto &data_manager = s.data_manager;
  const auto &static_graph = s.get_tasks();
  const auto &data = s.get_data();
  auto &comm_manager = s.get_communication_manager();
  auto &device_manager = s.get_device_manager();

  SPDLOG_DEBUG("Time:{} Attempting to launch data task {}:{} on device {}", current_time,
               static_graph.get_data_task_name(data_task_id), data_task_id, destination_device_id);

  assert(task_runtime.is_data_launchable(data_task_id));
  assert(task_runtime.get_data_task_mapped_device(data_task_id) == destination_device_id);

  const dataid_t data_id = static_graph.get_data_id(data_task_id);
  const auto &topology = s.get_topology();
  auto [found, source_device_id] =
      data_manager.request_source(topology, comm_manager, data_id, destination_device_id);

  if (!found) {
    SPDLOG_DEBUG("Time:{} Data task {}:{} missing available source", current_time,
                 static_graph.get_data_task_name(data_task_id), data_task_id);
    return false;
  }
  auto duration = data_manager.start_move(topology, comm_manager, device_manager, data, data_id,
                                          source_device_id, destination_device_id, current_time);

  if (duration.is_virtual) {
    SPDLOG_DEBUG("Time:{} Data task {}:{} is virtual", current_time,
                 static_graph.get_data_task_name(data_task_id), data_task_id);
    task_runtime.set_data_task_virtual(data_task_id, true);
  } else {
    SPDLOG_DEBUG("Time:{} Data task {}:{} moving from {} to {}", current_time,
                 static_graph.get_data_task_name(data_task_id), data_task_id, source_device_id,
                 destination_device_id);
  }

  // Record launching time
  task_runtime.data_notify_launched(data_task_id, source_device_id, current_time, static_graph);
  s.update_data_launched_cost(data_task_id, destination_device_id);
  success_count += 1;

  // Create completion event
  timecount_t completion_time = current_time + duration.duration;
  event_manager.create_event(EventType::DATA_COMPLETER, completion_time, data_task_id,
                             destination_device_id);

  return true;
}

bool Scheduler::launch_eviction_task(taskid_t eviction_task_id, devid_t destination_device_id,
                                     EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;
  auto &task_runtime = s.task_runtime;
  auto &data_manager = s.data_manager;
  const auto &static_graph = s.get_tasks();

  SPDLOG_DEBUG("Time:{} Attempting to launch eviction task {} on device {}", current_time,
               eviction_task_id, destination_device_id);

  assert(task_runtime.is_eviction_launchable(eviction_task_id));

  const dataid_t data_id = task_runtime.get_eviction_task_data_id(eviction_task_id);

  auto &comm_manager = s.get_communication_manager();
  const auto &topology = s.get_topology();
  auto [found, source_device_id] =
      data_manager.request_source(topology, comm_manager, data_id, destination_device_id);

  if (!found) {
    SPDLOG_DEBUG("Time:{} Eviction task {} missing available source for block {}", current_time,
                 eviction_task_id, data_id);
    return false;
  }

  SPDLOG_DEBUG("Time:{} Eviction task {} found source {} for block {}", current_time,
               eviction_task_id, source_device_id, data_id);

  task_runtime.set_eviction_task_source_device(eviction_task_id, source_device_id);
  auto duration =
      data_manager.start_move(topology, comm_manager, s.get_device_manager(), s.get_data(), data_id,
                              source_device_id, destination_device_id, current_time);

  if (duration.is_virtual) {
    SPDLOG_DEBUG("Time:{} Eviction task {} is virtual", current_time, eviction_task_id);
    task_runtime.set_eviction_task_virtual(eviction_task_id, true);
  } else {
    SPDLOG_DEBUG("Time:{} Eviction task {} moving from {} to {}", current_time, eviction_task_id,
                 source_device_id, destination_device_id);
  }

  // Record launching time
  task_runtime.eviction_notify_launched(eviction_task_id, source_device_id, current_time,
                                        static_graph);
  s.update_eviction_launched_cost(eviction_task_id, 0);
  success_count += 1;

  // Create completion event
  timecount_t completion_time = s.global_time + duration.duration;
  event_manager.create_event(EventType::EVICTOR_COMPLETER, completion_time, eviction_task_id,
                             destination_device_id);

  return true;
}

bool Scheduler::launch_compute_tasks(EventManager &event_manager) {
  ZoneScoped;

  auto &s = this->state;
  auto current_time = s.global_time;
  auto &launchable = queues.launchable;

  launchable.reset();
  launchable.current_or_next_active();

  SPDLOG_DEBUG("Time:{} Launching compute tasks", current_time);
  SPDLOG_DEBUG("Time:{} Launchable Queue Size: {}", current_time,
               queues.launchable.total_active_size());

  bool break_flag = false;

  while (queues.has_active_launchable() && conditions.get().should_launch(s, queues)) {

    SPDLOG_DEBUG("Time:{} Checking device queue {}", current_time, launchable.get_active_index());

    if (is_breakpoint()) {
      SPDLOG_DEBUG("Time:{} Breaking from launcher", current_time);
      break_flag = true;
      break;
    }

    if (launchable.get_active().empty()) {
      SPDLOG_DEBUG("Time:{} No active launchable tasks on device queue {}", current_time,
                   launchable.get_active_index());
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

  return break_flag;
}

bool Scheduler::launch_data_tasks(EventManager &event_manager) {
  ZoneScoped;

  auto &s = this->state;
  auto current_time = s.global_time;
  auto &data_launchable = queues.data_launchable;

  data_launchable.reset();
  data_launchable.current_or_next_active();

  SPDLOG_DEBUG("Time:{} Launching data tasks", current_time);
  SPDLOG_DEBUG("Time:{} Data Launchable Queue Size: {}", current_time,
               queues.data_launchable.total_active_size());

  bool break_flag = false;

  while (queues.has_active_data_launchable() && conditions.get().should_launch_data(s, queues)) {
    if (data_launchable.get_active().empty()) {
      data_launchable.next();
      continue;
    }

    taskid_t task_id = data_launchable.top();
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

  return break_flag;
}

bool Scheduler::launch_eviction_tasks(EventManager &event_manager) {
  ZoneScoped;

  auto &s = this->state;
  auto current_time = s.global_time;
  auto &eviction_launchable = queues.eviction_launchable;

  eviction_launchable.reset();
  eviction_launchable.current_or_next_active();

  SPDLOG_DEBUG("Time:{} Launching eviction tasks", current_time);
  SPDLOG_DEBUG("Time:{} Eviction Launchable Queue Size: {}", current_time,
               queues.eviction_launchable.total_active_size());

  // if (this->eviction_state == EvictionState::WAITING_FOR_COMPLETION &&
  //     queues.eviction_launchable.total_active_size() == 0) {
  //   // If the scheduler is waiting for eviction completion and there are no active eviction
  //   tasks,
  //   // it means that evictor has not been launched yet.
  //   // We need to launch the evictor to start the eviction process.
  //   event_manager.create_event(EventType::EVICTOR, current_time + SCHEDULER_TIME_GAP);
  // }

  bool break_flag = false;

  while (queues.has_active_eviction_launchable() &&
         conditions.get().should_launch_data(s, queues)) {
    if (eviction_launchable.get_active().empty()) {
      eviction_launchable.next();
      continue;
    }

    taskid_t task_id = eviction_launchable.top();
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

  return break_flag;
}

void Scheduler::launch_tasks(LauncherEvent &launch_event, EventManager &event_manager) {
  ZoneScoped;
  auto current_time = this->state.global_time;

  auto break_flag = launch_compute_tasks(event_manager);

  if (break_flag) [[unlikely]] {
    event_manager.create_event(EventType::LAUNCHER, current_time);
    return;
  }

  // Eviction tasks cannot break
  launch_eviction_tasks(event_manager);

  // Data tasks cannot break
  launch_data_tasks(event_manager);

  scheduler_event_count -= 1;
  assert(scheduler_event_count >= 0);

  if (scheduler_event_count == 0 and success_count > 0) {
    if (this->eviction_state != EvictionState::NONE) {
      if (this->eviction_state == EvictionState::RUNNING &&
          eviction_count == 0) { // Sometimes eviction completes before launcher
        SPDLOG_DEBUG("Time:{} Evictor finished", current_time);
        event_manager.create_event(EventType::RESERVER, current_time);
        this->eviction_state = EvictionState::NONE;
        clear_eviction_invalidation_cache();
      } else
        return;
    } else
      event_manager.create_event(EventType::MAPPER,
                                 current_time + SCHEDULER_TIME_GAP + TIME_TO_MAP);
    scheduler_event_count += 1;
  }
}

Scheduler::EvictionInvalidationInfo
Scheduler::get_eviction_invalidation_info(dataid_t data_id, devid_t invalidate_device) {
  const auto cache_key = pack_eviction_key(data_id, invalidate_device);
  if (auto it = eviction_invalidation_cache.find(cache_key);
      it != eviction_invalidation_cache.end()) {
    return {
        .future_usage = (it->second & 0x01) != 0,
        .write_after_read = (it->second & 0x02) != 0,
    };
  }

  const auto &mapped = state.mapped_but_not_reserved_tasks;
  auto &task_runtime = state.task_runtime;
  const auto &static_graph = state.get_tasks();

  // O(1): use incremental mapped_unique_count instead of scanning all mapped tasks
  const auto &ucount = state.mapped_unique_count[invalidate_device];
  bool future_usage = ucount.count(data_id) > 0;

  bool write_after_read = false;
  auto write_it = state.mapped_write_by_data.find(data_id);
  if (write_it != state.mapped_write_by_data.end() && !write_it->second.empty()) {
    taskid_t top;

    if (write_it->second.size() == 1) {
      top = *write_it->second.begin();
    } else {
      // Writers of the same data form a dependency chain — the topmost (earliest) writer
      // has the minimum topological depth. O(|buffer_set|) scan, no DFS needed.
      top = *write_it->second.begin();
      int32_t top_depth = static_graph.get_depth(top);
      for (auto tid : write_it->second) {
        int32_t d = static_graph.get_depth(tid);
        if (d < top_depth) {
          top_depth = d;
          top = tid;
        }
      }
    }

    if (task_runtime.get_compute_task_mapped_device(top) != invalidate_device) {
      // O(1) precheck: are there any mapped tasks on invalidate_device that read data_id?
      // If not, no dependency DFS is needed — write_after_read = true immediately.
      const auto &rcount = state.mapped_read_count_by_data_device[invalidate_device];
      if (rcount.count(data_id) > 0) {
        // Local readers exist — do interleaved DFS checking reader status during traversal.
        // Early exit as soon as a local reader is found among mapped predecessors.
        write_after_read = true;
        const auto readers_span = static_graph.get_tasks_reading_data(data_id);
        auto &stack = eviction_scratch_stack;
        reset_eviction_scratch_visited();
        stack.clear();
        stack.reserve(16);

        bool done = false;
        for (auto dep0 : static_graph.get_compute_task_dependencies(top)) {
          if (!state.is_mapped_not_reserved[static_cast<std::size_t>(dep0)]) continue;
          if (!mark_eviction_visited(dep0)) continue;
          if (task_runtime.get_compute_task_mapped_device(dep0) == invalidate_device &&
              std::binary_search(readers_span.begin(), readers_span.end(), dep0)) {
            write_after_read = false;
            done = true;
            break;
          }
          stack.push_back(dep0);
        }

        while (!done && !stack.empty()) {
          auto curr = stack.back();
          stack.pop_back();
          for (auto dep : static_graph.get_compute_task_dependencies(curr)) {
            if (!state.is_mapped_not_reserved[static_cast<std::size_t>(dep)]) continue;
            if (!mark_eviction_visited(dep)) continue;
            if (task_runtime.get_compute_task_mapped_device(dep) == invalidate_device &&
                std::binary_search(readers_span.begin(), readers_span.end(), dep)) {
              write_after_read = false;
              done = true;
              break;
            }
            stack.push_back(dep);
          }
        }
      } else {
        // No local readers — skip DFS, WAR = true by default
        write_after_read = true;
      }
    }
  }

  uint8_t encoded = 0;
  encoded |= static_cast<uint8_t>(future_usage ? 0x01 : 0);
  encoded |= static_cast<uint8_t>(write_after_read ? 0x02 : 0);
  eviction_invalidation_cache[cache_key] = encoded;
  return {.future_usage = future_usage, .write_after_read = write_after_read};
}

// TODO(wlr, jae): We need to work together to check this after the refactor
void Scheduler::evict(EvictorEvent &eviction_event, EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto &task_runtime = s.task_runtime;
  const auto &static_graph = s.get_tasks();
  const auto &data_manager = s.data_manager;
  const auto &lru_manager = s.data_manager.get_lru_manager();

  auto current_time = s.global_time;

  if (eviction_state == EvictionState::WAITING_FOR_COMPLETION) {
    if (s.counts.n_reserved() + s.counts.n_data_reserved() > 0) {
      SPDLOG_DEBUG("Time:{} Evictor waiting for all {} compute and {} data task to finish",
                   current_time, s.counts.n_reserved(), s.counts.n_data_reserved());
      event_manager.create_event(EventType::LAUNCHER, current_time);
      return;
    } else {
      SPDLOG_DEBUG("Starting evictor at {}", current_time);
      eviction_count = 0;
      clear_eviction_invalidation_cache();
      reset_eviction_planned_victims();

      for (auto &taskdevice : tasks_requesting_eviction) {
        auto [compute_task_id, device_id] = taskdevice;
        const auto [requested, missing] = s.request_reserve_resources(compute_task_id, device_id);
        if (missing.mem) { // There is still memory to evict
          const auto unique_data = static_graph.get_unique(compute_task_id);
          auto &data_ids = lru_manager.getLRUids(device_id, missing.mem, unique_data);
          for (auto data_id : data_ids) {
            // Skip victims already handled by an earlier requesting task in this cycle
            if (!mark_eviction_planned_victim(data_id, device_id)) {
              continue;
            }
            auto location_flags = data_manager.get_launched_location_flags(data_id);
            // count set bits in location_flags
            devid_t n_sources = __builtin_popcount(location_flags);
            assert(n_sources > 0);
            if (n_sources == 1) {
              eviction_count += 1;
              auto eviction_task_id =
                  task_runtime.add_eviction_task(compute_task_id, data_id, device_id);

              SPDLOG_DEBUG(
                  "Time:{} Launching eviction task {} to evict block {} for task {} on device {} ",
                  current_time, eviction_task_id, data_id,
                  static_graph.get_compute_task_name(compute_task_id), device_id);
              state.update_eviction_reserved_cost(eviction_task_id, 0);
              push_launchable_eviction(eviction_task_id);
            } else { // There are multiple sources for this data
              const auto invalidation =
                  get_eviction_invalidation_info(data_id, static_cast<devid_t>(device_id));

              SPDLOG_DEBUG("Time:{} Invalidating block {} for task {} on device {}", current_time,
                           data_id, static_graph.get_compute_task_name(compute_task_id), device_id);
              s.data_manager.evict_on_update_launched(s.get_data(), s.get_device_manager(), data_id,
                                                      device_id, current_time,
                                                      invalidation.future_usage,
                                                      invalidation.write_after_read);
            }
          }
        } else {
          SPDLOG_DEBUG("Time:{} No need to evict for task {} on device {}", current_time,
                       static_graph.get_compute_task_name(compute_task_id), device_id);
        }
      }

      SPDLOG_DEBUG("Time:{} Evictor pushed {} eviction tasks to launch queue", current_time,
                   eviction_count);
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
      clear_eviction_invalidation_cache();
    }
  }
}

void Scheduler::complete_task_postmatter(EventManager &event_manager) {
  auto &s = this->state;
  auto current_time = s.global_time;
  success_count += 1;

  SPDLOG_DEBUG("Time:{} Success count: {}, Scheduler event count: {}, Eviction state: {}",
               current_time, success_count, scheduler_event_count,
               static_cast<int>(this->eviction_state));

  const bool can_emit_scheduler_event = (scheduler_event_count == 0);
  if (can_emit_scheduler_event) {
    const auto eviction_state = this->eviction_state;
    if (eviction_state == EvictionState::WAITING_FOR_COMPLETION) {
      event_manager.create_event(EventType::EVICTOR, current_time + SCHEDULER_TIME_GAP);
    } else if (eviction_state == EvictionState::RUNNING) {
      if (eviction_count) {
        event_manager.create_event(EventType::LAUNCHER, current_time + SCHEDULER_TIME_GAP);
      } else {
        event_manager.create_event(EventType::RESERVER, current_time + SCHEDULER_TIME_GAP);
        this->eviction_state = EvictionState::NONE;
        clear_eviction_invalidation_cache();
      }
    } else {
      event_manager.create_event(EventType::MAPPER,
                                 s.global_time + SCHEDULER_TIME_GAP + TIME_TO_MAP);
    }
    scheduler_event_count += 1;
  }

  auto &device_manager = s.get_device_manager();
  auto &lru_manager = s.get_data_manager().get_lru_manager();

#ifdef DEBUG
  // check memory state, whether it is consistent for debugging purpose
  if (can_emit_scheduler_event) {
    bool flag = false;
    for (devid_t i = 0; i < device_manager.n_devices; i++) {
      mem_t launched_mem = device_manager.get_mem<TaskState::LAUNCHED>(i);
      mem_t reserved_mem = device_manager.get_mem<TaskState::RESERVED>(i);
      mem_t mapped_mem = device_manager.get_mem<TaskState::MAPPED>(i);
      mem_t lru_mem = lru_manager.get_mem(i);
      SPDLOG_DEBUG("Device {}: launched {}, lru {}, reserved {}, mapped {}", i, launched_mem, lru_mem,
                   reserved_mem, mapped_mem);
      if (i > 0 && mapped_mem < launched_mem)
        flag = true;
      assert(launched_mem == lru_mem);
    }
    if (flag) {
      SPDLOG_DEBUG("Memory state is inconsistent");
    }
  }
#endif
}

void Scheduler::complete_compute_task(ComputeCompleterEvent &event, EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto &static_graph = s.get_tasks();
  auto &task_runtime = s.task_runtime;
  auto current_time = s.global_time;
  auto &data_manager = s.data_manager;
  const auto &data = s.get_data();
  auto &device_manager = s.device_manager;

  const taskid_t compute_task_id = event.task;
  const devid_t device_id = event.device;

  SPDLOG_DEBUG("Time:{} Completing compute task {}:{} on device {}", current_time,
               static_graph.get_compute_task_name(compute_task_id), compute_task_id, device_id);
  assert(task_runtime.get_compute_task_mapped_device(compute_task_id) == device_id);

  // Free mapped, reserved, and launched resources (uses task static info, variants / data usage)
  s.free_task_resources(compute_task_id);

  // Remove retired data (uses task static info, data usage)
  for (const auto data_id : static_graph.get_retire(compute_task_id)) {
    data_manager.retire_data(data, device_manager, data_id, device_id, s.global_time);
  }

  // Notify dependents that the task has completed (uses task static info, dependents, and task
  // runtime info of dependents)
  task_runtime.compute_notify_completed(compute_task_id, current_time, static_graph,
                                        compute_task_buffer);

  SPDLOG_DEBUG("Time:{} Newly launchable compute tasks: {}", current_time,
               compute_task_buffer.size());
  push_launchable(compute_task_buffer);

  task_runtime.compute_notify_data_completed(compute_task_id, current_time, static_graph,
                                             data_task_buffer);

  SPDLOG_DEBUG("Time:{} Newly launchable data tasks: {}", current_time, data_task_buffer.size());

  push_launchable_data(data_task_buffer);

  // Updates task counter tables in scheduler
  s.update_completed_cost(compute_task_id, device_id);

  complete_task_postmatter(event_manager);
}

void Scheduler::complete_data_task(DataCompleterEvent &event, EventManager &event_manager) {
  ZoneScoped;
  auto &s = this->state;
  auto current_time = s.global_time;
  const auto &static_graph = s.get_tasks();
  auto &task_runtime = s.task_runtime;
  auto &comm_manager = s.get_communication_manager();
  taskid_t last_compute_idx = 0;

  const taskid_t data_task_id = event.task;
  const devid_t destination_id = event.device;

  SPDLOG_DEBUG("Time:{} Completing data task {}:{} on device {}", current_time,
               static_graph.get_data_task_name(data_task_id), data_task_id, destination_id);
  assert(task_runtime.get_data_task_mapped_device(data_task_id) == destination_id);

  // Updates data location and eviction manager (uses task runtime info of data task)
  const auto source_id = task_runtime.get_data_task_source_device(data_task_id);
  const auto is_virtual = task_runtime.is_data_task_virtual(data_task_id);
  const auto data_id = static_graph.get_data_id(data_task_id);
  s.data_manager.complete_move(comm_manager, data_id, source_id, destination_id, is_virtual,
                               current_time);

  // Notify dependents that the data task has completed
  // (uses task static info, dependents, and task runtime info of dependents)
  task_runtime.data_notify_completed(data_task_id, current_time, static_graph, compute_task_buffer);

  SPDLOG_DEBUG("Time:{} Newly launchable compute tasks: {}", current_time,
               compute_task_buffer.size());

  if (!compute_task_buffer.empty()) {
    push_launchable(compute_task_buffer);
  }

  // Updates task counter tables in scheduler
  s.update_data_completed_cost(data_task_id, destination_id);

  complete_task_postmatter(event_manager);
}

void Scheduler::complete_eviction_task(EvictorCompleterEvent &event, EventManager &event_manager) {
  auto &s = this->state;
  auto current_time = s.global_time;
  auto &task_runtime = s.task_runtime;
  auto &data_manager = s.data_manager;
  auto &device_manager = s.device_manager;
  auto &comm_manager = s.get_communication_manager();

  const taskid_t eviction_task_id = event.task;
  const devid_t destination_id = 0; // This is always the host device

  s.update_eviction_completed_cost(eviction_task_id, 0);

  SPDLOG_DEBUG("Time:{} Completing eviction task {}", current_time,
               task_runtime.get_eviction_task_name(eviction_task_id));

  auto source_id = task_runtime.get_eviction_task_source_device(eviction_task_id);
  auto is_virtual = task_runtime.is_eviction_task_virtual(eviction_task_id);
  auto data_id = task_runtime.get_eviction_task_data_id(eviction_task_id);

  data_manager.complete_eviction_move(comm_manager, data_id, source_id, destination_id, is_virtual,
                                      current_time);
  const auto data_size = s.get_data().get_size(data_id);
  device_manager.add_mem<TaskState::MAPPED>(destination_id, data_size, current_time);
  device_manager.add_mem<TaskState::RESERVED>(destination_id, data_size, current_time);

  auto invalidate_device_id = task_runtime.get_eviction_task_evicting_on(eviction_task_id);
  const auto invalidation =
      get_eviction_invalidation_info(data_id, static_cast<devid_t>(invalidate_device_id));

  data_manager.evict_on_update_launched(s.get_data(), device_manager, data_id, invalidate_device_id,
                                        current_time, invalidation.future_usage,
                                        invalidation.write_after_read);

  eviction_count -= 1;
  SPDLOG_DEBUG("Time:{} Eviction task {} completed {} left", current_time, eviction_task_id,
               eviction_count);
  task_runtime.eviction_notify_completed(eviction_task_id, current_time);
  complete_task_postmatter(event_manager);
}
