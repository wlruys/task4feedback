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
inline void schedule_event(EventManager &event_manager, EventType type, timecount_t event_time) {
  event_manager.create_event(type, event_time);
}
} // namespace

void Scheduler::finalize_mapping_phase(EventManager &event_manager, timecount_t current_time,
                                       bool hit_breakpoint) {
  if (hit_breakpoint) {
    schedule_event(event_manager, EventType::MAPPER, current_time);
    return;
  }
  schedule_event(event_manager, EventType::RESERVER, current_time + SCHEDULER_TIME_GAP);
}

ExecutionState Scheduler::finalize_external_mapping_phase(EventManager &event_manager) {
  const auto current_time = state.global_time;
  if (should_run_mapper_phase()) {
    return ExecutionState::EXTERNAL_MAPPING;
  }

  const bool hit_breakpoint = has_pending_step_breakpoint();
  if (hit_breakpoint) {
    SPDLOG_DEBUG("Time:{} Breaking from mapper", current_time);
  } else {
    SPDLOG_DEBUG("Time:{} Ending mapper", current_time);
  }
  finalize_mapping_phase(event_manager, current_time, hit_breakpoint);
  return hit_breakpoint ? ExecutionState::BREAKPOINT : ExecutionState::RUNNING;
}

void Scheduler::finalize_reserve_phase(EventManager &event_manager, timecount_t current_time,
                                       bool hit_breakpoint) {
  if (hit_breakpoint) [[unlikely]] {
    schedule_event(event_manager, EventType::RESERVER, current_time);
    return;
  }

  auto &s = this->state;
  if (!tasks_requesting_eviction.empty()) {
    // Eviction is a quiescent phase: we only enter it after all currently reserved work
    // has either launched or drained to avoid mixing reserve and eviction accounting.
    // Should not start eviction if there are tasks not launchable due to data movement.
    if (s.counts.n_unlaunched_reserved() == 0) {
      enter_eviction_waiting(event_manager, current_time);
      return;
    }

    SPDLOG_DEBUG("Time:{} Eviction will start after launching {} tasks", current_time,
                 s.counts.n_unlaunched_reserved());
  }

  schedule_event(event_manager, EventType::LAUNCHER, current_time + TIME_TO_LAUNCH);
}

void Scheduler::finalize_launch_phase(EventManager &event_manager, timecount_t current_time,
                                      bool hit_breakpoint) {
  if (hit_breakpoint) [[unlikely]] {
    schedule_event(event_manager, EventType::LAUNCHER, current_time);
    return;
  }

  scheduler_event_count -= 1;
  assert(scheduler_event_count >= 0);

  const auto &counts = state.counts;
  const uint64_t state_signature =
      (static_cast<uint64_t>(static_cast<uint32_t>(counts.n_active())) << 0) ^
      (static_cast<uint64_t>(static_cast<uint32_t>(counts.n_mapped())) << 8) ^
      (static_cast<uint64_t>(static_cast<uint32_t>(counts.n_reserved())) << 16) ^
      (static_cast<uint64_t>(static_cast<uint32_t>(counts.n_launched())) << 24) ^
      (static_cast<uint64_t>(static_cast<uint32_t>(counts.n_data_reserved())) << 32) ^
      (static_cast<uint64_t>(static_cast<uint32_t>(counts.n_data_launched())) << 40) ^
      (static_cast<uint64_t>(static_cast<uint8_t>(eviction_state)) << 48) ^
      (static_cast<uint64_t>(static_cast<uint16_t>(eviction_count)) << 56);
  const bool state_changed_since_last_dispatch =
      state_signature != last_launcher_dispatch_signature;

  if (scheduler_event_count == 0 && (success_count > 0 || state_changed_since_last_dispatch)) {
    // No-op return is intentional for launcher-triggered WAITING/RUNNING dispatch states.
    emit_post_completion_scheduler_event(event_manager, current_time,
                                         PostCompletionDispatchSource::FROM_LAUNCHER);
    last_launcher_dispatch_signature = state_signature;
  } else if (scheduler_event_count == 0) {
    SPDLOG_DEBUG(
        "Time:{} finalize_launch_phase made no dispatch (success_count={}, state_changed={}, "
        "eviction_state={}, eviction_count={}, active={}, mapped={}, reserved={}, launched={}, "
        "data_reserved={}, data_launched={})",
        current_time, success_count, state_changed_since_last_dispatch,
        static_cast<int>(eviction_state), eviction_count, counts.n_active(), counts.n_mapped(),
        counts.n_reserved(), counts.n_launched(), counts.n_data_reserved(), counts.n_data_launched());
  }
}

bool Scheduler::process_eviction_waiting_phase(EventManager &event_manager,
                                               timecount_t current_time) {
  auto &s = this->state;
  assert(eviction_state == eviction::State::WAITING_FOR_COMPLETION);
  if (s.counts.n_reserved() + s.counts.n_data_reserved() > 0) {
    SPDLOG_DEBUG("Time:{} Evictor waiting for all {} compute and {} data task to finish",
                 current_time, s.counts.n_reserved(), s.counts.n_data_reserved());
    schedule_event(event_manager, EventType::LAUNCHER, current_time);
    return false;
  }

  SPDLOG_DEBUG("Starting evictor at {}", current_time);
  select_and_enqueue_victims(current_time);
  return true;
}

void Scheduler::enter_eviction_waiting(EventManager &event_manager, timecount_t current_time) {
  SPDLOG_DEBUG("Time:{} Eviction will start for {} tasks", current_time,
               tasks_requesting_eviction.size());
  this->eviction_state = eviction::State::WAITING_FOR_COMPLETION;
  schedule_event(event_manager, EventType::EVICTOR, current_time + SCHEDULER_TIME_GAP);
}

void Scheduler::enter_eviction_running() {
  eviction_state = eviction::State::RUNNING;
}

void Scheduler::exit_eviction_running(EventManager &event_manager, timecount_t reserve_event_time) {
  schedule_event(event_manager, EventType::RESERVER, reserve_event_time);
  this->eviction_state = eviction::State::NONE;
  state.get_data_manager().eviction_runtime().clear_invalidation_caches();
}

bool Scheduler::request_and_start_transfer(dataid_t data_id, devid_t destination_device_id,
                                           timecount_t current_time, devid_t &source_device_id,
                                           MovementStatus &duration) {
  auto &s = this->state;
  auto &data_manager = s.data_manager;
  auto &comm_manager = s.get_communication_manager();
  const auto &topology = s.get_topology();

  auto [found, source] =
      data_manager.request_source(topology, comm_manager, data_id, destination_device_id);
  if (!found) {
    return false;
  }

  source_device_id = source;
  duration =
      data_manager.start_move(topology, comm_manager, s.get_device_manager(), s.get_data(), data_id,
                              source_device_id, destination_device_id, current_time);
  return true;
}

void Scheduler::complete_transfer_move(dataid_t data_id, devid_t source_id, devid_t destination_id,
                                       bool is_virtual, timecount_t current_time,
                                       bool is_eviction_move) {
  auto &s = this->state;
  auto &data_manager = s.data_manager;
  auto &comm_manager = s.get_communication_manager();
  const auto data_size = s.get_data().get_size(data_id);
  if (is_eviction_move) {
    data_manager.complete_eviction_move(comm_manager, data_id, source_id, destination_id, is_virtual,
                                        current_time, data_size);
    return;
  }
  data_manager.complete_move(comm_manager, data_id, source_id, destination_id, is_virtual, current_time,
                             data_size);
}

bool Scheduler::process_eviction_running_phase(EventManager &event_manager, timecount_t current_time) {
  assert(eviction_state == eviction::State::RUNNING);
  if (eviction_count > 0) {
    SPDLOG_DEBUG("Time:{} Evictor waiting for all eviction tasks to finish", current_time);
    schedule_event(event_manager, EventType::LAUNCHER, current_time);
    return false;
  }

  assert(eviction_count == 0);
  SPDLOG_DEBUG("Time:{} Evictor finished", current_time);
  exit_eviction_running(event_manager, current_time);
  return true;
}

bool Scheduler::should_run_mapper_phase() {
  return state.not_draining() && queues.has_mappable() && conditions.get().should_map(state, queues);
}

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

  s.get_data_manager().eviction_runtime().on_compute_mapped(static_graph, compute_task_id,
                                                            chosen_device);

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
    const auto n_candidates = top_k_tasks.size();
    for (auto &action : action_list) {
      if (action.pos >= n_candidates) {
        SPDLOG_CRITICAL("Invalid action position {}. Candidate count is {}", action.pos,
                        n_candidates);
        assert(false);
        return ExecutionState::ERROR;
      }
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

  return finalize_external_mapping_phase(event_manager);
}

ExecutionState Scheduler::map_tasks_from_python_soa(std::span<const int64_t> positions,
                                                    std::span<const int64_t> devices,
                                                    EventManager &event_manager) {
  if (positions.size() != devices.size()) {
    SPDLOG_CRITICAL("map_tasks_from_python_soa expects same-length positions/devices arrays ({} vs "
                    "{})",
                    positions.size(), devices.size());
    assert(false);
    return ExecutionState::ERROR;
  }

  ActionList action_list;
  action_list.reserve(positions.size());

  const auto top_k_tasks = queues.mappable.get_top_k();
  const auto n_candidates = top_k_tasks.size();
  const auto n_devices = static_cast<int64_t>(state.get_devices().size());

  for (std::size_t i = 0; i < positions.size(); ++i) {
    const auto pos = positions[i];
    const auto device = devices[i];
    if (pos < 0 || static_cast<std::size_t>(pos) >= n_candidates) {
      SPDLOG_CRITICAL("Invalid SOA action position {}. Candidate count is {}", pos, n_candidates);
      assert(false);
      return ExecutionState::ERROR;
    }
    if (device < 0 || device >= n_devices) {
      SPDLOG_CRITICAL("Invalid SOA mapped device {}. Device count is {}", device, n_devices);
      assert(false);
      return ExecutionState::ERROR;
    }

    const auto task_id = top_k_tasks[static_cast<std::size_t>(pos)];
    const auto mapping_priority = state.get_mapping_priority(task_id);
    action_list.push_back(Action{.pos = static_cast<std::size_t>(pos),
                                 .device = static_cast<devid_t>(device),
                                 .reservable_priority = mapping_priority,
                                 .launchable_priority = mapping_priority});
  }

  return map_tasks_from_python(action_list, event_manager);
}

void Scheduler::skip_map_tasks(MapperEvent &map_event, EventManager &event_manager) {
  MONUnusedParameter(map_event);
  success_count = 0;
  const auto current_time = state.global_time;
  SPDLOG_DEBUG("Time:{} Skipping mapper", current_time);
  finalize_mapping_phase(event_manager, current_time, false);
}

void Scheduler::skip_reserve_tasks(ReserverEvent &reserve_event, EventManager &event_manager) {
  MONUnusedParameter(reserve_event);
  const auto current_time = state.global_time;
  SPDLOG_DEBUG("Time:{} Skipping reserver", current_time);
  schedule_event(event_manager, EventType::LAUNCHER, current_time + SCHEDULER_TIME_GAP);
}

void Scheduler::map_tasks(MapperEvent &map_event, EventManager &event_manager, Mapper &mapper) {
  ZoneScoped;
  MONUnusedParameter(map_event);

  success_count = 0;
  auto &s = this->state;
  auto &task_runtime = s.task_runtime;
  auto current_time = s.global_time;

  SPDLOG_DEBUG("Time:{} Starting mapper", current_time);
  SPDLOG_DEBUG("Time:{} Mappable Queue Size: {}", current_time, queues.mappable.size());
  bool break_flag = false;

  while (queues.has_mappable() && conditions.get().should_map(s, queues)) {

    if (has_pending_step_breakpoint()) {
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

  finalize_mapping_phase(event_manager, current_time, break_flag);
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

  s.get_data_manager().eviction_runtime().on_compute_reserved(static_graph, compute_task_id,
                                                              device_id);

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
  MONUnusedParameter(reserve_event);
  // Can't reserve tasks if we are in the middle of an eviction
  auto current_time = this->state.global_time;
  assert(this->eviction_state == eviction::State::NONE);

  auto &s = this->state;

  auto &reservable = queues.reservable;
  reservable.reset();
  reservable.current_or_next_active();

  SPDLOG_DEBUG("Time:{} Reserving tasks", current_time);
  SPDLOG_DEBUG("Time:{} Reservable Queue Size: {}", current_time,
               queues.reservable.total_active_size());
  bool break_flag = false;
  tasks_requesting_eviction.clear();
  while (queues.has_active_reservable() && conditions.get().should_reserve(s, queues)) {

    if (has_pending_step_breakpoint()) {
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
    reservable.pop();

    push_reservable(compute_task_buffer);

    // Cycle to the next active device queue
    reservable.next();
  }

  finalize_reserve_phase(event_manager, current_time, break_flag);
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
  const auto &static_graph = s.get_tasks();

  SPDLOG_DEBUG("Time:{} Attempting to launch data task {}:{} on device {}", current_time,
               static_graph.get_data_task_name(data_task_id), data_task_id, destination_device_id);

  assert(task_runtime.is_data_launchable(data_task_id));
  assert(task_runtime.get_data_task_mapped_device(data_task_id) == destination_device_id);

  const dataid_t data_id = static_graph.get_data_id(data_task_id);
  devid_t source_device_id = 0;
  MovementStatus duration{};
  if (!request_and_start_transfer(data_id, destination_device_id, current_time, source_device_id,
                                  duration)) {
    SPDLOG_DEBUG("Time:{} Data task {}:{} missing available source", current_time,
                 static_graph.get_data_task_name(data_task_id), data_task_id);
    SPDLOG_DEBUG(
        "Time:{} Data task {}:{} source-miss detail: data_id={}, dst={}, launched_flags={:#x}, "
        "reserved_flags={:#x}, mapped_flags={:#x}, counts(res={}, launched={}, data_res={}, "
        "data_launched={})",
        current_time, static_graph.get_data_task_name(data_task_id), data_task_id, data_id,
        destination_device_id, static_cast<uint64_t>(s.data_manager.get_launched_location_flags(data_id)),
        static_cast<uint64_t>(s.data_manager.get_reserved_location_flags(data_id)),
        static_cast<uint64_t>(s.data_manager.get_mapped_location_flags(data_id)), s.counts.n_reserved(),
        s.counts.n_launched(), s.counts.n_data_reserved(), s.counts.n_data_launched());
    return false;
  }

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

  SPDLOG_DEBUG("Time:{} Attempting to launch eviction task {} on device {}", current_time,
               eviction_task_id, destination_device_id);

  assert(task_runtime.is_eviction_launchable(eviction_task_id));

  const dataid_t data_id = task_runtime.get_eviction_task_data_id(eviction_task_id);

  devid_t source_device_id = 0;
  MovementStatus duration{};
  if (!request_and_start_transfer(data_id, destination_device_id, current_time, source_device_id,
                                  duration)) {
    SPDLOG_DEBUG("Time:{} Eviction task {} missing available source for block {}", current_time,
                 eviction_task_id, data_id);
    SPDLOG_DEBUG(
        "Time:{} Eviction task {} source-miss detail: data_id={}, dst={}, launched_flags={:#x}, "
        "reserved_flags={:#x}, mapped_flags={:#x}, eviction_state={}, eviction_count={}",
        current_time, eviction_task_id, data_id, destination_device_id,
        static_cast<uint64_t>(s.data_manager.get_launched_location_flags(data_id)),
        static_cast<uint64_t>(s.data_manager.get_reserved_location_flags(data_id)),
        static_cast<uint64_t>(s.data_manager.get_mapped_location_flags(data_id)),
        static_cast<int>(eviction_state), eviction_count);
    return false;
  }

  SPDLOG_DEBUG("Time:{} Eviction task {} found source {} for block {}", current_time,
               eviction_task_id, source_device_id, data_id);

  task_runtime.set_eviction_task_source_device(eviction_task_id, source_device_id);

  if (duration.is_virtual) {
    SPDLOG_DEBUG("Time:{} Eviction task {} is virtual", current_time, eviction_task_id);
    task_runtime.set_eviction_task_virtual(eviction_task_id, true);
  } else {
    SPDLOG_DEBUG("Time:{} Eviction task {} moving from {} to {}", current_time, eviction_task_id,
                 source_device_id, destination_device_id);
  }

  // Record launching time
  task_runtime.eviction_notify_launched(eviction_task_id, source_device_id, current_time,
                                        s.get_tasks());
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

    if (has_pending_step_breakpoint()) {
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

void Scheduler::launch_data_tasks(EventManager &event_manager) {
  ZoneScoped;

  auto &s = this->state;
  auto current_time = s.global_time;
  auto &data_launchable = queues.data_launchable;

  data_launchable.reset();
  data_launchable.current_or_next_active();

  SPDLOG_DEBUG("Time:{} Launching data tasks", current_time);
  SPDLOG_DEBUG("Time:{} Data Launchable Queue Size: {}", current_time,
               queues.data_launchable.total_active_size());

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
}

void Scheduler::launch_eviction_tasks(EventManager &event_manager) {
  ZoneScoped;

  auto &s = this->state;
  auto current_time = s.global_time;
  auto &eviction_launchable = queues.eviction_launchable;

  eviction_launchable.reset();
  eviction_launchable.current_or_next_active();

  SPDLOG_DEBUG("Time:{} Launching eviction tasks", current_time);
  SPDLOG_DEBUG("Time:{} Eviction Launchable Queue Size: {}", current_time,
               queues.eviction_launchable.total_active_size());

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
}

bool Scheduler::emit_post_completion_scheduler_event(
    EventManager &event_manager, timecount_t current_time,
    PostCompletionDispatchSource source) {
  const bool from_launcher = (source == PostCompletionDispatchSource::FROM_LAUNCHER);
  const auto eviction_state = this->eviction_state;

  if (eviction_state == eviction::State::WAITING_FOR_COMPLETION) {
    // Preserve existing behavior:
    // - completer emits an EVICTOR event
    // - launcher emits nothing
    if (from_launcher) {
      return false;
    }
    event_manager.create_event(EventType::EVICTOR, current_time + SCHEDULER_TIME_GAP);
    scheduler_event_count += 1;
    return true;
  }

  if (eviction_state == eviction::State::RUNNING) {
    if (eviction_count > 0) {
      // Preserve existing behavior:
      // - completer emits a LAUNCHER event
      // - launcher emits nothing
      if (from_launcher) {
        return false;
      }
      event_manager.create_event(EventType::LAUNCHER, current_time + SCHEDULER_TIME_GAP);
      scheduler_event_count += 1;
      return true;
    }

    // Preserve timing difference:
    // - launcher path emits RESERVER at current_time
    // - completer path emits RESERVER at current_time + SCHEDULER_TIME_GAP
    SPDLOG_DEBUG("Time:{} Evictor finished", current_time);
    const auto event_time = current_time + (from_launcher ? 0 : SCHEDULER_TIME_GAP);
    exit_eviction_running(event_manager, event_time);
    scheduler_event_count += 1;
    return true;
  }

  event_manager.create_event(EventType::MAPPER, current_time + SCHEDULER_TIME_GAP + TIME_TO_MAP);
  scheduler_event_count += 1;
  return true;
}

void Scheduler::launch_tasks(LauncherEvent &launch_event, EventManager &event_manager) {
  ZoneScoped;
  MONUnusedParameter(launch_event);
  auto current_time = this->state.global_time;

  auto break_flag = launch_compute_tasks(event_manager);

  if (!break_flag) {
    // Eviction and data tasks cannot break the launcher phase.
    launch_eviction_tasks(event_manager);
    launch_data_tasks(event_manager);
  }

  finalize_launch_phase(event_manager, current_time, break_flag);
}

void Scheduler::enqueue_eviction_move_task(taskid_t compute_task_id, devid_t device_id,
                                           dataid_t data_id, timecount_t current_time) {
  auto &s = this->state;
  auto &task_runtime = s.task_runtime;
  const auto &static_graph = s.get_tasks();

  eviction_count += 1;
  auto eviction_task_id = task_runtime.add_eviction_task(compute_task_id, data_id, device_id);

  SPDLOG_DEBUG("Time:{} Launching eviction task {} to evict block {} for task {} on device {} ",
               current_time, eviction_task_id, data_id,
               static_graph.get_compute_task_name(compute_task_id), device_id);
  state.update_eviction_reserved_cost(eviction_task_id, 0);
  push_launchable_eviction(eviction_task_id);
}

void Scheduler::apply_eviction_invalidation(taskid_t compute_task_id, devid_t device_id,
                                            dataid_t data_id, timecount_t current_time,
                                            const eviction::InvalidationInfo *invalidation_info) {
  auto &s = this->state;
  const auto &static_graph = s.get_tasks();
  auto &eviction_runtime = s.get_data_manager().eviction_runtime();
  const eviction::InvalidationInfo invalidation = (invalidation_info != nullptr)
                                                      ? *invalidation_info
                                                      : eviction_runtime.get_invalidation_info(
                                                            s, data_id,
                                                            static_cast<devid_t>(device_id));

  SPDLOG_DEBUG("Time:{} Invalidating block {} for task {} on device {}", current_time, data_id,
               static_graph.get_compute_task_name(compute_task_id), device_id);
  s.data_manager.evict_on_update_launched(s.get_data(), s.get_device_manager(), data_id, device_id,
                                          current_time, invalidation);
}

void Scheduler::select_and_enqueue_victims(timecount_t current_time) {
  auto &s = this->state;
  auto &eviction_runtime = s.get_data_manager().eviction_runtime();

  // Preconditions:
  // - evictor is in WAITING_FOR_COMPLETION state
  // - no compute/data tasks remain reserved
  assert(eviction_state == eviction::State::WAITING_FOR_COMPLETION);
  assert(s.counts.n_reserved() + s.counts.n_data_reserved() == 0);
  eviction_count = 0;
  eviction_runtime.clear_cycle_state();
  SPDLOG_DEBUG("Time:{} Selecting eviction victims with {} policy", current_time,
               eviction_policy_name);
  assert(eviction_victim_selector_fn != nullptr);
  (this->*eviction_victim_selector_fn)(current_time);

  SPDLOG_DEBUG("Time:{} Evictor pushed {} eviction tasks to launch queue", current_time,
               eviction_count);
  enter_eviction_running();
}

// TODO(wlr, jae): We need to work together to check this after the refactor
void Scheduler::evict(EvictorEvent &eviction_event, EventManager &event_manager) {
  ZoneScoped;
  MONUnusedParameter(eviction_event);
  const auto current_time = state.global_time;

  if (eviction_state == eviction::State::WAITING_FOR_COMPLETION &&
      !process_eviction_waiting_phase(event_manager, current_time)) {
    return;
  }

  if (eviction_state == eviction::State::RUNNING) {
    process_eviction_running_phase(event_manager, current_time);
  }
}

bool Scheduler::can_emit_post_completion_scheduler_event() const {
  return scheduler_event_count == 0;
}

void Scheduler::log_completion_mismatch() const {
  const auto &counts = state.counts;
  const auto &runtime = state.task_runtime;
  spdlog::critical(
      "Completion mismatch: n_events=0 scheduler_event_count={} success_count={} "
      "eviction_count={} eviction_state={} "
      "compute_completed={}/{} data_completed={}/{} "
      "active={} mapped={} reserved={} launched={} data_reserved={} data_launched={}",
      scheduler_event_count, success_count, eviction_count, static_cast<int>(eviction_state),
      counts.n_completed(), runtime.get_n_compute_tasks(), counts.n_data_completed(),
      runtime.get_n_data_tasks() + runtime.get_n_eviction_tasks(), counts.n_active(),
      counts.n_mapped(), counts.n_reserved(), counts.n_launched(), counts.n_data_reserved(),
      counts.n_data_launched());
}

void Scheduler::debug_check_post_completion_memory_state(bool can_emit_scheduler_event) const {
#ifdef DEBUG
  if (!can_emit_scheduler_event) {
    return;
  }

  auto &s = this->state;
  const auto &device_manager = s.get_device_manager();
  const auto &residency_manager = s.get_data_manager().eviction_residency();

  bool flag = false;
  for (devid_t i = 0; i < device_manager.n_devices; i++) {
    mem_t launched_mem = device_manager.get_mem<TaskState::LAUNCHED>(i);
    mem_t reserved_mem = device_manager.get_mem<TaskState::RESERVED>(i);
    mem_t mapped_mem = device_manager.get_mem<TaskState::MAPPED>(i);
    mem_t residency_mem = residency_manager.get_mem(i);
    SPDLOG_DEBUG("Device {}: launched {}, residency {}, reserved {}, mapped {}", i, launched_mem,
                 residency_mem,
                 reserved_mem, mapped_mem);
    if (i > 0 && mapped_mem < launched_mem) {
      flag = true;
    }
    assert(launched_mem == residency_mem);
  }
  if (flag) {
    SPDLOG_DEBUG("Memory state is inconsistent");
  }
#else
  MONUnusedParameter(can_emit_scheduler_event);
#endif
}

void Scheduler::complete_task_postmatter(EventManager &event_manager) {
  auto &s = this->state;
  auto current_time = s.global_time;
  success_count += 1;

  SPDLOG_DEBUG("Time:{} Success count: {}, Scheduler event count: {}, Eviction state: {}",
               current_time, success_count, scheduler_event_count,
               static_cast<int>(this->eviction_state));

  const bool can_emit_scheduler_event = can_emit_post_completion_scheduler_event();
  if (can_emit_scheduler_event) {
    emit_post_completion_scheduler_event(event_manager, current_time,
                                         PostCompletionDispatchSource::FROM_COMPLETER);
  }
  debug_check_post_completion_memory_state(can_emit_scheduler_event);
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

  const taskid_t data_task_id = event.task;
  const devid_t destination_id = event.device;

  SPDLOG_DEBUG("Time:{} Completing data task {}:{} on device {}", current_time,
               static_graph.get_data_task_name(data_task_id), data_task_id, destination_id);
  assert(task_runtime.get_data_task_mapped_device(data_task_id) == destination_id);

  // Updates data location and eviction manager (uses task runtime info of data task)
  const auto source_id = task_runtime.get_data_task_source_device(data_task_id);
  const auto is_virtual = task_runtime.is_data_task_virtual(data_task_id);
  const auto data_id = static_graph.get_data_id(data_task_id);
  complete_transfer_move(data_id, source_id, destination_id, is_virtual, current_time,
                         /*is_eviction_move=*/false);

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

  const taskid_t eviction_task_id = event.task;
  const devid_t destination_id = 0; // This is always the host device

  s.update_eviction_completed_cost(eviction_task_id, 0);

  SPDLOG_DEBUG("Time:{} Completing eviction task {}", current_time,
               task_runtime.get_eviction_task_name(eviction_task_id));

  auto source_id = task_runtime.get_eviction_task_source_device(eviction_task_id);
  auto is_virtual = task_runtime.is_eviction_task_virtual(eviction_task_id);
  auto data_id = task_runtime.get_eviction_task_data_id(eviction_task_id);

  complete_transfer_move(data_id, source_id, destination_id, is_virtual, current_time,
                         /*is_eviction_move=*/true);
  const auto data_size = s.get_data().get_size(data_id);
  device_manager.add_mem<TaskState::MAPPED>(destination_id, data_size, current_time);
  device_manager.add_mem<TaskState::RESERVED>(destination_id, data_size, current_time);

  auto invalidate_device_id = task_runtime.get_eviction_task_evicting_on(eviction_task_id);
  auto &eviction_runtime = s.get_data_manager().eviction_runtime();
  const auto invalidation =
      eviction_runtime.get_invalidation_info(s, data_id, static_cast<devid_t>(invalidate_device_id));

  data_manager.evict_on_update_launched(s.get_data(), device_manager, data_id, invalidate_device_id,
                                        current_time, invalidation);

  eviction_count -= 1;
  assert(eviction_count >= 0);
  SPDLOG_DEBUG("Time:{} Eviction task {} completed {} left", current_time, eviction_task_id,
               eviction_count);
  task_runtime.eviction_notify_completed(eviction_task_id, current_time);
  complete_task_postmatter(event_manager);
}
