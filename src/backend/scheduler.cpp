#include "scheduler.hpp"
#include "event_manager.hpp"
#include "macros.hpp"
#include "settings.hpp"
#include "task_manager.hpp"

// SchedulerState

timecount_t SchedulerState::get_execution_time(taskid_t task_id) const {
  auto device_id = task_manager.state.get_mapping(task_id);
  auto arch = device_manager.devices.get_type(device_id);
  return task_manager.get_task_variant(task_id, arch).get_execution_time();
}

const Resources &SchedulerState::get_task_resources(taskid_t task_id,
                                                    devid_t device_id) const {
  const Resources &task_resources = task_manager.get_task_resources(
      task_id, device_manager.devices.get_type(device_id));
  return task_resources;
}

const Resources &SchedulerState::get_task_resources(taskid_t task_id) const {
  devid_t device_id = task_manager.state.get_mapping(task_id);
  const Resources &task_resources = task_manager.get_task_resources(
      task_id, device_manager.devices.get_type(device_id));
  return task_resources;
}

ResourceRequest SchedulerState::request_map_resources(taskid_t task_id,
                                                      devid_t device_id) const {

  const Resources &task_resources = get_task_resources(task_id, device_id);
  const auto &task = task_manager.get_tasks().get_compute_task(task_id);
  mem_t non_local_mem =
      data_manager.non_local_size_mapped(task.get_read(), device_id);
  non_local_mem += data_manager.non_local_size(task.get_write(), device_id);

  Resources requested = {task_resources.vcu,
                         task_resources.mem + non_local_mem};
  Resources missing;
  return {requested, missing};
}

bool SchedulerState::is_data_task(taskid_t task_id) const {
  return task_manager.tasks.is_data(task_id);
}

bool SchedulerState::is_compute_task(taskid_t task_id) const {
  return task_manager.tasks.is_compute(task_id);
}

ResourceRequest
SchedulerState::request_reserve_resources(taskid_t task_id,
                                          devid_t device_id) const {
  const auto &task = task_manager.get_tasks().get_compute_task(task_id);

  const Resources &task_resources = get_task_resources(task_id, device_id);
  mem_t non_local_mem =
      data_manager.non_local_size_reserved(task.get_read(), device_id);
  non_local_mem +=
      data_manager.non_local_size_reserved(task.get_write(), device_id);
  Resources requested = {task_resources.vcu,
                         task_resources.mem + non_local_mem};
  auto missing_mem = device_manager.overflow_mem<TaskState::RESERVED>(
      device_id, requested.mem);
  return {requested, Resources(0, missing_mem)};
}

ResourceRequest
SchedulerState::request_launch_resources(taskid_t task_id,
                                         devid_t device_id) const {
  const auto &task = task_manager.get_tasks().get_compute_task(task_id);
  const Resources &task_resources = get_task_resources(task_id, device_id);
  mem_t non_local_mem =
      data_manager.non_local_size_reserved(task.get_read(), device_id);
  assert(non_local_mem == 0);
  Resources requested = {task_resources.vcu,
                         task_resources.mem + non_local_mem};
  auto missing_vcu = device_manager.overflow_vcu<TaskState::LAUNCHED>(
      device_id, requested.vcu);
  return {requested, Resources(missing_vcu, 0)};
}

void SchedulerState::map_resources(taskid_t task_id, devid_t device_id,
                                   const Resources &requested) {
  MONUnusedParameter(task_id);
  device_manager.add_resources<TaskState::MAPPED>(device_id, requested);
}

void SchedulerState::reserve_resources(taskid_t task_id, devid_t device_id,
                                       const Resources &requested) {
  MONUnusedParameter(task_id);
  device_manager.add_resources<TaskState::RESERVED>(device_id, requested);
}

void SchedulerState::launch_resources(taskid_t task_id, devid_t device_id,
                                      const Resources &requested) {
  MONUnusedParameter(task_id);
  device_manager.add_resources<TaskState::LAUNCHED>(device_id, requested);
}

void SchedulerState::free_resources(taskid_t task_id) {
  devid_t device_id = task_manager.state.get_mapping(task_id);
  const auto &task_resources = get_task_resources(task_id);
  device_manager.remove_resources<TaskState::MAPPED>(device_id, task_resources);
  device_manager.remove_resources<TaskState::RESERVED>(device_id,
                                                       task_resources);
  device_manager.remove_resources<TaskState::LAUNCHED>(device_id,
                                                       task_resources);
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

const PriorityList &SchedulerState::get_mapping_priorities() const {
  return task_manager.state.get_mapping_priorities();
}

const PriorityList &SchedulerState::get_reserving_priorities() const {
  return task_manager.state.get_reserving_priorities();
}

const PriorityList &SchedulerState::get_launching_priorities() const {
  return task_manager.state.get_launching_priorities();
}

priority_t SchedulerState::get_reserving_priority(taskid_t task_id) const {
  return task_manager.state.get_reserving_priority(task_id);
}

priority_t SchedulerState::get_launching_priority(taskid_t task_id) const {
  return task_manager.state.get_launching_priority(task_id);
}

void SchedulerState::set_reserving_priority(taskid_t task_id,
                                            priority_t priority) {
  task_manager.state.set_reserving_priority(task_id, priority);
}

void SchedulerState::set_launching_priority(taskid_t task_id,
                                            priority_t priority) {
  task_manager.state.set_launching_priority(task_id, priority);
}

// Scheduler Queues
void SchedulerQueues::push_mappable(taskid_t id, priority_t p) {
  mappable.push(id, p);
}
void SchedulerQueues::push_mappable(const TaskIDList &ids,
                                    const PriorityList &ps) {
  assert(ps.size() > ids.size());
  for (auto id : ids) {
    assert(ps.size() > id);
    push_mappable(id, ps[id]);
  }
}

void SchedulerQueues::push_reservable(taskid_t id, priority_t p,
                                      devid_t device) {
  reservable.at(device).push(id, p);
}

void SchedulerQueues::push_reservable(const TaskIDList &ids,
                                      const PriorityList &ps, devid_t device) {
  assert(ps.size() > ids.size());
  for (auto id : ids) {
    push_reservable(id, ps[id], device);
  }
}

void SchedulerQueues::push_launchable(taskid_t id, priority_t p,
                                      devid_t device) {
  launchable.at(device).push(id, p);
}

void SchedulerQueues::push_launchable(const TaskIDList &ids,
                                      const PriorityList &ps, devid_t device) {
  assert(ps.size() > ids.size());
  for (auto id : ids) {
    push_launchable(id, ps[id], device);
  }
}

void SchedulerQueues::push_launchable_data(taskid_t id, priority_t p,
                                           devid_t device) {
  data_launchable.at(device).push(id, p);
}

void SchedulerQueues::push_launchable_data(const TaskIDList &ids,
                                           const PriorityList &ps,
                                           devid_t device) {
  assert(ps.size() > ids.size());
  for (auto id : ids) {
    push_launchable_data(id, ps[id], device);
  }
}

void SchedulerQueues::id_to_queue(taskid_t id, const TaskStateInfo &state) {
  if (state.is_mappable(id)) {
    push_mappable(id, state.get_mapping_priority(id));
  } else if (state.is_reservable(id)) {
    push_reservable(id, state.get_reserving_priority(id),
                    state.get_mapping(id));
  } else if (state.is_launchable(id)) {
    push_launchable(id, state.get_launching_priority(id),
                    state.get_mapping(id));
  }
}

TaskType SchedulerQueues::id_to_type(taskid_t id, const Tasks &tasks) {
  if (tasks.is_compute(id)) {
    return TaskType::COMPUTE;
  }
  return TaskType::DATA;
}

// TODO(wlr): Deal with data tasks
void SchedulerQueues::populate(const TaskManager &task_manager) {
  const auto &state = task_manager.get_state();
  const auto &compute_tasks = task_manager.get_tasks().get_compute_tasks();

  for (const auto &compute_task : compute_tasks) {
    id_to_queue(compute_task.id, state);
  }
}

void SchedulerQueues::populate(const TaskIDList &ids,
                               const TaskManager &task_manager) {
  const auto &state = task_manager.get_state();

  for (auto id : ids) {
    id_to_queue(id, state);
  }
}

// TaskCountInfo

TaskCountInfo::TaskCountInfo(std::size_t n_devices)
    : per_device_mapped_tasks(n_devices), per_device_reserved_tasks(n_devices),
      per_device_launched_tasks(n_devices),
      per_device_completed_tasks(n_devices),
      per_device_data_completed_tasks(n_devices) {}

void TaskCountInfo::count_mapped(devid_t device_id) {
  n_active_tasks += 1;
  n_mapped_tasks += 1;
  per_device_mapped_tasks[device_id] += 1;
}

void TaskCountInfo::count_reserved(devid_t device_id) {
  n_reserved_tasks += 1;
  per_device_reserved_tasks[device_id] += 1;
}

void TaskCountInfo::count_launched(devid_t device_id) {
  n_launched_tasks += 1;
  per_device_launched_tasks[device_id] += 1;
}

void TaskCountInfo::count_completed(devid_t device_id) {
  n_active_tasks -= 1;

  n_mapped_tasks -= 1;
  per_device_mapped_tasks[device_id] -= 1;

  n_reserved_tasks -= 1;
  per_device_reserved_tasks[device_id] -= 1;

  n_launched_tasks -= 1;
  per_device_launched_tasks[device_id] -= 1;

  n_completed_tasks += 1;
  per_device_completed_tasks[device_id] += 1;
}

void TaskCountInfo::count_data_completed(devid_t device_id) {
  n_data_completed_tasks += 1;
  per_device_data_completed_tasks[device_id] += 1;
}

// TaskCostInfo

TaskCostInfo::TaskCostInfo(std::size_t n_tasks, std::size_t n_devices)
    : per_device_mapped_time(n_devices), per_device_reserved_time(n_devices),
      per_device_launched_time(n_devices), per_device_completed_time(n_devices),
      per_device_data_completed_time(n_devices), eft_task_times(n_tasks) {}

void TaskCostInfo::count_mapped(devid_t device_id, timecount_t time) {
  per_device_mapped_time[device_id] += time;
}

void TaskCostInfo::count_reserved(devid_t device_id, timecount_t time) {
  per_device_reserved_time[device_id] += time;
}

void TaskCostInfo::count_launched(devid_t device_id, timecount_t time) {
  per_device_launched_time[device_id] += time;
}

void TaskCostInfo::count_completed(devid_t device_id, timecount_t time) {
  per_device_mapped_time[device_id] -= time;
  per_device_reserved_time[device_id] -= time;
  per_device_launched_time[device_id] -= time;
  per_device_completed_time[device_id] += time;
}

void TaskCostInfo::count_data_completed(devid_t device_id, timecount_t time) {
  per_device_data_completed_time[device_id] += time;
}

// TransitionConstraints

// Scheduler

TaskIDList &Scheduler::get_mappable_candidates() {
  auto &s = this->state;
  bool condition = queues.has_mappable() && conditions.should_map(s, queues);
  clear_task_buffer();

  if (!condition) {
    return task_buffer;
  }

  auto &mappable = queues.mappable;
  auto top_k_tasks = mappable.get_top_k();
  task_buffer.insert(task_buffer.end(), top_k_tasks.begin(), top_k_tasks.end());
  return task_buffer;
}

const TaskIDList &Scheduler::map_task(Action &action) {
  auto &s = this->state;

  taskid_t task_id = action.task_id;
  devid_t chosen_device = action.device;

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
  const ComputeTask &task =
      s.task_manager.get_tasks().get_compute_task(task_id);
  s.data_manager.read_update_mapped(task.get_read(), chosen_device);
  s.data_manager.read_update_mapped(task.get_write(), chosen_device);
  s.data_manager.write_update_mapped(task.get_write(), chosen_device);

  // Notify dependents and enqueue newly mappable tasks
  const auto &newly_mappable_tasks = s.notify_mapped(task_id);
  success_count += 1;
  state.counts.count_mapped(chosen_device);

  breakpoints.check_task_breakpoint(EventType::MAPPER, task_id);

  // std::cout << "Mapped task " << state.get_task_name(task_id) << " to device
  // "
  //           << state.device_manager.devices.get_name(chosen_device)
  //           << std::endl;

  // Check if the mapped task is reservable, and if so, enqueue it
  if (s.is_reservable(task_id)) {
    // std::cout << "Task " << state.get_task_name(task_id)
    //           << " is reservable at time " << s.global_time << std::endl;
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

void Scheduler::map_tasks_from_python(ActionList &action_list,
                                      EventManager &event_manager) {
  success_count = 0;
  if (!action_list.empty()) {
    TaskIDList nmt;
    for (auto &action : action_list) {
      const auto &tasks = map_task(action);
      nmt.insert(nmt.end(), tasks.begin(), tasks.end());
    }
    remove_mapped_tasks(action_list);
    push_mappable(nmt);
  }

  // The next event is a reserving event
  timecount_t reserver_time = state.global_time + TIME_TO_RESERVE;
  event_manager.create_event(EventType::RESERVER, reserver_time, TaskIDList());
}

void Scheduler::map_tasks(Event &map_event, EventManager &event_manager,
                          Mapper &mapper) {
  assert(map_event.get_type() == EventType::MAPPER);
  assert(map_event.get_tasks().empty());
  success_count = 0;

  auto &s = this->state;
  auto &task_states = state.task_manager.state;

  // spdlog::info("Mapping tasks at time {}", s.global_time);
  // spdlog::debug("Mappable Queue Size: {}", queues.mappable.size());

  while (queues.has_mappable() && conditions.should_map(s, queues)) {
    taskid_t task_id = queues.mappable.top();
    queues.mappable.pop();
    assert(task_states.is_mappable(task_id));
    Action action = mapper.map_task(task_id, s);
    // spdlog::info("Mapping task {} at time {} to device {}",
    //              s.get_task_name(task_id), s.global_time, action.device);
    const auto &newly_mappable_tasks = map_task(action);
    // spdlog::debug("Newly mappable tasks: {}", newly_mappable_tasks.size());
    push_mappable(newly_mappable_tasks);
  }

  // The next event is a reserving event
  timecount_t reserver_time = s.global_time + SCHEDULER_TIME_GAP;
  event_manager.create_event(EventType::RESERVER, reserver_time, TaskIDList());
}

void Scheduler::enqueue_data_tasks(taskid_t id) {
  auto &s = this->state;
  auto &task_manager = state.task_manager;
  const auto &tasks = state.task_manager.get_tasks();

  assert(tasks.is_compute(id));

  const auto &data_dependencies = tasks.get_data_dependencies(id);

  // spdlog::debug("Enqueueing {} data tasks for task {}",
  //               data_dependencies.size(), s.get_task_name(id));

  for (auto data_task_id : data_dependencies) {
    assert(tasks.is_data(data_task_id));
    task_manager.set_state(data_task_id, TaskState::RESERVED);
    if (s.is_launchable(data_task_id)) {
      // spdlog::debug("New data task {} is launchable",
      //               s.get_task_name(data_task_id));
      push_launchable_data(data_task_id);
    }
  }
}

SuccessPair Scheduler::reserve_task(taskid_t task_id, devid_t device_id) {
  auto &s = this->state;

  assert(s.is_compute_task(task_id));
  assert(s.is_reservable(task_id));
  assert(s.get_mapping(task_id) == device_id);

  // spdlog::debug("Attempting to reserve task {} at time {}",
  //               s.get_task_name(task_id), s.global_time);

  // Get total required task memory
  const auto [requested, missing] =
      s.request_reserve_resources(task_id, device_id);

  if (missing.mem > 0) {
    return {false, nullptr};
  }

  // spdlog::debug("Reserving task {} at time {}", s.get_task_name(task_id),
  //               s.global_time);

  // Update reserved resources
  s.reserve_resources(task_id, device_id, requested);

  // Update data locations
  const ComputeTask &task =
      s.task_manager.get_tasks().get_compute_task(task_id);
  s.data_manager.read_update_reserved(task.get_read(), device_id);
  s.data_manager.read_update_reserved(task.get_write(), device_id);
  s.data_manager.write_update_reserved(task.get_write(), device_id);

  const auto &newly_reservable_tasks = s.notify_reserved(task_id);

  success_count += 1;
  enqueue_data_tasks(task_id);
  s.counts.count_reserved(device_id);
  breakpoints.check_task_breakpoint(EventType::RESERVER, task_id);

  // Check if the reserved task is launchable, and if so, enqueue it
  if (s.is_launchable(task_id)) {
    // spdlog::debug("Task {} is launchable at time {}",
    // s.get_task_name(task_id),
    //               s.global_time);
    push_launchable(task_id, device_id);
  }

  return {true, &newly_reservable_tasks};
}

void Scheduler::reserve_tasks(Event &reserve_event,
                              EventManager &event_manager) {
  assert(reserve_event.get_type() == EventType::RESERVER);
  assert(reserve_event.get_tasks().empty());
  auto &s = this->state;

  auto &reservable = queues.reservable;
  reservable.reset();
  reservable.current_or_next_active();

  // spdlog::debug("Reserving tasks at time {}", s.global_time);
  // spdlog::debug("Reservable Queue Size: {}",
  //               queues.reservable.total_active_size());

  while (queues.has_active_reservable() &&
         conditions.should_reserve(s, queues)) {

    if (reservable.get_active().empty()) {
      reservable.next();
      continue;
    }

    auto device_id = static_cast<devid_t>(reservable.get_active_index());
    taskid_t task_id = reservable.top();
    auto [success, newly_reservable_tasks] = reserve_task(task_id, device_id);
    if (!success) {
      reservable.deactivate();
      reservable.next();
      continue;
    }

    std::cout << "Reserving task " << s.get_task_name(task_id) << " at time "
              << s.global_time << std::endl;

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

  // The next event is a launching event
  timecount_t launcher_time = s.global_time + TIME_TO_LAUNCH;
  event_manager.create_event(EventType::LAUNCHER, launcher_time, TaskIDList());
}

bool Scheduler::launch_task(taskid_t task_id, devid_t device_id) {
  auto &s = this->state;
  // spdlog::debug("Attempting to launch task {} at time {}",
  //               s.get_task_name(task_id), s.global_time);

  std::cout << "Attempting to launch task " << s.get_task_name(task_id)
            << " at time " << s.global_time << std::endl;

  assert(s.is_launchable(task_id));
  assert(s.get_mapping(task_id) == device_id);
  assert(s.is_compute_task(task_id));

  // Update data locations for WRITE data (create them here)
  const auto &task = s.task_manager.get_tasks().get_compute_task(task_id);
  s.data_manager.read_update_launched(task.get_write(), device_id);
  s.data_manager.write_update_launched(task.get_write(), device_id);

  // All READ data should already be here (prefetched by data tasks)
  s.data_manager.check_valid_launched(task.get_read(), device_id);

  // This should be equivalent to get_task_resources (non_local_mem = 0)
  // Calling this with an assert as an extra sanity check
  const auto [requested, missing] =
      s.request_launch_resources(task_id, device_id);

  if (missing.vcu > 0) {
    return false;
  }

  // spdlog::debug("Launching task {} at time {}", s.get_task_name(task_id),
  //               s.global_time);
  std::cout << "Launching task " << s.get_task_name(task_id) << " at time "
            << s.global_time << std::endl;

  // Update launched resources
  s.launch_resources(task_id, device_id, requested);

  // Record launching time
  s.notify_launched(task_id);
  success_count += 1;
  s.counts.count_launched(device_id);
  breakpoints.check_task_breakpoint(EventType::LAUNCHER, task_id);

  return true;
}

bool Scheduler::launch_data_task(taskid_t task_id, devid_t destination_id) {
  auto &s = this->state;
  // spdlog::debug("Attempting to launch data task {} at time {}",
  //               s.get_task_name(task_id), s.global_time);

  assert(s.is_launchable(task_id));
  assert(s.get_mapping(task_id) == destination_id);

  // spdlog::debug("Launching data task {} at time {}",
  // s.get_task_name(task_id),
  //               s.global_time);

  const auto &task = s.task_manager.get_tasks().get_data_task(task_id);
  const dataid_t data_id = task.get_data_id();

  auto [found, source_id] =
      s.data_manager.request_source(data_id, destination_id);

  if (!found) {
    return false;
  }

  s.data_manager.start_move(data_id, source_id, destination_id);

  // Record launching time
  s.notify_launched(task_id);
  success_count += 1;
  breakpoints.check_task_breakpoint(EventType::LAUNCHER, task_id);
  return true;
}

void Scheduler::launch_tasks(Event &launch_event, EventManager &event_manager) {
  assert(launch_event.get_type() == EventType::LAUNCHER);
  assert(launch_event.get_tasks().empty());

  auto &s = this->state;

  auto &launchable = queues.launchable;
  launchable.reset();
  launchable.current_or_next_active();

  // spdlog::debug("Launching tasks at time {}", s.global_time);
  // spdlog::debug("Launchable Queue Size: {}",
  //               queues.launchable.total_active_size());

  std::cout << "Launching tasks at time " << s.global_time << std::endl;
  std::cout << "Launchable Queue Size: "
            << queues.launchable.total_active_size() << std::endl;

  while (queues.has_active_launchable() &&
         conditions.should_launch(s, queues)) {

    if (launchable.get_active().empty()) {
      launchable.next();
      continue;
    }

    taskid_t task_id = launchable.top();
    auto device_id = static_cast<devid_t>(launchable.get_active_index());

    bool success = launch_task(task_id, device_id);
    if (!success) {
      launchable.deactivate();
      launchable.next();
      continue;
    }
    launchable.pop();
    launchable.next();

    // Create completion event
    timecount_t completion_time = s.global_time + s.get_execution_time(task_id);
    event_manager.create_event(EventType::COMPLETER, completion_time,
                               TaskIDList({task_id}));
  }

  // TODO(wlr): Launch data tasks (queues.data_launchable)
  // spdlog::debug("Data Launchable Queue Size: {}",
  //               queues.data_launchable.total_active_size());

  auto &data_launchable = queues.data_launchable;
  data_launchable.reset();
  data_launchable.current_or_next_active();

  while (queues.has_active_data_launchable() &&
         conditions.should_launch_data(s, queues)) {

    if (data_launchable.get_active().empty()) {
      data_launchable.next();
      continue;
    }

    taskid_t task_id = data_launchable.top();
    assert(s.is_data_task(task_id));
    auto device_id = static_cast<devid_t>(data_launchable.get_active_index());

    bool success = launch_data_task(task_id, device_id);
    if (!success) {
      data_launchable.deactivate();
      data_launchable.next();
      continue;
    }
    data_launchable.pop();
    data_launchable.next();

    // Create completion event
    timecount_t test_data_time = 100;
    timecount_t completion_time = s.global_time + test_data_time;
    event_manager.create_event(EventType::COMPLETER, completion_time,
                               TaskIDList({task_id}));
  }

  scheduler_event_count -= 1;

  if (scheduler_event_count == 0 and success_count > 0) {
    // std::cout << "Launcher is creating a new mapper event at time "
    //           << s.global_time << std::endl;
    event_manager.create_event(EventType::MAPPER,
                               s.global_time + SCHEDULER_TIME_GAP + TIME_TO_MAP,
                               TaskIDList());
    scheduler_event_count += 1;
  }
}

// TODO(wlr): implement eviction event
void Scheduler::evict(Event &eviction_event, EventManager &event_manager) {
  MONUnusedParameter(eviction_event);
  MONUnusedParameter(event_manager);
}

void Scheduler::complete_compute_task(taskid_t task_id, devid_t device_id) {
  auto &s = this->state;

  // Free mapped, reserved, and launched resources
  s.free_resources(task_id);
  s.counts.count_completed(device_id);

  const auto &newly_launchable_data_tasks = s.notify_data_completed(task_id);
  push_launchable_data(newly_launchable_data_tasks);
  // spdlog::debug("Newly launchable data tasks: {}",
  //               newly_launchable_data_tasks.size());
}

void Scheduler::complete_data_task(taskid_t task_id, devid_t device_id) {
  auto &s = this->state;
  s.counts.count_data_completed(device_id);
}

void Scheduler::complete_task(Event &complete_event,
                              EventManager &event_manager) {
  assert(complete_event.get_type() == EventType::COMPLETER);
  assert(complete_event.get_tasks().size() == 1);

  auto &s = this->state;
  taskid_t task_id = complete_event.get_tasks().front();
  devid_t device_id = s.get_mapping(task_id);

  // spdlog::debug("Completing task {} at time {}", s.get_task_name(task_id),
  //               s.global_time);

  if (s.is_compute_task(task_id)) {
    complete_compute_task(task_id, device_id);
  } else {
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
    event_manager.create_event(EventType::MAPPER,
                               s.global_time + TIME_TO_MAP + SCHEDULER_TIME_GAP,
                               TaskIDList());
    scheduler_event_count += 1;
  }
}