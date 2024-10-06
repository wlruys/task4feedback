#include "scheduler.hpp"
#include "events.hpp"
#include "settings.hpp"
#include "task_manager.hpp"
#include "tasks.hpp"
// SchedulerState

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
  Resources requested = task_resources;
  Resources missing;
  return {requested, missing};
}

ResourceRequest
SchedulerState::request_reserve_resources(taskid_t task_id,
                                          devid_t device_id) const {
  const Resources &task_resources = get_task_resources(task_id, device_id);
  Resources requested = task_resources;
  auto missing_mem = device_manager.overflow_mem<TaskState::RESERVED>(
      device_id, requested.mem);
  return {requested, Resources(0, missing_mem)};
}

ResourceRequest
SchedulerState::request_launch_resources(taskid_t task_id,
                                         devid_t device_id) const {
  const Resources &task_resources = get_task_resources(task_id, device_id);
  Resources requested = task_resources;
  auto missing_vcu = device_manager.overflow_vcu<TaskState::LAUNCHED>(
      device_id, requested.vcu);
  return {requested, Resources(missing_vcu, 0)};
}

void SchedulerState::map_resources(taskid_t task_id, devid_t device_id,
                                   const Resources &requested) {
  device_manager.add_resources<TaskState::MAPPED>(device_id, requested);
}

void SchedulerState::reserve_resources(taskid_t task_id, devid_t device_id,
                                       const Resources &requested) {
  device_manager.add_resources<TaskState::RESERVED>(device_id, requested);
}

void SchedulerState::launch_resources(taskid_t task_id, devid_t device_id,
                                      const Resources &requested) {
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

// TransitionConstraints

// Scheduler

void Scheduler::fill_mappable_targets(taskid_t task_id) {
  auto &task_manager = state.task_manager;
  const auto &tasks = task_manager.get_tasks();
  auto supported_architectures = tasks.get_supported_architectures(task_id);

  for (auto arch : supported_architectures) {
    const auto &device_ids = state.device_manager.devices.get_devices(arch);
    device_buffer.insert(device_buffer.end(), device_ids.begin(),
                         device_ids.end());
  }
}

devid_t Scheduler::choose_random_target() {
  std::uniform_int_distribution<std::size_t> dis(0, device_buffer.size() - 1);
  return device_buffer[dis(gen)];
}

template <TransitionConditionConcept Conditions>
EventList &Scheduler::map_tasks(Event &map_event) {
  assert(map_event.get_type() == EventType::MAPPER);
  assert(map_event.get_tasks().empty());
  EventList &next_events = get_clear_event_buffer();
  auto &s = this->state;
  auto &task_states = state.task_manager.state;

  while (queues.has_mappable()) {
    taskid_t task_id = queues.mappable.top();
    queues.mappable.pop();

    assert(task_states.is_mappable(task_id));

    // Choose a random target device
    fill_mappable_targets(task_id);
    devid_t chosen_device = choose_random_target();
    s.set_mapping(task_id, chosen_device);

    // TODO(wlr): Update mapped data locations

    // Update mapped resources
    auto [requested, missing] = s.request_map_resources(task_id, chosen_device);
    s.map_resources(task_id, chosen_device, requested);

    // Notify dependents and enqueue newly mappable tasks
    const auto &newly_mappable_tasks = s.notify_mapped(task_id);
    push_mappable(newly_mappable_tasks);

    // Check if the mapped task is reservable, and if so, enqueue it
    if (task_states.is_reservable(task_id)) {
      push_reservable(task_id, chosen_device);
    }
  }

  // The next event is a reserving event
  timecount_t reserver_time = s.global_time + 0;
  next_events.emplace_back(EventType::RESERVER, reserver_time, TaskIDList());

  return next_events;
}

template <TransitionConditionConcept Conditions>
EventList &Scheduler::reserve_tasks(Event &reserve_event) {
  assert(reserve_event.get_type() == EventType::RESERVER);
  assert(reserve_event.get_tasks().empty());
  EventList &next_events = get_clear_event_buffer();
  auto &s = this->state;

  auto &reservable = queues.reservable;
  reservable.reset();

  while (queues.has_active_reservable()) {
    auto device_id = static_cast<devid_t>(reservable.get_active_index());
    taskid_t task_id = reservable.top();

    assert(s.is_reservable(task_id));

    // Get total required task memory
    const auto [requested, missing] =
        s.request_reserve_resources(task_id, device_id);

    if (!missing.empty_mem()) {
      // Cannot fit the task at this time
      // Mark current device queue as inactive
      // and cycle to the next active device queue
      reservable.deactivate();
      reservable.next();
      continue;
    }

    // Pop the task from the queue
    reservable.pop();
    // Update reserved resources
    s.reserve_resources(task_id, device_id, requested);

    // Notify dependents and enqueue newly reservable tasks
    const auto &newly_reservable_tasks = s.notify_reserved(task_id);
    push_reservable(newly_reservable_tasks);

    // Check if the reserved task is launchable, and if so, enqueue it
    if (s.is_launchable(task_id)) {
      push_launchable(task_id, device_id);
    }

    // Cycle to the next active device queue
    reservable.next();
  }

  // The next event is a launching event
  timecount_t launcher_time = s.global_time + 0;
  next_events.emplace_back(EventType::LAUNCHER, launcher_time, TaskIDList());
  return next_events;
}

template <TransitionConditionConcept Conditions>
EventList &Scheduler::launch_tasks(Event &launch_event) {
  assert(launch_event.get_type() == EventType::LAUNCHER);
  assert(launch_event.get_tasks().empty());
  EventList &next_events = get_clear_event_buffer();

  auto &s = this->state;

  auto &launchable = queues.launchable;
  launchable.reset();

  while (queues.has_active_launchable()) {
    auto device_id = static_cast<devid_t>(launchable.get_active_index());
    taskid_t task_id = launchable.top();

    assert(s.is_launchable(task_id));
    assert(s.get_mapping(task_id) == device_id);

    const auto [requested, missing] =
        s.request_launch_resources(task_id, device_id);

    if (!missing.empty_vcu()) {
      // Cannot launch the task at this time
      // Mark current device queue as inactive
      // and cycle to the next active device queue
      launchable.deactivate();
      launchable.next();
      continue;
    }

    // Pop the task from the queue
    launchable.pop();
    // Update launched resources
    s.launch_resources(task_id, device_id, requested);

    // Record launching time
    s.notify_launched(task_id);
    // Cycle to the next active device queue
    launchable.next();
  }

  // TODO(wlr): Launch data tasks (queues.data_launchable)

  return next_events;
}

// TODO(wlr): implement eviction event
EventList &Scheduler::evict() {
  clear_event_buffer();
  EventList &next_events = event_buffer;
  return next_events;
}

EventList &Scheduler::complete_task(Event &complete_event) {
  assert(complete_event.get_type() == EventType::COMPLETER);
  assert(complete_event.get_tasks().size() == 1);
  EventList &next_events = get_clear_event_buffer();

  auto &s = this->state;
  taskid_t task_id = complete_event.get_tasks().front();

  // Free mapped, reserved, and launched resources
  s.free_resources(task_id);

  // Notify dependents and enqueue newly launchable tasks
  const auto &newly_launchable_tasks = s.notify_completed(task_id);
  push_launchable(newly_launchable_tasks);

  return next_events;
}

template EventList &
Scheduler::map_tasks<DefaultTransitionConditions>(Event &map_event);
template EventList &
Scheduler::reserve_tasks<DefaultTransitionConditions>(Event &reserve_event);
template EventList &
Scheduler::launch_tasks<DefaultTransitionConditions>(Event &launch_event);