#include "scheduler.hpp"
#include "events.hpp"
#include "settings.hpp"
#include "tasks.hpp"

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
  clear_event_buffer();
  EventList &next_events = event_buffer;
  auto &task_manager = state.task_manager;
  auto &device_manager = state.device_manager;
  const auto &devices = device_manager.devices;
  const auto &task_states = task_manager.get_state();
  timecount_t current_time = state.global_time;

  while (queues.n_mappable() != 0U) {
    taskid_t task_id = queues.mappable.top();
    queues.mappable.pop();

    // Choose a random target device
    fill_mappable_targets(task_id);
    devid_t device_id = choose_random_target();
    DeviceType device_type = devices.get_type(device_id);
    task_manager.set_mapping(task_id, device_id);

    // TODO(wlr): Update mapped data locations

    // Update mapped resources
    const auto &task_resources =
        task_manager.get_task_resources(task_id, device_type);
    device_manager.add_resources<TaskState::MAPPED>(device_id, task_resources);

    // Notify dependents and enqueue newly mappable tasks
    const auto &newly_mappable =
        task_manager.notify_mapped(task_id, current_time);
    queues.push_mappable(newly_mappable, task_states.get_mapping_priorities());

    // Check if the mapped task is reservable, and if so, enqueue it
    if (task_states.is_reservable(task_id)) {
      queues.push_reservable(
          task_id, task_states.get_reserving_priority(task_id), device_id);
    }
  }

  // The next event is a reserving event
  timecount_t reserver_time = current_time + 0;
  next_events.emplace_back(EventType::RESERVER, reserver_time, TaskIDList());

  return next_events;
}

template <TransitionConditionConcept Conditions>
EventList &Scheduler::reserve_tasks(Event &reserve_event) {
  assert(reserve_event.get_type() == EventType::RESERVER);
  clear_event_buffer();
  EventList &next_events = event_buffer;

  auto &task_manager = state.task_manager;
  auto &device_manager = state.device_manager;
  const auto &devices = device_manager.devices;
  const auto &task_states = task_manager.get_state();
  timecount_t current_time = state.global_time;

  auto &reservable = queues.reservable;
  reservable.reset();

  while (reservable.has_active()) {
    auto &device_queue = reservable.get_active();
    auto device_id = static_cast<devid_t>(reservable.get_active_index());
    DeviceType device_type = devices.get_type(device_id);
    taskid_t task_id = device_queue.top();

    // Get total required task memory
    // TODO(wlr): This should include non-local data, replace with function
    const Resources &task_resources =
        task_manager.get_task_resources(task_id, device_type);
    const mem_t total_task_mem = task_resources.mem;

    // Check if the task can be reserved
    const bool can_fit = device_manager.can_fit_mem<TaskState::RESERVED>(
        device_id, total_task_mem);

    if (!can_fit) {
      // Cannot fit the task at this time
      // TODO(wlr): Implement eviction triggers

      // Mark current device queue as inactive
      // and cycle to the next active device queue
      reservable.deactivate();
      reservable.next();
      continue;
    }

    // Pop the task from the queue
    device_queue.pop();
    // Update reserved resources
    device_manager.add_resources<TaskState::RESERVED>(device_id,
                                                      task_resources);

    // TODO(wlr): Update reserved data locations

    // Notify dependents and enqueue newly reservable tasks
    const auto &newly_reservable =
        task_manager.notify_reserved(task_id, current_time);
    queues.push_reservable(newly_reservable,
                           task_states.get_reserving_priorities(), device_id);

    // Check if the reserved task is launchable, and if so, enqueue it
    if (task_states.is_launchable(task_id)) {
      queues.push_launchable(
          task_id, task_states.get_launching_priority(task_id), device_id);
    }

    // TODO(wlr): Enqueue relevant data tasks
    // Cycle to the next active device queue
    reservable.next();
  }

  // The next event is a launching event
  timecount_t launcher_time = current_time + 0;
  next_events.emplace_back(EventType::LAUNCHER, launcher_time, TaskIDList());
  return next_events;
}

template <TransitionConditionConcept Conditions>
EventList &Scheduler::launch_tasks(Event &launch_event) {
  assert(launch_event.get_type() == EventType::LAUNCHER);
  clear_event_buffer();
  EventList &next_events = event_buffer;

  auto &task_manager = state.task_manager;
  auto &device_manager = state.device_manager;
  const auto &devices = device_manager.devices;
  // const auto &task_states = task_manager.get_state();
  timecount_t current_time = state.global_time;

  auto &launchable = queues.launchable;
  launchable.reset();

  while (launchable.has_active()) {
    auto &device_queue = launchable.get_active();
    auto device_id = static_cast<devid_t>(launchable.get_active_index());
    DeviceType device_type = devices.get_type(device_id);
    taskid_t task_id = device_queue.top();

    // Check if the task can be launched (check VCUs)
    const Resources &task_resources =
        task_manager.get_task_resources(task_id, device_type);
    const vcu_t total_task_vcu = task_resources.vcu;
    const bool can_fit = device_manager.can_fit_vcu<TaskState::LAUNCHED>(
        device_id, total_task_vcu);

    if (!can_fit) {
      // Cannot launch the task at this time
      // Mark current device queue as inactive
      // and cycle to the next active device queue
      launchable.deactivate();
      launchable.next();
      continue;
    }

    // Pop the task from the queue
    device_queue.pop();
    // Update launched resources
    device_manager.add_resources<TaskState::LAUNCHED>(device_id,
                                                      task_resources);
    // Record launching time
    task_manager.notify_launched(task_id, current_time);
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
  clear_event_buffer();
  EventList &next_events = event_buffer;

  auto &task_manager = state.task_manager;
  auto &device_manager = state.device_manager;
  const auto &devices = device_manager.devices;
  const auto &task_states = task_manager.get_state();
  timecount_t current_time = state.global_time;

  // Free mapped resources

  // Free reserved resources (does not include data blocks)

  // Free launched resources (does not include data blocks)

  return next_events;
}
