#include "scheduler.hpp"
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