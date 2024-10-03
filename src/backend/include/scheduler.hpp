#pragma once

#include "breakpoints.hpp"
#include "device_manager.hpp"
#include "events.hpp"
#include "graph.hpp"
#include "iterator.hpp"
#include "queues.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "task_manager.hpp"
#include "tasks.hpp"
#include <cassert>

using TaskQueue = ContainerQueue<taskid_t, std::priority_queue>;
using DeviceQueue = ActiveIterator<TaskQueue>;

using TaskIDTimeList = std::pair<TaskIDList, std::vector<timecount_t>>;

class Scheduler;
class TransitionConstraints;

class SchedulerQueues {
protected:
  TaskQueue mappable;
  DeviceQueue reservable;
  DeviceQueue launchable;

  void id_to_queue(taskid_t id, const TaskStateInfo &state);

public:
  SchedulerQueues(std::size_t num_devices)
      : reservable(num_devices), launchable(num_devices) {}

  void push_mappable(taskid_t id, priority_t p);
  void push_mappable(const TaskIDList &ids, const PriorityList &ps);

  void push_reservable(taskid_t id, priority_t p, devid_t device);
  void push_reservable(const TaskIDList &ids, const PriorityList &ps,
                       devid_t device);
  void push_launchable(taskid_t id, priority_t p, devid_t device);
  void push_launchable(const TaskIDList &ids, const PriorityList &ps,
                       devid_t device);

  [[nodiscard]] std::size_t n_mappable() const { return mappable.size(); }
  [[nodiscard]] std::size_t n_reservable(devid_t device) const {
    const auto &device_queue = reservable[device];
    return device_queue.size();
  }
  [[nodiscard]] std::size_t n_launchable(devid_t device) const {
    const auto &device_queue = launchable[device];
    return device_queue.size();
  }

  void populate(const TaskManager &task_manager);
  void populate(const TaskIDList &ids, const TaskManager &task_manager);

  friend class Scheduler;
};

class SchedulerState {
protected:
  timecount_t global_time = 0;
  TaskManager task_manager;
  DeviceManager device_manager;

public:
  SchedulerState(Tasks &tasks, Devices &devices)
      : task_manager(TaskManager(tasks)),
        device_manager(DeviceManager(devices)) {}

  void update_time(timecount_t time) {
    global_time = std::max(global_time, time);
    assert(global_time > 0);
  }

  void initialize() { task_manager.initialize(); }

  friend class Scheduler;
  friend class TransitionConstraints;
};

class TransitionConstraints {
public:
  bool should_map(const SchedulerState &state, const SchedulerQueues &queues) {
    return true;
  };
  bool should_reserve(const SchedulerState &state,
                      const SchedulerQueues &queues) {
    return true;
  };
  bool should_launch(const SchedulerState &state,
                     const SchedulerQueues &queues) {
    return true;
  };
};

#define INITIAL_BUFFER_SIZE 10

class Scheduler {

protected:
  SchedulerState state;
  SchedulerQueues queues;
  TransitionConstraints constraints;
  BreakpointManager breakpoints;

  bool can_map = true;
  bool can_reserve = true;
  bool can_launch = true;

  TaskIDList task_buffer;

public:
  bool initialized = false;
  Scheduler(Tasks &tasks, Devices &devices)
      : state(tasks, devices), queues(SchedulerQueues(devices.size())) {
    task_buffer.reserve(INITIAL_BUFFER_SIZE);
  }

  TaskIDList initially_mappable_tasks() {
    const auto &compute_tasks =
        state.task_manager.get_tasks().get_compute_tasks();
    return GraphManager::initial_tasks(compute_tasks);
  }

  void initialize() {
    state.initialize();
    const auto &task_states = state.task_manager.state;
    auto initial_tasks = initially_mappable_tasks();
    queues.push_mappable(initial_tasks, task_states.get_mapping_priorities());
    initialized = true;
  }

  TaskIDList &get_mappable_candidates();

  EventList map_tasks();
  EventList map_tasks(std::vector<std::size_t> pos, DeviceIDList &devices);

  EventList reserve_tasks();
  EventList launch_tasks();

  EventList evict();

  EventList complete_tasks(const TaskIDList &completed_tasks);

  void update_time(timecount_t time) { state.update_time(time); }

  friend class SchedulerState;
  friend class SchedulerQueues;
};
