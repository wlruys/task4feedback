#pragma once

#include "device_manager.hpp"
#include "iterator.hpp"
#include "queues.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "task_manager.hpp"

using TaskQueue = ContainerQueue<taskid_t, std::priority_queue>;
using DeviceQueue = ActiveIterator<TaskQueue>;

using TaskIDTimeList = std::pair<TaskIDList, std::vector<timecount_t>>;

class SchedulerQueues {
public:
  TaskQueue mappable;
  DeviceQueue reservable;
  DeviceQueue launchable;

  SchedulerQueues(std::size_t num_devices)
      : reservable(num_devices), launchable(num_devices) {}

  void populate(TaskManager &task_manager);
  void populate(TaskIDList &active_tasks, TaskManager &task_manager);
};

class SchedulerState {
public:
  TaskManager tasks;
  DeviceManager devices;

  SchedulerState(std::size_t num_devices, Tasks &tasks)
      : tasks(TaskManager(tasks)), devices(DeviceManager(num_devices)) {}
};

#define INITIAL_BUFFER_SIZE 10

class Scheduler {

public:
  SchedulerState state;
  SchedulerQueues queues;

  bool can_map = true;
  bool can_reserve = true;
  bool can_launch = true;

  TaskIDList task_buffer;

  Scheduler(std::size_t num_devices, Tasks &tasks)
      : state(num_devices, tasks), queues(SchedulerQueues(num_devices)) {
    task_buffer.reserve(INITIAL_BUFFER_SIZE);
  }

  TaskIDList &get_mappable_candidates(timecount_t time);

  TaskIDList &map_tasks(std::vector<std::size_t> pos, DeviceIDList &devices);

  TaskIDList &reserve_tasks(timecount_t time);
  TaskIDTimeList &launch_tasks(timecount_t time);

  void complete_task(taskid_t id, timecount_t time);
};
