#pragma once
#include "events.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <spdlog/spdlog.h>

class Breakpoint {};

class BreakpointManager {
private:
  std::map<EventType, TaskIDList> breakpoints;
  timecount_t max_time = MAX_TIME;
  int32_t steps_to_go = -1;

  volatile bool breakpoint_status = false;

  static bool check_task(TaskIDList &tasks, taskid_t task) {
    if (tasks.empty()) {
      return false;
    }
    bool found = std::find(tasks.begin(), tasks.end(), task) != tasks.end();

    if (found) {
      // Remove task from list
      tasks.erase(std::remove(tasks.begin(), tasks.end(), task), tasks.end());
    }
    return found;
  }

  static bool check_tasks(TaskIDList &tasks, const TaskIDList &other) {
    if (tasks.empty() || other.empty()) {
      return false;
    }

    return std::ranges::all_of(other, [&tasks](taskid_t task) { return check_task(tasks, task); });
  }

  static bool check_event(const std::map<EventType, TaskIDList> &map, EventType type) {
    return map.find(type) != map.end();
  }

public:
  [[nodiscard]] bool check_breakpoint() const {
    SPDLOG_DEBUG("Breakpoint status: {}", breakpoint_status);
    return breakpoint_status;
  }

  void reset_breakpoint() {
    SPDLOG_DEBUG("Resetting breakpoint status.");
    breakpoint_status = false;
  }

  bool check_task_breakpoint(EventType type, taskid_t task) {
    SPDLOG_DEBUG("Checking task breakpoint for type: {} and task: {}", static_cast<int>(type),
                 task);
    bool is_breakpoint = check_event(breakpoints, type);
    if (is_breakpoint) {
      is_breakpoint = check_task(breakpoints[type], task);
    }
    breakpoint_status |= is_breakpoint;
    return is_breakpoint;
  }

  bool check_time_breakpoint(timecount_t time) {
    SPDLOG_DEBUG("Checking time breakpoint for time: {} with max_time: {}", time, max_time);
    breakpoint_status |= (time >= max_time);
    return breakpoint_status;
  }

  void set_steps_to_go(int32_t steps) {
    SPDLOG_DEBUG("Setting steps to go: {}", steps);
    steps_to_go = steps;
  }

  bool decrement_steps() {
    SPDLOG_DEBUG("Decrementing steps: {}", steps_to_go);
    breakpoint_status |= (steps_to_go > 0) ? (--steps_to_go == 0) : false;
    SPDLOG_DEBUG("Breakpoint status after decrement: {}", breakpoint_status);
    return breakpoint_status;
  }

  void add_breakpoint(EventType type, taskid_t task) {
    if (!check_event(breakpoints, type)) {
      breakpoints[type] = TaskIDList();
    }
    breakpoints[type].push_back(task);
  }

  void add_time_breakpoint(timecount_t time) {
    max_time = time;
  }

  void clear() {
    breakpoints.clear();
    reset_breakpoint();
  }
};