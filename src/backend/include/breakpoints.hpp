#pragma once
#include "events.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include <algorithm>
#include <limits>
#include <map>

class Breakpoint {};

class BreakpointManager {
private:
  timecount_t max_time = MAX_TIME;
  // TODO(wlr): Replace with vector for better performance
  // TODO(wlr): Add fast path for empty breakpoints
  std::map<TaskIDList> breakpoints;
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
    return breakpoint_status;
  }

  void reset_breakpoint() {
    breakpoint_status = false;
  }

  bool check_task_breakpoint(EventType type, taskid_t task) {
    bool is_breakpoint = check_event(breakpoints, type);
    if (is_breakpoint) {
      is_breakpoint = check_task(breakpoints[type], task);
    }
    if (is_breakpoint) {
      breakpoint_status = true;
    }
    return is_breakpoint;
  }

  bool check_time_breakpoint(timecount_t time) {
    bool is_breakpoint = (time >= max_time);
    if (is_breakpoint) {
      breakpoint_status = true;
    }
    return is_breakpoint;
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