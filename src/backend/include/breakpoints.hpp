#include "events.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include <limits>
#include <map>

class Breakpoint {};

class BreakpointManager {
private:
  timecount_t max_time = MAX_TIME;
  std::map<EventType, TaskIDList> breakpoints;
  bool breakpoint_status = false;

  static bool check_task(const TaskIDList &tasks, taskid_t task) {
    return std::find(tasks.begin(), tasks.end(), task) != tasks.end();
  }

  static bool check_tasks(const TaskIDList &tasks, const TaskIDList &other) {
    return std::ranges::all_of(
        other, [&tasks](taskid_t task) { return check_task(tasks, task); });
  }

  static bool check_event(const std::map<EventType, TaskIDList> &map,
                          EventType type) {
    return map.find(type) != map.end();
  }

public:
  [[nodiscard]] bool check_breakpoint() const { return breakpoint_status; }

  [[nodiscard]] bool check_task_breakpoint(EventType type, taskid_t task) {
    bool is_breakpoint = check_event(breakpoints, type);
    if (is_breakpoint) {
      is_breakpoint = check_task(breakpoints.at(type), task);
    }
    breakpoint_status = is_breakpoint;
    return is_breakpoint;
  }

  [[nodiscard]] bool check_time_breakpoint(timecount_t time) {
    bool is_breakpoint = (time >= max_time);
    breakpoint_status = is_breakpoint;
    return is_breakpoint;
  }

  void add_breakpoint(EventType type, taskid_t task) {
    if (!check_event(breakpoints, type)) {
      breakpoints[type] = TaskIDList();
    }
    breakpoints[type].push_back(task);
  }

  void add_time_breakpoint(timecount_t time) { max_time = time; }
};