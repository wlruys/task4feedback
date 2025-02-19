#pragma once
#include "events.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include <algorithm>
#include <limits>
#include <map>

class Breakpoint {};


class TaskIDGroup {
protected:
  TaskIDList tasks;

public:
  [[nodiscard]] TaskIDList get_tasks() const {
    return tasks;
  }

  void add_task(taskid_t task) {
    tasks.push_back(task);
  }

  void add_tasks(const TaskIDList &other) {
    tasks.insert(tasks.end(), other.begin(), other.end());
  }

  void remove_task(taskid_t task) {
    tasks.erase(std::remove(tasks.begin(), tasks.end(), task), tasks.end());
  }

  void clear() {
    tasks.clear();
  }

  [[nodiscard]] bool empty() const {
    return tasks.empty();
  }

  [[nodiscard]] size_t size() const {
    return tasks.size();
  }

  [[nodiscard]] bool contains(taskid_t task) const {
    return std::find(tasks.begin(), tasks.end(), task) != tasks.end();
  }

  [[nodiscard]] bool contains_all(const TaskIDList &other) const {
    return std::ranges::all_of(other, [this](taskid_t task) { return contains(task); });
  }

  [[nodiscard]] bool contains_any(const TaskIDList &other) const {
    return std::ranges::any_of(other, [this](taskid_t task) { return contains(task); });
  }

  [[nodiscard]] bool contains_none(const TaskIDList &other) const {
    return std::ranges::none_of(other, [this](taskid_t task) { return contains(task); });
  }
};

class IndividualTasks : public TaskIDGroup {
  public:
    bool check_task(taskid_t task) {
        if (tasks.empty()) {
            return false;
        }
        bool found = std::find(tasks.begin(), tasks.end(), task) != tasks.end();

        if (found) {
            tasks.erase(std::remove(tasks.begin(), tasks.end(), task), tasks.end());
        }

        //If the task is found, then the breakpoint is triggered
        return found;
    }

};

class CollectiveTasks : public TaskIDGroup {
  public: 
  bool check_task(taskid_t task) {
    if (tasks.empty()) {
      return false;
    }

    bool found = std::find(tasks.begin(), tasks.end(), task) != tasks.end();

    if (found) {
      tasks.erase(std::remove(tasks.begin(), tasks.end(), task), tasks.end());

      //If all tasks have been found, then the breakpoint is triggered
      //Only triggers when the last task is found
      if (tasks.empty()) {
        return true;
      }

    }

    return false;
  }
};

class BreakpointManager {
private:
  timecount_t max_time = MAX_TIME;
  std::map<EventType, IndividualTasks> breakpoints;
  std::map<EventType, CollectiveTasks> group_breakpoints;

  volatile bool breakpoint_status = false;

  template<typename T>
  static bool check_event(const std::map<EventType, T> &map, EventType type) {
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
    bool is_breakpoint_individual = false;
    bool is_breakpoint_collective = false;

    if (is_breakpoint) {
      is_breakpoint_individual = breakpoints.at(type).check_task(task);
      is_breakpoint_collective = group_breakpoints.at(type).check_task(task);
      //both need to run in order to remove triggering tasks 
      is_breakpoint = is_breakpoint_individual || is_breakpoint_collective;
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
      breakpoints[type] = IndividualTasks();
    }
    breakpoints[type].add_task(task);
  }

  void add_breakpoint(EventType type, const TaskIDList &tasks) {
    if (!check_event(breakpoints, type)) {
      breakpoints[type] = IndividualTasks();
    }
    breakpoints[type].add_tasks(tasks);
  }

  void add_collective_breakpoint(EventType type, taskid_t task) {
    if (!check_event(group_breakpoints, type)) {
      group_breakpoints[type] = CollectiveTasks();
    }
    group_breakpoints[type].add_task(task);
  }

  void add_collective_breakpoint(EventType type, const TaskIDList &tasks) {
    if (!check_event(group_breakpoints, type)) {
      group_breakpoints[type] = CollectiveTasks();
    }
    group_breakpoints[type].add_tasks(tasks);
  }


  void add_time_breakpoint(timecount_t time) {
    max_time = time;
  }

  void clear() {
    breakpoints.clear();
    reset_breakpoint();
  }
};