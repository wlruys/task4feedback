#pragma once
#include "events.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include <array>
#include <ankerl/unordered_dense.h>
#include <limits>
#include <spdlog/spdlog.h>

class Breakpoint {};

class BreakpointManager {
private:
  std::array<ankerl::unordered_dense::set<taskid_t>, num_event_types> task_breakpoints;
  std::array<uint8_t, num_event_types> task_breakpoint_enabled{};
  timecount_t max_time = MAX_TIME;
  int32_t steps_to_go = -1;
  int32_t mapper_boundaries_to_go = -1;
  bool step_stop_pending = false;

public:
  [[nodiscard]] bool needs_event_poll() const {
    if (max_time != MAX_TIME) {
      return true;
    }
    for (auto enabled : task_breakpoint_enabled) {
      if (enabled) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool has_pending_step_stop() const {
    return step_stop_pending;
  }

  bool consume_step_stop() {
    const bool pending = step_stop_pending;
    step_stop_pending = false;
    return pending;
  }

  bool check_task_breakpoint(EventType type, taskid_t task) {
    const auto idx = static_cast<std::size_t>(type);
    if (!task_breakpoint_enabled[idx]) {
      return false;
    }
    auto &set = task_breakpoints[idx];
    const auto erased = set.erase(task) > 0;
    if (set.empty()) {
      task_breakpoint_enabled[idx] = 0;
    }
    return erased;
  }

  [[nodiscard]] bool has_time_breakpoint() const {
    return max_time != MAX_TIME;
  }

  bool check_time_breakpoint(timecount_t time) const {
    return max_time != MAX_TIME && time >= max_time;
  }

  void set_steps_to_go(int32_t steps) {
    steps_to_go = steps;
    step_stop_pending = false;
  }

  bool decrement_steps() {
    if (steps_to_go <= 0) {
      return false;
    }
    if (--steps_to_go == 0) {
      step_stop_pending = true;
      return true;
    }
    return false;
  }

  void set_mapper_boundaries_to_go(int32_t boundaries) {
    mapper_boundaries_to_go = boundaries;
  }

  bool decrement_mapper_boundaries() {
    if (mapper_boundaries_to_go <= 0) {
      return false;
    }
    return --mapper_boundaries_to_go == 0;
  }

  void add_breakpoint(EventType type, taskid_t task) {
    const auto idx = static_cast<std::size_t>(type);
    task_breakpoints[idx].insert(task);
    task_breakpoint_enabled[idx] = 1;
  }

  void add_time_breakpoint(timecount_t time) {
    max_time = time;
  }

  void clear() {
    for (auto &bp_set : task_breakpoints) {
      bp_set.clear();
    }
    task_breakpoint_enabled.fill(0);
    max_time = MAX_TIME;
    steps_to_go = -1;
    mapper_boundaries_to_go = -1;
    step_stop_pending = false;
  }
};
