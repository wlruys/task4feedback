#pragma once
#include "action.hpp"
#include "event_manager.hpp"
#include "events.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include <cstddef>

enum class ExecutionState {
  NONE = 0,
  RUNNING = 1,
  COMPLETE = 2,
  BREAKPOINT = 3,
  PYTHON_MAPPING = 4,
  ERROR = 5,
};
constexpr std::size_t num_execution_states = 6;

inline std::string to_string(const ExecutionState &state) {
  switch (state) {
  case ExecutionState::NONE:
    return "NONE";
    break;
  case ExecutionState::RUNNING:
    return "RUNNING";
    break;
  case ExecutionState::COMPLETE:
    return "COMPLETE";
    break;
  case ExecutionState::BREAKPOINT:
    return "BREAKPOINT";
    break;
  case ExecutionState::PYTHON_MAPPING:
    return "PYTHON_MAPPING";
    break;
  case ExecutionState::ERROR:
    return "ERROR";
    break;
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const ExecutionState &state) {
  os << to_string(state);
  return os;
}

class Simulator {
protected:
  void add_initial_event() {
    event_manager.create_event(EventType::MAPPER, 0, TaskIDList());
  }

  ExecutionState dispatch_mapper(Event &event) {
    if (use_python_mapper) {
      return ExecutionState::PYTHON_MAPPING;
    }
    // otherwise just run the mapper from C++
    scheduler.map_tasks(event, event_manager, mapper);
    return ExecutionState::RUNNING;
  }

public:
  EventManager event_manager;
  Scheduler scheduler;
  Mapper &mapper;

  bool initialized = false;
  volatile bool use_python_mapper = true;

  ExecutionState last_state = ExecutionState::NONE;
  Event last_event = Event(EventType::MAPPER, 0, TaskIDList());

  Simulator(Tasks &tasks, Devices &devices, Mapper &mapper_)
      : event_manager(EventManager()), scheduler(Scheduler(tasks, devices)),
        mapper(mapper_) {}

  void set_use_python_mapper(bool use_python_mapper_) {
    use_python_mapper = use_python_mapper_;
  }

  void set_mapper(Mapper &mapper_) { mapper = mapper_; }

  void initialize(unsigned int seed, bool create_data_tasks = false) {
    add_initial_event();
    scheduler.initialize(seed, create_data_tasks);
    initialized = true;
  }

  void set_transition_conditions(TransitionConditions &conditions_) {
    scheduler.set_transition_conditions(conditions_);
  }

  ExecutionState handle_event(Event &event) {
    auto event_type = event.get_type();

    switch (event_type) {
    case EventType::MAPPER:
      return dispatch_mapper(event);
      break;
    case EventType::RESERVER:
      scheduler.reserve_tasks(event, event_manager);
      return ExecutionState::RUNNING;
      break;
    case EventType::LAUNCHER:
      scheduler.launch_tasks(event, event_manager);
      return ExecutionState::RUNNING;
      break;
    case EventType::EVICTOR:
      scheduler.evict(event, event_manager);
      return ExecutionState::RUNNING;
      break;
    case EventType::COMPLETER:
      scheduler.complete_task(event, event_manager);
      return ExecutionState::RUNNING;
      break;
    }
    return {};
  }

  void update_time(Event &event) { scheduler.update_time(event.get_time()); }

  const TaskIDList &get_mappable_candidates() {
    return scheduler.get_mappable_candidates();
  }

  void map_tasks(ActionList &action_list) {
    scheduler.map_tasks_from_python(action_list, event_manager);
    // Set the state back to running
    last_state = ExecutionState::RUNNING;
  }

  [[nodiscard]] ExecutionState check_breakpoints(ExecutionState ex_state) {
    if (scheduler.is_breakpoint()) {
      scheduler.breakpoints.reset_breakpoint();
      return ExecutionState::BREAKPOINT;
    }
    return ex_state;
  }

  [[nodiscard]] ExecutionState check_complete(ExecutionState ex_state) const {
    // event list has events
    if (!event_manager.has_events()) {
      if (scheduler.is_complete()) {
        return ExecutionState::COMPLETE;
      }
      return ExecutionState::ERROR;
    }

    return ex_state;
  }

  ExecutionState run() {

    if (last_state == ExecutionState::NONE) {
      last_state = ExecutionState::RUNNING;
    }

    if (!initialized) {
      last_state = ExecutionState::ERROR;
      return ExecutionState::ERROR;
    }

    if (last_state == ExecutionState::ERROR) {
      return ExecutionState::ERROR;
    }

    if (last_state == ExecutionState::COMPLETE) {
      return ExecutionState::COMPLETE;
    }

    if (last_state == ExecutionState::PYTHON_MAPPING) {
      return ExecutionState::ERROR;
    }

    Event current_event = Event(EventType::MAPPER, 0, TaskIDList());
    ExecutionState execution_state = ExecutionState::RUNNING;

    while (execution_state == ExecutionState::RUNNING) {
      execution_state = check_complete(execution_state);
      execution_state = check_breakpoints(execution_state);

      if (execution_state != ExecutionState::RUNNING) {
        break;
      }

      current_event = event_manager.pop_event();
      update_time(current_event);
      execution_state = handle_event(current_event);
      scheduler.check_time_breakpoint();
    }

    last_state = execution_state;
    last_event = current_event;

    return execution_state;
  }

  [[nodiscard]] timecount_t get_current_time() const {
    return scheduler.get_state().get_global_time();
  }

  void add_task_breakpoint(EventType type, taskid_t task) {
    scheduler.breakpoints.add_breakpoint(type, task);
  }

  void add_time_breakpoint(timecount_t time) {
    scheduler.breakpoints.add_time_breakpoint(time);
  }
};