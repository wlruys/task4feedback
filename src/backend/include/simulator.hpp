#pragma once
#include "action.hpp"
#include "communication_manager.hpp"
#include "event_manager.hpp"
#include "events.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include "spdlog/cfg/env.h"
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include <cstddef>
#include <functional>
#include <memory>

void logger_setup() {
  auto new_logger = spdlog::stdout_color_mt("console");
  spdlog::set_default_logger(new_logger);
  spdlog::set_level(spdlog::level::debug);
}

enum class ExecutionState {
  NONE = 0,
  RUNNING = 1,
  COMPLETE = 2,
  BREAKPOINT = 3,
  EXTERNAL_MAPPING = 4,
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
  case ExecutionState::EXTERNAL_MAPPING:
    return "EXTERNAL_MAPPING";
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
    if (use_python_mapper && scheduler.get_queues().has_mappable() &&
        scheduler.conditions->should_map(scheduler.get_state(), scheduler.get_queues())) {
      return ExecutionState::EXTERNAL_MAPPING;
    }
    // otherwise just run the mapper from C++
    scheduler.map_tasks(event, event_manager, mapper.get());
    return ExecutionState::RUNNING;
  }

public:
  EventManager event_manager;
  Scheduler scheduler;
  std::reference_wrapper<Mapper> mapper;

  bool initialized = false;
  volatile bool use_python_mapper = false;

  ExecutionState last_state = ExecutionState::NONE;
  Event last_event = Event(EventType::MAPPER, 0, TaskIDList());

  Simulator(SchedulerInput &input)
      : event_manager(EventManager()), scheduler(Scheduler(input)), mapper(input.mapper) {
  }

  void set_use_python_mapper(bool use_python_mapper_) {
    use_python_mapper = use_python_mapper_;
  }

  void set_mapper(Mapper &mapper_) {
    mapper = mapper_;
  }

  const SchedulerState &get_state() const {
    return scheduler.get_state();
  }
  SchedulerState &get_state() {
    return scheduler.get_state();
  }

  void initialize(bool create_data_tasks = false, bool use_transition_conditions = true) {
    add_initial_event();
    scheduler.initialize(create_data_tasks, use_transition_conditions);
    initialized = true;
  }

  void set_transition_conditions(std::shared_ptr<TransitionConditions> conditions_) {
    scheduler.set_transition_conditions(std::move(conditions_));
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

  void update_time(Event &event) {
    scheduler.update_time(event.get_time());
  }

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
      spdlog::critical("No more events and not complete.");
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
      spdlog::critical("Simulator not initialized.");
      return ExecutionState::ERROR;
    }

    if (last_state == ExecutionState::ERROR) {
      spdlog::critical("Simulator in error state.");
      return ExecutionState::ERROR;
    }

    if (last_state == ExecutionState::COMPLETE) {
      return ExecutionState::COMPLETE;
    }

    if (last_state == ExecutionState::EXTERNAL_MAPPING) {
      spdlog::debug("Python Mapping has not been completed.");
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

  [[nodiscard]] devid_t get_mapping(taskid_t task_id) const {
    /*Get location */
    return scheduler.get_state().get_mapping(task_id);
  }

  [[nodiscard]] timecount_t get_mapped_time(taskid_t task_id) const {
    /*Get the time a task was mapped */
    const auto& s = scheduler.get_state();
    const auto& records = s.get_task_manager().get_records();
    return records.get_mapped_time(task_id);
  }

  [[nodiscard]] timecount_t get_reserved_time(taskid_t task_id) const {
    /*Get the time a task was reserved */
    const auto& s = scheduler.get_state();
    const auto& records = s.get_task_manager().get_records();
    return records.get_reserved_time(task_id);
  }

  [[nodiscard]] timecount_t get_launched_time(taskid_t task_id) const {
    /*Get the time a task was launched */
    const auto& s = scheduler.get_state();
    const auto& records = s.get_task_manager().get_records();
    return records.get_launched_time(task_id);
  }

  [[nodiscard]] timecount_t get_completed_time(taskid_t task_id) const {
    /*Get the time a task was completed */
    const auto& s = scheduler.get_state();
    const auto& records = s.get_task_manager().get_records();
    return records.get_completed_time(task_id);
  }

  bool track_resource_guard() const {
    /* Compilation guard for when resource tracking (memory and vcu usage over time) is disabled */
    #ifndef SIM_TRACK_RESOURCES
    spdlog::warn("SIM_TRACK_RESOURCES not defined. Resource tracking is disabled.");
    return true;
    #else
    return false;
    #endif
  }

  bool track_location_guard() const {
    /* Compilation guard for when location tracking (data location over time) is disabled */
    #ifndef SIM_TRACK_LOCATION
    spdlog::warn("SIM_TRACK_LOCATION not defined. Location tracking is disabled.");
    return true;
    #else 
    return false;
    #endif
  }

  [[nodiscard]] vcu_t get_mapped_vcu_at_time(devid_t device_id, timecount_t time) const {
    /* Get the VCU mapped to a device at a given time */
    if(track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_vcu_at_time<TaskState::MAPPED>(device_id, time);
  }

  [[nodiscard]] mem_t get_mapped_mem_at_time(devid_t device_id, timecount_t time) const {
    /* Get the memory mapped to a device at a given time */
    if(track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_mem_at_time<TaskState::MAPPED>(device_id, time);
  }

  [[nodiscard]] vcu_t get_reserved_vcu_at_time(devid_t device_id, timecount_t time) const {
    /* Get the VCU reserved to a device at a given time */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_vcu_at_time<TaskState::RESERVED>(device_id, time);
  }

  [[nodiscard]] mem_t get_reserved_mem_at_time(devid_t device_id, timecount_t time) const{
    /* Get the memory reserved to a device at a given time */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_mem_at_time<TaskState::RESERVED>(device_id, time);
  }

  [[nodiscard]] vcu_t get_launched_vcu_at_time(devid_t device_id, timecount_t time) const{
    /* Get the VCU launched to a device at a given time */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_vcu_at_time<TaskState::LAUNCHED>(device_id, time);
  }

  [[nodiscard]] mem_t get_launched_mem_at_time(devid_t device_id, timecount_t time) const{
    /* Get the memory launched to a device at a given time */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_mem_at_time<TaskState::LAUNCHED>(device_id, time);
  }

  ResourceEventArray<vcu_t> get_vcu_events_mapped(devid_t device_id) const {
    /* Get the VCU events for a device */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_vcu_events<TaskState::MAPPED>(device_id);
  }

  ResourceEventArray<vcu_t> get_vcu_events_reserved(devid_t device_id) const {
    /* Get the VCU events for a device */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_vcu_events<TaskState::RESERVED>(device_id);
  }

  ResourceEventArray<vcu_t> get_vcu_events_launched(devid_t device_id) const {
    /* Get the VCU events for a device */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_vcu_events<TaskState::LAUNCHED>(device_id);
  }

  ResourceEventArray<mem_t> get_mem_events_mapped(devid_t device_id) const {
    /* Get the memory events for a device */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_mem_events<TaskState::MAPPED>(device_id);
  }

  ResourceEventArray<mem_t> get_mem_events_reserved(devid_t device_id) const {
    /* Get the memory events for a device */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_mem_events<TaskState::RESERVED>(device_id);
  }

  ResourceEventArray<mem_t> get_mem_events_launched(devid_t device_id) const {
    /* Get the memory events for a device */
    if (track_resource_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& device_manager = s.get_device_manager();
    return device_manager.get_mem_events<TaskState::LAUNCHED>(device_id);
  }

  [[nodiscard]] TaskState get_state_at_time(taskid_t task_id, timecount_t time) const {
    /* Get the state of a task at a given time */
    if (track_resource_guard()){
      return {};
    }
    const auto& s = scheduler.get_state();
    const auto& records = s.get_task_manager().get_records();
    return records.get_state_at_time(task_id, time);
  }

  ValidEventArray get_valid_intervals_mapped(dataid_t data_id, devid_t device_id) const {
    /* Get the valid intervals for a data mapped to a device */
    if (track_location_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& data_manager = s.get_data_manager();
    return data_manager.get_valid_intervals_mapped(data_id, device_id);
  }

  ValidEventArray get_valid_intervals_reserved(dataid_t data_id, devid_t device_id) const {
    /* Get the valid intervals for a data reserved to a device*/
    if (track_location_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& data_manager = s.get_data_manager();
    return data_manager.get_valid_intervals_reserved(data_id, device_id);
  }

  ValidEventArray get_valid_intervals_launched(dataid_t data_id, devid_t device_id) const {
    /* Get the valid intervals for a data launched to a device */
    if (track_location_guard()){
      return {};
    }

    const auto& s = scheduler.get_state();
    const auto& data_manager = s.get_data_manager();
    return data_manager.get_valid_intervals_launched(data_id, device_id);
  }

  bool check_valid_mapped(dataid_t data_id, devid_t device_id, timecount_t query_time) const {
    /* Check if a data is valid at a given time (in the mapping location table) */
    if (track_location_guard()){
      return false;
    }

    const auto& s = scheduler.get_state();
    const auto& data_manager = s.get_data_manager();
    return data_manager.check_valid_at_time_mapped(data_id, device_id, query_time);
  }

  bool check_valid_reserved(dataid_t data_id, devid_t device_id, timecount_t query_time) const {
    /* Check if a data is valid at a given time (in the reservation location table) */
    if (track_location_guard()){
      return false;
    }

    const auto& s = scheduler.get_state();
    const auto& data_manager = s.get_data_manager();
    return data_manager.check_valid_at_time_reserved(data_id, device_id, query_time);
  }

  bool check_valid_launched(dataid_t data_id, devid_t device_id, timecount_t query_time) const {
    /* Check if a data is valid at a given time (in the launch location table) */
    if (track_location_guard()){
      return false;
    }

    const auto& s = scheduler.get_state();
    const auto& data_manager = s.get_data_manager();
    return data_manager.check_valid_at_time_launched(data_id, device_id, query_time);
  }

};