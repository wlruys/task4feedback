#pragma once
#include "action.hpp"
#include "communication.hpp"
#include "devices.hpp"
#include "events.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include "spdlog/cfg/env.h"
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>
#include <tracy/Tracy.hpp>
#include <unistd.h>
#include <vector>

void init_simulator_logger() {
  try {
    static bool logger_initialized = false;
    if (logger_initialized) {
      return;
    }

    std::string logger_name = "simulator_console";
    spdlog::drop(logger_name);

    auto logger = spdlog::stdout_color_mt(logger_name);
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::debug);

    logger_initialized = true;
  } catch (const spdlog::spdlog_ex &ex) {
    std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown error during logger initialization" << std::endl;
  }
}

template <typename TransitionConditionT = TransitionConditions>
class SimulatorT {
protected:
  void add_initial_event() {
    ZoneScoped;
    event_manager.create_mapper(0);
  }

  ExecutionState dispatch_mapper(MapperEvent &event) {
    ZoneScoped;

    if (scheduler.hit_mapper_boundary_breakpoint()) {
      SPDLOG_DEBUG("Time: {} Mapper-boundary breakpoint hit", event.time);
      event_manager.create_mapper(event.time);
      return ExecutionState::BREAKPOINT;
    }

    auto &queues = scheduler.get_queues();
    auto &state = scheduler.get_state();

    if (state.not_draining() && queues.has_mappable() &&
        scheduler.conditions.should_map(state, queues)) {
      if (use_python_mapper) {
        SPDLOG_DEBUG("Time: {} Releasing control to Python mapper", event.time);
        return ExecutionState::EXTERNAL_MAPPING;
      } else {
        SPDLOG_DEBUG("Time: {} Running C++ mapper", event.time);
        scheduler.map_tasks(event, event_manager, mapper.get(), true);
        return ExecutionState::RUNNING;
      }
    } else {
      SPDLOG_DEBUG("Skipping mapping tasks, conditions not met.");
      scheduler.skip_map_tasks(event, event_manager);
      return ExecutionState::RUNNING;
    }
  }

public:
  EventManager event_manager;
  SchedulerT<TransitionConditionT> scheduler;
  std::reference_wrapper<Mapper> mapper;
  uint64_t events_processed{0};
  bool initialized{false};
  bool data_initialized{false};
  bool use_python_mapper{false};
  // uint8_t flags{0};

  // constexpr static uint8_t FLAG_USE_PYTHON_MAPPER = 0x01;
  // constexpr static uint8_t FLAG_INITIALIZED = 0x02;
  // constexpr static uint8_t FLAG_DATA_INITIALIZED = 0x04;

  ExecutionState last_state{ExecutionState::NONE};
  Event last_event{.type = EventType::MAPPER, .time = 0, .task = 0, .device = 0};

  SimulatorT(SchedulerInputT<TransitionConditionT> &input, Mapper &mapper)
      : event_manager(), scheduler(input), mapper(mapper) {
    const auto n_compute = static_cast<std::size_t>(input.tasks.get().get_n_compute_tasks());
    const auto n_data = static_cast<std::size_t>(input.tasks.get().get_n_data_tasks());
    std::size_t reserve_hint = input.expected_inflight_events;
    if (reserve_hint == 0) {
      reserve_hint = std::max<std::size_t>(INITIAL_EVENT_BUFFER_SIZE, (n_compute + n_data) / 2);
    }
    event_manager.reserve(reserve_hint);
  }

  void set_use_python_mapper(bool use_python_mapper_) {
    use_python_mapper = use_python_mapper_;
  }

  void set_steps(int32_t steps) {
    scheduler.set_steps(steps);
  }

  void set_mapper_boundary_steps(int32_t boundaries) {
    scheduler.set_mapper_boundary_steps(boundaries);
  }

  void start_drain() {
    scheduler.start_drain();
  }

  void stop_drain() {
    scheduler.stop_drain();

    if (!event_manager.has_events()) {
      const auto current_time = scheduler.get_state().get_global_time();
      event_manager.create_mapper(current_time);
      scheduler.scheduler_event_count += 1;
    }
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

  void initialize(bool create_data_tasks = false, bool initialize_data_manager = false) {
    ZoneScoped;
    if (initialized) {
      SPDLOG_WARN("Simulator already initialized ...skipping.");
      return;
    }
    // std::cout << "Initializing simulator with create_data_tasks: " << create_data_tasks
    //           << " and initialize_data_manager: " << initialize_data_manager << std::endl;
    add_initial_event();
    scheduler.initialize(create_data_tasks, initialize_data_manager);
    initialized = true;
    data_initialized = initialize_data_manager;
  }

  void initialize_data_manager() {
    ZoneScoped;
    if (!initialized) {
      SPDLOG_CRITICAL("Simulator not initialized.");
      T4F_INVARIANT(false);
      return;
    }

    if (data_initialized) {
      SPDLOG_WARN("Data Manager already initialized. ...skipping.");
      T4F_INVARIANT(false);
      return;
    }
    scheduler.initialize_data_manager();
    data_initialized = true;
  }

  void initialize_data_replicate(dataid_t data_id, devid_t device_id) {
    ZoneScoped;
    if (!initialized) {
      SPDLOG_CRITICAL("Simulator not initialized.");
      T4F_INVARIANT(false);
      return;
    }

    if (!data_initialized) {
      SPDLOG_CRITICAL("Data Manager not initialized.");
      T4F_INVARIANT(false);
      return;
    }

    scheduler.initialize_data_replicate(data_id, device_id);
  }

  void set_transition_conditions(const TransitionConditionT &conditions) {
    scheduler.set_transition_conditions(conditions);
  }

  ExecutionState handle_event(const Event &event) {
    ZoneScoped;
    switch (event.type) {
    case EventType::MAPPER: {
      MapperEvent mapper_event{event.time};
      return dispatch_mapper(mapper_event);
    }
    case EventType::RESERVER: {
      ReserverEvent reserve_event{event.time};
      scheduler.reserve_tasks(reserve_event, event_manager);
      return ExecutionState::RUNNING;
    }
    case EventType::LAUNCHER: {
      LauncherEvent launch_event{event.time};
      scheduler.launch_tasks(launch_event, event_manager);
      return ExecutionState::RUNNING;
    }
    case EventType::EVICTOR: {
      EvictorEvent evictor_event{event.time};
      scheduler.evict(evictor_event, event_manager);
      return ExecutionState::RUNNING;
    }
    case EventType::COMPUTE_COMPLETER: {
      ComputeCompleterEvent complete_event{event.time, event.task, event.device};
      scheduler.complete_compute_task(complete_event, event_manager);
      return ExecutionState::RUNNING;
    }
    case EventType::DATA_COMPLETER: {
      DataCompleterEvent complete_event{event.time, event.task, event.device};
      scheduler.complete_data_task(complete_event, event_manager);
      return ExecutionState::RUNNING;
    }
    case EventType::EVICTOR_COMPLETER: {
      EvictorCompleterEvent complete_event{event.time, event.task};
      scheduler.complete_eviction_task(complete_event, event_manager);
      return ExecutionState::RUNNING;
    }
    default:
      spdlog::critical("Unknown event type: {}", static_cast<int>(event.type));
      return ExecutionState::ERROR;
    }
  }

  void update_time(const Event &event) {
    scheduler.update_time(event.time);
  }

  size_t get_mappable_candidates(std::span<int64_t> v) {
    return scheduler.get_mappable_candidates(v);
  }

  void map_tasks(ActionList &action_list) {
    if (this->last_state != ExecutionState::EXTERNAL_MAPPING) {
      spdlog::critical("Simulator not in external mapping state.");
      T4F_INVARIANT(false);
      return;
    }

    ExecutionState new_state = scheduler.map_tasks_from_python(action_list, event_manager);
    // Set the state back to running
    this->last_state = new_state;
  }

  void skip_external_mapping(bool enqueue_mapping_event = true) {
    if (last_state != ExecutionState::EXTERNAL_MAPPING) {
      spdlog::critical("Simulator not in external mapping state.");
      T4F_INVARIANT(false);
      return;
    }

    // Set the state back to running
    this->last_state = ExecutionState::RUNNING;

    // Create a new event to run the mapper
    if (enqueue_mapping_event) {
      const auto current_time = scheduler.get_state().get_global_time();
      event_manager.create_mapper(current_time);
    }
  }

  [[nodiscard]] ExecutionState check_complete(ExecutionState ex_state) const {
    // event list has events
    if (!event_manager.has_events()) {
      if (scheduler.is_complete()) {
        return ExecutionState::COMPLETE;
      } else if (scheduler.is_drain_complete()) {
        return ExecutionState::BREAKPOINT;
      }
      spdlog::critical("No more events and not complete.");
      T4F_INVARIANT(false);
      return ExecutionState::ERROR;
    }

    return ex_state;
  }
  ExecutionState run() {
    ZoneScoped;

    SPDLOG_DEBUG("Running simulator");

    if (last_state == ExecutionState::NONE) {
      last_state = ExecutionState::RUNNING;
    }

    if (!initialized) {
      last_state = ExecutionState::ERROR;
      spdlog::critical("Simulator not initialized.");
      T4F_INVARIANT(false);
      return ExecutionState::ERROR;
    }

    if (!data_initialized) {
      last_state = ExecutionState::ERROR;
      spdlog::critical("Data Manager not initialized.");
      T4F_INVARIANT(false);
      return ExecutionState::ERROR;
    }

    if (last_state == ExecutionState::ERROR) {
      spdlog::critical("Simulator in error state.");
      T4F_INVARIANT(false);
      return ExecutionState::ERROR;
    }

    if (last_state == ExecutionState::COMPLETE) {
      return ExecutionState::COMPLETE;
    }

    if (last_state == ExecutionState::EXTERNAL_MAPPING) {
      SPDLOG_DEBUG(
          "Time:{} Python Mapping has not been completed. Returning control to Python layer.",
          this->get_current_time());
      return ExecutionState::EXTERNAL_MAPPING;
    }

    Event current_event{.type = EventType::MAPPER, .time = 0, .task = 0, .device = 0};
    ExecutionState execution_state = ExecutionState::RUNNING;

    while (execution_state == ExecutionState::RUNNING) {
      if (scheduler.consume_step_breakpoint()) {
        SPDLOG_DEBUG("Breakpoint hit at time: {}", this->get_current_time());
        execution_state = ExecutionState::BREAKPOINT;
        break;
      }

      execution_state = check_complete(execution_state);

      if (execution_state != ExecutionState::RUNNING) {
        SPDLOG_DEBUG("Exiting run loop with state: {}", static_cast<int>(execution_state));
        break;
      }

      current_event = event_manager.pop_event();
      events_processed++;
      update_time(current_event);
      execution_state = handle_event(current_event);

      if (execution_state != ExecutionState::RUNNING) {
        continue;
      }

      if (scheduler.consume_step_breakpoint()) {
        SPDLOG_DEBUG("Breakpoint hit at time: {}", this->get_current_time());
        execution_state = ExecutionState::BREAKPOINT;
        break;
      }

      if (scheduler.needs_event_breakpoint_poll()) {
        const auto event_time = current_event.time;
        if (scheduler.has_time_breakpoint() && scheduler.hit_time_breakpoint(event_time)) {
          execution_state = ExecutionState::BREAKPOINT;
          break;
        }

        const bool is_completer = current_event.type == EventType::COMPUTE_COMPLETER ||
                                  current_event.type == EventType::DATA_COMPLETER ||
                                  current_event.type == EventType::EVICTOR_COMPLETER;
        if (is_completer &&
            scheduler.hit_task_breakpoint(current_event.type, current_event.task)) {
          execution_state = ExecutionState::BREAKPOINT;
          break;
        }
      }
    }

    last_state = execution_state;
    last_event = current_event;

    return execution_state;
  }

  [[nodiscard]] timecount_t get_current_time() const {
    return scheduler.get_state().get_global_time();
  }

  [[nodiscard]] mem_t get_evicted_memory_size() const {
    return scheduler.get_state().get_data_manager().get_lru_manager().get_evicted_memory_size();
  }

  [[nodiscard]] mem_t get_max_memory_usage() const {
    const auto &dm = scheduler.get_state().get_device_manager();
    mem_t overall_max = 0;
    for (devid_t i = 1; i < dm.n_devices; ++i) {
      overall_max = std::max(overall_max, dm.launched.get_mem_peak(i));
    }
    return overall_max;
  }

  [[nodiscard]] const std::vector<mem_t> &get_eviction_data_movement() const {
    return scheduler.get_state()
        .get_data_manager()
        .get_movement_counter()
        .get_eviction_data_movement();
  }

  [[nodiscard]] const std::vector<mem_t> &get_total_data_movement() const {
    return scheduler.get_state()
        .get_data_manager()
        .get_movement_counter()
        .get_total_data_movement();
  }

  void add_task_breakpoint(EventType type, taskid_t task) {
    scheduler.breakpoints.add_breakpoint(type, task);
  }

  void clear_breakpoints() {
    scheduler.breakpoints.clear();
  }

  void add_time_breakpoint(timecount_t time) {
    scheduler.breakpoints.add_time_breakpoint(time);
  }
};

using Simulator = SimulatorT<TransitionConditions>;
