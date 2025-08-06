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
    // Use a simpler, static logger name to avoid potential threading issues
    static bool logger_initialized = false;
    if (logger_initialized) {
      return;
    }

    // Use a simple static name instead of dynamic generation
    std::string logger_name = "simulator_console";

    // Drop existing logger if it exists
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

class Simulator {
protected:
  void add_initial_event() {
    ZoneScoped;
    event_manager.create_event(EventType::MAPPER, 0);
  }

  ExecutionState dispatch_mapper(MapperEvent &event) {
    ZoneScoped;

    auto &queues = scheduler.get_queues();
    auto &state = scheduler.get_state();

    if (state.not_draining() && queues.has_mappable() &&
        scheduler.conditions.get().should_map(state, queues)) {
      if (use_python_mapper) {
        SPDLOG_DEBUG("Time: {} Releasing control to Python mapper", event.time);
        return ExecutionState::EXTERNAL_MAPPING;
      } else {
        SPDLOG_DEBUG("Time: {} Running C++ mapper", event.time);
        scheduler.map_tasks(event, event_manager, mapper.get());
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
  Scheduler scheduler;
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
  EventVariant last_event{MapperEvent(0)};

  Simulator(SchedulerInput &input, Mapper &mapper)
      : event_manager(EventManager()), scheduler(Scheduler(input)), mapper(mapper) {
  }

  void set_use_python_mapper(bool use_python_mapper_) {
    use_python_mapper = use_python_mapper_;
  }

  void set_steps(int32_t steps) {
    scheduler.set_steps(steps);
  }

  void start_drain() {
    scheduler.start_drain();
  }

  void stop_drain() {
    scheduler.stop_drain();

    if (!event_manager.has_events()) {
      const auto current_time = scheduler.get_state().get_global_time();
      event_manager.create_event(EventType::MAPPER, current_time);
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
      assert(false);
      return;
    }

    if (data_initialized) {
      SPDLOG_WARN("Data Manager already initialized. ...skipping.");
      assert(false);
      return;
    }
    scheduler.initialize_data_manager();
    data_initialized = true;
  }

  void set_transition_conditions(TransitionConditions &conditions) {
    scheduler.set_transition_conditions(conditions);
  }

  ExecutionState handle_event(EventVariant &event) {
    ZoneScoped;
    return std::visit(
        [this](auto &e) -> ExecutionState {
          using T = std::decay_t<decltype(e)>;

          if constexpr (std::is_same_v<T, MapperEvent>) {
            return dispatch_mapper(e);
          } else if constexpr (std::is_same_v<T, ReserverEvent>) {
            scheduler.reserve_tasks(e, event_manager);
            return ExecutionState::RUNNING;
          } else if constexpr (std::is_same_v<T, LauncherEvent>) {
            scheduler.launch_tasks(e, event_manager);
            return ExecutionState::RUNNING;
          } else if constexpr (std::is_same_v<T, EvictorEvent>) {
            scheduler.evict(e, event_manager);
            return ExecutionState::RUNNING;
          } else if constexpr (std::is_same_v<T, CompleterVariant>) {
            return std::visit(
                [this](auto &completer_event) -> ExecutionState {
                  using CT = std::decay_t<decltype(completer_event)>;

                  if constexpr (std::is_same_v<CT, ComputeCompleterEvent>) {
                    scheduler.complete_compute_task(completer_event, event_manager);
                    return ExecutionState::RUNNING;
                  } else if constexpr (std::is_same_v<CT, DataCompleterEvent>) {
                    scheduler.complete_data_task(completer_event, event_manager);
                    return ExecutionState::RUNNING;
                  } else if constexpr (std::is_same_v<CT, EvictorCompleterEvent>) {
                    scheduler.complete_eviction_task(completer_event, event_manager);
                    return ExecutionState::RUNNING;
                  } else {
                    spdlog::critical("Unknown completer event type: {}",
                                     typeid(completer_event).name());
                    return ExecutionState::ERROR;
                  }
                },
                e);
          } else {
            spdlog::critical("Unknown event type: {}", typeid(e).name());
            return ExecutionState::ERROR;
          }
        },
        event);
  }

  void update_time(EventVariant &event) {
    scheduler.update_time(get_time(event));
  }

  size_t get_mappable_candidates(std::span<int64_t> v) {
    return scheduler.get_mappable_candidates(v);
  }

  void map_tasks(ActionList &action_list) {
    if (this->last_state != ExecutionState::EXTERNAL_MAPPING) {
      spdlog::critical("Simulator not in external mapping state.");
      assert(false);
      return;
    }

    ExecutionState new_state = scheduler.map_tasks_from_python(action_list, event_manager);
    // Set the state back to running
    this->last_state = new_state;
  }

  void skip_external_mapping(bool enqueue_mapping_event = true) {
    if (last_state != ExecutionState::EXTERNAL_MAPPING) {
      spdlog::critical("Simulator not in external mapping state.");
      assert(false);
      return;
    }

    // Set the state back to running
    this->last_state = ExecutionState::RUNNING;

    // Create a new event to run the mapper
    if (enqueue_mapping_event) {
      const auto current_time = scheduler.get_state().get_global_time();
      event_manager.create_event(EventType::MAPPER, current_time);
    }
  }

  [[nodiscard]] ExecutionState check_breakpoints(ExecutionState ex_state) {
    if (scheduler.is_breakpoint()) {
      SPDLOG_DEBUG("Breakpoint hit at time: {}", this->get_current_time());
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
      } else if (scheduler.is_drain_complete()) {
        return ExecutionState::BREAKPOINT;
      }
      spdlog::critical("No more events and not complete.");
      assert(false);
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
      assert(false);
      return ExecutionState::ERROR;
    }

    if (!data_initialized) {
      last_state = ExecutionState::ERROR;
      spdlog::critical("Data Manager not initialized.");
      assert(false);
      return ExecutionState::ERROR;
    }

    if (last_state == ExecutionState::ERROR) {
      spdlog::critical("Simulator in error state.");
      assert(false);
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

    EventVariant current_event = MapperEvent(0);
    ExecutionState execution_state = ExecutionState::RUNNING;

    while (execution_state == ExecutionState::RUNNING) {
      execution_state = check_complete(execution_state);
      execution_state = check_breakpoints(execution_state);

      if (execution_state != ExecutionState::RUNNING) {
        SPDLOG_DEBUG("Exiting run loop with state: {}", static_cast<int>(execution_state));
        break;
      }

      current_event = event_manager.pop_event();
      events_processed++;
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

  [[nodiscard]] mem_t get_evicted_memory_size() const {
    return scheduler.get_state().get_data_manager().get_lru_manager().get_evicted_memory_size();
  }

  [[nodiscard]] mem_t get_max_memory_usage() const {
    const auto &dm = scheduler.get_state().get_device_manager();
    mem_t overall_max = 0;
    for (devid_t i = 1; i < dm.n_devices; ++i) {
      const auto &tracker = dm.launched.mem_tracker[i];
      if (!tracker.empty()) {
        mem_t device_max = *std::max_element(tracker.resources.begin(), tracker.resources.end());
        overall_max = std::max(overall_max, device_max);
      }
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
