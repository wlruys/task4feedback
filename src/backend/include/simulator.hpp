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
#include <span>
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

enum class StopReason : int8_t {
  NONE = 0,
  BREAKPOINT_MAPPER_BOUNDARY = 1,
  BREAKPOINT_STEPS = 2,
  BREAKPOINT_TIME = 3,
  BREAKPOINT_TASK = 4,
  BREAKPOINT_DRAIN = 5,
  EXTERNAL_MAPPING = 6,
  COMPLETE = 7,
  ERROR = 8,
};

struct StopInfo {
  ExecutionState state = ExecutionState::NONE;
  StopReason reason = StopReason::NONE;
  EventType event_type = EventType::MAPPER;
  timecount_t time = 0;

  operator ExecutionState() const {
    return state;
  }
};

class Simulator {
protected:
  StopReason pending_stop_reason{StopReason::NONE};
  EventType pending_stop_event_type{EventType::MAPPER};
  timecount_t pending_stop_time{0};

  void clear_pending_stop() {
    pending_stop_reason = StopReason::NONE;
    pending_stop_event_type = EventType::MAPPER;
    pending_stop_time = get_current_time();
  }

  void set_pending_stop(StopReason reason, EventType event_type, timecount_t time) {
    pending_stop_reason = reason;
    pending_stop_event_type = event_type;
    pending_stop_time = time;
  }

  [[nodiscard]] StopInfo make_stop_info(ExecutionState state) const {
    StopInfo info;
    info.state = state;
    info.reason = pending_stop_reason;
    info.event_type = pending_stop_event_type;
    info.time = pending_stop_time;
    return info;
  }

  void add_initial_event() {
    ZoneScoped;
    event_manager.create_event(EventType::MAPPER, 0);
  }

  ExecutionState dispatch_mapper(MapperEvent &event) {
    ZoneScoped;
    if (scheduler.hit_mapper_boundary_breakpoint()) {
      SPDLOG_DEBUG("Time: {} Mapper-boundary breakpoint hit", event.time);
      event_manager.create_event(EventType::MAPPER, event.time);
      set_pending_stop(StopReason::BREAKPOINT_MAPPER_BOUNDARY, EventType::MAPPER, event.time);
      return ExecutionState::BREAKPOINT;
    }

    if (scheduler.should_run_mapper_phase()) {
      if (use_python_mapper) {
        SPDLOG_DEBUG("Time: {} Releasing control to Python mapper", event.time);
        set_pending_stop(StopReason::EXTERNAL_MAPPING, EventType::MAPPER, event.time);
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

  [[nodiscard]] ExecutionState validate_run_state() {
    if (last_state == ExecutionState::NONE) {
      last_state = ExecutionState::RUNNING;
    }

    if (!initialized) {
      last_state = ExecutionState::ERROR;
      spdlog::critical("Simulator not initialized.");
      assert(false);
      set_pending_stop(StopReason::ERROR, pending_stop_event_type, get_current_time());
      return ExecutionState::ERROR;
    }

    if (!data_initialized) {
      last_state = ExecutionState::ERROR;
      spdlog::critical("Data Manager not initialized.");
      assert(false);
      set_pending_stop(StopReason::ERROR, pending_stop_event_type, get_current_time());
      return ExecutionState::ERROR;
    }

    if (last_state == ExecutionState::ERROR) {
      spdlog::critical("Simulator in error state.");
      assert(false);
      set_pending_stop(StopReason::ERROR, pending_stop_event_type, get_current_time());
      return ExecutionState::ERROR;
    }

    if (last_state == ExecutionState::COMPLETE) {
      set_pending_stop(StopReason::COMPLETE, pending_stop_event_type, get_current_time());
      return ExecutionState::COMPLETE;
    }

    if (last_state == ExecutionState::EXTERNAL_MAPPING) {
      SPDLOG_DEBUG(
          "Time:{} Python Mapping has not been completed. Returning control to Python layer.",
          this->get_current_time());
      set_pending_stop(StopReason::EXTERNAL_MAPPING, EventType::MAPPER, get_current_time());
      return ExecutionState::EXTERNAL_MAPPING;
    }

    return ExecutionState::RUNNING;
  }

  template <typename DispatchFn>
  ExecutionState dispatch_event_frame(timecount_t event_time, EventType event_type,
                                      EventVariant &event, DispatchFn &&dispatch_fn) {
    event_manager.begin_dispatch(event_time, event_type);
    ExecutionState state = dispatch_fn();
    event_manager.end_dispatch();

    if (state == ExecutionState::BREAKPOINT && scheduler.consume_step_breakpoint()) {
      set_pending_stop(StopReason::BREAKPOINT_STEPS, event_type, event_time);
      return state;
    }
    if (state == ExecutionState::ERROR) {
      set_pending_stop(StopReason::ERROR, event_type, event_time);
      return state;
    }
    if (state == ExecutionState::EXTERNAL_MAPPING && pending_stop_reason == StopReason::NONE) {
      set_pending_stop(StopReason::EXTERNAL_MAPPING, event_type, event_time);
      return state;
    }
    if (state != ExecutionState::RUNNING) {
      return state;
    }

    if (scheduler.consume_step_breakpoint()) {
      set_pending_stop(StopReason::BREAKPOINT_STEPS, event_type, event_time);
      return ExecutionState::BREAKPOINT;
    }

    if (scheduler.needs_event_breakpoint_poll()) {
      if (scheduler.has_time_breakpoint() && scheduler.hit_time_breakpoint(event_time)) {
        set_pending_stop(StopReason::BREAKPOINT_TIME, event_type, event_time);
        return ExecutionState::BREAKPOINT;
      }
      if (auto *completer = std::get_if<CompleterVariant>(&event)) {
        const bool hit_task_breakpoint = std::visit(
            [&](const auto &ce) { return scheduler.hit_task_breakpoint(ce.type, ce.task); },
            *completer);
        if (hit_task_breakpoint) {
          set_pending_stop(StopReason::BREAKPOINT_TASK, event_type, event_time);
          return ExecutionState::BREAKPOINT;
        }
      }
    }

    return state;
  }

  ExecutionState dispatch_regular_event(EventVariant &event) {
    const auto [event_type, event_time] = get_event_info(event);
    update_time(event_time);
    return dispatch_event_frame(event_time, event_type, event,
                                [&]() { return handle_event(event); });
  }

  [[nodiscard]] bool has_pending_same_time_data_completer(timecount_t event_time) const {
    if (!event_manager.has_events()) {
      return false;
    }
    const auto &next_event = event_manager.peek_next_event();
    return get_time(next_event) == event_time && get_type(next_event) == EventType::DATA_COMPLETER;
  }

  ExecutionState run_inline_phase_events(EventVariant &current_event, ExecutionState state) {
    while (state == ExecutionState::RUNNING && event_manager.has_inline_phase_events()) {
      current_event = event_manager.pop_inline_phase_event();
      events_processed++;
      state = dispatch_regular_event(current_event);
    }
    return state;
  }

  ExecutionState run_data_completer_batch(EventVariant &current_event, timecount_t event_time,
                                          ExecutionState state) {
    while (state == ExecutionState::RUNNING) {
      auto *completer = std::get_if<CompleterVariant>(&current_event);
      assert(completer != nullptr && std::holds_alternative<DataCompleterEvent>(*completer));

      state = dispatch_event_frame(event_time, EventType::DATA_COMPLETER, current_event, [&]() {
        scheduler.complete_data_task(std::get<DataCompleterEvent>(*completer), event_manager);
        return ExecutionState::RUNNING;
      });
      if (state != ExecutionState::RUNNING) {
        break;
      }

      state = run_inline_phase_events(current_event, state);
      if (state != ExecutionState::RUNNING || !has_pending_same_time_data_completer(event_time)) {
        break;
      }

      current_event = event_manager.pop_event();
      events_processed++;
      update_time(get_time(current_event));
    }

    return state;
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
  StopInfo last_stop_info{};
  EventVariant last_event{MapperEvent(0)};

  Simulator(SchedulerInput &input, Mapper &mapper)
      : event_manager(EventManager()), scheduler(Scheduler(input)), mapper(mapper) {
    init_simulator_logger();
  }

  void set_use_python_mapper(bool use_python_mapper_) {
    use_python_mapper = use_python_mapper_;
  }

  void enable_profiling(bool enabled = true) {
    (void)enabled;
  }

  void reset_profiling() {
  }

  [[nodiscard]] bool is_profiling_enabled() const {
    return false;
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

  void initialize_data_replicate(dataid_t data_id, devid_t device_id) {
    ZoneScoped;
    if (!initialized) {
      SPDLOG_CRITICAL("Simulator not initialized.");
      assert(false);
      return;
    }

    if (!data_initialized) {
      SPDLOG_CRITICAL("Data Manager not initialized.");
      assert(false);
      return;
    }

    scheduler.initialize_data_replicate(data_id, device_id);
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

  void update_time(timecount_t time) {
    scheduler.update_time(time);
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
    if (new_state == ExecutionState::BREAKPOINT && scheduler.consume_step_breakpoint()) {
      set_pending_stop(StopReason::BREAKPOINT_STEPS, EventType::MAPPER, get_current_time());
    } else if (new_state == ExecutionState::EXTERNAL_MAPPING) {
      set_pending_stop(StopReason::EXTERNAL_MAPPING, EventType::MAPPER, get_current_time());
    } else if (new_state == ExecutionState::RUNNING) {
      clear_pending_stop();
    }
    this->last_state = new_state;
    this->last_stop_info = make_stop_info(new_state);
  }

  void map_tasks_soa(std::span<const int64_t> positions, std::span<const int64_t> devices) {
    if (this->last_state != ExecutionState::EXTERNAL_MAPPING) {
      spdlog::critical("Simulator not in external mapping state.");
      assert(false);
      return;
    }

    ExecutionState new_state = scheduler.map_tasks_from_python_soa(positions, devices, event_manager);
    if (new_state == ExecutionState::BREAKPOINT && scheduler.consume_step_breakpoint()) {
      set_pending_stop(StopReason::BREAKPOINT_STEPS, EventType::MAPPER, get_current_time());
    } else if (new_state == ExecutionState::EXTERNAL_MAPPING) {
      set_pending_stop(StopReason::EXTERNAL_MAPPING, EventType::MAPPER, get_current_time());
    } else if (new_state == ExecutionState::RUNNING) {
      clear_pending_stop();
    }
    this->last_state = new_state;
    this->last_stop_info = make_stop_info(new_state);
  }

  void skip_external_mapping(bool enqueue_mapping_event = true) {
    if (last_state != ExecutionState::EXTERNAL_MAPPING) {
      spdlog::critical("Simulator not in external mapping state.");
      assert(false);
      return;
    }

    // Set the state back to running
    this->last_state = ExecutionState::RUNNING;
    clear_pending_stop();
    this->last_stop_info = make_stop_info(ExecutionState::RUNNING);

    // Create a new event to run the mapper
    if (enqueue_mapping_event) {
      const auto current_time = scheduler.get_state().get_global_time();
      event_manager.create_event(EventType::MAPPER, current_time);
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
      scheduler.log_completion_mismatch();
      spdlog::critical("No more events and not complete.");
      assert(false);
      return ExecutionState::ERROR;
    }

    return ex_state;
  }

  StopInfo run() {
    ZoneScoped;

    SPDLOG_DEBUG("Running simulator");
    clear_pending_stop();
    ExecutionState execution_state = validate_run_state();
    if (execution_state != ExecutionState::RUNNING) {
      last_state = execution_state;
      last_stop_info = make_stop_info(execution_state);
      return last_stop_info;
    }

    EventVariant current_event = MapperEvent(0);

    while (execution_state == ExecutionState::RUNNING) {
      execution_state = check_complete(execution_state);
      if (execution_state == ExecutionState::COMPLETE) {
        set_pending_stop(StopReason::COMPLETE, get_type(last_event), get_current_time());
      } else if (execution_state == ExecutionState::BREAKPOINT) {
        set_pending_stop(StopReason::BREAKPOINT_DRAIN, get_type(last_event), get_current_time());
      } else if (execution_state == ExecutionState::ERROR) {
        set_pending_stop(StopReason::ERROR, get_type(last_event), get_current_time());
      }

      if (execution_state != ExecutionState::RUNNING) {
        SPDLOG_DEBUG("Exiting run loop with state: {}", static_cast<int>(execution_state));
        break;
      }

      current_event = event_manager.pop_event();
      events_processed++;
      const auto [event_type, event_time] = get_event_info(current_event);
      update_time(event_time);

      if (event_type == EventType::DATA_COMPLETER) {
        execution_state = run_data_completer_batch(current_event, event_time, execution_state);
      } else {
        execution_state = dispatch_event_frame(event_time, event_type, current_event,
                                               [&]() { return handle_event(current_event); });
        execution_state = run_inline_phase_events(current_event, execution_state);
      }
    }

    last_state = execution_state;
    last_event = current_event;
    last_stop_info = make_stop_info(execution_state);
    return last_stop_info;
  }

  [[nodiscard]] timecount_t get_current_time() const {
    return scheduler.get_state().get_global_time();
  }

  [[nodiscard]] mem_t get_evicted_memory_size() const {
    return scheduler.get_state()
        .get_data_manager()
        .eviction_residency()
        .get_evicted_memory_size();
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

  [[nodiscard]] const StopInfo &get_last_stop_info() const {
    return last_stop_info;
  }
};
