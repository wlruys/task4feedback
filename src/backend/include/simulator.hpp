#include "event_manager.hpp"
#include "events.hpp"
#include "scheduler.hpp"
#include <cstddef>

enum class StopReason {
  ERROR = 0,
  COMPLETE = 1,
  BREAKPOINT = 2,
  MAPPING = 3,
};

class Simulator {
protected:
  void add_initial_event() {
    event_manager.create_event(EventType::MAPPER, 0, TaskIDList());
  }

  volatile bool is_running = true;
  volatile bool pause_request = true;

public:
  EventManager event_manager;
  Scheduler scheduler;

  bool initialized = false;

  Simulator(Tasks &tasks, Devices &devices)
      : event_manager(EventManager()), scheduler(Scheduler(tasks, devices)) {}

  void initialize() {
    add_initial_event();
    scheduler.initialize();
    initialized = true;
  }

  EventList handle_event(Event &event) {
    auto event_type = event.get_type();
    const auto &tasks = event.get_tasks();

    switch (event_type) {
    case EventType::MAPPER:
      return scheduler.map_tasks<DefaultTransitionConditions>(event);
      break;
    case EventType::RESERVER:
      return scheduler.reserve_tasks<DefaultTransitionConditions>(event);
      break;
    case EventType::LAUNCHER:
      return scheduler.launch_tasks<DefaultTransitionConditions>(event);
      break;
    case EventType::EVICTOR:
      return scheduler.evict();
      break;
    case EventType::COMPLETER:
      return scheduler.complete_task(event);
      break;
    }
  }

  void update_time(Event &event) { scheduler.update_time(event.get_time()); }

  bool check_status() {
    // event list has events
    if (!event_manager.has_events()) {
      is_running = false;
      return false;
    }

    // TODO(wlr): check breakpoints

    return true;
  }

  void run() {
    while (is_running) {
      Event next_event = event_manager.pop_next_event();
      update_time(next_event);
      handle_event(next_event);
      check_status();
    }
  }
};