#pragma once
#include "events.hpp"
#include <queue>
#include <vector>

class EventManager {
  EventQueue events_;

public:
  // No-payload events:
  inline void create_event(EventType t, timecount_t time) {
    switch (t) {
    case EventType::MAPPER:
      events_.push(MapperEvent{time});
      break;
    case EventType::RESERVER:
      events_.push(ReserverEvent{time});
      break;
    case EventType::LAUNCHER:
      events_.push(LauncherEvent{time});
      break;
    case EventType::EVICTOR:
      events_.push(EvictorEvent{time});
      break;
    default:
      throw std::invalid_argument("create_event(type,time) only for MAPPER/RESERVER/LAUNCHER");
    }
  }

  // Completer with TaskIDList:
  inline void create_event(EventType t, timecount_t time, taskid_t task_id) {
    if (t != EventType::COMPLETER) {
      std::cout << "create_event: expected COMPLETER got " << t;
      throw std::invalid_argument("create_event: expected COMPLETER");
    }
    events_.push(CompleterEvent{time, task_id});
  }

  // Push an existing variant:
  inline void add_event(EventVariant ev) {
    events_.push(std::move(ev));
  }

  [[nodiscard]] bool has_events() const {
    return !events_.empty();
  }
  std::size_t num_events() const {
    return events_.size();
  }

  // Peek at the next event (by const‐ref):
  [[nodiscard]] inline EventVariant const &peek_next_event() const {
    return events_.top();
  }

  // Pop and return by value:
  inline EventVariant pop_event() {
    auto ev = events_.top();
    events_.pop();
    return ev;
  }
};