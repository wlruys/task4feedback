#pragma once

#include "resources.hpp"
#include "settings.hpp"
#include <cstddef>
#include <queue>
#include <utility>
#include <variant>
#include <vector>

enum class EventType : int8_t {
  MAPPER = 0,
  RESERVER = 1,
  LAUNCHER = 2,
  EVICTOR = 3,
  COMPUTE_COMPLETER = 4,
  DATA_COMPLETER = 5,
  EVICTOR_COMPLETER = 6
};
constexpr std::size_t num_event_types = 7;

inline std::string to_string(EventType t) {
  switch (t) {
  case EventType::MAPPER:
    return "MAPPER";
  case EventType::RESERVER:
    return "RESERVER";
  case EventType::LAUNCHER:
    return "LAUNCHER";
  case EventType::EVICTOR:
    return "EVICTOR";
  case EventType::COMPUTE_COMPLETER:
    return "COMPUTE_COMPLETER";
  case EventType::DATA_COMPLETER:
    return "DATA_COMPLETER";
  case EventType::EVICTOR_COMPLETER:
    return "EVICTOR_COMPLETER";
  }
  return "UNKNOWN";
}

inline std::ostream &operator<<(std::ostream &os, EventType t) {
  return os << to_string(t);
}

struct MapperEvent {
  static constexpr EventType type = EventType::MAPPER;
  timecount_t time;
  explicit MapperEvent(timecount_t t) : time(t) {
  }
};

struct ReserverEvent {
  static constexpr EventType type = EventType::RESERVER;
  timecount_t time;
  explicit ReserverEvent(timecount_t t) : time(t) {
  }
};

struct LauncherEvent {
  static constexpr EventType type = EventType::LAUNCHER;
  timecount_t time;
  explicit LauncherEvent(timecount_t t) : time(t) {
  }
};

struct EvictorEvent {
  static constexpr EventType type = EventType::EVICTOR;
  timecount_t time;
  explicit EvictorEvent(timecount_t t) : time(t) {
  }
};

struct CompleterEvent {
  timecount_t time;
  taskid_t task;
  devid_t device;
  CompleterEvent(timecount_t t, taskid_t tid, devid_t did) : time(t), task(tid), device(did) {
  }
};

struct ComputeCompleterEvent : public CompleterEvent {
  static constexpr EventType type = EventType::COMPUTE_COMPLETER;
  ComputeCompleterEvent(timecount_t t, taskid_t tid, devid_t did) : CompleterEvent(t, tid, did) {
  }
};

struct DataCompleterEvent : public CompleterEvent {
  static constexpr EventType type = EventType::DATA_COMPLETER;
  DataCompleterEvent(timecount_t t, taskid_t tid, devid_t did) : CompleterEvent(t, tid, did) {
  }
};

struct EvictorCompleterEvent : public CompleterEvent {
  static constexpr EventType type = EventType::EVICTOR_COMPLETER;
  EvictorCompleterEvent(timecount_t t, taskid_t tid) : CompleterEvent(t, tid, 0) {
  }
};

using CompleterVariant =
    std::variant<ComputeCompleterEvent, DataCompleterEvent, EvictorCompleterEvent>;

using EventVariant =
    std::variant<MapperEvent, ReserverEvent, LauncherEvent, EvictorEvent, CompleterVariant>;

struct TypeExtractor {
  EventType operator()(MapperEvent const &) const noexcept {
    return EventType::MAPPER;
  }
  EventType operator()(ReserverEvent const &) const noexcept {
    return EventType::RESERVER;
  }
  EventType operator()(LauncherEvent const &) const noexcept {
    return EventType::LAUNCHER;
  }
  EventType operator()(EvictorEvent const &) const noexcept {
    return EventType::EVICTOR;
  }
  EventType operator()(CompleterVariant const &cv) const noexcept {
    return std::visit([](auto const &e) -> EventType { return e.type; }, cv);
  }
};

inline EventType get_type(EventVariant const &v) {
  return std::visit(TypeExtractor{}, v);
}

inline timecount_t get_time(EventVariant const &v) {
  return std::visit(
      [](auto const &e) -> timecount_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(e)>, CompleterVariant>) {
          return std::visit([](auto const &ce) -> timecount_t { return ce.time; }, e);
        } else {
          return e.time;
        }
      },
      v);
}

// Comparator for our min-heap priority queue (earlier time ⇒ higher priority);
// on tie, larger EventType -> higher priority so COMPLETER (4) goes first)
struct EventVariantCompare {
  bool operator()(EventVariant const &a, EventVariant const &b) const {
    auto ta = get_time(a), tb = get_time(b);
    if (ta != tb)
      return ta > tb;
    // if same time, we want the one with larger EventType first:
    return get_type(a) < get_type(b);
  }
};

using EventQueue =
    std::priority_queue<EventVariant, std::vector<EventVariant>, EventVariantCompare>;

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
  inline void create_event(EventType t, timecount_t time, taskid_t task_id, devid_t device_id) {
    // if (t != EventType::COMPLETER) {
    //   std::cout << "create_event: expected COMPLETER got " << t;
    //   throw std::invalid_argument("create_event: expected COMPLETER");
    // }
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