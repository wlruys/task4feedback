#pragma once

#include "resources.hpp"
#include "settings.hpp"
#include <cstddef>
#include <utility>
#include <variant>

enum class EventType {
  MAPPER = 0,
  RESERVER = 1,
  LAUNCHER = 2,
  EVICTOR = 3,
  COMPLETER = 4,
};
constexpr std::size_t num_event_types = 5;

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
  case EventType::COMPLETER:
    return "COMPLETER";
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
  TaskDeviceList tasks;
  EvictorEvent(timecount_t t, TaskDeviceList ts) : time(t), tasks(std::move(ts)) {
  }
};

struct CompleterEvent {
  static constexpr EventType type = EventType::COMPLETER;
  timecount_t time;
  TaskIDList tasks;
  CompleterEvent(timecount_t t, TaskIDList ts) : time(t), tasks(std::move(ts)) {
  }
};

using EventVariant =
    std::variant<MapperEvent, ReserverEvent, LauncherEvent, EvictorEvent, CompleterEvent>;

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
  EventType operator()(CompleterEvent const &) const noexcept {
    return EventType::COMPLETER;
  }
};

inline EventType get_type(EventVariant const &v) {
  return std::visit(TypeExtractor{}, v);
}

inline timecount_t get_time(EventVariant const &v) {
  return std::visit([](auto const &e) -> timecount_t { return e.time; }, v);
}

// Comparator for a min-heap (earlier time ⇒ higher priority;
// on tie, larger EventType ⇒ higher priority so COMPLETER (4) goes first)
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
