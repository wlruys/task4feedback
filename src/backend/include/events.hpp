#pragma once

#include "resources.hpp"
#include "settings.hpp"
#include <cstddef>
#include <utility>

enum class EventType {
  MAPPER = 0,
  RESERVER = 1,
  LAUNCHER = 2,
  EVICTOR = 3,
  COMPLETER = 4,
};
constexpr std::size_t num_event_types = 5;

// Define << operator for EventType
inline std::string to_string(const EventType &type) {
  switch (type) {
  case EventType::MAPPER:
    return "MAPPER";
    break;
  case EventType::RESERVER:
    return "RESERVER";
    break;
  case EventType::LAUNCHER:
    return "LAUNCHER";
    break;
  case EventType::EVICTOR:
    return "EVICTOR";
    break;
  case EventType::COMPLETER:
    return "COMPLETER";
    break;
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const EventType &type) {
  os << to_string(type);
  return os;
}

class Event {
protected:
  EventType type = EventType::MAPPER;
  timecount_t time = 0;
  TaskIDList tasks;

public:
  Event(EventType type, timecount_t time, TaskIDList tasks)
      : type(type), time(time), tasks(std::move(tasks)) {
  }

  [[nodiscard]] EventType get_type() const {
    return type;
  }
  [[nodiscard]] timecount_t get_time() const {
    return time;
  }

  [[nodiscard]] const TaskIDList &get_tasks() {
    return tasks;
  }

  [[nodiscard]] bool operator<(const Event &other) const {

    if (time == other.time) {
      // larger events are processed first
      // Always process COMPLETER events before scheduler events
      return type > other.type;
    }

    return time < other.time;
  }

  [[nodiscard]] bool operator>(const Event &other) const {
    if (time == other.time) {
      return type < other.type;
    }
    return time > other.time;
  }
};

class MapperEvent : public Event {
public:
  MapperEvent(timecount_t time, TaskIDList tasks)
      : Event{EventType::MAPPER, time, std::move(tasks)} {
  }
};

class ReserverEvent : public Event {
public:
  ReserverEvent(timecount_t time, TaskIDList tasks)
      : Event{EventType::RESERVER, time, std::move(tasks)} {
  }
};

class LauncherEvent : public Event {
public:
  LauncherEvent(timecount_t time, TaskIDList tasks)
      : Event{EventType::LAUNCHER, time, std::move(tasks)} {
  }
};

class EvictorEvent : public Event {
public:
  EvictorEvent(timecount_t time, TaskIDList tasks)
      : Event{EventType::EVICTOR, time, std::move(tasks)} {
  }
};

class CompleterEvent : public Event {
public:
  CompleterEvent(timecount_t time, TaskIDList tasks)
      : Event{EventType::COMPLETER, time, std::move(tasks)} {
  }
};

using EventList = std::vector<Event>;