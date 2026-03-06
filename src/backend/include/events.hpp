#pragma once

#include "resources.hpp"
#include "settings.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

enum class EventType : int8_t {
  MAPPER = 0,
  RESERVER = 1,
  LAUNCHER = 2,
  EVICTOR = 4,
  COMPUTE_COMPLETER = 8,
  DATA_COMPLETER = 16,
  EVICTOR_COMPLETER = 32
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
  explicit MapperEvent(timecount_t t = 0) : time(t) {
  }
};

struct ReserverEvent {
  static constexpr EventType type = EventType::RESERVER;
  timecount_t time;
  explicit ReserverEvent(timecount_t t = 0) : time(t) {
  }
};

struct LauncherEvent {
  static constexpr EventType type = EventType::LAUNCHER;
  timecount_t time;
  explicit LauncherEvent(timecount_t t = 0) : time(t) {
  }
};

struct EvictorEvent {
  static constexpr EventType type = EventType::EVICTOR;
  timecount_t time;
  explicit EvictorEvent(timecount_t t = 0) : time(t) {
  }
};

struct CompleterEvent {
  timecount_t time;
  taskid_t task;
  devid_t device;
  CompleterEvent(timecount_t t = 0, taskid_t tid = 0, devid_t did = 0)
      : time(t), task(tid), device(did) {
  }
};

struct ComputeCompleterEvent : public CompleterEvent {
  static constexpr EventType type = EventType::COMPUTE_COMPLETER;
  ComputeCompleterEvent(timecount_t t = 0, taskid_t tid = 0, devid_t did = 0)
      : CompleterEvent(t, tid, did) {
  }
};

struct DataCompleterEvent : public CompleterEvent {
  static constexpr EventType type = EventType::DATA_COMPLETER;
  DataCompleterEvent(timecount_t t = 0, taskid_t tid = 0, devid_t did = 0)
      : CompleterEvent(t, tid, did) {
  }
};

struct EvictorCompleterEvent : public CompleterEvent {
  static constexpr EventType type = EventType::EVICTOR_COMPLETER;
  EvictorCompleterEvent(timecount_t t = 0, taskid_t tid = 0) : CompleterEvent(t, tid, 0) {
  }
};

struct Event {
  EventType type{EventType::MAPPER};
  timecount_t time{0};
  taskid_t task{0};
  devid_t device{0};
};

class EventHeap {
  std::vector<Event> heap_;

  [[nodiscard]] static bool higher_priority(const Event &lhs, const Event &rhs) noexcept {
    if (lhs.time != rhs.time) {
      return lhs.time < rhs.time;
    }
    return lhs.type > rhs.type;
  }

  void sift_up(std::size_t idx) noexcept {
    while (idx > 0) {
      const std::size_t parent = (idx - 1) / 2;
      if (!higher_priority(heap_[idx], heap_[parent])) {
        break;
      }
      std::swap(heap_[idx], heap_[parent]);
      idx = parent;
    }
  }

  void sift_down(std::size_t idx) noexcept {
    const std::size_t n = heap_.size();
    while (true) {
      const std::size_t left = idx * 2 + 1;
      const std::size_t right = left + 1;
      std::size_t best = idx;
      if (left < n && higher_priority(heap_[left], heap_[best])) {
        best = left;
      }
      if (right < n && higher_priority(heap_[right], heap_[best])) {
        best = right;
      }
      if (best == idx) {
        return;
      }
      std::swap(heap_[idx], heap_[best]);
      idx = best;
    }
  }

public:
  EventHeap() = default;
  explicit EventHeap(std::size_t reserve_hint) {
    heap_.reserve(reserve_hint);
  }
  EventHeap(const EventHeap &other) {
    const auto active = other.heap_.size();
    const auto headroom = std::max<std::size_t>(64, active / 2);
    heap_.reserve(active + headroom);
    heap_.insert(heap_.end(), other.heap_.begin(), other.heap_.end());
  }
  EventHeap &operator=(const EventHeap &other) {
    if (this == &other) {
      return *this;
    }
    const auto active = other.heap_.size();
    const auto headroom = std::max<std::size_t>(64, active / 2);
    heap_.clear();
    heap_.reserve(active + headroom);
    heap_.insert(heap_.end(), other.heap_.begin(), other.heap_.end());
    return *this;
  }

  void reserve(std::size_t reserve_hint) {
    heap_.reserve(reserve_hint);
  }

  void push(Event event) noexcept {
    heap_.push_back(event);
    sift_up(heap_.size() - 1);
  }

  [[nodiscard]] const Event &top() const noexcept {
    T4F_INVARIANT(!heap_.empty());
    return heap_.front();
  }

  Event pop() noexcept {
    T4F_INVARIANT(!heap_.empty());
    Event next = heap_.front();
    if (heap_.size() == 1) {
      heap_.pop_back();
      return next;
    }
    heap_.front() = heap_.back();
    heap_.pop_back();
    sift_down(0);
    return next;
  }

  [[nodiscard]] bool empty() const noexcept {
    return heap_.empty();
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return heap_.size();
  }
};

class EventManager {
  EventHeap events_;

public:
  EventManager() = default;
  explicit EventManager(std::size_t reserve_hint) : events_(reserve_hint) {
  }

  void reserve(std::size_t reserve_hint) {
    events_.reserve(reserve_hint);
  }

  inline void create_mapper(timecount_t time) {
    events_.push(Event{.type = EventType::MAPPER, .time = time});
  }

  inline void create_reserver(timecount_t time) {
    events_.push(Event{.type = EventType::RESERVER, .time = time});
  }

  inline void create_launcher(timecount_t time) {
    events_.push(Event{.type = EventType::LAUNCHER, .time = time});
  }

  inline void create_evictor(timecount_t time) {
    events_.push(Event{.type = EventType::EVICTOR, .time = time});
  }

  inline void create_compute_completer(timecount_t time, taskid_t task_id, devid_t device_id) {
    events_.push(Event{
        .type = EventType::COMPUTE_COMPLETER, .time = time, .task = task_id, .device = device_id});
  }

  inline void create_data_completer(timecount_t time, taskid_t task_id, devid_t device_id) {
    events_.push(
        Event{.type = EventType::DATA_COMPLETER, .time = time, .task = task_id, .device = device_id});
  }

  inline void create_evictor_completer(timecount_t time, taskid_t task_id) {
    events_.push(Event{.type = EventType::EVICTOR_COMPLETER, .time = time, .task = task_id});
  }

  // No-payload events:
  inline void create_event(EventType t, timecount_t time) {
    switch (t) {
    case EventType::MAPPER:
      create_mapper(time);
      break;
    case EventType::RESERVER:
      create_reserver(time);
      break;
    case EventType::LAUNCHER:
      create_launcher(time);
      break;
    case EventType::EVICTOR:
      create_evictor(time);
      break;
    default:
      throw std::invalid_argument(
          "create_event(type,time) only for MAPPER/RESERVER/LAUNCHER/EVICTOR");
    }
  }

  inline void create_event(EventType t, timecount_t time, taskid_t task_id, devid_t device_id) {
    switch (t) {
    case EventType::COMPUTE_COMPLETER:
      create_compute_completer(time, task_id, device_id);
      break;
    case EventType::DATA_COMPLETER:
      create_data_completer(time, task_id, device_id);
      break;
    case EventType::EVICTOR_COMPLETER:
      create_evictor_completer(time, task_id);
      break;
    default:
      throw std::invalid_argument("create_event(type,time, task_id, device_id) only for "
                                  "COMPUTE_COMPLETER/DATA_COMPLETER/EVICTOR_COMPLETER");
    }
  }

  inline void add_event(Event ev) {
    events_.push(ev);
  }

  [[nodiscard]] bool has_events() const {
    return !events_.empty();
  }
  std::size_t num_events() const {
    return events_.size();
  }

  // Peek at the next event (by const‐ref):
  [[nodiscard]] inline Event const &peek_next_event() const {
    return events_.top();
  }

  // Pop and return by value:
  inline Event pop_event() {
    return events_.pop();
  }
};
