#pragma once

#include "resources.hpp"
#include "settings.hpp"
#include <ankerl/unordered_dense.h>
#include <array>
#include <cassert>
#include <cstddef>
#include <deque>
#include <functional>
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

inline constexpr bool is_phase_event_type(EventType t) {
  return t == EventType::MAPPER || t == EventType::RESERVER || t == EventType::LAUNCHER;
}

struct EventInfo {
  EventType type;
  timecount_t time;
};

inline EventInfo get_event_info(const EventVariant &v) {
  return std::visit(
      [](const auto &e) -> EventInfo {
        if constexpr (std::is_same_v<std::decay_t<decltype(e)>, CompleterVariant>) {
          return std::visit(
              [](const auto &ce) -> EventInfo { return {ce.type, ce.time}; }, e);
        } else {
          return {e.type, e.time};
        }
      },
      v);
}

class EventQueue {
  using TimeMinHeap =
      std::priority_queue<timecount_t, std::vector<timecount_t>, std::greater<timecount_t>>;

  struct TimeBucket {
    std::array<std::deque<EventVariant>, num_event_types> by_type;
    std::size_t size = 0;

    [[nodiscard]] bool empty() const {
      return size == 0;
    }
  };

  TimeMinHeap times_;
  ankerl::unordered_dense::map<timecount_t, TimeBucket> buckets_;
  std::size_t event_count_{0};

  [[nodiscard]] static std::size_t event_type_index(const EventVariant &ev) {
    return static_cast<std::size_t>(get_type(ev));
  }

public:
  EventQueue() = default;

  void push(EventVariant ev) {
    const auto t = get_time(ev);
    auto it = buckets_.find(t);
    if (it == buckets_.end()) {
      times_.push(t);
      it = buckets_.emplace(t, TimeBucket{}).first;
    }
    const auto type_idx = event_type_index(ev);
    it->second.by_type[type_idx].push_back(std::move(ev));
    it->second.size += 1;
    event_count_ += 1;
  }

  [[nodiscard]] bool empty() const {
    return event_count_ == 0;
  }

  [[nodiscard]] std::size_t size() const {
    return event_count_;
  }

  [[nodiscard]] const EventVariant &top() const {
    assert(event_count_ > 0);
    const auto t = times_.top();
    const auto &bucket = buckets_.at(t);
    for (std::size_t type_idx = num_event_types; type_idx > 0; --type_idx) {
      const auto &q = bucket.by_type[type_idx - 1];
      if (!q.empty()) {
        return q.front();
      }
    }
    assert(false && "EventQueue::top() encountered empty bucket");
    return bucket.by_type[0].front();
  }

  EventVariant pop() {
    assert(event_count_ > 0);
    const auto t = times_.top();
    auto it = buckets_.find(t);
    assert(it != buckets_.end());
    auto &bucket = it->second;

    for (std::size_t type_idx = num_event_types; type_idx > 0; --type_idx) {
      auto &q = bucket.by_type[type_idx - 1];
      if (q.empty()) {
        continue;
      }
      auto ev = std::move(q.front());
      q.pop_front();
      bucket.size -= 1;
      event_count_ -= 1;
      if (bucket.empty()) {
        buckets_.erase(it);
        times_.pop();
      }
      return ev;
    }

    assert(false && "EventQueue::pop() encountered empty bucket");
    return MapperEvent(0);
  }
};

class EventManager {
  EventQueue events_;
  std::deque<EventVariant> inline_phase_events_;
  bool dispatch_active_{false};
  timecount_t dispatch_time_{0};
  bool inline_phase_allowed_{false};

  inline bool should_inline_phase(EventType t, timecount_t time) const {
    return dispatch_active_ && inline_phase_allowed_ && (time == dispatch_time_) &&
           is_phase_event_type(t);
  }

public:
  void begin_dispatch(timecount_t time, EventType dispatch_type) {
    dispatch_active_ = true;
    dispatch_time_ = time;
    inline_phase_allowed_ = is_phase_event_type(dispatch_type);
  }

  void end_dispatch() {
    dispatch_active_ = false;
    inline_phase_allowed_ = false;
  }

  [[nodiscard]] bool has_inline_phase_events() const {
    return !inline_phase_events_.empty();
  }

  EventVariant pop_inline_phase_event() {
    assert(!inline_phase_events_.empty());
    auto ev = std::move(inline_phase_events_.front());
    inline_phase_events_.pop_front();
    return ev;
  }

  // No-payload events:
  inline void create_event(EventType t, timecount_t time) {
    switch (t) {
    case EventType::MAPPER:
      if (should_inline_phase(t, time)) {
        inline_phase_events_.push_back(MapperEvent{time});
      } else {
        events_.push(MapperEvent{time});
      }
      break;
    case EventType::RESERVER:
      if (should_inline_phase(t, time)) {
        inline_phase_events_.push_back(ReserverEvent{time});
      } else {
        events_.push(ReserverEvent{time});
      }
      break;
    case EventType::LAUNCHER:
      if (should_inline_phase(t, time)) {
        inline_phase_events_.push_back(LauncherEvent{time});
      } else {
        events_.push(LauncherEvent{time});
      }
      break;
    case EventType::EVICTOR:
      events_.push(EvictorEvent{time});
      break;
    default:
      throw std::invalid_argument("create_event(type,time) only for MAPPER/RESERVER/LAUNCHER");
    }
  }

  inline void create_event(EventType t, timecount_t time, taskid_t task_id, devid_t device_id) {
    switch (t) {
    case EventType::COMPUTE_COMPLETER: {
      events_.push(ComputeCompleterEvent{time, task_id, device_id});
      break;
    }
    case EventType::DATA_COMPLETER: {
      events_.push(DataCompleterEvent{time, task_id, device_id});
      break;
    }
    case EventType::EVICTOR_COMPLETER: {
      events_.push(EvictorCompleterEvent{time, task_id});
      break;
    }
    default:
      throw std::invalid_argument("create_event(type,time, task_id, device_id) only for "
                                  "COMPUTE_COMPLETER/DATA_COMPLETER/EVICTOR_COMPLETER");
    }
  }

  // Push an existing variant:
  inline void add_event(EventVariant ev) {
    const auto t = get_type(ev);
    const auto time = get_time(ev);
    if (should_inline_phase(t, time)) {
      inline_phase_events_.push_back(std::move(ev));
    } else {
      events_.push(std::move(ev));
    }
  }

  [[nodiscard]] bool has_events() const {
    return !inline_phase_events_.empty() || !events_.empty();
  }
  std::size_t num_events() const {
    return inline_phase_events_.size() + events_.size();
  }

  // Peek at the next event (by const‐ref):
  [[nodiscard]] inline EventVariant const &peek_next_event() const {
    if (!inline_phase_events_.empty()) {
      return inline_phase_events_.front();
    }
    return events_.top();
  }

  // Pop and return by value:
  inline EventVariant pop_event() {
    if (!inline_phase_events_.empty()) {
      auto ev = std::move(inline_phase_events_.front());
      inline_phase_events_.pop_front();
      return ev;
    }
    return events_.pop();
  }
};
