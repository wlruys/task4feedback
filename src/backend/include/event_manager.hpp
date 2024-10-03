#include "events.hpp"
#include "queues.hpp"
#include <queue>
#include <vector>

using EventQueue = std::priority_queue<Event, std::vector<Event>, std::less<>>;

class EventManager {
protected:
  EventQueue events;

public:
  void create_event(EventType type, timecount_t time, TaskIDList tasks) {
    events.emplace(type, time, tasks);
  }

  void add_event(const Event &event) { events.push(event); }

  const Event &peek_next_event() { return events.top(); }

  Event pop_next_event() {
    Event next_event = events.top();
    events.pop();
    return next_event;
  }

  [[nodiscard]] bool has_events() const { return !events.empty(); }
};