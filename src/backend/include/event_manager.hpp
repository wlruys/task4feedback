#pragma once
#include "events.hpp"
#include "queues.hpp"
#include <queue>
#include <vector>

using EventQueue =
    std::priority_queue<Event, std::vector<Event>, std::greater<>>;

class EventManager {
protected:
  EventQueue events;

public:
  void create_event(EventType type, timecount_t time, TaskIDList tasks) {
    events.emplace(type, time, tasks);
  }

  void add_event(const Event &event) { events.push(event); }

  void add_events(const EventList &event_list) {
    // std::cout << "Adding " << event_list.size() << " events" << std::endl;
    for (const auto &event : event_list) {
      // std::cout << "Adding event of type " << event.get_type() << std::endl;
      events.push(event);
    }
  }

  const Event &peek_next_event() { return events.top(); }

  std::size_t num_events() const { return events.size(); }

  Event pop_event() {
    Event next_event = events.top();
    events.pop();
    return next_event;
  }

  [[nodiscard]] bool has_events() const { return !events.empty(); }
};