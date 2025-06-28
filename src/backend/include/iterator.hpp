#pragma once
#include "queues.hpp"
#include "settings.hpp"
#include <vector>

template <typename T> class ActiveIterator {
protected:
  std::vector<T> containers;
  std::vector<int32_t> active;
  std::size_t active_index;
  std::size_t num_active;

public:
  ActiveIterator() = default;
  ActiveIterator(std::size_t num_containers)
      : containers(num_containers), active(num_containers, true), active_index(0),
        num_active(num_containers) {
  }

  bool is_active(std::size_t index) noexcept {
    return active[index] == 1;
  }
  bool has_active() {
    return num_active > 0;
  }

  T &get_active() {
    return containers[active_index];
  }
  T &operator[](std::size_t index) noexcept {
    return containers[index];
  }
  T &at(std::size_t index) {
    return containers[index];
  }
  const T &operator[](std::size_t index) const noexcept {
    return containers[index];
  }
  const T &at(std::size_t index) const noexcept {
    return containers[index];
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return containers.size();
  }

  [[nodiscard]] std::size_t active_size() const noexcept {
    return num_active;
  }
  [[nodiscard]] std::size_t total_size() const noexcept {
    std::size_t tsize = 0;
    for (auto &c : containers) {
      tsize += c.size();
    }
    return tsize;
  }

  [[nodiscard]] std::size_t total_active_size() const noexcept {
    int32_t tsize = 0;
    const auto n = containers.size();
    for (int32_t i = 0; i < containers.size(); i++) {
      tsize += containers[i].size() * std::size_t(active[i]);
    }
    return tsize;
  }

  void set_active_queue(int index) noexcept {
    active_index = index;
  }
  int get_active_index() noexcept {
    return active_index;
  }

  void deactivate(int index) noexcept {
    active[index] = false;
    num_active--;
  }

  void deactivate() {
    active[active_index] = false;
    num_active--;
  }

  void activate(std::size_t index) noexcept {
    active[index] = 1;
    num_active++;
  }

  void activate() noexcept {
    active[active_index] = 1;
    num_active++;
  }

  void next() noexcept {
    active_index = (active_index + 1) % containers.size();
  }

  void current_or_next_active() noexcept {
    if (!active[active_index]) {
      next_active();
    }
  }

  void prev() noexcept {
    active_index = (active_index == 0) ? containers.size() - 1 : active_index - 1;
  }

  void next_active() noexcept {
    next();
    while (!active[active_index]) {
      next();
    }
  }

  void prev_active() noexcept {
    prev();
    while (!active[active_index]) {
      prev();
    }
  }

  void reset() noexcept {
    for (int32_t i = 0; i < active.size(); i++) {
      active[i] = 1;
    }
    num_active = containers.size();
  }
};

template <PriorityQueueConcept Q> class ActiveQueueIterator : public ActiveIterator<Q> {

public:
  void push(Q::value_type value) noexcept {
    this->containers[this->active_index].push(value);
  }

  void push(Q::value_type value, priority_t priority) noexcept {
    this->containers[this->active_index].push(value, priority);
  }

  void push_at(std::size_t index, Q::value_type value) noexcept {
    this->containers[index].push(value);
  }

  void push_priority_at(std::size_t index, Q::value_type value, priority_t priority) noexcept {
    this->activate(index);
    this->containers[index].push(value, priority);
  }

  void push_random(Q::value_type value) noexcept {
    this->containers[this->active_index].push_random(value);
  }

  void push_random_at(std::size_t index, Q::value_type value) noexcept {
    this->containers[index].push_random(value);
  }

  [[nodiscard]] const Q::value_type &top() const noexcept {
    return this->containers[this->active_index].top();
  }

  [[nodiscard]] const Element<typename Q::value_type> &top_element() const noexcept {
    return this->containers[this->active_index].top_element();
  };

  void pop() noexcept {
    this->containers[this->active_index].pop();
  }
};