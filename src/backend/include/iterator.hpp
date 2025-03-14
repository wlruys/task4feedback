#pragma once
#include "queues.hpp"
#include "settings.hpp"
#include <vector>

template <typename T> class ActiveIterator {
protected:
  std::vector<T> containers;
  std::vector<bool> active;
  std::size_t active_index;
  std::size_t num_active;

public:
  ActiveIterator() = default;
  ActiveIterator(std::size_t num_containers)
      : containers(num_containers), active(num_containers, true), active_index(0),
        num_active(num_containers) {
  }

  bool is_active(std::size_t index) {
    return active.at(index);
  }
  bool has_active() {
    return num_active > 0;
  }

  T &get_active() {
    return containers.at(active_index);
  }
  T &operator[](std::size_t index) {
    return containers.at(index);
  }
  T &at(std::size_t index) {
    return containers.at(index);
  }
  const T &operator[](std::size_t index) const {
    return containers.at(index);
  }
  const T &at(std::size_t index) const {
    return containers.at(index);
  }

  [[nodiscard]] std::size_t size() const {
    return containers.size();
  }

  [[nodiscard]] std::size_t active_size() const {
    return num_active;
  }
  [[nodiscard]] std::size_t total_size() const {
    std::size_t tsize = 0;
    for (auto &c : containers) {
      tsize += c.size();
    }
    return tsize;
  }

  [[nodiscard]] std::size_t total_active_size() const {
    std::size_t tsize = 0;
    for (std::size_t i = 0; i < containers.size(); i++) {
      if (active[i]) {
        tsize += containers.at(i).size();
      }
    }
    return tsize;
  }

  void set_active_queue(std::size_t index) {
    active_index = index;
  }
  std::size_t get_active_index() {
    return active_index;
  }

  void deactivate(std::size_t index) {
    active.at(index) = false;
    num_active--;
  }

  void deactivate() {
    active.at(active_index) = false;
    num_active--;
  }

  void activate(std::size_t index) {
    active.at(index) = true;
    num_active++;
  }

  void activate() {
    active.at(active_index) = true;
    num_active++;
  }

  void next() {
    active_index = (active_index + 1) % containers.size();
  }

  void current_or_next_active() {
    if (!active.at(active_index)) {
      next_active();
    }
  }

  void prev() {
    if (active_index == 0) {
      active_index = containers.size() - 1;
    } else {
      active_index--;
    }
  }

  void next_active() {
    next();
    while (!active.at(active_index)) {
      next();
    }
  }

  void prev_active() {
    prev();
    while (!active.at(active_index)) {
      prev();
    }
  }

  void reset() {
    for (auto &&a : active) {
      a = true;
    }
    num_active = containers.size();
  }
};

template <PriorityQueueConcept Q> class ActiveQueueIterator : public ActiveIterator<Q> {

public:
  void push(Q::value_type value) {
    this->containers.at(this->active_index).push(value);
  }

  void push(Q::value_type value, priority_t priority) {
    this->containers.at(this->active_index).push(value, priority);
  }

  void push_at(std::size_t index, Q::value_type value) {
    this->containers.at(index).push(value);
  }

  void push_priority_at(std::size_t index, Q::value_type value, priority_t priority) {
    this->activate(index);
    this->containers.at(index).push(value, priority);
  }

  void push_random(Q::value_type value) {
    this->containers.at(this->active_index).push_random(value);
  }

  void push_random_at(std::size_t index, Q::value_type value) {
    this->containers.at(index).push_random(value);
  }

  [[nodiscard]] const Q::value_type &top() const {
    return this->containers.at(this->active_index).top();
  }

  [[nodiscard]] const Element<typename Q::value_type> &top_element() const {
    return this->containers.at(this->active_index).top_element();
  };

  void pop() {
    this->containers.at(this->active_index).pop();
  }
};