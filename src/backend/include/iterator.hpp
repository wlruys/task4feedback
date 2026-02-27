#pragma once
#include "queues.hpp"
#include "settings.hpp"
#include <vector>

template <typename T> class ActiveIterator {
protected:
  std::vector<T> containers;
  std::vector<uint8_t> active;
  std::size_t active_index;
  std::size_t num_active;
  std::size_t element_count{0};
  std::size_t active_element_count{0};

public:
  ActiveIterator() = default;
  ActiveIterator(std::size_t num_containers)
      : containers(num_containers), active(num_containers, 1), active_index(0),
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
    return element_count;
  }

  [[nodiscard]] std::size_t total_active_size() const noexcept {
    return active_element_count;
  }

  void set_active_queue(int index) noexcept {
    active_index = index;
  }
  int get_active_index() noexcept {
    return active_index;
  }

  void deactivate(int index) noexcept {
    if (active[index]) {
      active[index] = 0;
      num_active--;
      active_element_count -= containers[index].size();
    }
  }

  void deactivate() {
    deactivate(static_cast<int>(active_index));
  }

  void activate(std::size_t index) noexcept {
    if (!active[index]) {
      active[index] = 1;
      num_active++;
      active_element_count += containers[index].size();
    }
  }

  void activate() noexcept {
    activate(active_index);
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
    active_element_count = element_count;
  }
};

template <PriorityQueueConcept Q> class ActiveQueueIterator : public ActiveIterator<Q> {

public:
  void push(Q::value_type value) noexcept {
    this->containers[this->active_index].push(value);
    this->element_count++;
    if (this->active[this->active_index]) {
      this->active_element_count++;
    }
  }

  void push(Q::value_type value, priority_t priority) noexcept {
    this->containers[this->active_index].push(value, priority);
    this->element_count++;
    if (this->active[this->active_index]) {
      this->active_element_count++;
    }
  }

  void push_at(std::size_t index, Q::value_type value) noexcept {
    this->containers[index].push(value);
    this->element_count++;
    if (this->active[index]) {
      this->active_element_count++;
    }
  }

  void push_priority_at(std::size_t index, Q::value_type value, priority_t priority) noexcept {
    this->activate(index);
    this->containers[index].push(value, priority);
    this->element_count++;
    if (this->active[index]) {
      this->active_element_count++;
    }
  }

  void push_random(Q::value_type value) noexcept {
    this->containers[this->active_index].push_random(value);
    this->element_count++;
    if (this->active[this->active_index]) {
      this->active_element_count++;
    }
  }

  void push_random_at(std::size_t index, Q::value_type value) noexcept {
    this->containers[index].push_random(value);
    this->element_count++;
    if (this->active[index]) {
      this->active_element_count++;
    }
  }

  [[nodiscard]] const Q::value_type &top() const noexcept {
    return this->containers[this->active_index].top();
  }

  [[nodiscard]] const Element<typename Q::value_type> &top_element() const noexcept {
    return this->containers[this->active_index].top_element();
  };

  void pop() noexcept {
    assert(!this->containers[this->active_index].empty());
    this->containers[this->active_index].pop();
    this->element_count--;
    if (this->active[this->active_index]) {
      this->active_element_count--;
    }
  }
};
