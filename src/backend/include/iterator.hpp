#pragma once
#include "queues.hpp"
#include "settings.hpp"
#include <bit>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

template <typename T> class ActiveIterator {
protected:
  std::vector<T> containers;
  uint64_t tasks_mask{0};  // bit i: containers[i] is non-empty
  uint64_t viable_mask{0}; // bit i: device i has not been deactivated this phase
  uint32_t n_devices{0};
  uint32_t active_index{0};
  std::size_t element_count{0};

  [[nodiscard]] uint64_t drainable_mask() const noexcept { return tasks_mask & viable_mask; }

  [[nodiscard]] uint64_t all_viable_mask() const noexcept {
    // Branch prevents UB when n_devices exactly == 64
    return (n_devices == 64) ? ~uint64_t{0} : (1ULL << n_devices) - 1;
  }

public:
  ActiveIterator() = default;
  ActiveIterator(std::size_t num_containers)
      : containers(num_containers), n_devices(static_cast<uint32_t>(num_containers)) {
    T4F_INVARIANT(num_containers <= 64);
    viable_mask = all_viable_mask();
  }

  [[nodiscard]] bool has_active() const noexcept { return drainable_mask() != 0; }
  [[nodiscard]] bool has_active_elements() const noexcept { return tasks_mask != 0; }
  [[nodiscard]] bool is_active(std::size_t index) const noexcept {
    return (viable_mask >> index) & 1;
  }

  T &get_active() noexcept { return containers[active_index]; }
  const T &get_active() const noexcept { return containers[active_index]; }
  T &operator[](std::size_t index) noexcept { return containers[index]; }
  const T &operator[](std::size_t index) const noexcept { return containers[index]; }
  T &at(std::size_t index) noexcept { return containers[index]; }
  const T &at(std::size_t index) const noexcept { return containers[index]; }

  [[nodiscard]] std::size_t size() const noexcept { return n_devices; }
  [[nodiscard]] std::size_t total_size() const noexcept { return element_count; }
  [[nodiscard]] std::size_t active_size() const noexcept {
    return static_cast<std::size_t>(std::popcount(viable_mask));
  }

  [[nodiscard]] std::size_t total_active_size() const noexcept {
    std::size_t count = 0;
    for (uint64_t d = drainable_mask(); d; d &= d - 1) {
      count += containers[std::countr_zero(d)].size();
    }
    return count;
  }

  [[nodiscard]] std::size_t get_active_index() const noexcept { return active_index; }
  void set_active_queue(uint32_t index) noexcept { active_index = index; }

  void deactivate(uint32_t index) noexcept { viable_mask &= ~(1ULL << index); }
  void deactivate() noexcept { viable_mask &= ~(1ULL << active_index); }
  void deactivate_mask(uint64_t mask) noexcept { viable_mask &= ~mask; }
  void activate(std::size_t index) noexcept { viable_mask |= (1ULL << index); }
  void activate() noexcept { viable_mask |= (1ULL << active_index); }

  void reset() noexcept {
    viable_mask = all_viable_mask();
  }

  void next() noexcept {
    active_index = (active_index + 1 == n_devices) ? 0 : active_index + 1;
  }

  void prev() noexcept {
    active_index = (active_index == 0) ? n_devices - 1 : active_index - 1;
  }

  void seek_drainable() noexcept {
    if (uint64_t d = drainable_mask()) {
      active_index = std::countr_zero(d);
    }
  }

  // Advance active_index to the next drainable device in round-robin order.
  void next_drainable() noexcept {
    uint64_t d = drainable_mask();
    if (!d) return;
    uint32_t next = (active_index + 1 == n_devices) ? 0 : active_index + 1;
    uint64_t hi = d >> next;
    active_index = hi ? next + std::countr_zero(hi) : std::countr_zero(d);
  }

  // Advance to the next viable device (ignores tasks_mask). Used outside drain loops.
  void next_active() noexcept {
    uint64_t v = viable_mask;
    if (!v) return;
    uint32_t next = (active_index + 1 == n_devices) ? 0 : active_index + 1;
    uint64_t hi = v >> next;
    active_index = hi ? next + std::countr_zero(hi) : std::countr_zero(v);
  }

  // Moves backward to the previous viable device.
  void prev_active() noexcept {
    uint64_t v = viable_mask;
    if (!v) return;
    uint64_t lo = v & ((1ULL << active_index) - 1);
    active_index = lo ? 63 - std::countl_zero(lo) : 63 - std::countl_zero(v);
  }

  // Position at the current device if viable, else advance to the next viable device.
  void current_or_next_active() noexcept {
    if (!((viable_mask >> active_index) & 1)) next_active();
  }
};

template <PriorityQueueConcept Q> class ActiveQueueIterator : public ActiveIterator<Q> {
public:
  void push(Q::value_type value) noexcept {
    this->containers[this->active_index].push(std::move(value));
    this->element_count++;
    this->tasks_mask |= (1ULL << this->active_index);
  }

  void push(Q::value_type value, priority_t priority) noexcept {
    this->containers[this->active_index].push(std::move(value), priority);
    this->element_count++;
    this->tasks_mask |= (1ULL << this->active_index);
  }

  void push_at(std::size_t index, Q::value_type value) noexcept {
    this->containers[index].push(std::move(value));
    this->element_count++;
    this->tasks_mask |= (1ULL << index);
  }

  // Pushing new work to a device re-enables it as viable
  void push_priority_at(std::size_t index, Q::value_type value, priority_t priority) noexcept {
    this->viable_mask |= (1ULL << index);
    this->containers[index].push(std::move(value), priority);
    this->element_count++;
    this->tasks_mask |= (1ULL << index);
  }

  void push_random(Q::value_type value) noexcept {
    this->containers[this->active_index].push_random(std::move(value));
    this->element_count++;
    this->tasks_mask |= (1ULL << this->active_index);
  }

  void push_random_at(std::size_t index, Q::value_type value) noexcept {
    this->containers[index].push_random(std::move(value));
    this->element_count++;
    this->tasks_mask |= (1ULL << index);
  }

  [[nodiscard]] Q::value_type top() const noexcept {
    return this->containers[this->active_index].top();
  }

  [[nodiscard]] const typename Q::element_type &top_element() const noexcept {
    return this->containers[this->active_index].top_element();
  }

  void pop() noexcept {
    auto &container = this->containers[this->active_index];
    T4F_INVARIANT(!container.empty());
    container.pop();
    this->element_count--;
    if (container.empty()) {
      this->tasks_mask &= ~(1ULL << this->active_index);
    }
  }
};
