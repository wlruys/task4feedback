#pragma once
#include "queues.hpp"
#include "settings.hpp"
#include <bit>
#include <cassert>
#include <cstdint>
#include <vector>

// ActiveIterator manages a collection of per-device containers with round-robin
// cycling and per-phase device viability tracking. Two bitmasks replace the
// old byte-array active flags:
//
//   tasks_mask  — bit i is set when containers[i] is non-empty
//   viable_mask — bit i is set when device i has not been deactivated this phase
//
// A device is "drainable" when both bits are set. Using bit operations keeps
// all state transitions to single instructions and eliminates array scans.
//
// Supports up to 64 devices (one uint64_t per mask).

template <typename T> class ActiveIterator {
protected:
  std::vector<T> containers;
  uint64_t tasks_mask{0};  // bit i: containers[i] is non-empty
  uint64_t viable_mask{0}; // bit i: device i has not been deactivated this phase
  uint32_t n_devices{0};
  uint32_t active_index{0};
  std::size_t element_count{0};

  [[nodiscard]] uint64_t drainable_mask() const noexcept { return tasks_mask & viable_mask; }

public:
  ActiveIterator() = default;
  ActiveIterator(std::size_t num_containers)
      : containers(num_containers), n_devices(static_cast<uint32_t>(num_containers)) {
    assert(num_containers <= 64);
    viable_mask = num_containers ? (1ULL << num_containers) - 1 : 0;
  }

  // --- Observation ---

  // True when at least one device has both tasks and viability.
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
    return std::popcount(viable_mask);
  }

  // Sums elements across currently drainable queues. Iterates set bits — fine
  // for debug logging, not intended for the hot path.
  [[nodiscard]] std::size_t total_active_size() const noexcept {
    std::size_t count = 0;
    for (uint64_t d = drainable_mask(); d; d &= d - 1) {
      count += containers[std::countr_zero(d)].size();
    }
    return count;
  }

  [[nodiscard]] std::size_t get_active_index() const noexcept { return active_index; }
  void set_active_queue(uint32_t index) noexcept { active_index = index; }

  // --- Viability control ---

  void deactivate(uint32_t index) noexcept { viable_mask &= ~(1ULL << index); }
  void deactivate() noexcept { deactivate(active_index); }
  void activate(std::size_t index) noexcept { viable_mask |= (1ULL << index); }
  void activate() noexcept { activate(active_index); }

  // Mark all devices viable for a new phase.
  void reset() noexcept {
    viable_mask = n_devices ? (1ULL << n_devices) - 1 : 0;
  }

  // --- Cursor movement ---

  void next() noexcept {
    if (++active_index >= n_devices) active_index = 0;
  }

  void prev() noexcept {
    active_index = (active_index == 0) ? n_devices - 1 : active_index - 1;
  }

  // Position active_index at the lowest-indexed drainable device to seed a drain loop.
  void seek_drainable() noexcept {
    uint64_t d = drainable_mask();
    if (d) active_index = std::countr_zero(d);
  }

  // Advance active_index to the next drainable device in round-robin order.
  // When has_active() is true, the result always lands on a drainable device.
  void next_drainable() noexcept {
    uint64_t d = drainable_mask();
    if (!d) return;
    uint32_t next = active_index + 1;
    if (next >= n_devices) next = 0;
    uint64_t hi = d >> next;
    active_index = hi ? next + std::countr_zero(hi) : std::countr_zero(d);
  }

  // Advance to the next viable device (ignores tasks_mask). Used outside drain loops.
  void next_active() noexcept {
    uint64_t v = viable_mask;
    if (!v) return;
    uint32_t next = active_index + 1;
    if (next >= n_devices) next = 0;
    uint64_t hi = v >> next;
    active_index = hi ? next + std::countr_zero(hi) : std::countr_zero(v);
  }

  void prev_active() noexcept {
    for (uint32_t i = 0; i < n_devices; ++i) {
      prev();
      if ((viable_mask >> active_index) & 1) return;
    }
  }

  // Position at the current device if viable, else advance to the next viable device.
  void current_or_next_active() noexcept {
    if (!((viable_mask >> active_index) & 1)) next_active();
  }
};

template <PriorityQueueConcept Q> class ActiveQueueIterator : public ActiveIterator<Q> {
public:
  void push(Q::value_type value) noexcept {
    this->containers[this->active_index].push(value);
    this->element_count++;
    this->tasks_mask |= (1ULL << this->active_index);
  }

  void push(Q::value_type value, priority_t priority) noexcept {
    this->containers[this->active_index].push(value, priority);
    this->element_count++;
    this->tasks_mask |= (1ULL << this->active_index);
  }

  void push_at(std::size_t index, Q::value_type value) noexcept {
    this->containers[index].push(value);
    this->element_count++;
    this->tasks_mask |= (1ULL << index);
  }

  // Pushing new work to a device re-enables it as viable: a device that was
  // deactivated due to resource pressure may succeed on a task with different
  // requirements pushed mid-phase.
  void push_priority_at(std::size_t index, Q::value_type value, priority_t priority) noexcept {
    this->viable_mask |= (1ULL << index);
    this->containers[index].push(value, priority);
    this->element_count++;
    this->tasks_mask |= (1ULL << index);
  }

  void push_random(Q::value_type value) noexcept {
    this->containers[this->active_index].push_random(value);
    this->element_count++;
    this->tasks_mask |= (1ULL << this->active_index);
  }

  void push_random_at(std::size_t index, Q::value_type value) noexcept {
    this->containers[index].push_random(value);
    this->element_count++;
    this->tasks_mask |= (1ULL << index);
  }

  [[nodiscard]] const Q::value_type &top() const noexcept {
    return this->containers[this->active_index].top();
  }

  [[nodiscard]] const Element<typename Q::value_type> &top_element() const noexcept {
    return this->containers[this->active_index].top_element();
  }

  void pop() noexcept {
    assert(!this->containers[this->active_index].empty());
    this->containers[this->active_index].pop();
    this->element_count--;
    if (this->containers[this->active_index].empty()) {
      this->tasks_mask &= ~(1ULL << this->active_index);
    }
  }
};
