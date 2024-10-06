#pragma once
#include <vector>

template <typename T> class ActiveIterator {
private:
  std::vector<T> containers;
  std::vector<bool> active;
  std::size_t active_index;
  std::size_t num_active;

public:
  ActiveIterator() = default;
  ActiveIterator(std::size_t num_containers) : containers(num_containers) {
    active.resize(num_containers, true);
    active_index = 0;
    num_active = num_containers;
  }

  bool is_active(std::size_t index) { return active[index]; }
  bool has_active() { return num_active > 0; }

  T &get_active() { return containers[active_index]; }
  T &operator[](std::size_t index) { return containers[index]; }
  const T &operator[](std::size_t index) const { return containers[index]; }

  T &at(std::size_t index) { return containers[index]; }

  [[nodiscard]] std::size_t size() const { return containers.size(); }
  [[nodiscard]] std::size_t active_size() const { return num_active; }
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
        tsize += containers[i].size();
      }
    }
    return tsize;
  }

  void set_active_queue(std::size_t index) { active_index = index; }
  std::size_t get_active_index() { return active_index; }

  void deactivate(std::size_t index) {
    active[index] = false;
    num_active--;
  }

  void deactivate() {
    active[active_index] = false;
    num_active--;
  }

  void activate(std::size_t index) {
    active[index] = true;
    num_active++;
  }

  void activate() {
    active[active_index] = true;
    num_active++;
  }

  void next() { active_index = (active_index + 1) % containers.size(); }

  void prev() {
    if (active_index == 0) {
      active_index = containers.size() - 1;
    } else {
      active_index--;
    }
  }

  void next_active() {
    next();
    while (!active[active_index]) {
      next();
    }
  }

  void prev_active() {
    prev();
    while (!active[active_index]) {
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