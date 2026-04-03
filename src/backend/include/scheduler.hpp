#pragma once

#include "action.hpp"
#include "breakpoints.hpp"
#include "communication.hpp"
#include "data.hpp"
#include "devices.hpp"
#include "events.hpp"
#include "kahypar_wrapper.hpp"
#include "iterator.hpp"
#include "macros.hpp"
#include "metis_wrapper.hpp"
#include "noise.hpp"
#include "queues.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include "tasks.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stack>
#include <stdexcept>
#include <type_traits>
#include <tracy/Tracy.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#define TIME_TO_MAP 0
#define TIME_TO_RESERVE 0
#define TIME_TO_LAUNCH 0
#define SCHEDULER_TIME_GAP 0
#define INITIAL_TASK_BUFFER_SIZE 10
#define INITIAL_DEVICE_BUFFER_SIZE 10
#define INITIAL_EVENT_BUFFER_SIZE 5000

// using MappableTaskQueue = ContainerQueue<taskid_t, TopKQueueHelper<1>::queue_type>;
using MappableTaskQueue = ContainerQueue<taskid_t, DynamicTopKQueue>;
using TaskQueue = ContainerQueue<taskid_t, std::priority_queue>;
using DeviceQueue = ActiveQueueIterator<TaskQueue>;

struct DataLaunchQueueEntry {
  taskid_t task_id = 0;
  priority_t parent_priority = 0;
  timecount_t remaining_transfer_time = 0;
  uint64_t seq = 0;
  bool transfer_aware = false;
};

struct DataLaunchQueueCompare {
  [[nodiscard]] bool operator()(const DataLaunchQueueEntry &lhs,
                                const DataLaunchQueueEntry &rhs) const {
    if (lhs.transfer_aware != rhs.transfer_aware) {
      return lhs.transfer_aware < rhs.transfer_aware;
    }

    if (!lhs.transfer_aware) {
      return lhs.seq < rhs.seq;
    }

    if (lhs.parent_priority != rhs.parent_priority) {
      return lhs.parent_priority > rhs.parent_priority;
    }
    if (lhs.remaining_transfer_time != rhs.remaining_transfer_time) {
      return lhs.remaining_transfer_time < rhs.remaining_transfer_time;
    }
    if (lhs.task_id != rhs.task_id) {
      return lhs.task_id < rhs.task_id;
    }
    return lhs.seq < rhs.seq;
  }
};

class DataLaunchTaskQueue {
private:
  using Container = std::multiset<DataLaunchQueueEntry, DataLaunchQueueCompare>;
  Container entries;

public:
  using value_type = DataLaunchQueueEntry;
  using element_type = value_type;
  using value_compare = DataLaunchQueueCompare;

  DataLaunchTaskQueue() = default;

  void push(value_type value) {
    entries.insert(std::move(value));
  }

  void push(value_type value, priority_t /*priority*/) {
    entries.insert(std::move(value));
  }

  void push_random(value_type value) {
    entries.insert(std::move(value));
  }

  [[nodiscard]] value_type top() const {
    T4F_INVARIANT(!entries.empty() && "top() called on an empty DataLaunchTaskQueue");
    return *entries.begin();
  }

  [[nodiscard]] const element_type &top_element() const {
    T4F_INVARIANT(!entries.empty() && "top_element() called on an empty DataLaunchTaskQueue");
    return *entries.begin();
  }

  void pop() {
    T4F_INVARIANT(!entries.empty() && "pop() called on an empty DataLaunchTaskQueue");
    entries.erase(entries.begin());
  }

  [[nodiscard]] bool empty() const noexcept {
    return entries.empty();
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return entries.size();
  }
};

using DataLaunchDeviceQueue = ActiveQueueIterator<DataLaunchTaskQueue>;

using TaskIDTimeList = std::pair<TaskIDList, std::vector<timecount_t>>;

class TransitionConditions;
class Scheduler;
class Mapper;

enum class ExecutionState : int8_t {
  NONE = 0,
  RUNNING = 1,
  COMPLETE = 2,
  BREAKPOINT = 3,
  EXTERNAL_MAPPING = 4,
  ERROR = 5,
};
constexpr std::size_t num_execution_states = 6;

enum class EvictionPolicy : int8_t {
  LRU = 0,
  LEAST_USED_MAPPED = 1,
};
constexpr std::size_t num_eviction_policies = 2;

inline std::string to_string(const EvictionPolicy &policy) {
  switch (policy) {
  case EvictionPolicy::LRU:
    return "LRU";
  case EvictionPolicy::LEAST_USED_MAPPED:
    return "LEAST_USED_MAPPED";
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const EvictionPolicy &policy) {
  os << to_string(policy);
  return os;
}

inline std::string to_string(const ExecutionState &state) {
  switch (state) {
  case ExecutionState::NONE:
    return "NONE";
    break;
  case ExecutionState::RUNNING:
    return "RUNNING";
    break;
  case ExecutionState::COMPLETE:
    return "COMPLETE";
    break;
  case ExecutionState::BREAKPOINT:
    return "BREAKPOINT";
    break;
  case ExecutionState::EXTERNAL_MAPPING:
    return "EXTERNAL_MAPPING";
    break;
  case ExecutionState::ERROR:
    return "ERROR";
    break;
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const ExecutionState &state) {
  os << to_string(state);
  return os;
}

class SchedulerQueues {
protected:
  MappableTaskQueue mappable;
  DeviceQueue reservable;
  DeviceQueue launchable;
  DataLaunchDeviceQueue data_launchable;
  DeviceQueue eviction_launchable;

  // static TaskType id_to_type(taskid_t id, const Tasks &tasks);

  // void id_to_queue(taskid_t id, const TaskStateInfo &state);

public:
  SchedulerQueues() = default;
  uint64_t data_queue_seq = 0;

  SchedulerQueues(Devices &devices)
      : reservable(devices.size()), launchable(devices.size()), data_launchable(devices.size()),
        eviction_launchable(devices.size()) {
  }

  SchedulerQueues(const SchedulerQueues &other) = default;
  SchedulerQueues &operator=(const SchedulerQueues &other) = default;

  void push_mappable(taskid_t id, priority_t p) {
    mappable.push(id, p);
  }

  void push_mappable(const std::span<const taskid_t> ids, const std::span<const priority_t> ps) {
    for (int32_t i = 0; i < ids.size(); i++) {
      push_mappable(ids[i], ps[i]);
    }
  }

  void push_reservable(taskid_t id, priority_t p, devid_t device) {
    reservable.push_priority_at(device, id, p);
    SPDLOG_DEBUG("Pushing reservable compute task {} with priority {} on device {} top {}", id, p,
                 device, reservable[device].top());
  }

  void push_launchable(taskid_t id, priority_t p, devid_t device) {
    SPDLOG_DEBUG("Pushing launchable compute task {} with priority {} on device {}", id, p, device);
    launchable.push_priority_at(device, id, p);
  }

  [[nodiscard]] uint64_t next_data_queue_seq() {
    return data_queue_seq++;
  }

  void push_launchable_data(DataLaunchQueueEntry entry, devid_t device) {
    const auto task_id = entry.task_id;
    const auto parent_priority = entry.parent_priority;
    const auto remaining_transfer_time = entry.remaining_transfer_time;
    const auto seq = entry.seq;
    const auto transfer_aware = entry.transfer_aware;
    data_launchable.push_priority_at(device, std::move(entry), 0);
    SPDLOG_DEBUG(
        "Pushing launchable data task {} with parent priority {} on device {} seq {} "
        "transfer_aware {} remaining_transfer_time {} current_top {}",
        task_id, parent_priority, device, seq, transfer_aware, remaining_transfer_time,
        data_launchable[device].top().task_id);
  }

  void push_launchable_eviction(taskid_t id, priority_t p, devid_t device) {
    eviction_launchable.push_priority_at(device, id, p);
  }

  [[nodiscard]] std::size_t n_mappable() const {
    return mappable.size();
  }
  [[nodiscard]] bool has_mappable() const {
    return !mappable.empty();
  }

  [[nodiscard]] std::size_t n_reservable(devid_t device) const {
    const auto &device_queue = reservable.at(device);
    return device_queue.size();
  }

  [[nodiscard]] bool has_reservable(devid_t device) const {
    const auto &device_queue = reservable.at(device);
    return !device_queue.empty();
  }

  [[nodiscard]] bool has_active_reservable() const {
    return reservable.has_active();
  }

  [[nodiscard]] bool has_reservable() const {
    return reservable.total_size() > 0;
  }

  [[nodiscard]] std::size_t n_launchable(devid_t device) const {
    const auto &device_queue = launchable.at(device);
    return device_queue.size();
  }

  [[nodiscard]] bool has_launchable() const {
    return launchable.total_size() > 0;
  }

  [[nodiscard]] bool has_launchable(devid_t device) const {
    const auto &device_queue = launchable.at(device);
    return !device_queue.empty();
  }

  [[nodiscard]] bool has_active_launchable() const {
    return launchable.has_active();
  }

  [[nodiscard]] std::size_t n_data_launchable(devid_t device) const {
    const auto &device_queue = data_launchable.at(device);
    return device_queue.size();
  }
  [[nodiscard]] bool has_data_launchable(devid_t device) const {
    const auto &device_queue = data_launchable.at(device);
    return !device_queue.empty();
  }

  [[nodiscard]] bool has_active_data_launchable() const {
    return data_launchable.has_active();
  }

  [[nodiscard]] std::size_t n_eviction_launchable(devid_t device) const {
    const auto &device_queue = eviction_launchable.at(device);
    return device_queue.size();
  }
  [[nodiscard]] bool has_eviction_launchable(devid_t device) const {
    const auto &device_queue = eviction_launchable.at(device);
    return !device_queue.empty();
  }

  [[nodiscard]] bool has_active_eviction_launchable() const {
    return eviction_launchable.has_active();
  }

  [[nodiscard]] bool has_eviction_launchable() const {
    return eviction_launchable.total_size() > 0;
  }

  friend class Scheduler;
};

class TaskCountInfo {
public:
  ankerl::unordered_dense::set<taskid_t> active_tasks;
  using precision_t = int32_t;

  constexpr static precision_t n_per_device_counts =
      7; // active, mapped, reserved, launched, completed, data_reserved, data_launched

  constexpr static precision_t mapped_offset = 0;
  constexpr static precision_t reserved_offset = 1;
  constexpr static precision_t launched_offset = 2;
  constexpr static precision_t completed_offset = 3;
  constexpr static precision_t data_reserved_offset = 4;
  constexpr static precision_t data_launched_offset = 5;
  constexpr static precision_t data_completed_offset = 6;

  TaskCountInfo() = default;

  TaskCountInfo(std::size_t n_devices)
      : n_devices(n_devices), per_device_counts(n_devices * n_per_device_counts) {};

  void count_mapped(taskid_t task_id, devid_t device_id) {
    n_active_tasks += 1;
    n_mapped_tasks += 1;
    per_device_counts[mapped_offset * n_devices + device_id] += 1;
    active_tasks.insert(task_id);
  }
  void count_reserved(taskid_t task_id, devid_t device_id) {
    n_reserved_tasks += 1;
    per_device_counts[reserved_offset * n_devices + device_id] += 1;
  }
  void count_launched(taskid_t task_id, devid_t device_id) {
    n_launched_tasks += 1;
    per_device_counts[launched_offset * n_devices + device_id] += 1;
  }

  void count_completed(taskid_t task_id, devid_t device_id) {
    n_active_tasks -= 1;
    n_mapped_tasks -= 1;
    n_reserved_tasks -= 1;
    n_launched_tasks -= 1;
    n_completed_tasks += 1;
    per_device_counts[mapped_offset * n_devices + device_id] -= 1;
    per_device_counts[reserved_offset * n_devices + device_id] -= 1;
    per_device_counts[launched_offset * n_devices + device_id] -= 1;
    per_device_counts[completed_offset * n_devices + device_id] += 1;
    active_tasks.erase(task_id);
  }

  void count_data_reserved(taskid_t task_id, devid_t device_id) {
    n_reserved_data_tasks += 1;
    per_device_counts[data_reserved_offset * n_devices + device_id] += 1;
  }

  void count_data_launched(taskid_t task_id, devid_t device_id) {
    n_launched_data_tasks += 1;
    per_device_counts[data_launched_offset * n_devices + device_id] += 1;
  }

  void count_data_completed(taskid_t task_id, devid_t device_id) {
    n_reserved_data_tasks -= 1;
    n_launched_data_tasks -= 1;
    n_data_completed_tasks += 1;
    per_device_counts[data_reserved_offset * n_devices + device_id] -= 1;
    per_device_counts[data_launched_offset * n_devices + device_id] -= 1;
    per_device_counts[data_completed_offset * n_devices + device_id] += 1;
  }

  auto get_active_task_list() const {
    return TaskIDList(active_tasks.begin(), active_tasks.end());
  }

  auto get_active_tasks() const {
    return active_tasks;
  }

  [[nodiscard]] auto n_active() const {
    return n_active_tasks;
  }
  [[nodiscard]] auto n_mapped() const {
    return n_mapped_tasks;
  }
  [[nodiscard]] auto n_reserved() const {
    return n_reserved_tasks;
  }
  [[nodiscard]] auto n_launched() const {
    return n_launched_tasks;
  }
  [[nodiscard]] auto n_completed() const {
    return n_completed_tasks;
  }

  [[nodiscard]] auto n_data_reserved() const {
    return n_reserved_data_tasks;
  }

  [[nodiscard]] auto n_data_launched() const {
    return n_launched_data_tasks;
  }

  [[nodiscard]] auto n_data_completed() const {
    return n_data_completed_tasks;
  }

  [[nodiscard]] auto n_unlaunched_reserved() const {
    return n_reserved_tasks - n_launched_tasks;
  }

  [[nodiscard]] auto n_unlaunched_mapped() const {
    return n_mapped_tasks - n_launched_tasks;
  }

  [[nodiscard]] auto n_unreserved_mapped() const {
    return n_mapped_tasks - n_reserved_tasks;
  }

  [[nodiscard]] auto n_active(devid_t device_id) const {
    return per_device_counts[mapped_offset * n_devices + device_id];
  }

  [[nodiscard]] auto n_mapped(devid_t device_id) const {
    return per_device_counts[mapped_offset * n_devices + device_id];
  }

  [[nodiscard]] auto n_reserved(devid_t device_id) const {
    return per_device_counts[reserved_offset * n_devices + device_id];
  }

  [[nodiscard]] auto n_unreserved_mapped(devid_t device_id) const {
    return n_mapped(device_id) - n_reserved(device_id);
  }

  [[nodiscard]] auto n_launched(devid_t device_id) const {
    return per_device_counts[launched_offset * n_devices + device_id];
  }

  [[nodiscard]] auto n_completed(devid_t device_id) const {
    return per_device_counts[completed_offset * n_devices + device_id];
  }

  [[nodiscard]] auto n_data_completed(devid_t device_id) const {
    return per_device_counts[data_completed_offset * n_devices + device_id];
  }

protected:
  precision_t n_devices{};
  precision_t n_active_tasks{};
  precision_t n_mapped_tasks{};
  precision_t n_reserved_tasks{};
  precision_t n_launched_tasks{};
  precision_t n_reserved_data_tasks{};
  precision_t n_launched_data_tasks{};
  precision_t n_completed_tasks{};
  precision_t n_data_completed_tasks{};
  std::vector<precision_t> per_device_counts{};
};

class TaskDevicePhaseInfo {
public:
  using TaskSet = ankerl::unordered_dense::set<taskid_t>;

  TaskDevicePhaseInfo() = default;

  explicit TaskDevicePhaseInfo(std::size_t n_devices)
      : mapped_tasks(n_devices), reserved_tasks(n_devices) {
  }

  void reserve(std::size_t expected_tasks_per_device) {
    if (expected_tasks_per_device == 0) {
      return;
    }

    for (auto &tasks : mapped_tasks) {
      tasks.reserve(expected_tasks_per_device);
    }
    for (auto &tasks : reserved_tasks) {
      tasks.reserve(expected_tasks_per_device);
    }
  }

  void clear() {
    for (auto &tasks : mapped_tasks) {
      tasks.clear();
    }
    for (auto &tasks : reserved_tasks) {
      tasks.clear();
    }
  }

  [[nodiscard]] std::size_t size() const {
    return mapped_tasks.size();
  }

  void on_mapped(taskid_t task_id, devid_t device_id) {
    reserved_at(device_id).erase(task_id);
    mapped_at(device_id).insert(task_id);
  }

  void on_reserved(taskid_t task_id, devid_t device_id) {
    mapped_at(device_id).erase(task_id);
    reserved_at(device_id).insert(task_id);
  }

  void on_completed(taskid_t task_id, devid_t device_id) {
    mapped_at(device_id).erase(task_id);
    reserved_at(device_id).erase(task_id);
  }

  [[nodiscard]] bool has_mapped(taskid_t task_id, devid_t device_id) const {
    const auto &tasks = mapped_at(device_id);
    return tasks.find(task_id) != tasks.end();
  }

  [[nodiscard]] bool has_reserved(taskid_t task_id, devid_t device_id) const {
    const auto &tasks = reserved_at(device_id);
    return tasks.find(task_id) != tasks.end();
  }

  [[nodiscard]] const TaskSet &get_mapped_tasks(devid_t device_id) const {
    return mapped_at(device_id);
  }

  [[nodiscard]] const TaskSet &get_reserved_tasks(devid_t device_id) const {
    return reserved_at(device_id);
  }

private:
  [[nodiscard]] TaskSet &mapped_at(devid_t device_id) {
    T4F_INVARIANT(device_id >= 0);
    const auto idx = static_cast<std::size_t>(device_id);
    T4F_INVARIANT(idx < mapped_tasks.size());
    return mapped_tasks[idx];
  }

  [[nodiscard]] const TaskSet &mapped_at(devid_t device_id) const {
    T4F_INVARIANT(device_id >= 0);
    const auto idx = static_cast<std::size_t>(device_id);
    T4F_INVARIANT(idx < mapped_tasks.size());
    return mapped_tasks[idx];
  }

  [[nodiscard]] TaskSet &reserved_at(devid_t device_id) {
    T4F_INVARIANT(device_id >= 0);
    const auto idx = static_cast<std::size_t>(device_id);
    T4F_INVARIANT(idx < reserved_tasks.size());
    return reserved_tasks[idx];
  }

  [[nodiscard]] const TaskSet &reserved_at(devid_t device_id) const {
    T4F_INVARIANT(device_id >= 0);
    const auto idx = static_cast<std::size_t>(device_id);
    T4F_INVARIANT(idx < reserved_tasks.size());
    return reserved_tasks[idx];
  }

  std::vector<TaskSet> mapped_tasks;
  std::vector<TaskSet> reserved_tasks;
};

class TaskDataUsageInfo {
public:
  TaskDataUsageInfo() = default;

  explicit TaskDataUsageInfo(std::size_t n_data)
      : mapped_usage(n_data, 0), reserved_usage(n_data, 0) {
  }

  void clear() {
    std::fill(mapped_usage.begin(), mapped_usage.end(), int32_t{0});
    std::fill(reserved_usage.begin(), reserved_usage.end(), int32_t{0});
  }

  [[nodiscard]] std::size_t size() const {
    return mapped_usage.size();
  }

  void on_mapped(std::span<const dataid_t> data_ids) {
    for (const auto data_id : data_ids) {
      auto &mapped = mapped_at(data_id);
      auto &reserved = reserved_at(data_id);
      reserved = std::max(reserved, int32_t{0});
      mapped += 1;
    }
  }

  void on_reserved(std::span<const dataid_t> data_ids) {
    for (const auto data_id : data_ids) {
      auto &mapped = mapped_at(data_id);
      auto &reserved = reserved_at(data_id);
      if (mapped > 0) {
        mapped -= 1;
      }
      reserved += 1;
    }
  }

  void on_completed(std::span<const dataid_t> data_ids) {
    for (const auto data_id : data_ids) {
      auto &mapped = mapped_at(data_id);
      auto &reserved = reserved_at(data_id);
      if (reserved > 0) {
        reserved -= 1;
      } else if (mapped > 0) {
        mapped -= 1;
      }
    }
  }

  [[nodiscard]] int32_t get_mapped_usage(dataid_t data_id) const {
    return mapped_at(data_id);
  }

  [[nodiscard]] int32_t get_reserved_usage(dataid_t data_id) const {
    return reserved_at(data_id);
  }

private:
  [[nodiscard]] int32_t &mapped_at(dataid_t data_id) {
    T4F_INVARIANT(data_id >= 0);
    const auto idx = static_cast<std::size_t>(data_id);
    T4F_INVARIANT(idx < mapped_usage.size());
    return mapped_usage[idx];
  }

  [[nodiscard]] const int32_t &mapped_at(dataid_t data_id) const {
    T4F_INVARIANT(data_id >= 0);
    const auto idx = static_cast<std::size_t>(data_id);
    T4F_INVARIANT(idx < mapped_usage.size());
    return mapped_usage[idx];
  }

  [[nodiscard]] int32_t &reserved_at(dataid_t data_id) {
    T4F_INVARIANT(data_id >= 0);
    const auto idx = static_cast<std::size_t>(data_id);
    T4F_INVARIANT(idx < reserved_usage.size());
    return reserved_usage[idx];
  }

  [[nodiscard]] const int32_t &reserved_at(dataid_t data_id) const {
    T4F_INVARIANT(data_id >= 0);
    const auto idx = static_cast<std::size_t>(data_id);
    T4F_INVARIANT(idx < reserved_usage.size());
    return reserved_usage[idx];
  }

  std::vector<int32_t> mapped_usage;
  std::vector<int32_t> reserved_usage;
};

class TaskDataDeviceUsageInfo {
public:
  TaskDataDeviceUsageInfo() = default;

  explicit TaskDataDeviceUsageInfo(std::size_t n_data, std::size_t n_devices)
      : n_data(n_data), n_devices(n_devices), mapped_usage(n_data * n_devices, 0) {
  }

  void clear() {
    std::fill(mapped_usage.begin(), mapped_usage.end(), int32_t{0});
  }

  [[nodiscard]] std::size_t data_size() const {
    return n_data;
  }

  [[nodiscard]] std::size_t device_size() const {
    return n_devices;
  }

  void on_mapped(std::span<const dataid_t> data_ids, devid_t device_id) {
    for (const auto data_id : data_ids) {
      mapped_at(data_id, device_id) += 1;
    }
  }

  void on_reserved(std::span<const dataid_t> data_ids, devid_t device_id) {
    for (const auto data_id : data_ids) {
      auto &mapped = mapped_at(data_id, device_id);
      if (mapped > 0) {
        mapped -= 1;
      }
    }
  }

  void on_completed(std::span<const dataid_t> data_ids, devid_t device_id, bool was_mapped) {
    if (!was_mapped) {
      return;
    }
    on_reserved(data_ids, device_id);
  }

  [[nodiscard]] int32_t get_mapped_usage(dataid_t data_id, devid_t device_id) const {
    return mapped_at(data_id, device_id);
  }

private:
  [[nodiscard]] std::size_t offset(dataid_t data_id, devid_t device_id) const {
    T4F_INVARIANT(data_id >= 0);
    T4F_INVARIANT(device_id >= 0);
    const auto data_idx = static_cast<std::size_t>(data_id);
    const auto device_idx = static_cast<std::size_t>(device_id);
    T4F_INVARIANT(data_idx < n_data);
    T4F_INVARIANT(device_idx < n_devices);
    return device_idx * n_data + data_idx;
  }

  [[nodiscard]] int32_t &mapped_at(dataid_t data_id, devid_t device_id) {
    return mapped_usage[offset(data_id, device_id)];
  }

  [[nodiscard]] const int32_t &mapped_at(dataid_t data_id, devid_t device_id) const {
    return mapped_usage[offset(data_id, device_id)];
  }

  std::size_t n_data = 0;
  std::size_t n_devices = 0;
  std::vector<int32_t> mapped_usage;
};

class TaskCostInfo {
public:
  TaskCostInfo() = default;

  TaskCostInfo(std::size_t n_devices) : n_devices(n_devices), per_device_costs(n_devices * 5) {};
  void count_mapped(devid_t device_id, timecount_t time) {
    per_device_costs[device_id] += time;
  }
  void count_reserved(devid_t device_id, timecount_t time) {
    per_device_costs[n_devices + device_id] += time;
  }
  void count_launched(devid_t device_id, timecount_t time) {
    per_device_costs[2 * n_devices + device_id] += time;
  }

  void count_completed(devid_t device_id, timecount_t time) {
    per_device_costs[device_id] -= time;
    per_device_costs[n_devices + device_id] -= time;
    per_device_costs[2 * n_devices + device_id] -= time;
    per_device_costs[3 * n_devices + device_id] += time;
  }

  void count_data_completed(devid_t device_id, timecount_t time) {
    per_device_costs[4 * n_devices + device_id] += time;
  }

  [[nodiscard]] timecount_t get_mapped_time(devid_t device_id) const {
    return per_device_costs[device_id];
  }
  [[nodiscard]] timecount_t get_reserved_time(devid_t device_id) const {
    return per_device_costs[n_devices + device_id];
  }
  [[nodiscard]] timecount_t get_launched_time(devid_t device_id) const {
    return per_device_costs[2 * n_devices + device_id];
  }

  [[nodiscard]] timecount_t get_completed_time(devid_t device_id) const {
    return per_device_costs[3 * n_devices + device_id];
  }
  [[nodiscard]] timecount_t get_data_completed_time(devid_t device_id) const {
    return per_device_costs[4 * n_devices + device_id];
  }

protected:
  int32_t n_devices{};
  std::vector<timecount_t> per_device_costs;
};

struct ResourceRequest {
  Resources requested{0, 0};
  Resources missing{0, 0};
};

struct SchedulerInput {
  std::reference_wrapper<Graph> graph;
  std::reference_wrapper<StaticTaskInfo> tasks;
  std::reference_wrapper<Data> data;
  std::reference_wrapper<Devices> devices;
  std::reference_wrapper<Topology> topology;
  std::reference_wrapper<TaskNoise> task_noise;
  std::reference_wrapper<TransitionConditions> conditions;
  int32_t top_k_candidates = 0;
  EvictionPolicy eviction_policy = EvictionPolicy::LRU;
  bool transfer_aware_data_launch_order = false;

  SchedulerInput(Graph &graph, StaticTaskInfo &tasks, Data &data, Devices &devices,
                 Topology &topology, TaskNoise &task_noise, TransitionConditions &conditions,
                 int32_t top_k_candidates = 1,
                 EvictionPolicy eviction_policy = EvictionPolicy::LRU,
                 bool transfer_aware_data_launch_order = false)
      : graph(graph), tasks(tasks), data(data), devices(devices), topology(topology),
        task_noise(task_noise), conditions(conditions), top_k_candidates(top_k_candidates),
        eviction_policy(eviction_policy),
        transfer_aware_data_launch_order(transfer_aware_data_launch_order) {
  }

  SchedulerInput(const SchedulerInput &other) = default;

  SchedulerInput &operator=(const SchedulerInput &other) = default;

  // Shallow copy constructor
  SchedulerInput(SchedulerInput &&other) noexcept
      : graph(other.graph), tasks(other.tasks), data(other.data), devices(other.devices),
        topology(other.topology), task_noise(other.task_noise), conditions(other.conditions),
        top_k_candidates(other.top_k_candidates), eviction_policy(other.eviction_policy),
        transfer_aware_data_launch_order(other.transfer_aware_data_launch_order) {
  }

  SchedulerInput &operator=(SchedulerInput &&other) noexcept {
    if (this != &other) {
      graph = other.graph;
      tasks = other.tasks;
      data = other.data;
      devices = other.devices;
      topology = other.topology;
      task_noise = other.task_noise;
      conditions = other.conditions;
      top_k_candidates = other.top_k_candidates;
      eviction_policy = other.eviction_policy;
      transfer_aware_data_launch_order = other.transfer_aware_data_launch_order;
    }
    return *this;
  }
};

class SchedulerState {
protected:
  timecount_t global_time = 0;
  RuntimeTaskInfo task_runtime;
  DeviceManager device_manager;
  CommunicationManager communication_manager;
  DataManager data_manager;
  std::optional<TaskDevicePhaseInfo> task_device_phase_info;
  std::optional<TaskDataUsageInfo> task_data_usage_info;
  std::optional<TaskDataDeviceUsageInfo> task_data_device_usage_info;
  std::reference_wrapper<Graph> graph;
  std::reference_wrapper<StaticTaskInfo> tasks;
  std::reference_wrapper<Data> data;
  std::reference_wrapper<Devices> devices;
  std::reference_wrapper<Topology> topology;
  std::reference_wrapper<TaskNoise> task_noise;

  static constexpr uint8_t DRAIN_FLAG = 0b00000001;
  static constexpr uint8_t RECORD_FLAG = 0b00000010;

  [[nodiscard]] ResourceRequest request_map_resources(taskid_t task_id, devid_t device_id) const {
    const auto &static_graph = get_tasks();
    const auto arch = get_devices().get_type(device_id);
    const Resources &task_resources = static_graph.get_compute_task_resources(task_id, arch);
    mem_t non_local_memory =
        data_manager.non_local_size_mapped(data.get(), static_graph.get_unique(task_id), device_id);
    Resources requested = {task_resources.vcu, task_resources.mem + non_local_memory};
    Resources missing;
    return {.requested = requested, .missing = missing};
  }

  [[nodiscard]] ResourceRequest request_reserve_resources(taskid_t task_id,
                                                          devid_t device_id) const {
    const auto &static_graph = get_tasks();
    const auto arch = get_devices().get_type(device_id);
    const Resources &task_resources = static_graph.get_compute_task_resources(task_id, arch);
    mem_t non_local_memory = data_manager.non_local_size_reserved(
        data.get(), static_graph.get_unique(task_id), device_id);
    Resources requested = {task_resources.vcu, task_resources.mem + non_local_memory};
    auto missing_memory =
        device_manager.overflow_mem<TaskState::RESERVED>(device_id, requested.mem);
    return {.requested = requested, .missing = Resources(0, missing_memory)};
  }

  [[nodiscard]] ResourceRequest request_launch_resources(taskid_t compute_task_id,
                                                         devid_t device_id) const {
    const auto &static_graph = get_tasks();
    const auto arch = get_devices().get_type(device_id);
    const Resources &task_resources =
        static_graph.get_compute_task_resources(compute_task_id, arch);
    SPDLOG_DEBUG("Requesting launch resources for task {} on device {}",
                 static_graph.get_compute_task_name(compute_task_id), device_id);
    SPDLOG_DEBUG("Task resources: VCU: {}, MEM: {}", task_resources.vcu, task_resources.mem);
    Resources requested = {task_resources.vcu, task_resources.mem};
    auto missing_vcu = device_manager.overflow_vcu<TaskState::LAUNCHED>(device_id, requested.vcu);
    return {requested, Resources(missing_vcu, 0)};
  }

  void map_resources(taskid_t task_id, devid_t device_id, const Resources &requested) {
    device_manager.add_resources<TaskState::MAPPED>(device_id, requested, global_time);
  }

  void reserve_resources(taskid_t task_id, devid_t device_id, const Resources &requested) {
    device_manager.add_resources<TaskState::RESERVED>(device_id, requested, global_time);
  }

  void launch_resources(taskid_t task_id, devid_t device_id, const Resources &requested) {
    device_manager.add_resources<TaskState::LAUNCHED>(device_id, requested, global_time);
  }

  void free_task_resources(taskid_t task_id) {
    auto mapped_device_id = task_runtime.get_compute_task_mapped_device(task_id);
    const auto &task_resources = get_task_resources(task_id, mapped_device_id);
    device_manager.remove_resources<TaskState::MAPPED>(mapped_device_id, task_resources,
                                                       global_time);
    device_manager.remove_resources<TaskState::RESERVED>(mapped_device_id, task_resources,
                                                         global_time);
    device_manager.remove_resources<TaskState::LAUNCHED>(mapped_device_id, task_resources,
                                                         global_time);
  }

public:
  TaskCountInfo counts;
  TaskCostInfo costs;
  uint8_t flags = 0;

  SchedulerState(SchedulerInput &input)
      : global_time(0), graph(input.graph), tasks(input.tasks), data(input.data),
        devices(input.devices), topology(input.topology), task_noise(input.task_noise),
        task_runtime(RuntimeTaskInfo(input.tasks)), device_manager(DeviceManager(input.devices)),
        communication_manager(input.topology, input.devices),
        data_manager(input.data, input.devices), counts(input.devices.get().size()),
        costs(input.devices.get().size()) {
  }

  SchedulerState(const SchedulerState &other)
      : global_time(other.global_time), task_runtime(other.task_runtime),
        device_manager(other.device_manager), communication_manager(other.communication_manager),
        data_manager(other.data_manager), task_device_phase_info(other.task_device_phase_info),
        task_data_usage_info(other.task_data_usage_info),
        task_data_device_usage_info(other.task_data_device_usage_info), graph(other.graph),
        tasks(other.tasks), data(other.data), devices(other.devices), topology(other.topology),
        task_noise(other.task_noise), counts(other.counts), costs(other.costs), flags(other.flags) {
    // ZoneScoped;
    // {
    //   ZoneScopedN("copy_task_runtime");
    //   task_runtime = other.task_runtime;
    // }

    // {
    //   ZoneScopedN("copy_device_manager");
    //   device_manager = other.device_manager;
    // }

    // {
    //   ZoneScopedN("copy_communication_manager");
    //   communication_manager =
    //       other.communication_manager;
    // }

    // {
    //   ZoneScopedN("copy_data_manager");
    //   data_manager = other.data_manager;
    // }

    // {
    //   ZoneScopedN("copy_counts");
    //   counts = other.counts;
    // }

    // {
    //   ZoneScopedN("copy_costs");
    //   costs = other.costs;
    // }
  }

  void start_drain() {
    flags |= DRAIN_FLAG;
  }

  void stop_drain() {
    flags &= ~DRAIN_FLAG;
  }

  void start_record() {
    flags |= RECORD_FLAG;
  }

  void stop_record() {
    flags &= ~RECORD_FLAG;
  }

  [[nodiscard]] bool is_recording() const {
    return (flags & RECORD_FLAG) != 0;
  }

  [[nodiscard]] bool is_draining() const {
    return (flags & DRAIN_FLAG) != 0;
  }

  [[nodiscard]] bool not_draining() const {
    return (flags & DRAIN_FLAG) == 0;
  }

  void update_time(timecount_t time) {
    global_time = std::max(global_time, time);
    T4F_INVARIANT(time >= global_time);
    T4F_INVARIANT(global_time >= 0);
  }

  void initialize(bool create_data_tasks = false, bool initialize_data_manager = true) {
    // task_runtime.initialize();
    device_manager.initialize(get_devices());
    if (initialize_data_manager) {
      data_manager.initialize(get_data(), get_devices(), device_manager);
    }
    if (task_device_phase_info.has_value()) {
      task_device_phase_info->clear();
    }
    if (task_data_usage_info.has_value()) {
      task_data_usage_info->clear();
    }
    if (task_data_device_usage_info.has_value()) {
      task_data_device_usage_info->clear();
    }
  }

  void initialize_data_replicate(dataid_t data_id, devid_t device_id) {
    data_manager.initialize_data_replicate(data.get(), devices.get(), device_manager, data_id,
                                           device_id);
  }

  void randomize_durations() {
    get_task_noise().generate_duration(tasks);
  }

  void randomize_priority() {
    // get_task_noise().generate_priority(tasks);
  }

  void initialize_data_manager() {
    data_manager.initialize(get_data(), get_devices(), device_manager);
  }

  [[nodiscard]] bool is_complete() const {
    int32_t n_data_tasks = task_runtime.get_n_data_tasks();
    int32_t n_compute_tasks = task_runtime.get_n_compute_tasks();
    int32_t n_eviction_tasks = task_runtime.get_n_eviction_tasks();

    bool data_complete = counts.n_data_completed() == (n_data_tasks + n_eviction_tasks);
    bool compute_complete = counts.n_completed() == n_compute_tasks;
    return data_complete && compute_complete;
  }

  [[nodiscard]] bool is_drain_complete() const {
    bool no_mapped = counts.n_mapped() == 0;
    bool no_reserved = counts.n_reserved() == 0;
    bool no_launched = counts.n_launched() == 0;
    bool no_data_reserved = counts.n_data_reserved() == 0;
    bool no_data_launched = counts.n_data_launched() == 0;
    bool no_active = counts.n_active() == 0;

    SPDLOG_DEBUG("Drain complete check: Mapped: {}, Reserved: {}, Launched: {}, "
                 "Data Reserved: {}, Data Launched: {}, Active: {}",
                 no_mapped, no_reserved, no_launched, no_data_reserved, no_data_launched,
                 no_active);
    return no_mapped && no_reserved && no_launched && no_data_reserved && no_data_launched &&
           no_active;
  }

  [[nodiscard]] const Resources &get_task_resources(taskid_t compute_task_id,
                                                    devid_t device_id) const {
    DeviceType arch = get_devices().get_type(device_id);
    const Resources &task_resources = get_tasks().get_compute_task_resources(compute_task_id, arch);
    return task_resources;
  }

  [[nodiscard]] const Resources &get_task_resources(taskid_t compute_task_id) const {
    devid_t device_id = task_runtime.get_compute_task_mapped_device(compute_task_id);
    DeviceType arch = get_devices().get_type(device_id);
    const Resources &task_resources = get_tasks().get_compute_task_resources(compute_task_id, arch);
    return task_resources;
  }

  void update_mapped_cost(taskid_t compute_task_id, devid_t device_id) {
    if (task_device_phase_info.has_value()) {
      task_device_phase_info->on_mapped(compute_task_id, device_id);
    }
    if (task_data_usage_info.has_value()) {
      task_data_usage_info->on_mapped(get_tasks().get_unique(compute_task_id));
    }
    if (task_data_device_usage_info.has_value()) {
      task_data_device_usage_info->on_mapped(get_tasks().get_unique(compute_task_id), device_id);
    }
    DeviceType arch = get_devices().get_type(device_id);
    timecount_t time = get_tasks().get_mean_duration(compute_task_id, arch);
    costs.count_mapped(device_id, time);
    counts.count_mapped(compute_task_id, device_id);
  }

  void update_reserved_cost(taskid_t compute_task_id, devid_t device_id) {
    if (task_device_phase_info.has_value()) {
      task_device_phase_info->on_reserved(compute_task_id, device_id);
    }
    if (task_data_usage_info.has_value()) {
      task_data_usage_info->on_reserved(get_tasks().get_unique(compute_task_id));
    }
    if (task_data_device_usage_info.has_value()) {
      task_data_device_usage_info->on_reserved(get_tasks().get_unique(compute_task_id), device_id);
    }
    DeviceType arch = get_devices().get_type(device_id);
    timecount_t time = get_tasks().get_mean_duration(compute_task_id, arch);
    costs.count_reserved(device_id, time);
    counts.count_reserved(compute_task_id, device_id);
  }

  void update_launched_cost(taskid_t compute_task_id, devid_t device_id) {
    DeviceType arch = get_devices().get_type(device_id);
    timecount_t time = get_tasks().get_mean_duration(compute_task_id, arch);
    costs.count_launched(device_id, time);
    counts.count_launched(compute_task_id, device_id);
  }

  void update_completed_cost(taskid_t compute_task_id, devid_t device_id) {
    const bool was_mapped = task_device_phase_info.has_value() &&
                            task_device_phase_info->has_mapped(compute_task_id, device_id);
    if (task_device_phase_info.has_value()) {
      task_device_phase_info->on_completed(compute_task_id, device_id);
    }
    if (task_data_usage_info.has_value()) {
      task_data_usage_info->on_completed(get_tasks().get_unique(compute_task_id));
    }
    if (task_data_device_usage_info.has_value()) {
      task_data_device_usage_info->on_completed(get_tasks().get_unique(compute_task_id), device_id,
                                                was_mapped);
    }
    DeviceType arch = get_devices().get_type(device_id);
    timecount_t time = get_tasks().get_mean_duration(compute_task_id, arch);
    costs.count_completed(device_id, time);
    counts.count_completed(compute_task_id, device_id);
  }

  void update_data_reserved_cost(taskid_t data_task_id, devid_t device_id) {
    counts.count_data_reserved(data_task_id, device_id);
  }

  void update_data_launched_cost(taskid_t data_task_id, devid_t device_id) {
    counts.count_data_launched(data_task_id, device_id);
  }

  void update_data_completed_cost(taskid_t data_task_id, devid_t device_id) {
    counts.count_data_completed(data_task_id, device_id);
  }

  void update_eviction_reserved_cost(taskid_t eviction_task_id, devid_t device_id) {
    // Note(wlr): Eviction tasks are just treated as data tasks for now
    counts.count_data_reserved(eviction_task_id, device_id);
  }

  void update_eviction_launched_cost(taskid_t eviction_task_id, devid_t device_id) {
    // Note(wlr): Eviction tasks are just treated as data tasks for now
    counts.count_data_launched(eviction_task_id, device_id);
  }

  void update_eviction_completed_cost(taskid_t eviction_task_id, devid_t device_id) {
    // Note(wlr): Eviction tasks are just treated as data tasks for now
    counts.count_data_completed(eviction_task_id, device_id);
  }

  // TODO(wlr): Reenable these interfaces when refactor is complete
  //   bool track_resource_guard() const {
  // /* Compilation guard for when resource tracking (memory and vcu usage over time) is disabled */
  // #ifndef SIM_TRACK_RESOURCES
  //     spdlog::warn("SIM_TRACK_RESOURCES not defined. Resource tracking is disabled.");
  //     return true;
  // #else
  //     return false;
  // #endif
  //   }

  //   bool track_location_guard() const {
  // /* Compilation guard for when location tracking (data location over time) is disabled */
  // #ifndef SIM_TRACK_LOCATION
  //     spdlog::warn("SIM_TRACK_LOCATION not defined. Location tracking is disabled.");
  //     return true;
  // #else
  //     return false;
  // #endif
  //   }

  //  [[nodiscard]] vcu_t get_mapped_vcu_at(devid_t device_id, timecount_t time) const;
  //  [[nodiscard]] vcu_t get_reserved_vcu_at(devid_t device_id, timecount_t time) const;
  //  [[nodiscard]] vcu_t get_launched_vcu_at(devid_t device_id, timecount_t time) const;

  // [[nodiscard]] mem_t get_mapped_mem_at(devid_t device_id, timecount_t time) const;
  // [[nodiscard]] mem_t get_reserved_mem_at(devid_t device_id, timecount_t time) const;
  // [[nodiscard]] mem_t get_launched_mem_at(devid_t device_id, timecount_t time) const;

  // [[nodiscard]] ResourceEventArray<vcu_t> get_mapped_vcu_events(devid_t device_id) const;
  // [[nodiscard]] ResourceEventArray<vcu_t> get_reserved_vcu_events(devid_t device_id) const;
  // [[nodiscard]] ResourceEventArray<vcu_t> get_launched_vcu_events(devid_t device_id) const;

  // [[nodiscard]] ResourceEventArray<mem_t> get_mapped_mem_events(devid_t device_id) const;
  // [[nodiscard]] ResourceEventArray<mem_t> get_reserved_mem_events(devid_t device_id) const;
  // [[nodiscard]] ResourceEventArray<mem_t> get_launched_mem_events(devid_t device_id) const;

  // [[nodiscard]] TaskState get_state_at(taskid_t task_id, timecount_t time) const;

  // [[nodiscard]] ValidEventArray get_valid_intervals_mapped(dataid_t data_id,
  //                                                          devid_t device_id) const;
  // [[nodiscard]] ValidEventArray get_valid_intervals_reserved(dataid_t data_id,
  //                                                            devid_t device_id) const;
  // [[nodiscard]] ValidEventArray get_valid_intervals_launched(dataid_t data_id,
  //                                                            devid_t device_id) const;

  // [[nodiscard]] bool check_valid_mapped_at(dataid_t data_id, devid_t device_id,
  //                                          timecount_t query_time) const;
  // [[nodiscard]] bool check_valid_reserved_at(dataid_t data_id, devid_t device_id,
  //                                            timecount_t query_time) const;
  // [[nodiscard]] bool check_valid_launched_at(dataid_t data_id, devid_t device_id,
  //                                            timecount_t query_time) const;

  // [[nodiscard]] check_valid_mapped(dataid_t data_id, devid_t device_id) const;
  // [[nodiscard]] check_valid_reserved(dataid_t data_id, devid_t device_id) const;
  // [[nodiscard]] check_valid_launched(dataid_t data_id, devid_t device_id) const;

  [[nodiscard]] priority_t get_mapping_priority(taskid_t compute_task_id) const {
    return get_task_noise().get_priority(compute_task_id);
  }
  [[nodiscard]] priority_t get_reserving_priority(taskid_t compute_task_id) const {
    return task_runtime.get_compute_task_reserve_priority(compute_task_id);
  }
  [[nodiscard]] priority_t get_launching_priority(taskid_t compute_task_id) const {
    return task_runtime.get_compute_task_launch_priority(compute_task_id);
  }

  [[nodiscard]] timecount_t get_execution_time(taskid_t compute_task_id) const {
    auto mapped_device_id = task_runtime.get_compute_task_mapped_device(compute_task_id);
    auto arch = get_devices().get_type(mapped_device_id);
    return get_task_noise().get(compute_task_id, arch);
  }

  [[nodiscard]] timecount_t get_global_time() const {
    T4F_INVARIANT(global_time >= 0);
    return global_time;
  }

  [[nodiscard]] const StaticTaskInfo &get_tasks() const {
    return tasks.get();
  }

  [[nodiscard]] StaticTaskInfo &get_tasks() {
    return tasks.get();
  }

  [[nodiscard]] const Data &get_data() const {
    return data.get();
  }

  [[nodiscard]] Data &get_data() {
    return data.get();
  }

  [[nodiscard]] const Graph &get_graph() const {
    return graph.get();
  }

  [[nodiscard]] Graph &get_graph() {
    return graph.get();
  }

  [[nodiscard]] const Topology &get_topology() const {
    return topology.get();
  }

  [[nodiscard]] const Devices &get_devices() const {
    return devices.get();
  }

  [[nodiscard]] const TaskNoise &get_task_noise() const {
    return task_noise.get();
  }

  [[nodiscard]] TaskNoise &get_task_noise() {
    return task_noise.get();
  }

  void enable_task_device_phase_info(std::size_t expected_tasks_per_device = 0) {
    task_device_phase_info.emplace(get_devices().size());
    task_device_phase_info->reserve(expected_tasks_per_device);
  }

  void disable_task_device_phase_info() {
    task_device_phase_info.reset();
  }

  [[nodiscard]] bool has_task_device_phase_info() const {
    return task_device_phase_info.has_value();
  }

  [[nodiscard]] TaskDevicePhaseInfo *get_task_device_phase_info() {
    if (!task_device_phase_info.has_value()) {
      return nullptr;
    }
    return &task_device_phase_info.value();
  }

  [[nodiscard]] const TaskDevicePhaseInfo *get_task_device_phase_info() const {
    if (!task_device_phase_info.has_value()) {
      return nullptr;
    }
    return &task_device_phase_info.value();
  }

  void enable_task_data_usage_info() {
    task_data_usage_info.emplace(get_data().size());
  }

  void disable_task_data_usage_info() {
    task_data_usage_info.reset();
  }

  [[nodiscard]] bool has_task_data_usage_info() const {
    return task_data_usage_info.has_value();
  }

  [[nodiscard]] TaskDataUsageInfo *get_task_data_usage_info() {
    if (!task_data_usage_info.has_value()) {
      return nullptr;
    }
    return &task_data_usage_info.value();
  }

  [[nodiscard]] const TaskDataUsageInfo *get_task_data_usage_info() const {
    if (!task_data_usage_info.has_value()) {
      return nullptr;
    }
    return &task_data_usage_info.value();
  }

  void enable_task_data_device_usage_info() {
    task_data_device_usage_info.emplace(get_data().size(), get_devices().size());
  }

  void disable_task_data_device_usage_info() {
    task_data_device_usage_info.reset();
  }

  [[nodiscard]] bool has_task_data_device_usage_info() const {
    return task_data_device_usage_info.has_value();
  }

  [[nodiscard]] TaskDataDeviceUsageInfo *get_task_data_device_usage_info() {
    if (!task_data_device_usage_info.has_value()) {
      return nullptr;
    }
    return &task_data_device_usage_info.value();
  }

  [[nodiscard]] const TaskDataDeviceUsageInfo *get_task_data_device_usage_info() const {
    if (!task_data_device_usage_info.has_value()) {
      return nullptr;
    }
    return &task_data_device_usage_info.value();
  }

  [[nodiscard]] RuntimeTaskInfo &get_task_runtime() {
    return task_runtime;
  }

  [[nodiscard]] const RuntimeTaskInfo &get_task_runtime() const {
    return task_runtime;
  }

  [[nodiscard]] const DeviceManager &get_device_manager() const {
    return device_manager;
  }

  [[nodiscard]] DeviceManager &get_device_manager() {
    return device_manager;
  }

  [[nodiscard]] CommunicationManager &get_communication_manager() {
    return communication_manager;
  }

  [[nodiscard]] const CommunicationManager &get_communication_manager() const {
    return communication_manager;
  }

  [[nodiscard]] const DataManager &get_data_manager() const {
    return data_manager;
  }

  [[nodiscard]] DataManager &get_data_manager() {
    return data_manager;
  }

  friend class Scheduler;
  friend class TransitionConstraints;
};

template <typename T>
concept TransitionConditionConcept = requires(T t, SchedulerState &state, SchedulerQueues &queues) {
  { t.should_map(state, queues) } -> std::convertible_to<bool>;
  { t.update_map(state, queues) } -> std::convertible_to<bool>;
  { t.should_reserve(state, queues) } -> std::convertible_to<bool>;
  { t.should_launch(state, queues) } -> std::convertible_to<bool>;
  { t.should_launch_data(state, queues) } -> std::convertible_to<bool>;
};

class TransitionConditions {
public:
  virtual bool should_map(SchedulerState &state, SchedulerQueues &queues) {
    MONUnusedParameter(state);
    MONUnusedParameter(queues);
    return true;
  }

  virtual bool update_map(SchedulerState &state, SchedulerQueues &queues) {
    MONUnusedParameter(state);
    MONUnusedParameter(queues);
    return true;
  }

  virtual bool should_reserve(SchedulerState &state, SchedulerQueues &queues) {
    MONUnusedParameter(state);
    MONUnusedParameter(queues);
    return true;
  }

  virtual bool should_launch(SchedulerState &state, SchedulerQueues &queues) {
    MONUnusedParameter(state);
    MONUnusedParameter(queues);
    return true;
  }

  virtual bool should_launch_data(SchedulerState &state, SchedulerQueues &queues) {
    MONUnusedParameter(state);
    MONUnusedParameter(queues);
    return true;
  }
};

static_assert(TransitionConditionConcept<TransitionConditions>);

class DefaultTransitionConditions : public TransitionConditions {};

class RangeTransitionConditions : public TransitionConditions {
public:
  int32_t mapped_reserved_gap = 1;
  int32_t reserved_launched_gap = 1;
  int32_t total_in_flight = 1;

  RangeTransitionConditions(int32_t mapped_reserved_gap_, int32_t reserved_launched_gap_,
                            int32_t total_in_flight_)
      : mapped_reserved_gap(mapped_reserved_gap_), reserved_launched_gap(reserved_launched_gap_),
        total_in_flight(total_in_flight_) {
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto n_mapped = state.counts.n_mapped();
    auto n_reserved = state.counts.n_reserved();
    T4F_INVARIANT(n_mapped >= n_reserved);
    return ((n_mapped - n_reserved) <= mapped_reserved_gap) && (n_mapped <= total_in_flight);
  }

  bool should_reserve(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto n_reserved = state.counts.n_reserved();
    auto n_launched = state.counts.n_launched();
    T4F_INVARIANT(n_reserved >= n_launched);
    return (n_reserved - n_launched) <= reserved_launched_gap;
  }
};

class BatchTransitionConditions : public TransitionConditions {
public:
  timecount_t last_accessed = 0;
  int32_t batch_size = 20;
  int32_t queue_threshold = 2;
  int32_t max_in_flight = 16;
  int32_t active_batch = 0;

  BatchTransitionConditions(int32_t batch_size_, int32_t queue_threshold_, int32_t max_in_flight_)
      : batch_size(batch_size_), queue_threshold(queue_threshold_), max_in_flight(max_in_flight_) {
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto &counts = state.counts;
    auto n_mapped = counts.n_mapped();
    bool space_flag = (n_mapped <= max_in_flight + active_batch);
    bool workqueue_flag = false;
    const devid_t n_devices = state.get_devices().size();
    for (int i = 1; i < n_devices; i++) {
      if (counts.n_mapped(i) < queue_threshold) {
        workqueue_flag = true;
        break;
      }
    }

    bool flag = space_flag || workqueue_flag;

    if (flag) {
      if (active_batch == 0) {
        last_accessed = state.get_global_time();
        active_batch = batch_size;
      }
    } else {
      active_batch = 0;
    }

    return flag;
  }
};

class DeviceThresholdState {
public:
  static constexpr int32_t disabled = -1;

private:
  int32_t mapped_threshold_ = 0;
  int32_t reserved_threshold_ = disabled;

  static void validate_thresholds(int32_t mapped_threshold, int32_t reserved_threshold) {
    if (mapped_threshold >= 0 && reserved_threshold >= 0) {
      throw std::invalid_argument(
          "Device threshold settings are exclusive: enable either mapped or reserved checks, "
          "not both");
    }
  }

public:
  DeviceThresholdState() = default;

  DeviceThresholdState(int32_t mapped_threshold, int32_t reserved_threshold) {
    set_thresholds(mapped_threshold, reserved_threshold);
  }

  void set_thresholds(int32_t mapped_threshold, int32_t reserved_threshold) {
    validate_thresholds(mapped_threshold, reserved_threshold);
    mapped_threshold_ = mapped_threshold;
    reserved_threshold_ = reserved_threshold;
  }

  void use_mapped_threshold(int32_t mapped_threshold) {
    set_thresholds(mapped_threshold, disabled);
  }

  void use_reserved_threshold(int32_t reserved_threshold) {
    set_thresholds(disabled, reserved_threshold);
  }

  void disable_thresholds() {
    set_thresholds(disabled, disabled);
  }

  [[nodiscard]] int32_t get_mapped_threshold() const {
    return mapped_threshold_;
  }

  [[nodiscard]] int32_t get_reserved_threshold() const {
    return reserved_threshold_;
  }

  void set_mapped_threshold(int32_t mapped_threshold) {
    set_thresholds(mapped_threshold, reserved_threshold_);
  }

  void set_reserved_threshold(int32_t reserved_threshold) {
    set_thresholds(mapped_threshold_, reserved_threshold);
  }

  [[nodiscard]] bool has_mapped_threshold() const {
    return mapped_threshold_ >= 0;
  }

  [[nodiscard]] bool has_reserved_threshold() const {
    return reserved_threshold_ >= 0;
  }

  [[nodiscard]] bool is_device_under_threshold(const SchedulerState &state,
                                               devid_t device_id) const {
    if (has_mapped_threshold()) {
      return state.counts.n_unreserved_mapped(device_id) <= mapped_threshold_;
    }
    if (has_reserved_threshold()) {
      return state.counts.n_reserved(device_id) <= reserved_threshold_;
    }
    return false;
  }

  void append_under_threshold_gpu_devices(const SchedulerState &state,
                                          std::vector<devid_t> &out) const {
    if (!has_mapped_threshold() && !has_reserved_threshold()) {
      return;
    }

    const auto &devices = state.get_devices();
    const devid_t n_devices = devices.size();
    for (devid_t device_id = 1; device_id < n_devices; ++device_id) {
      if (devices.get_type(device_id) != DeviceType::GPU) {
        continue;
      }
      if (is_device_under_threshold(state, device_id)) {
        out.push_back(device_id);
      }
    }
  }
};

class DeviceThresholdTransitionConditions : public TransitionConditions {
public:
  DeviceThresholdState thresholds;

  DeviceThresholdTransitionConditions() = default;

  DeviceThresholdTransitionConditions(int32_t mapped_threshold_, int32_t reserved_threshold_)
      : thresholds(mapped_threshold_, reserved_threshold_) {
  }

  [[nodiscard]] int32_t get_mapped_threshold() const {
    return thresholds.get_mapped_threshold();
  }

  [[nodiscard]] int32_t get_reserved_threshold() const {
    return thresholds.get_reserved_threshold();
  }

  void set_mapped_threshold(int32_t mapped_threshold) {
    thresholds.set_mapped_threshold(mapped_threshold);
  }

  void set_reserved_threshold(int32_t reserved_threshold) {
    thresholds.set_reserved_threshold(reserved_threshold);
  }

  void set_thresholds(int32_t mapped_threshold, int32_t reserved_threshold) {
    thresholds.set_thresholds(mapped_threshold, reserved_threshold);
  }

  void use_mapped_threshold(int32_t mapped_threshold) {
    thresholds.use_mapped_threshold(mapped_threshold);
  }

  void use_reserved_threshold(int32_t reserved_threshold) {
    thresholds.use_reserved_threshold(reserved_threshold);
  }

  void disable_thresholds() {
    thresholds.disable_thresholds();
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    std::vector<devid_t> selected_devices;
    thresholds.append_under_threshold_gpu_devices(state, selected_devices);
    return !selected_devices.empty();
  }
};

// struct SuccessPair {
//   bool success = false;
//   taskid_t last_idx = 0;
// };

enum class EvictionState : int8_t {
  NONE = 0,
  WAITING_FOR_COMPLETION = 2,
  RUNNING = 4,
};

class Scheduler {

protected:
  struct EvictionInvalidationInfo {
    bool future_usage = false;
    bool write_after_read = false;
  };

  SchedulerState state;
  SchedulerQueues queues;
  EvictionPolicy eviction_policy = EvictionPolicy::LRU;
  bool transfer_aware_data_launch_order = false;
  TaskDeviceList tasks_requesting_eviction;
  int64_t success_count = 0;
  int64_t eviction_count = 0;
  EvictionState eviction_state = EvictionState::NONE;
  ankerl::unordered_dense::map<uint64_t, uint8_t> eviction_invalidation_cache;
  ankerl::unordered_dense::set<uint64_t> eviction_planned_victim_keys;
  DataIDList eviction_victim_buffer;
  std::vector<DataIDList> eviction_candidate_buckets;
  std::vector<taskid_t> eviction_nonempty_buckets;

  void enqueue_data_tasks(taskid_t task_id);
  [[nodiscard]] EvictionInvalidationInfo
  get_eviction_invalidation_info(const StaticTaskInfo &static_graph,
                                 const RuntimeTaskInfo &task_runtime, dataid_t data_id,
                                 devid_t device_id);
  [[nodiscard]] std::span<const dataid_t>
  select_eviction_victims(devid_t device_id, mem_t missing_memory,
                          std::span<const dataid_t> used_ids);
  [[nodiscard]] timecount_t estimate_data_task_remaining_transfer_time(taskid_t data_task_id) const;
  void clear_eviction_invalidation_cache() {
    eviction_invalidation_cache.clear();
  }

public:
  BreakpointManager breakpoints;
  TaskIDList compute_task_buffer;
  TaskIDList data_task_buffer;
  TaskIDList newly_mappable_buffer;
  TaskIDList candidates_buffer;
  std::reference_wrapper<TransitionConditions> conditions;
  int64_t scheduler_event_count = 1;
  bool initialized = false;

  Scheduler(SchedulerInput &input)
      : state(input), queues(input.devices), conditions(input.conditions),
        eviction_policy(input.eviction_policy),
        transfer_aware_data_launch_order(input.transfer_aware_data_launch_order) {
    compute_task_buffer.reserve(INITIAL_TASK_BUFFER_SIZE);
    data_task_buffer.reserve(INITIAL_TASK_BUFFER_SIZE);
    candidates_buffer.reserve(INITIAL_TASK_BUFFER_SIZE);
    tasks_requesting_eviction.reserve(INITIAL_TASK_BUFFER_SIZE);
    eviction_invalidation_cache.reserve(INITIAL_TASK_BUFFER_SIZE * 8);
    eviction_planned_victim_keys.reserve(INITIAL_TASK_BUFFER_SIZE * 8);
    eviction_victim_buffer.reserve(INITIAL_TASK_BUFFER_SIZE);
    eviction_candidate_buckets.reserve(INITIAL_TASK_BUFFER_SIZE);
    eviction_nonempty_buckets.reserve(INITIAL_TASK_BUFFER_SIZE);
    if (eviction_policy == EvictionPolicy::LEAST_USED_MAPPED) {
      state.enable_task_data_device_usage_info();
    }
    if (input.top_k_candidates > 0) {
      queues.mappable.set_k(static_cast<int>(input.top_k_candidates));
    }
  }

  Scheduler(const Scheduler &other) = default;

  void set_transition_conditions(TransitionConditions &conditions_) {
    conditions = conditions_;
  }

  void set_steps(int32_t steps) {
    breakpoints.set_steps_to_go(steps);
  }

  void set_mapper_boundary_steps(int32_t boundaries) {
    breakpoints.set_mapper_boundaries_to_go(boundaries);
  }

  void start_drain() {
    state.start_drain();
  }

  void stop_drain() {
    state.stop_drain();
  }

  const std::span<const taskid_t> initially_mappable_tasks() {
    return state.get_graph().get_initial_tasks();
  }

  void initialize(bool create_data_tasks = false, bool initialize_data_manager = false) {
    if (initialized) {
      SPDLOG_WARN("Scheduler already initialized. Skipping re-initialization.");
      return;
    }
    state.initialize(create_data_tasks, initialize_data_manager);
    auto initial_tasks = initially_mappable_tasks();
    push_mappable(initial_tasks);
    initialized = true;
  }

  void initialize_data_manager() {
    state.initialize_data_manager();
  }

  void initialize_data_replicate(dataid_t data_id, devid_t device_id) {
    state.initialize_data_replicate(data_id, device_id);
  }

  size_t get_mappable_candidates(std::span<int64_t> v);

  std::span<const taskid_t> collect_candidates() {
    candidates_buffer = queues.mappable.get_top_k();
    return std::span<const taskid_t>(candidates_buffer);
  }

  taskid_t map_task(taskid_t task_id, Action &action);
  void apply_mapped_actions(std::span<const taskid_t> candidates, ActionList &actions);
  void skip_map_tasks(MapperEvent &map_event, EventManager &event_manager);
  void map_tasks(MapperEvent &map_event, EventManager &event_manager, Mapper &mapper);
  ExecutionState map_tasks_from_python(ActionList &action_list, EventManager &event_manager);
  void remove_mapped_tasks(ActionList &action_list);

  bool reserve_task(taskid_t task_id, devid_t device_id);
  void skip_reserve_tasks(ReserverEvent &reserve_event, EventManager &event_manager);
  void reserve_tasks(ReserverEvent &reserve_event, EventManager &event_manager);

  bool launch_compute_task(taskid_t task_id, devid_t device_id, EventManager &event_manager);
  bool launch_data_task(taskid_t task_id, devid_t device_id, EventManager &event_manager);
  bool launch_eviction_task(taskid_t task_id, devid_t device_id, EventManager &event_manager);
  bool launch_compute_tasks(EventManager &event_manager);
  bool launch_data_tasks(EventManager &event_manager);
  bool launch_eviction_tasks(EventManager &event_manager);
  void launch_tasks(LauncherEvent &launch_event, EventManager &event_manager);

  void evict(EvictorEvent &eviction_event, EventManager &event_manager);

  void complete_compute_task(ComputeCompleterEvent &complete_event, EventManager &event_manager);
  void complete_data_task(DataCompleterEvent &complete_event, EventManager &event_manager);
  void complete_eviction_task(EvictorCompleterEvent &complete_event, EventManager &event_manager);
  void complete_task_postmatter(EventManager &event_manager);

  void update_time(timecount_t time) {
    state.update_time(time);
  }

  [[nodiscard]] const SchedulerState &get_state() const {
    return state;
  }
  [[nodiscard]] SchedulerState &get_state() {
    return state;
  }
  [[nodiscard]] const SchedulerQueues &get_queues() const {
    return queues;
  }
  [[nodiscard]] SchedulerQueues &get_queues() {
    return queues;
  }

  void push_mappable(taskid_t compute_task_id) {
    priority_t p = state.get_task_noise().get_priority(compute_task_id);
    SPDLOG_DEBUG("Pushing mappable compute task {} with priority {}", compute_task_id, p);
    queues.push_mappable(compute_task_id, p);
  }

  void push_mappable(const std::span<const taskid_t> compute_task_id) {
    const auto &ps = state.get_task_noise().get_priorities();
    for (int i = 0; i < compute_task_id.size(); ++i) {
      const taskid_t id = compute_task_id[i];
      const priority_t p = ps[id];
      SPDLOG_DEBUG("Pushing mappable compute task {} with priority {}", id, p);
      queues.push_mappable(id, p);
    }
  }

  void push_reservable(taskid_t compute_task_id, devid_t device) {
    priority_t p = state.task_runtime.get_compute_task_reserve_priority(compute_task_id);
    queues.push_reservable(compute_task_id, p, device);
  }

  void push_reservable(const std::span<const taskid_t> compute_task_ids) {
    for (auto id : compute_task_ids) {
      const priority_t p = state.task_runtime.get_compute_task_reserve_priority(id);
      const devid_t device = state.task_runtime.get_compute_task_mapped_device(id);
      SPDLOG_DEBUG("Time:{} Pushing reservable compute task {} with priority {} on device {}",
                   state.get_global_time(), id, p, device);
      queues.push_reservable(id, p, device);
    }
  }

  void push_launchable(taskid_t compute_task_id, devid_t device) {
    const priority_t p = state.task_runtime.get_compute_task_launch_priority(compute_task_id);
    queues.push_launchable(compute_task_id, p, device);
  }

  void push_launchable(const std::span<const taskid_t> compute_task_ids) {
    for (auto id : compute_task_ids) {
      const priority_t p = state.task_runtime.get_compute_task_launch_priority(id);
      const devid_t device = state.task_runtime.get_compute_task_mapped_device(id);
      queues.push_launchable(id, p, device);
    }
  }

  void push_launchable_data(taskid_t data_task_id) {
    const priority_t p = state.task_runtime.get_data_task_launch_priority(data_task_id);
    const devid_t device = state.task_runtime.get_data_task_mapped_device(data_task_id);
    const auto remaining_transfer_time = estimate_data_task_remaining_transfer_time(data_task_id);
    DataLaunchQueueEntry entry{
        .task_id = data_task_id,
        .parent_priority = p,
        .remaining_transfer_time = remaining_transfer_time,
        .seq = queues.next_data_queue_seq(),
        .transfer_aware = transfer_aware_data_launch_order,
    };
    queues.push_launchable_data(entry, device);
  }

  void push_launchable_data(const std::span<const taskid_t> data_task_ids) {
    for (auto data_task_id : data_task_ids) {
      push_launchable_data(data_task_id);
    }
  }

  void push_launchable_eviction(taskid_t eviction_task_id) {
    SPDLOG_DEBUG("Time:{} Pushing launchable eviction task {}", state.get_global_time(),
                 eviction_task_id);
    queues.push_launchable_eviction(eviction_task_id, 0, 0);
  }

  [[nodiscard]] bool is_complete() const {
    return state.is_complete();
  }

  [[nodiscard]] bool is_drain_complete() const {
    return state.is_drain_complete();
  }

  [[nodiscard]] bool has_pending_step_breakpoint() const {
    return breakpoints.has_pending_step_stop();
  }

  bool consume_step_breakpoint() {
    return breakpoints.consume_step_stop();
  }

  bool hit_mapper_boundary_breakpoint() {
    return breakpoints.decrement_mapper_boundaries();
  }

  [[nodiscard]] bool has_time_breakpoint() const {
    return breakpoints.has_time_breakpoint();
  }

  [[nodiscard]] bool hit_time_breakpoint(timecount_t time) const {
    return breakpoints.check_time_breakpoint(time);
  }

  [[nodiscard]] bool hit_task_breakpoint(EventType type, taskid_t task_id) {
    return breakpoints.check_task_breakpoint(type, task_id);
  }

  [[nodiscard]] bool needs_event_breakpoint_poll() const {
    return breakpoints.needs_event_poll();
  }

  friend class SchedulerState;
  friend class SchedulerQueues;
};

class LRUEvictionPolicy {
public:
  [[nodiscard]] std::span<const dataid_t>
  select_victims(const LRU_manager &lru_manager, devid_t device_id, mem_t missing_memory,
                 std::span<const dataid_t> used_ids) const {
    return lru_manager.getLRUids(device_id, static_cast<std::size_t>(missing_memory), used_ids);
  }
};

class LeastUsedMappedEvictionPolicy {
public:
  [[nodiscard]] std::span<const dataid_t>
  select_victims(const SchedulerState &state, const Data &data, const LRU_manager &lru_manager,
                 devid_t device_id, mem_t missing_memory, std::span<const dataid_t> used_ids,
                 std::vector<DataIDList> &candidate_buckets,
                 std::vector<taskid_t> &nonempty_buckets, DataIDList &victim_buffer) const;
};

class Mapper {

protected:
  devid_t n_devices = 0;
  devicemask_t device_flags = 0;
  uint8_t arch_flags = 0;
  std::vector<devid_t> device_buffer;
  std::vector<DeviceType> arch_buffer;
  ActionList action_buffer;

  void fill_arch_targets(taskid_t compute_task_id, const SchedulerState &state) {
    arch_buffer.clear();
    const auto arch_mask = state.get_tasks().get_supported_architecture_mask(compute_task_id);
    for (uint8_t i = 0; i < num_device_types; ++i) {
      const uint8_t arch_flag = 1 << i;
      if (arch_mask & arch_flag) {
        arch_buffer.push_back(static_cast<DeviceType>(arch_flag));
      }
    }
  }

  void fill_device_targets(taskid_t compute_task_id, const SchedulerState &state) {
    device_buffer.clear();
    const std::size_t n_devices = state.get_devices().size();
    auto device_mask = state.get_tasks().get_supported_devices_mask(compute_task_id);

    using UMask = std::make_unsigned_t<devicemask_t>;
    constexpr std::size_t mask_bits = std::numeric_limits<UMask>::digits;
    auto mask = static_cast<UMask>(device_mask);
    if (n_devices < mask_bits) {
      const auto limit_mask = (UMask{1} << n_devices) - UMask{1};
      mask &= limit_mask;
    }

    while (mask) {
      const auto bit = static_cast<devid_t>(std::countr_zero(mask));
      device_buffer.push_back(bit);
      mask &= (mask - UMask{1});
    }
  }

  static const DeviceIDList &get_devices_from_arch(DeviceType arch, SchedulerState &state) {
    const auto &devices = state.get_devices();
    return devices.get_devices(arch);
  }

public:
  Mapper() = default;

  Mapper(const Mapper &other) = default;

  void initialize() {
    device_buffer.reserve(INITIAL_DEVICE_BUFFER_SIZE);
    arch_buffer.reserve(INITIAL_DEVICE_BUFFER_SIZE);
    action_buffer.reserve(5);
  }

  virtual Action map_task(taskid_t task_id, const SchedulerState &state) {
    MONUnusedParameter(state);
    SPDLOG_WARN("Mapping task {} with unset mapper", task_id);
    return Action(0, 0);
  }

  virtual ActionList &map_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    action_buffer.clear();
    action_buffer.reserve(task_ids.size());
    for (std::size_t i = 0; i < task_ids.size(); ++i) {
      auto action = map_task(task_ids[i], state);
      action.pos = i;
      action_buffer.emplace_back(action);
    }
    return action_buffer;
  }
};

class RandomMapper : public Mapper {
protected:
  std::random_device rd;
  std::mt19937 gen;

  DeviceType choose_random_architecture(std::vector<DeviceType> &arch_buffer) {
    std::uniform_int_distribution<std::size_t> dist(0, arch_buffer.size() - 1);
    return arch_buffer[dist(gen)];
  }

  devid_t choose_random_device(DeviceIDList &device_buffer) {
    std::uniform_int_distribution<std::size_t> dist(0, device_buffer.size() - 1);
    return device_buffer[dist(gen)];
  }

public:
  RandomMapper(unsigned int seed = 0) : gen(seed) {
  }

  RandomMapper(const RandomMapper &other) {
    gen = other.gen;
  }

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    fill_device_targets(task_id, state);
    devid_t device_id = choose_random_device(device_buffer);
    return Action(0, device_id);
  }
};

class RoundRobinMapper : public Mapper {
protected:
  std::size_t device_index = 0;

public:
  RoundRobinMapper() = default;
  RoundRobinMapper(const RoundRobinMapper &other) = default;
  Action map_task(taskid_t compute_task_id, const SchedulerState &state) override {
    fill_device_targets(compute_task_id, state);
    auto mp = state.get_mapping_priority(compute_task_id);
    devid_t device_id = device_buffer[device_index];
    device_index = (device_index + 1) % device_buffer.size();
    return Action(0, device_id, mp, mp);
  }
};

class StaticMapper : public Mapper {
protected:
  DeviceIDList mapping;
  PriorityList reserving_priorities;
  PriorityList launching_priorities;

public:
  StaticMapper() = default;

  StaticMapper(const StaticMapper &other) = default;

  StaticMapper(DeviceIDList device_ids_) : mapping(std::move(device_ids_)) {
  }

  StaticMapper(DeviceIDList device_ids_, PriorityList reserving_priorities_,
               PriorityList launching_priorities_)
      : mapping(std::move(device_ids_)), reserving_priorities(std::move(reserving_priorities_)),
        launching_priorities(std::move(launching_priorities_)) {
  }

  void set_reserving_priorities(PriorityList reserving_priorities_) {
    reserving_priorities = std::move(reserving_priorities_);
  }

  void set_launching_priorities(PriorityList launching_priorities_) {
    launching_priorities = std::move(launching_priorities_);
  }

  void set_mapping(DeviceIDList device_ids_) {
    mapping = std::move(device_ids_);
  }

  Action map_task(taskid_t compute_task_id, const SchedulerState &state) override {
    devid_t device_id = 0;
    auto mp = state.get_mapping_priority(compute_task_id);
    priority_t rp = mp;
    priority_t lp = mp;

    if (!mapping.empty()) {
      device_id = mapping.at(compute_task_id % mapping.size());
    }
    if (!reserving_priorities.empty()) {
      rp = reserving_priorities.at(compute_task_id % reserving_priorities.size());
    }
    if (!launching_priorities.empty()) {
      lp = launching_priorities.at(compute_task_id % launching_priorities.size());
    }

    T4F_INVARIANT(state.get_tasks().is_architecture_supported(compute_task_id,
                                                       state.get_devices().get_type(device_id)));
    T4F_INVARIANT(device_id < state.get_devices().size());

    return Action(0, device_id, rp, lp);
  }
};

class StaticActionMapper : public Mapper {
protected:
  ActionList actions;

public:
  StaticActionMapper(ActionList actions_) : actions(std::move(actions_)) {
  }

  StaticActionMapper(const StaticActionMapper &other) = default;

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    MONUnusedParameter(state);
    return actions.at(task_id);
  }
};

class DeviceTime {
public:
  devid_t device_id;
  timecount_t time;
};

class EFTMapper : public Mapper {

protected:
  // Records the finish time by task id
  std::vector<timecount_t> finish_time_record;
  // Records the max predecessor finish time for each task id
  std::vector<timecount_t> max_dependency_finish_time;

  void ensure_task_buffers_size(std::size_t n_tasks) {
    if (finish_time_record.size() < n_tasks) {
      finish_time_record.resize(n_tasks, 0);
    }
    if (max_dependency_finish_time.size() < n_tasks) {
      max_dependency_finish_time.resize(n_tasks, 0);
    }
  }

  void reset_task_buffers() {
    std::fill(finish_time_record.begin(), finish_time_record.end(), 0);
    std::fill(max_dependency_finish_time.begin(), max_dependency_finish_time.end(), 0);
  }

  [[nodiscard]] bool should_reset_for_new_run(const SchedulerState &state) const {
    return state.get_global_time() == 0 && state.counts.n_mapped() == 0 &&
           state.counts.n_reserved() == 0 && state.counts.n_launched() == 0 &&
           state.counts.n_completed() == 0;
  }

public:
  void record_finish_time(taskid_t task_id, timecount_t time, const SchedulerState &state) {
    finish_time_record[task_id] = time;
    for (auto dependent_id : state.get_tasks().get_compute_task_dependents(task_id)) {
      max_dependency_finish_time[dependent_id] =
          std::max(max_dependency_finish_time[dependent_id], time);
    }
  }

  timecount_t time_for_transfer(dataid_t data_id, devid_t destination,
                                const SchedulerState &state) {
    auto &data_manager = state.get_data_manager();
    const auto &communication_manager = state.get_communication_manager();
    auto location_flags = data_manager.get_mapped_location_flags(data_id);

    const auto &topology = state.get_topology();
    const mem_t data_size = state.get_data().get_size(data_id);
    SourceRequest req =
        communication_manager.get_best_source(topology, destination, location_flags);
    T4F_INVARIANT(req.found);
    return communication_manager.ideal_time_to_transfer(topology, data_size, req.source,
                                                        destination);
  }

  timecount_t get_finish_time(taskid_t compute_task_id, devid_t device_id, timecount_t start_t,
                              const SchedulerState &state) {
    const auto read_set = state.get_tasks().get_read(compute_task_id);
    const DeviceType arch = state.get_devices().get_type(device_id);
    timecount_t duration = state.get_tasks().get_mean_duration(compute_task_id, arch);
    timecount_t data_time = 0;

    for (auto data_id : read_set) {
      timecount_t transfer_time = time_for_transfer(data_id, device_id, state);
      data_time += transfer_time;
    }

    return start_t + data_time + duration;
  }

  timecount_t virtual get_device_available_time(devid_t device_id, const SchedulerState &state) {
    timecount_t reserved_workload = state.costs.get_reserved_time(device_id);
    return state.get_global_time() + reserved_workload;
  }

  timecount_t get_dependency_finish_time(taskid_t compute_task_id, const SchedulerState &state) {
    MONUnusedParameter(state);
    return max_dependency_finish_time[compute_task_id];
  }

  virtual DeviceTime get_best_device(taskid_t task_id, const SchedulerState &state) {
    fill_device_targets(task_id, state);
    T4F_INVARIANT(!device_buffer.empty());
    const timecount_t dep_time = get_dependency_finish_time(task_id, state);

    auto min_time = MAX_TIME;
    auto best_device = device_buffer[0];
    for (auto device_id : device_buffer) {
      const timecount_t device_available = get_device_available_time(device_id, state);
      const timecount_t start_time = std::max(device_available, dep_time);
      const timecount_t finish_time = get_finish_time(task_id, device_id, start_time, state);
      SPDLOG_DEBUG("Task {} on device {}: start_time = {}, finish_time = {}, dep_time = {}, "
                   "device_available = {}",
                   task_id, device_id, start_time, finish_time, dep_time, device_available);
      if (finish_time < min_time) {
        min_time = finish_time;
        best_device = device_id;
      }
    }

    return {best_device, min_time};
  }

  EFTMapper() = default;

  EFTMapper(const EFTMapper &other) = default;

  EFTMapper(std::size_t n_tasks, std::size_t /*n_devices*/)
      : finish_time_record(n_tasks, 0), max_dependency_finish_time(n_tasks, 0) {}

  void initialize(std::size_t n_tasks, std::size_t /*n_devices*/) {
    finish_time_record = std::vector<timecount_t>(n_tasks, 0);
    max_dependency_finish_time = std::vector<timecount_t>(n_tasks, 0);
  }

  Action map_task(taskid_t compute_task_id, const SchedulerState &state) override {
    ensure_task_buffers_size(static_cast<std::size_t>(state.get_tasks().get_n_compute_tasks()));
    if (should_reset_for_new_run(state)) {
      reset_task_buffers();
    }
    auto [best_device, min_time] = get_best_device(compute_task_id, state);
    record_finish_time(compute_task_id, min_time, state);
    auto mp = state.get_mapping_priority(compute_task_id);
    return Action(0, best_device, mp, mp);
  }
};

class MemoryAwareEFTMapper : public EFTMapper {

protected:
  mutable DataIDList sorted_unique_buf_;

  timecount_t estimate_eviction_cost(devid_t device_id, mem_t bytes_to_evict,
                                     std::span<const dataid_t> protected_ids,
                                     const SchedulerState &state) const {
    if (bytes_to_evict <= 0) {
      return 0;
    }
    const auto &lru = state.get_data_manager().get_lru_manager();
    const auto &comm = state.get_communication_manager();
    const auto &topo = state.get_topology();
    const devid_t host = 0;

    timecount_t cost = 0;
    mem_t accumulated = 0;

    lru.visitLRUCandidates(device_id, protected_ids,
                           [&](dataid_t victim_id, mem_t victim_bytes) {
                             if (accumulated >= bytes_to_evict) {
                               return;
                             }
                             accumulated += victim_bytes;
                             auto flags =
                                 state.get_data_manager().get_mapped_location_flags(victim_id);
                             if (__builtin_popcount(flags) <= 1) {
                               // Only copy — must write back to host
                               cost +=
                                   comm.ideal_time_to_transfer(topo, victim_bytes, device_id, host);
                             }
                             // else: free invalidation
                           });

    if (accumulated < bytes_to_evict) {
      return MAX_TIME; // infeasible: not enough evictable data
    }
    return cost;
  }

public:
  double alpha = 1.0; // weight for eviction cost term (public for Python exposure)

  MemoryAwareEFTMapper() = default;
  MemoryAwareEFTMapper(const MemoryAwareEFTMapper &) = default;

  MemoryAwareEFTMapper(std::size_t n_tasks, std::size_t n_devices, double alpha_val = 1.0)
      : EFTMapper(n_tasks, n_devices), alpha(alpha_val) {}

  DeviceTime get_best_device(taskid_t task_id, const SchedulerState &state) override {
    fill_device_targets(task_id, state);
    T4F_INVARIANT(!device_buffer.empty());
    const timecount_t dep_time = get_dependency_finish_time(task_id, state);
    const auto &tasks = state.get_tasks();
    const auto &dm = state.get_device_manager();

    // Build sorted unique data IDs for LRU binary_search exclusion
    auto unique = tasks.get_unique(task_id);
    sorted_unique_buf_.assign(unique.begin(), unique.end());
    std::sort(sorted_unique_buf_.begin(), sorted_unique_buf_.end());

    auto min_score = MAX_TIME;
    auto best_device = device_buffer[0];

    for (auto dev : device_buffer) {
      const timecount_t dev_avail = get_device_available_time(dev, state);
      const timecount_t start_t = std::max(dev_avail, dep_time);
      const timecount_t ft = get_finish_time(task_id, dev, start_t, state);

      // Estimate memory overflow on this device using RESERVED state,
      // which matches what the LRU tracks (physically resident data)
      // and what the actual eviction trigger at reserve time will compute.
      const DeviceType arch = state.get_devices().get_type(dev);
      const Resources &res = tasks.get_compute_task_resources(task_id, arch);
      mem_t non_local = state.get_data_manager().non_local_size_reserved(
          state.get_data(), unique, dev);
      mem_t task_mem = res.mem + non_local;
      mem_t overflow = dm.overflow_mem<TaskState::RESERVED>(dev, task_mem);

      timecount_t evict_cost = 0;
      if (overflow > 0) {
        evict_cost = estimate_eviction_cost(dev, overflow, sorted_unique_buf_, state);
        if (evict_cost >= MAX_TIME) {
          SPDLOG_DEBUG("Task {} on device {}: INFEASIBLE (cannot evict enough memory)", task_id,
                       dev);
          continue; // skip infeasible device
        }
      }

      timecount_t score =
          ft + static_cast<timecount_t>(alpha * static_cast<double>(evict_cost));
      SPDLOG_DEBUG("Task {} on device {}: ft={}, evict_cost={}, score={}, overflow={}", task_id,
                   dev, ft, evict_cost, score, overflow);

      if (score < min_score) {
        min_score = score;
        best_device = dev;
      }
    }

    // Fallback: if all devices are infeasible, pick least-loaded and let reactive eviction handle it
    if (min_score >= MAX_TIME) {
      SPDLOG_WARN("Task {}: all devices infeasible for memory, falling back to least-loaded",
                  task_id);
      mem_t min_mem = std::numeric_limits<mem_t>::max();
      for (auto dev : device_buffer) {
        mem_t usage = dm.get_mem<TaskState::MAPPED>(dev);
        if (usage < min_mem) {
          min_mem = usage;
          best_device = dev;
        }
      }
      const timecount_t dev_avail = get_device_available_time(best_device, state);
      min_score = get_finish_time(task_id, best_device, std::max(dev_avail, dep_time), state);
    }

    return {best_device, min_score};
  }
};

class DequeueEFTMapper : public EFTMapper {

  std::vector<timecount_t> device_available_time_buffer;

public:
  timecount_t get_device_available_time(devid_t device_id, const SchedulerState &state) override {
    return std::max(state.get_global_time(), device_available_time_buffer.at(device_id));
  }

  void set_device_available_time(devid_t device_id, timecount_t time) {
    device_available_time_buffer.at(device_id) = time;
  }

  DequeueEFTMapper() = default;

  DequeueEFTMapper(const DequeueEFTMapper &other) = default;

  DequeueEFTMapper(std::size_t n_tasks, std::size_t n_devices)
      : EFTMapper(n_tasks, n_devices), device_available_time_buffer(n_devices) {
  }

  void initialize(std::size_t n_tasks, std::size_t n_devices) {
    EFTMapper::initialize(n_tasks, n_devices);
    device_available_time_buffer = std::vector<timecount_t>(n_devices, 0);
  }

  Action map_task(taskid_t compute_task_id, const SchedulerState &state) override {
    SPDLOG_DEBUG("Mapping compute task {} with DequeueEFTMapper", compute_task_id);
    ensure_task_buffers_size(static_cast<std::size_t>(state.get_tasks().get_n_compute_tasks()));
    if (should_reset_for_new_run(state)) {
      reset_task_buffers();
      std::fill(device_available_time_buffer.begin(), device_available_time_buffer.end(),
                state.get_global_time());
    }
    if (device_available_time_buffer.size() < state.get_devices().size()) {
      device_available_time_buffer.resize(state.get_devices().size(), state.get_global_time());
    }

    auto [best_device, min_time] = get_best_device(compute_task_id, state);
    record_finish_time(compute_task_id, min_time, state);
    set_device_available_time(best_device, min_time);
    const auto mp = state.get_mapping_priority(compute_task_id);
    return Action(0, best_device, mp, mp);
  }
};

class DataAwareMapper : public Mapper {
private:
  struct TaskRec {
    taskid_t task_id = -1;
    std::size_t input_pos = 0;
    priority_t priority = 0;
    DeviceIDList supported_devices;
    std::vector<int32_t> read_data_indices;
    std::vector<int32_t> unique_data_indices;
    bool active = true;
  };

  struct DataRec {
    dataid_t data_id = -1;
    std::vector<int32_t> reader_task_indices;
    std::vector<int32_t> planned_count;
    std::vector<uint8_t> local_present;
  };

  struct DevicePlan {
    std::vector<int32_t> planned_task_indices;
    std::vector<int32_t> frontier_data_indices;
  };

  struct DataChoice {
    bool valid = false;
    int32_t data_index = -1;
    int32_t newly_free_count = 0;
    priority_t newly_free_priority_sum = 0;
    timecount_t transfer_time = MAX_TIME;
    dataid_t data_id = -1;
  };

  struct Proposal {
    bool valid = false;
    int32_t task_index = -1;
    devid_t device_id = -1;
    int32_t newly_free_count = 0;
    timecount_t transfer_time = MAX_TIME;
    priority_t priority = 0;
    taskid_t task_id = -1;
  };

  [[nodiscard]] static bool device_has_data(devicemask_t location_flags, devid_t device_id) {
    using UMask = std::make_unsigned_t<devicemask_t>;
    constexpr std::size_t mask_bits = std::numeric_limits<UMask>::digits;
    if (device_id < 0 || static_cast<std::size_t>(device_id) >= mask_bits) {
      return false;
    }
    const auto mask = static_cast<UMask>(location_flags);
    const auto bit = static_cast<UMask>(UMask{1} << static_cast<std::size_t>(device_id));
    return (mask & bit) != 0;
  }

  [[nodiscard]] static devicemask_t device_bit(devid_t device_id) {
    using UMask = std::make_unsigned_t<devicemask_t>;
    constexpr std::size_t mask_bits = std::numeric_limits<UMask>::digits;
    if (device_id < 0 || static_cast<std::size_t>(device_id) >= mask_bits) {
      return 0;
    }
    return static_cast<devicemask_t>(UMask{1} << static_cast<std::size_t>(device_id));
  }

  ActionList &plan_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    const auto &static_graph = state.get_tasks();
    const auto &data_manager = state.get_data_manager();
    const auto &communication_manager = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto n_devices = static_cast<std::size_t>(state.get_devices().size());

    action_buffer.clear();
    action_buffer.reserve(task_ids.size());
    if (task_ids.empty()) {
      return action_buffer;
    }

    std::vector<TaskRec> task_records;
    task_records.reserve(task_ids.size());

    ankerl::unordered_dense::map<dataid_t, int32_t> data_index;
    data_index.reserve(task_ids.size() * 4);

    std::vector<DataRec> data_records;
    data_records.reserve(task_ids.size() * 4);

    auto get_or_create_data = [&](dataid_t data_id) -> int32_t {
      const auto it = data_index.find(data_id);
      if (it != data_index.end()) {
        return it->second;
      }

      const auto idx = static_cast<int32_t>(data_records.size());
      data_index.emplace(data_id, idx);
      DataRec rec;
      rec.data_id = data_id;
      rec.planned_count.assign(n_devices, 0);
      rec.local_present.assign(n_devices, 0);
      data_records.push_back(std::move(rec));
      return idx;
    };

    for (std::size_t input_pos = 0; input_pos < task_ids.size(); ++input_pos) {
      const auto task_id = task_ids[input_pos];
      TaskRec rec;
      rec.task_id = task_id;
      rec.input_pos = input_pos;
      rec.priority = state.get_mapping_priority(task_id);

      fill_device_targets(task_id, state);
      rec.supported_devices = device_buffer;
      T4F_INVARIANT(!rec.supported_devices.empty());

      const auto task_index = static_cast<int32_t>(task_records.size());
      for (const auto data_id : static_graph.get_read(task_id)) {
        const auto data_idx = get_or_create_data(data_id);
        rec.read_data_indices.push_back(data_idx);
      }

      for (const auto data_id : static_graph.get_unique(task_id)) {
        const auto it = data_index.find(data_id);
        if (it != data_index.end()) {
          rec.unique_data_indices.push_back(it->second);
        }
      }

      task_records.push_back(std::move(rec));
      for (const auto data_idx : task_records.back().read_data_indices) {
        data_records[static_cast<std::size_t>(data_idx)].reader_task_indices.push_back(task_index);
      }
    }

    std::vector<DevicePlan> device_plans(n_devices);
    for (const auto &task_rec : task_records) {
      for (const auto device_id : task_rec.supported_devices) {
        auto &frontier =
            device_plans[static_cast<std::size_t>(device_id)].frontier_data_indices;
        frontier.insert(frontier.end(), task_rec.read_data_indices.begin(),
                        task_rec.read_data_indices.end());
      }
    }

    for (auto &plan : device_plans) {
      std::sort(plan.frontier_data_indices.begin(), plan.frontier_data_indices.end());
      plan.frontier_data_indices.erase(
          std::unique(plan.frontier_data_indices.begin(), plan.frontier_data_indices.end()),
          plan.frontier_data_indices.end());
    }

    auto task_supports_device = [](const TaskRec &task_rec, devid_t device_id) {
      return std::binary_search(task_rec.supported_devices.begin(),
                                task_rec.supported_devices.end(), device_id);
    };

    auto effective_location_flags = [&](int32_t data_idx) {
      const auto &data_rec = data_records[static_cast<std::size_t>(data_idx)];
      auto flags = data_manager.get_mapped_location_flags(data_rec.data_id);
      for (std::size_t device = 0; device < n_devices; ++device) {
        if (data_rec.local_present[device]) {
          flags = static_cast<devicemask_t>(
              flags | device_bit(static_cast<devid_t>(device)));
        }
      }
      return flags;
    };

    auto is_local_on_device = [&](int32_t data_idx, devid_t device_id) {
      const auto &data_rec = data_records[static_cast<std::size_t>(data_idx)];
      const auto device_index = static_cast<std::size_t>(device_id);
      return device_has_data(data_manager.get_mapped_location_flags(data_rec.data_id), device_id) ||
             data_rec.local_present[device_index] != 0;
    };

    auto transfer_time_for = [&](int32_t data_idx, devid_t device_id) {
      if (is_local_on_device(data_idx, device_id)) {
        return timecount_t{0};
      }

      const auto &data_rec = data_records[static_cast<std::size_t>(data_idx)];
      const auto flags = effective_location_flags(data_idx);
      const mem_t data_size = state.get_data().get_size(data_rec.data_id);
      const auto req = communication_manager.get_best_source(topology, device_id, flags);
      T4F_INVARIANT(req.found);
      return communication_manager.ideal_time_to_transfer(topology, data_size, req.source,
                                                          device_id);
    };

    auto total_transfer_time = [&](const TaskRec &task_rec, devid_t device_id) {
      timecount_t total = 0;
      for (const auto data_idx : task_rec.read_data_indices) {
        total += transfer_time_for(data_idx, device_id);
      }
      return total;
    };

    auto count_local_reads = [&](const TaskRec &task_rec, devid_t device_id) {
      int32_t total = 0;
      for (const auto data_idx : task_rec.read_data_indices) {
        total += static_cast<int32_t>(is_local_on_device(data_idx, device_id));
      }
      return total;
    };

    auto is_task_free = [&](const TaskRec &task_rec, devid_t device_id) {
      for (const auto data_idx : task_rec.read_data_indices) {
        if (!is_local_on_device(data_idx, device_id)) {
          return false;
        }
      }
      return true;
    };

    auto would_be_free_if_data_present =
        [&](const TaskRec &task_rec, devid_t device_id, int32_t data_idx) {
          if (!task_supports_device(task_rec, device_id) || is_task_free(task_rec, device_id)) {
            return false;
          }

          bool uses_data = false;
          for (const auto read_idx : task_rec.read_data_indices) {
            if (read_idx == data_idx) {
              uses_data = true;
              continue;
            }
            if (!is_local_on_device(read_idx, device_id)) {
              return false;
            }
          }
          return uses_data;
        };

    auto better_seed_task =
        [&](int32_t lhs_index, int32_t rhs_index, devid_t device_id) {
          if (rhs_index < 0) {
            return true;
          }
          const auto &lhs = task_records[static_cast<std::size_t>(lhs_index)];
          const auto &rhs = task_records[static_cast<std::size_t>(rhs_index)];
          const auto lhs_local = count_local_reads(lhs, device_id);
          const auto rhs_local = count_local_reads(rhs, device_id);
          if (lhs_local != rhs_local) {
            return lhs_local > rhs_local;
          }
          const auto lhs_transfer = total_transfer_time(lhs, device_id);
          const auto rhs_transfer = total_transfer_time(rhs, device_id);
          if (lhs_transfer != rhs_transfer) {
            return lhs_transfer < rhs_transfer;
          }
          if (lhs.priority != rhs.priority) {
            return lhs.priority > rhs.priority;
          }
          return lhs.task_id < rhs.task_id;
        };

    auto best_seed_task_for_device = [&](devid_t device_id) {
      int32_t best_index = -1;
      for (std::size_t i = 0; i < task_records.size(); ++i) {
        const auto &task_rec = task_records[i];
        if (!task_rec.active || !task_supports_device(task_rec, device_id)) {
          continue;
        }
        if (better_seed_task(static_cast<int32_t>(i), best_index, device_id)) {
          best_index = static_cast<int32_t>(i);
        }
      }
      return best_index;
    };

    auto best_free_task_for_device = [&](devid_t device_id) {
      int32_t best_index = -1;
      for (std::size_t i = 0; i < task_records.size(); ++i) {
        const auto &task_rec = task_records[i];
        if (!task_rec.active || !task_supports_device(task_rec, device_id) ||
            !is_task_free(task_rec, device_id)) {
          continue;
        }

        if (best_index < 0) {
          best_index = static_cast<int32_t>(i);
          continue;
        }

        const auto &best_task = task_records[static_cast<std::size_t>(best_index)];
        if (task_rec.priority != best_task.priority) {
          if (task_rec.priority > best_task.priority) {
            best_index = static_cast<int32_t>(i);
          }
          continue;
        }

        const auto task_transfer = total_transfer_time(task_rec, device_id);
        const auto best_transfer = total_transfer_time(best_task, device_id);
        if (task_transfer != best_transfer) {
          if (task_transfer < best_transfer) {
            best_index = static_cast<int32_t>(i);
          }
          continue;
        }

        if (task_rec.task_id < best_task.task_id) {
          best_index = static_cast<int32_t>(i);
        }
      }
      return best_index;
    };

    auto best_data_choice_for_device = [&](devid_t device_id) {
      DataChoice best;
      const auto device_index = static_cast<std::size_t>(device_id);
      for (const auto data_idx : device_plans[device_index].frontier_data_indices) {
        if (is_local_on_device(data_idx, device_id)) {
          continue;
        }

        const auto &data_rec = data_records[static_cast<std::size_t>(data_idx)];
        DataChoice current;
        current.valid = true;
        current.data_index = data_idx;
        current.transfer_time = transfer_time_for(data_idx, device_id);
        current.data_id = data_rec.data_id;

        for (const auto task_index : data_rec.reader_task_indices) {
          const auto &task_rec = task_records[static_cast<std::size_t>(task_index)];
          if (!task_rec.active) {
            continue;
          }
          if (would_be_free_if_data_present(task_rec, device_id, data_idx)) {
            ++current.newly_free_count;
            current.newly_free_priority_sum += task_rec.priority;
          }
        }

        if (current.newly_free_count == 0) {
          continue;
        }

        if (!best.valid || current.newly_free_count > best.newly_free_count ||
            (current.newly_free_count == best.newly_free_count &&
             current.transfer_time < best.transfer_time) ||
            (current.newly_free_count == best.newly_free_count &&
             current.transfer_time == best.transfer_time &&
             current.newly_free_priority_sum > best.newly_free_priority_sum) ||
            (current.newly_free_count == best.newly_free_count &&
             current.transfer_time == best.transfer_time &&
             current.newly_free_priority_sum == best.newly_free_priority_sum &&
             current.data_id < best.data_id)) {
          best = current;
        }
      }
      return best;
    };

    auto best_task_for_data_choice = [&](const DataChoice &choice, devid_t device_id) {
      int32_t best_index = -1;
      const auto &data_rec = data_records[static_cast<std::size_t>(choice.data_index)];
      for (const auto task_index : data_rec.reader_task_indices) {
        const auto &task_rec = task_records[static_cast<std::size_t>(task_index)];
        if (!task_rec.active ||
            !would_be_free_if_data_present(task_rec, device_id, choice.data_index)) {
          continue;
        }

        if (best_index < 0) {
          best_index = task_index;
          continue;
        }

        const auto &best_task = task_records[static_cast<std::size_t>(best_index)];
        if (task_rec.priority != best_task.priority) {
          if (task_rec.priority > best_task.priority) {
            best_index = task_index;
          }
          continue;
        }

        const auto task_transfer = total_transfer_time(task_rec, device_id);
        const auto best_transfer = total_transfer_time(best_task, device_id);
        if (task_transfer != best_transfer) {
          if (task_transfer < best_transfer) {
            best_index = task_index;
          }
          continue;
        }

        if (task_rec.task_id < best_task.task_id) {
          best_index = task_index;
        }
      }
      return best_index;
    };

    auto proposal_for_device = [&](devid_t device_id) {
      Proposal proposal;
      const auto device_index = static_cast<std::size_t>(device_id);

      if (device_plans[device_index].planned_task_indices.empty()) {
        const auto seed_task = best_seed_task_for_device(device_id);
        if (seed_task >= 0) {
          const auto &task_rec = task_records[static_cast<std::size_t>(seed_task)];
          proposal.valid = true;
          proposal.task_index = seed_task;
          proposal.device_id = device_id;
          proposal.newly_free_count = 1;
          proposal.transfer_time = total_transfer_time(task_rec, device_id);
          proposal.priority = task_rec.priority;
          proposal.task_id = task_rec.task_id;
        }
        return proposal;
      }

      const auto free_task = best_free_task_for_device(device_id);
      if (free_task >= 0) {
        const auto &task_rec = task_records[static_cast<std::size_t>(free_task)];
        proposal.valid = true;
        proposal.task_index = free_task;
        proposal.device_id = device_id;
        proposal.newly_free_count = 0;
        proposal.transfer_time = total_transfer_time(task_rec, device_id);
        proposal.priority = task_rec.priority;
        proposal.task_id = task_rec.task_id;
        return proposal;
      }

      const auto data_choice = best_data_choice_for_device(device_id);
      if (data_choice.valid) {
        const auto task_index = best_task_for_data_choice(data_choice, device_id);
        if (task_index >= 0) {
          const auto &task_rec = task_records[static_cast<std::size_t>(task_index)];
          proposal.valid = true;
          proposal.task_index = task_index;
          proposal.device_id = device_id;
          proposal.newly_free_count = data_choice.newly_free_count;
          proposal.transfer_time = total_transfer_time(task_rec, device_id);
          proposal.priority = task_rec.priority;
          proposal.task_id = task_rec.task_id;
          return proposal;
        }
      }

      const auto fallback_task = best_seed_task_for_device(device_id);
      if (fallback_task >= 0) {
        const auto &task_rec = task_records[static_cast<std::size_t>(fallback_task)];
        proposal.valid = true;
        proposal.task_index = fallback_task;
        proposal.device_id = device_id;
        proposal.newly_free_count = 0;
        proposal.transfer_time = total_transfer_time(task_rec, device_id);
        proposal.priority = task_rec.priority;
        proposal.task_id = task_rec.task_id;
      }
      return proposal;
    };

    auto better_global_proposal = [](const Proposal &lhs, const Proposal &rhs) {
      if (!rhs.valid) {
        return lhs.valid;
      }
      if (!lhs.valid) {
        return false;
      }
      if (lhs.priority != rhs.priority) {
        return lhs.priority > rhs.priority;
      }
      if (lhs.newly_free_count != rhs.newly_free_count) {
        return lhs.newly_free_count > rhs.newly_free_count;
      }
      if (lhs.transfer_time != rhs.transfer_time) {
        return lhs.transfer_time < rhs.transfer_time;
      }
      if (lhs.device_id != rhs.device_id) {
        return lhs.device_id < rhs.device_id;
      }
      return lhs.task_id < rhs.task_id;
    };

    while (action_buffer.size() < task_records.size()) {
      Proposal best_proposal;
      for (std::size_t device = 0; device < n_devices; ++device) {
        const auto current = proposal_for_device(static_cast<devid_t>(device));
        if (better_global_proposal(current, best_proposal)) {
          best_proposal = current;
        }
      }

      T4F_INVARIANT(best_proposal.valid);
      auto &task_rec = task_records[static_cast<std::size_t>(best_proposal.task_index)];
      task_rec.active = false;

      const auto device_index = static_cast<std::size_t>(best_proposal.device_id);
      device_plans[device_index].planned_task_indices.push_back(best_proposal.task_index);

      for (const auto data_idx : task_rec.read_data_indices) {
        auto &data_rec = data_records[static_cast<std::size_t>(data_idx)];
        ++data_rec.planned_count[device_index];
        data_rec.local_present[device_index] = 1;
      }

      for (const auto data_idx : task_rec.unique_data_indices) {
        data_records[static_cast<std::size_t>(data_idx)].local_present[device_index] = 1;
      }

      action_buffer.push_back(
          Action{task_rec.input_pos, best_proposal.device_id, task_rec.priority, task_rec.priority});
    }

    return action_buffer;
  }

public:
  DataAwareMapper() = default;

  DataAwareMapper(const DataAwareMapper &other) = default;

  DataAwareMapper(std::size_t n_tasks, std::size_t n_devices) {
    MONUnusedParameter(n_tasks);
    MONUnusedParameter(n_devices);
  }

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    auto &actions = plan_tasks(std::span<const taskid_t>(&task_id, 1), state);
    T4F_INVARIANT(actions.size() == 1);
    return actions.front();
  }

  ActionList &map_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) override {
    return plan_tasks(task_ids, state);
  }
};

class KaHyParMapper : public EFTMapper {
private:
  struct CandidateRec {
    taskid_t task_id = -1;
    std::size_t input_pos = 0;
    priority_t priority = 0;
    DeviceIDList supported_devices;
    int32_t gpu_vertex = -1;
    bool gpu_eligible = false;
  };

  struct PartitionRec {
    int32_t label = -1;
    int32_t aligned_label = -1;
    std::vector<int32_t> candidate_indices;
    DataIDList unique_read_data;
    priority_t total_priority = 0;
    taskid_t min_task_id = -1;
    devid_t previous_device = -1;
  };

  struct HistoricalPartitionRec {
    int32_t aligned_label = -1;
    DataIDList unique_read_data;
    devid_t device_id = -1;
    taskid_t min_task_id = -1;
  };

  KaHyPar_wrapper kahypar;
  std::vector<CandidateRec> candidate_records;
  std::vector<int32_t> task_to_candidate_index;
  std::vector<int32_t> task_to_gpu_vertex;
  TaskIDList touched_candidate_tasks;
  TaskIDList touched_gpu_tasks;
  std::vector<int32_t> vertex_to_candidate_index;
  std::vector<int32_t> partition_labels;
  std::vector<PartitionRec> partitions;
  std::vector<timecount_t> partition_costs;
  std::vector<devid_t> partition_devices;
  std::vector<devid_t> eligible_devices_buffer;
  std::vector<timecount_t> batch_device_available_time;
  ankerl::unordered_dense::map<dataid_t, timecount_t> average_transfer_cost_cache;
  std::vector<HistoricalPartitionRec> previous_partitions;
  KaHyParHypergraph last_hypergraph;
  int32_t next_aligned_label = 0;

  void ensure_scratch_sizes(std::size_t n_candidates, std::size_t n_compute_tasks,
                            std::size_t n_devices) {
    candidate_records.reserve(n_candidates);
    touched_candidate_tasks.reserve(n_candidates);
    touched_gpu_tasks.reserve(n_candidates);
    vertex_to_candidate_index.reserve(n_candidates);
    partition_labels.reserve(n_candidates);
    partitions.reserve(n_candidates);
    eligible_devices_buffer.reserve(n_devices);
    average_transfer_cost_cache.reserve(n_candidates);

    if (task_to_candidate_index.size() < n_compute_tasks) {
      task_to_candidate_index.resize(n_compute_tasks, -1);
    }
    if (task_to_gpu_vertex.size() < n_compute_tasks) {
      task_to_gpu_vertex.resize(n_compute_tasks, -1);
    }
    if (batch_device_available_time.size() < n_devices) {
      batch_device_available_time.resize(n_devices, 0);
    }
  }

  void clear_candidate_maps() {
    for (const auto task_id : touched_candidate_tasks) {
      const auto idx = static_cast<std::size_t>(task_id);
      T4F_INVARIANT(idx < task_to_candidate_index.size());
      task_to_candidate_index[idx] = -1;
    }
    for (const auto task_id : touched_gpu_tasks) {
      const auto idx = static_cast<std::size_t>(task_id);
      T4F_INVARIANT(idx < task_to_gpu_vertex.size());
      task_to_gpu_vertex[idx] = -1;
    }
    touched_candidate_tasks.clear();
    touched_gpu_tasks.clear();
  }

  void collect_eligible_devices(const SchedulerState &state) {
    eligible_devices_buffer.clear();
    const auto n_devices = state.get_devices().size();
    for (devid_t device_id = 1; device_id < n_devices; ++device_id) {
      if (state.get_devices().get_type(device_id) != DeviceType::GPU) {
        continue;
      }
      eligible_devices_buffer.push_back(device_id);
    }
  }

  void reset_partition_history() {
    average_transfer_cost_cache.clear();
    previous_partitions.clear();
    next_aligned_label = 0;
  }

  void prepare_candidates(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    clear_candidate_maps();
    candidate_records.clear();
    vertex_to_candidate_index.clear();
    last_hypergraph.clear();
    average_transfer_cost_cache.clear();

    const auto n_compute_tasks =
        static_cast<std::size_t>(state.get_tasks().get_n_compute_tasks());
    const auto n_devices = static_cast<std::size_t>(state.get_devices().size());
    ensure_scratch_sizes(task_ids.size(), n_compute_tasks, n_devices);
    collect_eligible_devices(state);

    for (std::size_t input_pos = 0; input_pos < task_ids.size(); ++input_pos) {
      const auto task_id = task_ids[input_pos];
      CandidateRec candidate;
      candidate.task_id = task_id;
      candidate.input_pos = input_pos;
      candidate.priority = state.get_mapping_priority(task_id);
      fill_device_targets(task_id, state);
      candidate.supported_devices = device_buffer;
      T4F_INVARIANT(!candidate.supported_devices.empty());

      for (const auto device_id : candidate.supported_devices) {
        if (device_id > 0 && std::binary_search(eligible_devices_buffer.begin(),
                                                eligible_devices_buffer.end(), device_id)) {
          candidate.gpu_eligible = true;
          break;
        }
      }

      const auto candidate_index = static_cast<int32_t>(candidate_records.size());
      candidate_records.push_back(std::move(candidate));

      const auto task_idx = static_cast<std::size_t>(task_id);
      T4F_INVARIANT(task_idx < task_to_candidate_index.size());
      task_to_candidate_index[task_idx] = candidate_index;
      touched_candidate_tasks.push_back(task_id);
    }

    last_hypergraph.eptr.push_back(0);
    for (auto &candidate : candidate_records) {
      if (!candidate.gpu_eligible) {
        continue;
      }
      candidate.gpu_vertex = static_cast<int32_t>(vertex_to_candidate_index.size());
      vertex_to_candidate_index.push_back(
          task_to_candidate_index[static_cast<std::size_t>(candidate.task_id)]);
      last_hypergraph.vertex_task_ids.push_back(candidate.task_id);
      last_hypergraph.vertex_input_positions.push_back(candidate.input_pos);
      last_hypergraph.vwgts.push_back(KaHyPar_wrapper::clamp_weight(
          static_cast<uint64_t>(state.get_tasks().get_mean_duration(candidate.task_id,
                                                                    DeviceType::GPU))));

      const auto task_idx = static_cast<std::size_t>(candidate.task_id);
      T4F_INVARIANT(task_idx < task_to_gpu_vertex.size());
      task_to_gpu_vertex[task_idx] = candidate.gpu_vertex;
      touched_gpu_tasks.push_back(candidate.task_id);
    }

    last_hypergraph.num_vertices = static_cast<int32_t>(vertex_to_candidate_index.size());
  }

  [[nodiscard]] timecount_t average_transfer_cost_for_data(dataid_t data_id,
                                                           const SchedulerState &state) {
    if (const auto it = average_transfer_cost_cache.find(data_id);
        it != average_transfer_cost_cache.end()) {
      return it->second;
    }

    if (eligible_devices_buffer.empty()) {
      const auto fallback_cost = static_cast<timecount_t>(state.get_data().get_size(data_id));
      average_transfer_cost_cache.emplace(data_id, fallback_cost);
      return fallback_cost;
    }

    timecount_t total_cost = 0;
    for (const auto device_id : eligible_devices_buffer) {
      const auto transfer_cost = transfer_time_for_data(data_id, device_id, state);
      if (transfer_cost >= MAX_TIME) {
        average_transfer_cost_cache.emplace(data_id, MAX_TIME);
        return MAX_TIME;
      }
      total_cost = saturating_add(total_cost, transfer_cost);
    }

    const auto average_cost = total_cost / static_cast<timecount_t>(eligible_devices_buffer.size());
    average_transfer_cost_cache.emplace(data_id, average_cost);
    return average_cost;
  }

  void build_hypergraph(const SchedulerState &state) {
    if (last_hypergraph.num_vertices <= 0) {
      return;
    }

    const auto &tasks = state.get_tasks();
    const auto &data = state.get_data();

    for (const auto data_id : tasks.get_read_usage_data_ids()) {
      const auto readers = tasks.get_tasks_reading_data_by_gen(data_id);
      const auto generations = tasks.get_read_generations_for_data(data_id);
      T4F_INVARIANT(readers.size() == generations.size());

      std::size_t group_begin = 0;
      while (group_begin < readers.size()) {
        const auto generation = generations[group_begin];
        std::size_t group_end = group_begin + 1;
        while (group_end < readers.size() && generations[group_end] == generation) {
          ++group_end;
        }

        const auto edge_begin = last_hypergraph.eind.size();
        for (std::size_t idx = group_begin; idx < group_end; ++idx) {
          const auto task_id = readers[idx];
          if (task_id < 0) {
            continue;
          }
          const auto task_idx = static_cast<std::size_t>(task_id);
          if (task_idx >= task_to_gpu_vertex.size()) {
            continue;
          }
          const auto vertex = task_to_gpu_vertex[task_idx];
          if (vertex >= 0) {
            last_hypergraph.eind.push_back(vertex);
          }
        }

        const auto edge_size =
            static_cast<int32_t>(last_hypergraph.eind.size() - edge_begin);
        if (edge_size >= 2) {
          last_hypergraph.eptr.push_back(static_cast<int32_t>(last_hypergraph.eind.size()));
          last_hypergraph.hewgts.push_back(KaHyPar_wrapper::clamp_weight(
              static_cast<uint64_t>(average_transfer_cost_for_data(data_id, state))));
          last_hypergraph.edge_data_ids.push_back(data_id);
          last_hypergraph.edge_generations.push_back(generation);
        } else {
          last_hypergraph.eind.resize(edge_begin);
        }

        group_begin = group_end;
      }
    }
  }

  void reset_batch_device_times(const SchedulerState &state) {
    const auto n_devices = state.get_devices().size();
    batch_device_available_time.resize(n_devices, 0);
    for (devid_t device_id = 0; device_id < n_devices; ++device_id) {
      batch_device_available_time[static_cast<std::size_t>(device_id)] =
          EFTMapper::get_device_available_time(device_id, state);
    }
  }

  [[nodiscard]] timecount_t saturating_add(timecount_t lhs, timecount_t rhs) const {
    if (lhs >= MAX_TIME || rhs >= MAX_TIME) {
      return MAX_TIME;
    }
    if (lhs > MAX_TIME - rhs) {
      return MAX_TIME;
    }
    return lhs + rhs;
  }

  [[nodiscard]] timecount_t transfer_time_for_data(dataid_t data_id, devid_t device_id,
                                                   const SchedulerState &state) const {
    const auto &data_manager = state.get_data_manager();
    if (data_manager.check_valid_mapped(data_id, device_id)) {
      return 0;
    }

    const auto &communication_manager = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto flags = data_manager.get_mapped_location_flags(data_id);
    const auto req = communication_manager.get_best_source(topology, device_id, flags);
    if (!req.found) {
      return MAX_TIME;
    }

    return communication_manager.ideal_time_to_transfer(
        topology, state.get_data().get_size(data_id), req.source, device_id);
  }

  [[nodiscard]] DeviceTime select_best_device(const CandidateRec &candidate,
                                              const SchedulerState &state) {
    const auto dep_time = get_dependency_finish_time(candidate.task_id, state);
    timecount_t best_finish = MAX_TIME;
    devid_t best_device = candidate.supported_devices.front();

    for (const auto device_id : candidate.supported_devices) {
      const auto start_time =
          std::max(batch_device_available_time[static_cast<std::size_t>(device_id)], dep_time);
      const auto finish_time = get_finish_time(candidate.task_id, device_id, start_time, state);
      if (finish_time < best_finish ||
          (finish_time == best_finish && device_id < best_device)) {
        best_finish = finish_time;
        best_device = device_id;
      }
    }

    return {best_device, best_finish};
  }

  void record_assignment(taskid_t task_id, devid_t device_id, const SchedulerState &state) {
    const auto dep_time = get_dependency_finish_time(task_id, state);
    const auto start_time =
        std::max(batch_device_available_time[static_cast<std::size_t>(device_id)], dep_time);
    const auto finish_time = get_finish_time(task_id, device_id, start_time, state);
    record_finish_time(task_id, finish_time, state);
    batch_device_available_time[static_cast<std::size_t>(device_id)] = finish_time;
  }

  void emit_fallback_actions(const std::vector<int32_t> &candidate_indices,
                             const SchedulerState &state) {
    std::vector<int32_t> ordered = candidate_indices;
    std::sort(ordered.begin(), ordered.end(), [&](int32_t lhs, int32_t rhs) {
      const auto &lhs_candidate = candidate_records[static_cast<std::size_t>(lhs)];
      const auto &rhs_candidate = candidate_records[static_cast<std::size_t>(rhs)];
      return lhs_candidate.input_pos < rhs_candidate.input_pos;
    });

    for (const auto candidate_index : ordered) {
      const auto &candidate = candidate_records[static_cast<std::size_t>(candidate_index)];
      const auto selection = select_best_device(candidate, state);
      action_buffer.push_back(
          Action{candidate.input_pos, selection.device_id, candidate.priority, candidate.priority});
      record_assignment(candidate.task_id, selection.device_id, state);
    }
  }

  void build_partitions(const SchedulerState &state) {
    partitions.clear();
    partition_devices.clear();
    partition_costs.clear();

    ankerl::unordered_dense::map<int32_t, int32_t> partition_index_by_label;
    partition_index_by_label.reserve(vertex_to_candidate_index.size());
    const auto &tasks = state.get_tasks();

    for (std::size_t vertex = 0; vertex < vertex_to_candidate_index.size(); ++vertex) {
      const auto raw_label = partition_labels[vertex];
      const auto [it, inserted] = partition_index_by_label.try_emplace(
          raw_label, static_cast<int32_t>(partitions.size()));
      if (inserted) {
        PartitionRec rec;
        rec.label = raw_label;
        rec.aligned_label = raw_label;
        partitions.push_back(std::move(rec));
      }

      auto &partition = partitions[static_cast<std::size_t>(it->second)];
      const auto candidate_index = vertex_to_candidate_index[vertex];
      const auto &candidate = candidate_records[static_cast<std::size_t>(candidate_index)];
      partition.candidate_indices.push_back(candidate_index);
      partition.total_priority += candidate.priority;
      if (partition.min_task_id < 0 || candidate.task_id < partition.min_task_id) {
        partition.min_task_id = candidate.task_id;
      }

      const auto reads = tasks.get_read(candidate.task_id);
      partition.unique_read_data.insert(partition.unique_read_data.end(), reads.begin(), reads.end());
    }

    for (auto &partition : partitions) {
      std::sort(partition.unique_read_data.begin(), partition.unique_read_data.end());
      partition.unique_read_data.erase(
          std::unique(partition.unique_read_data.begin(), partition.unique_read_data.end()),
          partition.unique_read_data.end());
    }
  }

  [[nodiscard]] timecount_t weighted_data_overlap(std::span<const dataid_t> lhs,
                                                  std::span<const dataid_t> rhs,
                                                  const SchedulerState &state) {
    std::size_t lhs_index = 0;
    std::size_t rhs_index = 0;
    timecount_t overlap = 0;

    while (lhs_index < lhs.size() && rhs_index < rhs.size()) {
      const auto lhs_data = lhs[lhs_index];
      const auto rhs_data = rhs[rhs_index];
      if (lhs_data == rhs_data) {
        overlap = saturating_add(overlap, average_transfer_cost_for_data(lhs_data, state));
        ++lhs_index;
        ++rhs_index;
      } else if (lhs_data < rhs_data) {
        ++lhs_index;
      } else {
        ++rhs_index;
      }
    }

    return overlap;
  }

  void align_partitions(const SchedulerState &state) {
    if (partitions.empty()) {
      return;
    }

    struct AlignmentChoice {
      std::size_t current_partition = 0;
      std::size_t previous_partition = 0;
      timecount_t overlap = 0;
    };

    std::vector<AlignmentChoice> choices;
    choices.reserve(partitions.size() * previous_partitions.size());
    for (std::size_t current_index = 0; current_index < partitions.size(); ++current_index) {
      auto &partition = partitions[current_index];
      partition.previous_device = -1;
      for (std::size_t previous_index = 0; previous_index < previous_partitions.size();
           ++previous_index) {
        const auto overlap = weighted_data_overlap(partition.unique_read_data,
                                                   previous_partitions[previous_index].unique_read_data,
                                                   state);
        if (overlap > 0) {
          choices.push_back({current_index, previous_index, overlap});
        }
      }
    }

    std::sort(choices.begin(), choices.end(), [&](const AlignmentChoice &lhs,
                                                  const AlignmentChoice &rhs) {
      if (lhs.overlap != rhs.overlap) {
        return lhs.overlap > rhs.overlap;
      }
      const auto lhs_label = previous_partitions[lhs.previous_partition].aligned_label;
      const auto rhs_label = previous_partitions[rhs.previous_partition].aligned_label;
      if (lhs_label != rhs_label) {
        return lhs_label < rhs_label;
      }
      return partitions[lhs.current_partition].min_task_id <
             partitions[rhs.current_partition].min_task_id;
    });

    std::vector<uint8_t> current_matched(partitions.size(), 0);
    std::vector<uint8_t> previous_matched(previous_partitions.size(), 0);
    for (const auto &choice : choices) {
      if (current_matched[choice.current_partition] ||
          previous_matched[choice.previous_partition]) {
        continue;
      }

      auto &partition = partitions[choice.current_partition];
      const auto &previous = previous_partitions[choice.previous_partition];
      partition.aligned_label = previous.aligned_label;
      partition.previous_device = previous.device_id;
      current_matched[choice.current_partition] = 1;
      previous_matched[choice.previous_partition] = 1;
    }

    for (auto &partition : partitions) {
      if (partition.aligned_label >= 0 && partition.previous_device >= 0) {
        continue;
      }
      partition.aligned_label = next_aligned_label++;
    }
  }

  void score_partitions(const SchedulerState &state) {
    partition_costs.assign(partitions.size() * eligible_devices_buffer.size(), MAX_TIME);

    for (std::size_t partition_index = 0; partition_index < partitions.size(); ++partition_index) {
      const auto &partition = partitions[partition_index];
      for (std::size_t device_index = 0; device_index < eligible_devices_buffer.size();
           ++device_index) {
        const auto device_id = eligible_devices_buffer[device_index];
        timecount_t total_cost = 0;
        for (const auto data_id : partition.unique_read_data) {
          total_cost = saturating_add(total_cost, transfer_time_for_data(data_id, device_id, state));
        }
        partition_costs[partition_index * eligible_devices_buffer.size() + device_index] =
            total_cost;
      }
    }
  }

  [[nodiscard]] bool better_partition_device_choice(std::size_t lhs_partition,
                                                    std::size_t lhs_device,
                                                    std::size_t rhs_partition,
                                                    std::size_t rhs_device) const {
    if (rhs_partition == static_cast<std::size_t>(-1)) {
      return true;
    }

    const auto lhs_cost =
        partition_costs[lhs_partition * eligible_devices_buffer.size() + lhs_device];
    const auto rhs_cost =
        partition_costs[rhs_partition * eligible_devices_buffer.size() + rhs_device];
    if (lhs_cost != rhs_cost) {
      return lhs_cost < rhs_cost;
    }

    const auto lhs_size = partitions[lhs_partition].candidate_indices.size();
    const auto rhs_size = partitions[rhs_partition].candidate_indices.size();
    if (lhs_size != rhs_size) {
      return lhs_size > rhs_size;
    }

    if (partitions[lhs_partition].total_priority != partitions[rhs_partition].total_priority) {
      return partitions[lhs_partition].total_priority > partitions[rhs_partition].total_priority;
    }

    const auto lhs_device_id = eligible_devices_buffer[lhs_device];
    const auto rhs_device_id = eligible_devices_buffer[rhs_device];
    const bool lhs_matches_previous = partitions[lhs_partition].previous_device == lhs_device_id;
    const bool rhs_matches_previous = partitions[rhs_partition].previous_device == rhs_device_id;
    if (lhs_matches_previous != rhs_matches_previous) {
      return lhs_matches_previous;
    }

    if (lhs_device_id != rhs_device_id) {
      return lhs_device_id < rhs_device_id;
    }

    if (partitions[lhs_partition].aligned_label != partitions[rhs_partition].aligned_label) {
      return partitions[lhs_partition].aligned_label < partitions[rhs_partition].aligned_label;
    }

    return partitions[lhs_partition].min_task_id < partitions[rhs_partition].min_task_id;
  }

  void assign_partition_devices(const SchedulerState &state) {
    MONUnusedParameter(state);
    score_partitions(state);
    partition_devices.assign(partitions.size(), devid_t{-1});
    std::vector<uint8_t> partition_assigned(partitions.size(), 0);
    std::vector<uint8_t> device_used(eligible_devices_buffer.size(), 0);

    for (std::size_t assigned_count = 0; assigned_count < partitions.size(); ++assigned_count) {
      std::size_t best_partition = static_cast<std::size_t>(-1);
      std::size_t best_device = static_cast<std::size_t>(-1);

      for (std::size_t partition_index = 0; partition_index < partitions.size(); ++partition_index) {
        if (partition_assigned[partition_index]) {
          continue;
        }
        for (std::size_t device_index = 0; device_index < eligible_devices_buffer.size();
             ++device_index) {
          if (device_used[device_index]) {
            continue;
          }
          if (better_partition_device_choice(partition_index, device_index, best_partition,
                                             best_device)) {
            best_partition = partition_index;
            best_device = device_index;
          }
        }
      }

      T4F_INVARIANT(best_partition != static_cast<std::size_t>(-1));
      T4F_INVARIANT(best_device != static_cast<std::size_t>(-1));
      partition_assigned[best_partition] = 1;
      device_used[best_device] = 1;
      partition_devices[best_partition] = eligible_devices_buffer[best_device];
    }
  }

  void emit_partition_actions(const SchedulerState &state) {
    std::vector<std::size_t> partition_order(partitions.size(), 0);
    std::iota(partition_order.begin(), partition_order.end(), std::size_t{0});
    std::sort(partition_order.begin(), partition_order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                if (partition_devices[lhs] != partition_devices[rhs]) {
                  return partition_devices[lhs] < partition_devices[rhs];
                }
                if (partitions[lhs].aligned_label != partitions[rhs].aligned_label) {
                  return partitions[lhs].aligned_label < partitions[rhs].aligned_label;
                }
                return partitions[lhs].min_task_id < partitions[rhs].min_task_id;
              });

    for (const auto partition_index : partition_order) {
      auto &partition = partitions[partition_index];
      std::sort(partition.candidate_indices.begin(), partition.candidate_indices.end(),
                [&](int32_t lhs, int32_t rhs) {
                  const auto &lhs_candidate =
                      candidate_records[static_cast<std::size_t>(lhs)];
                  const auto &rhs_candidate =
                      candidate_records[static_cast<std::size_t>(rhs)];
                  if (lhs_candidate.priority != rhs_candidate.priority) {
                    return lhs_candidate.priority > rhs_candidate.priority;
                  }
                  return lhs_candidate.task_id < rhs_candidate.task_id;
                });

      const auto device_id = partition_devices[partition_index];
      for (const auto candidate_index : partition.candidate_indices) {
        const auto &candidate = candidate_records[static_cast<std::size_t>(candidate_index)];
        action_buffer.push_back(
            Action{candidate.input_pos, device_id, candidate.priority, candidate.priority});
        record_assignment(candidate.task_id, device_id, state);
      }
    }
  }

  void persist_partition_history() {
    previous_partitions.clear();
    previous_partitions.reserve(partitions.size());
    for (std::size_t partition_index = 0; partition_index < partitions.size(); ++partition_index) {
      HistoricalPartitionRec rec;
      rec.aligned_label = partitions[partition_index].aligned_label;
      rec.unique_read_data = partitions[partition_index].unique_read_data;
      rec.device_id = partition_devices[partition_index];
      rec.min_task_id = partitions[partition_index].min_task_id;
      previous_partitions.push_back(std::move(rec));
    }
    std::sort(previous_partitions.begin(), previous_partitions.end(),
              [](const HistoricalPartitionRec &lhs, const HistoricalPartitionRec &rhs) {
                if (lhs.aligned_label != rhs.aligned_label) {
                  return lhs.aligned_label < rhs.aligned_label;
                }
                return lhs.min_task_id < rhs.min_task_id;
              });
  }

  ActionList &plan_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    action_buffer.clear();
    action_buffer.reserve(task_ids.size());
    if (task_ids.empty()) {
      last_hypergraph.clear();
      clear_candidate_maps();
      return action_buffer;
    }

    ensure_task_buffers_size(static_cast<std::size_t>(state.get_tasks().get_n_compute_tasks()));
    if (should_reset_for_new_run(state)) {
      reset_task_buffers();
      reset_partition_history();
    }

    prepare_candidates(task_ids, state);
    build_hypergraph(state);
    reset_batch_device_times(state);

    std::vector<int32_t> fallback_candidate_indices;
    fallback_candidate_indices.reserve(candidate_records.size());
    for (std::size_t i = 0; i < candidate_records.size(); ++i) {
      if (!candidate_records[i].gpu_eligible) {
        fallback_candidate_indices.push_back(static_cast<int32_t>(i));
      }
    }

    const auto nparts = static_cast<int32_t>(
        std::min(eligible_devices_buffer.size(), vertex_to_candidate_index.size()));
    const bool should_fallback_all =
        eligible_devices_buffer.empty() || last_hypergraph.num_vertices <= 1 ||
        last_hypergraph.num_hyperedges() == 0 || nparts <= 1;
    if (should_fallback_all) {
      fallback_candidate_indices.clear();
      fallback_candidate_indices.reserve(candidate_records.size());
      for (std::size_t i = 0; i < candidate_records.size(); ++i) {
        fallback_candidate_indices.push_back(static_cast<int32_t>(i));
      }
      emit_fallback_actions(fallback_candidate_indices, state);
      return action_buffer;
    }

    partition_labels.assign(vertex_to_candidate_index.size(), 0);
    const bool partitioned =
        kahypar.call_kahypar_partition(last_hypergraph, nparts, partition_labels);
    if (!partitioned) {
      SPDLOG_WARN("KaHyPar call failed, falling back to EFT-style mapping for this batch");
      fallback_candidate_indices.clear();
      fallback_candidate_indices.reserve(candidate_records.size());
      for (std::size_t i = 0; i < candidate_records.size(); ++i) {
        fallback_candidate_indices.push_back(static_cast<int32_t>(i));
      }
      emit_fallback_actions(fallback_candidate_indices, state);
      return action_buffer;
    }

    build_partitions(state);
    align_partitions(state);
    assign_partition_devices(state);
    persist_partition_history();
    emit_partition_actions(state);
    emit_fallback_actions(fallback_candidate_indices, state);
    return action_buffer;
  }

public:
  KaHyParMapper() {
#ifndef ENABLE_KAHYPAR
    throw std::runtime_error("KaHyParMapper requires ENABLE_KAHYPAR at build time");
#endif
  }

  KaHyParMapper(const KaHyParMapper &other) = default;

  KaHyParMapper(std::size_t n_tasks, std::size_t n_devices) {
#ifndef ENABLE_KAHYPAR
    MONUnusedParameter(n_tasks);
    MONUnusedParameter(n_devices);
    throw std::runtime_error("KaHyParMapper requires ENABLE_KAHYPAR at build time");
#else
    candidate_records.reserve(n_tasks);
    touched_candidate_tasks.reserve(n_tasks);
    touched_gpu_tasks.reserve(n_tasks);
    vertex_to_candidate_index.reserve(n_tasks);
    partition_labels.reserve(n_tasks);
    partitions.reserve(std::min(n_tasks, n_devices));
    eligible_devices_buffer.reserve(n_devices);
    batch_device_available_time.reserve(n_devices);
#endif
  }

  [[nodiscard]] const KaHyParHypergraph &get_last_hypergraph() const {
    return last_hypergraph;
  }

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    auto &actions = plan_tasks(std::span<const taskid_t>(&task_id, 1), state);
    T4F_INVARIANT(actions.size() == 1);
    return actions.front();
  }

  ActionList &map_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) override {
    return plan_tasks(task_ids, state);
  }
};

class METISMapper : public EFTMapper {
private:
  struct CandidateRec {
    taskid_t task_id = -1;
    std::size_t input_pos = 0;
    priority_t priority = 0;
    DeviceIDList supported_devices;
    int32_t gpu_vertex = -1;
    bool gpu_eligible = false;
  };

  struct PartitionRec {
    int32_t label = -1;
    std::vector<int32_t> candidate_indices;
    DataIDList unique_read_data;
    priority_t total_priority = 0;
    taskid_t min_task_id = -1;
  };

  struct AssignmentSolution {
    bool feasible = false;
    timecount_t total_cost = MAX_TIME;
    std::vector<uint8_t> device_suffix;
  };

  METIS_wrapper metis;
  std::vector<CandidateRec> candidate_records;
  std::vector<int32_t> task_to_candidate_index;
  std::vector<int32_t> task_to_gpu_vertex;
  TaskIDList touched_candidate_tasks;
  TaskIDList touched_gpu_tasks;
  std::vector<int32_t> vertex_to_candidate_index;
  std::vector<int32_t> partition_labels;
  std::vector<PartitionRec> partitions;
  std::vector<timecount_t> partition_costs;
  std::vector<devid_t> partition_devices;
  std::vector<devid_t> eligible_devices_buffer;
  std::vector<timecount_t> batch_device_available_time;
  std::vector<std::size_t> assignment_partition_order;
  std::vector<AssignmentSolution> assignment_memo;
  std::vector<uint8_t> assignment_memo_ready;
  MetisGraph last_graph;

  void ensure_scratch_sizes(std::size_t n_candidates, std::size_t n_compute_tasks,
                            std::size_t n_devices) {
    candidate_records.reserve(n_candidates);
    touched_candidate_tasks.reserve(n_candidates);
    touched_gpu_tasks.reserve(n_candidates);
    vertex_to_candidate_index.reserve(n_candidates);
    partition_labels.reserve(n_candidates);
    partitions.reserve(n_candidates);
    eligible_devices_buffer.reserve(n_devices);
    assignment_partition_order.reserve(n_candidates);

    if (task_to_candidate_index.size() < n_compute_tasks) {
      task_to_candidate_index.resize(n_compute_tasks, -1);
    }
    if (task_to_gpu_vertex.size() < n_compute_tasks) {
      task_to_gpu_vertex.resize(n_compute_tasks, -1);
    }
    if (batch_device_available_time.size() < n_devices) {
      batch_device_available_time.resize(n_devices, 0);
    }
  }

  void clear_candidate_maps() {
    for (const auto task_id : touched_candidate_tasks) {
      const auto idx = static_cast<std::size_t>(task_id);
      T4F_INVARIANT(idx < task_to_candidate_index.size());
      task_to_candidate_index[idx] = -1;
    }
    for (const auto task_id : touched_gpu_tasks) {
      const auto idx = static_cast<std::size_t>(task_id);
      T4F_INVARIANT(idx < task_to_gpu_vertex.size());
      task_to_gpu_vertex[idx] = -1;
    }
    touched_candidate_tasks.clear();
    touched_gpu_tasks.clear();
  }

  void collect_eligible_devices(const SchedulerState &state) {
    eligible_devices_buffer.clear();
    const auto n_devices = state.get_devices().size();
    for (devid_t device_id = 1; device_id < n_devices; ++device_id) {
      if (state.get_devices().get_type(device_id) != DeviceType::GPU) {
        continue;
      }
      eligible_devices_buffer.push_back(device_id);
    }
  }

  void prepare_candidates(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    clear_candidate_maps();
    candidate_records.clear();
    vertex_to_candidate_index.clear();
    last_graph.clear();

    const auto n_compute_tasks =
        static_cast<std::size_t>(state.get_tasks().get_n_compute_tasks());
    const auto n_devices = static_cast<std::size_t>(state.get_devices().size());
    ensure_scratch_sizes(task_ids.size(), n_compute_tasks, n_devices);
    collect_eligible_devices(state);

    for (std::size_t input_pos = 0; input_pos < task_ids.size(); ++input_pos) {
      const auto task_id = task_ids[input_pos];
      CandidateRec candidate;
      candidate.task_id = task_id;
      candidate.input_pos = input_pos;
      candidate.priority = state.get_mapping_priority(task_id);
      fill_device_targets(task_id, state);
      candidate.supported_devices = device_buffer;
      T4F_INVARIANT(!candidate.supported_devices.empty());

      for (const auto device_id : candidate.supported_devices) {
        if (device_id > 0 && std::binary_search(eligible_devices_buffer.begin(),
                                                eligible_devices_buffer.end(), device_id)) {
          candidate.gpu_eligible = true;
          break;
        }
      }

      const auto candidate_index = static_cast<int32_t>(candidate_records.size());
      candidate_records.push_back(std::move(candidate));

      const auto task_idx = static_cast<std::size_t>(task_id);
      T4F_INVARIANT(task_idx < task_to_candidate_index.size());
      task_to_candidate_index[task_idx] = candidate_index;
      touched_candidate_tasks.push_back(task_id);
    }

    for (auto &candidate : candidate_records) {
      if (!candidate.gpu_eligible) {
        continue;
      }
      candidate.gpu_vertex = static_cast<int32_t>(vertex_to_candidate_index.size());
      vertex_to_candidate_index.push_back(
          task_to_candidate_index[static_cast<std::size_t>(candidate.task_id)]);
      last_graph.vertex_task_ids.push_back(candidate.task_id);
      last_graph.vertex_input_positions.push_back(candidate.input_pos);
      last_graph.vwgts.push_back(METIS_wrapper::clamp_weight(
          static_cast<uint64_t>(state.get_tasks().get_mean_duration(candidate.task_id,
                                                                    DeviceType::GPU))));

      const auto task_idx = static_cast<std::size_t>(candidate.task_id);
      T4F_INVARIANT(task_idx < task_to_gpu_vertex.size());
      task_to_gpu_vertex[task_idx] = candidate.gpu_vertex;
      touched_gpu_tasks.push_back(candidate.task_id);
    }

    last_graph.num_vertices = static_cast<int32_t>(vertex_to_candidate_index.size());
  }

  [[nodiscard]] timecount_t saturating_add(timecount_t lhs, timecount_t rhs) const {
    if (lhs >= MAX_TIME || rhs >= MAX_TIME) {
      return MAX_TIME;
    }
    if (lhs > MAX_TIME - rhs) {
      return MAX_TIME;
    }
    return lhs + rhs;
  }

  [[nodiscard]] timecount_t transfer_time_for_data(dataid_t data_id, devid_t device_id,
                                                   const SchedulerState &state) const {
    const auto &data_manager = state.get_data_manager();
    if (data_manager.check_valid_mapped(data_id, device_id)) {
      return 0;
    }

    const auto &communication_manager = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto flags = data_manager.get_mapped_location_flags(data_id);
    const auto req = communication_manager.get_best_source(topology, device_id, flags);
    if (!req.found) {
      return MAX_TIME;
    }

    return communication_manager.ideal_time_to_transfer(
        topology, state.get_data().get_size(data_id), req.source, device_id);
  }

  [[nodiscard]] timecount_t average_transfer_time_for_data(dataid_t data_id,
                                                           const SchedulerState &state) const {
    if (eligible_devices_buffer.empty()) {
      return MAX_TIME;
    }

    timecount_t total_cost = 0;
    for (const auto device_id : eligible_devices_buffer) {
      total_cost = saturating_add(total_cost, transfer_time_for_data(data_id, device_id, state));
    }
    if (total_cost >= MAX_TIME) {
      return MAX_TIME;
    }
    return total_cost / static_cast<timecount_t>(eligible_devices_buffer.size());
  }

  void build_metis_graph(const SchedulerState &state) {
    last_graph.xadj.clear();
    last_graph.adjncy.clear();
    last_graph.adjwgt.clear();
    if (last_graph.num_vertices <= 0) {
      return;
    }

    const auto &tasks = state.get_tasks();
    ankerl::unordered_dense::map<uint64_t, uint64_t> edge_weights;
    edge_weights.reserve(static_cast<std::size_t>(last_graph.num_vertices) * 4);
    std::vector<int32_t> group_vertices;

    for (const auto data_id : tasks.get_read_usage_data_ids()) {
      const auto readers = tasks.get_tasks_reading_data_by_gen(data_id);
      const auto generations = tasks.get_read_generations_for_data(data_id);
      T4F_INVARIANT(readers.size() == generations.size());

      std::size_t group_begin = 0;
      while (group_begin < readers.size()) {
        const auto generation = generations[group_begin];
        std::size_t group_end = group_begin + 1;
        while (group_end < readers.size() && generations[group_end] == generation) {
          ++group_end;
        }

        group_vertices.clear();
        for (std::size_t idx = group_begin; idx < group_end; ++idx) {
          const auto task_id = readers[idx];
          if (task_id < 0) {
            continue;
          }
          const auto task_idx = static_cast<std::size_t>(task_id);
          if (task_idx >= task_to_gpu_vertex.size()) {
            continue;
          }
          const auto vertex = task_to_gpu_vertex[task_idx];
          if (vertex >= 0) {
            group_vertices.push_back(vertex);
          }
        }

        std::sort(group_vertices.begin(), group_vertices.end());
        group_vertices.erase(std::unique(group_vertices.begin(), group_vertices.end()),
                             group_vertices.end());

        if (group_vertices.size() >= 2) {
          const auto avg_transfer_cost = average_transfer_time_for_data(data_id, state);
          const auto group_weight = METIS_wrapper::clamp_weight(
              static_cast<uint64_t>(std::max<timecount_t>(avg_transfer_cost, 1)));

          for (std::size_t i = 0; i + 1 < group_vertices.size(); ++i) {
            for (std::size_t j = i + 1; j < group_vertices.size(); ++j) {
              const auto lhs = static_cast<uint32_t>(group_vertices[i]);
              const auto rhs = static_cast<uint32_t>(group_vertices[j]);
              const auto key = (static_cast<uint64_t>(lhs) << 32) | rhs;
              auto &accum = edge_weights[key];
              accum += static_cast<uint64_t>(group_weight);
            }
          }
        }

        group_begin = group_end;
      }
    }

    std::vector<std::vector<std::pair<int32_t, int32_t>>> adjacency(
        static_cast<std::size_t>(last_graph.num_vertices));
    for (const auto &entry : edge_weights) {
      const auto lhs = static_cast<int32_t>(entry.first >> 32);
      const auto rhs = static_cast<int32_t>(entry.first & 0xffffffffU);
      const auto weight = METIS_wrapper::clamp_weight(entry.second);
      adjacency[static_cast<std::size_t>(lhs)].push_back({rhs, weight});
      adjacency[static_cast<std::size_t>(rhs)].push_back({lhs, weight});
    }

    last_graph.xadj.reserve(static_cast<std::size_t>(last_graph.num_vertices) + 1);
    last_graph.xadj.push_back(0);
    for (auto &neighbors : adjacency) {
      std::sort(neighbors.begin(), neighbors.end(),
                [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
      for (const auto &[neighbor, weight] : neighbors) {
        last_graph.adjncy.push_back(neighbor);
        last_graph.adjwgt.push_back(weight);
      }
      last_graph.xadj.push_back(static_cast<int32_t>(last_graph.adjncy.size()));
    }
  }

  void reset_batch_device_times(const SchedulerState &state) {
    const auto n_devices = state.get_devices().size();
    batch_device_available_time.resize(n_devices, 0);
    for (devid_t device_id = 0; device_id < n_devices; ++device_id) {
      batch_device_available_time[static_cast<std::size_t>(device_id)] =
          EFTMapper::get_device_available_time(device_id, state);
    }
  }

  [[nodiscard]] DeviceTime select_best_device(const CandidateRec &candidate,
                                              const SchedulerState &state) {
    const auto dep_time = get_dependency_finish_time(candidate.task_id, state);
    timecount_t best_finish = MAX_TIME;
    devid_t best_device = candidate.supported_devices.front();

    for (const auto device_id : candidate.supported_devices) {
      const auto start_time =
          std::max(batch_device_available_time[static_cast<std::size_t>(device_id)], dep_time);
      const auto finish_time = get_finish_time(candidate.task_id, device_id, start_time, state);
      if (finish_time < best_finish || (finish_time == best_finish && device_id < best_device)) {
        best_finish = finish_time;
        best_device = device_id;
      }
    }

    return {best_device, best_finish};
  }

  void record_assignment(taskid_t task_id, devid_t device_id, const SchedulerState &state) {
    const auto dep_time = get_dependency_finish_time(task_id, state);
    const auto start_time =
        std::max(batch_device_available_time[static_cast<std::size_t>(device_id)], dep_time);
    const auto finish_time = get_finish_time(task_id, device_id, start_time, state);
    record_finish_time(task_id, finish_time, state);
    batch_device_available_time[static_cast<std::size_t>(device_id)] = finish_time;
  }

  void emit_fallback_actions(const std::vector<int32_t> &candidate_indices,
                             const SchedulerState &state) {
    std::vector<int32_t> ordered = candidate_indices;
    std::sort(ordered.begin(), ordered.end(), [&](int32_t lhs, int32_t rhs) {
      const auto &lhs_candidate = candidate_records[static_cast<std::size_t>(lhs)];
      const auto &rhs_candidate = candidate_records[static_cast<std::size_t>(rhs)];
      return lhs_candidate.input_pos < rhs_candidate.input_pos;
    });

    for (const auto candidate_index : ordered) {
      const auto &candidate = candidate_records[static_cast<std::size_t>(candidate_index)];
      const auto selection = select_best_device(candidate, state);
      action_buffer.push_back(
          Action{candidate.input_pos, selection.device_id, candidate.priority, candidate.priority});
      record_assignment(candidate.task_id, selection.device_id, state);
    }
  }

  void build_partitions(const SchedulerState &state) {
    partitions.clear();
    partition_devices.clear();
    partition_costs.clear();

    ankerl::unordered_dense::map<int32_t, int32_t> partition_index_by_label;
    partition_index_by_label.reserve(vertex_to_candidate_index.size());
    const auto &tasks = state.get_tasks();

    for (std::size_t vertex = 0; vertex < vertex_to_candidate_index.size(); ++vertex) {
      const auto raw_label = partition_labels[vertex];
      const auto [it, inserted] = partition_index_by_label.try_emplace(
          raw_label, static_cast<int32_t>(partitions.size()));
      if (inserted) {
        PartitionRec rec;
        rec.label = raw_label;
        partitions.push_back(std::move(rec));
      }

      auto &partition = partitions[static_cast<std::size_t>(it->second)];
      const auto candidate_index = vertex_to_candidate_index[vertex];
      const auto &candidate = candidate_records[static_cast<std::size_t>(candidate_index)];
      partition.candidate_indices.push_back(candidate_index);
      partition.total_priority += candidate.priority;
      if (partition.min_task_id < 0 || candidate.task_id < partition.min_task_id) {
        partition.min_task_id = candidate.task_id;
      }

      const auto reads = tasks.get_read(candidate.task_id);
      partition.unique_read_data.insert(partition.unique_read_data.end(), reads.begin(), reads.end());
    }

    for (auto &partition : partitions) {
      std::sort(partition.unique_read_data.begin(), partition.unique_read_data.end());
      partition.unique_read_data.erase(
          std::unique(partition.unique_read_data.begin(), partition.unique_read_data.end()),
          partition.unique_read_data.end());
    }
  }

  void score_partitions(const SchedulerState &state) {
    partition_costs.assign(partitions.size() * eligible_devices_buffer.size(), MAX_TIME);

    for (std::size_t partition_index = 0; partition_index < partitions.size(); ++partition_index) {
      const auto &partition = partitions[partition_index];
      for (std::size_t device_index = 0; device_index < eligible_devices_buffer.size();
           ++device_index) {
        const auto device_id = eligible_devices_buffer[device_index];
        timecount_t total_cost = 0;
        for (const auto data_id : partition.unique_read_data) {
          total_cost = saturating_add(total_cost, transfer_time_for_data(data_id, device_id, state));
        }
        partition_costs[partition_index * eligible_devices_buffer.size() + device_index] =
            total_cost;
      }
    }
  }

  [[nodiscard]] timecount_t get_partition_cost(std::size_t partition_index,
                                               std::size_t device_index) const {
    return partition_costs[partition_index * eligible_devices_buffer.size() + device_index];
  }

  void prepare_assignment_order() {
    assignment_partition_order.resize(partitions.size());
    std::iota(assignment_partition_order.begin(), assignment_partition_order.end(), std::size_t{0});
    std::sort(assignment_partition_order.begin(), assignment_partition_order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                const auto &lhs_partition = partitions[lhs];
                const auto &rhs_partition = partitions[rhs];
                if (lhs_partition.candidate_indices.size() != rhs_partition.candidate_indices.size()) {
                  return lhs_partition.candidate_indices.size() >
                         rhs_partition.candidate_indices.size();
                }
                if (lhs_partition.total_priority != rhs_partition.total_priority) {
                  return lhs_partition.total_priority > rhs_partition.total_priority;
                }
                if (lhs_partition.min_task_id != rhs_partition.min_task_id) {
                  return lhs_partition.min_task_id < rhs_partition.min_task_id;
                }
                return lhs_partition.label < rhs_partition.label;
              });
  }

  [[nodiscard]] bool better_assignment_sequence(const std::vector<uint8_t> &lhs,
                                                const std::vector<uint8_t> &rhs) const {
    if (rhs.empty()) {
      return true;
    }
    const auto common = std::min(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < common; ++i) {
      const auto lhs_device = eligible_devices_buffer[static_cast<std::size_t>(lhs[i])];
      const auto rhs_device = eligible_devices_buffer[static_cast<std::size_t>(rhs[i])];
      if (lhs_device != rhs_device) {
        return lhs_device < rhs_device;
      }
    }
    return lhs.size() < rhs.size();
  }

  const AssignmentSolution &solve_assignment(uint64_t used_mask) {
    const auto mask_index = static_cast<std::size_t>(used_mask);
    if (assignment_memo_ready[mask_index]) {
      return assignment_memo[mask_index];
    }

    auto &best = assignment_memo[mask_index];
    assignment_memo_ready[mask_index] = 1;
    best = AssignmentSolution{};

    const auto position =
        static_cast<std::size_t>(std::popcount(static_cast<unsigned long long>(used_mask)));
    if (position == assignment_partition_order.size()) {
      best.feasible = true;
      best.total_cost = 0;
      return best;
    }

    const auto partition_index = assignment_partition_order[position];
    for (std::size_t device_index = 0; device_index < eligible_devices_buffer.size(); ++device_index) {
      const auto bit = (uint64_t{1} << device_index);
      if ((used_mask & bit) != 0) {
        continue;
      }

      const auto direct_cost = get_partition_cost(partition_index, device_index);
      if (direct_cost >= MAX_TIME) {
        continue;
      }

      const auto &suffix = solve_assignment(used_mask | bit);
      if (!suffix.feasible) {
        continue;
      }

      AssignmentSolution candidate;
      candidate.feasible = true;
      candidate.total_cost = saturating_add(direct_cost, suffix.total_cost);
      if (candidate.total_cost >= MAX_TIME) {
        continue;
      }
      candidate.device_suffix = suffix.device_suffix;
      candidate.device_suffix.insert(candidate.device_suffix.begin(),
                                     static_cast<uint8_t>(device_index));

      if (!best.feasible || candidate.total_cost < best.total_cost ||
          (candidate.total_cost == best.total_cost &&
           better_assignment_sequence(candidate.device_suffix, best.device_suffix))) {
        best = std::move(candidate);
      }
    }

    return best;
  }

  [[nodiscard]] bool assign_partition_devices(const SchedulerState &state) {
    MONUnusedParameter(state);
    score_partitions(state);
    partition_devices.assign(partitions.size(), devid_t{-1});
    if (partitions.empty()) {
      return true;
    }

    T4F_INVARIANT(partitions.size() <= eligible_devices_buffer.size());
    constexpr std::size_t mask_bits = std::numeric_limits<uint64_t>::digits;
    if (eligible_devices_buffer.size() >= mask_bits) {
      SPDLOG_WARN("METISMapper exact assignment requires fewer than {} GPU devices; got {}",
                  mask_bits, eligible_devices_buffer.size());
      return false;
    }

    prepare_assignment_order();
    const auto state_count = static_cast<std::size_t>(uint64_t{1} << eligible_devices_buffer.size());
    assignment_memo.assign(state_count, AssignmentSolution{});
    assignment_memo_ready.assign(state_count, 0);

    const auto &best = solve_assignment(0);
    if (!best.feasible || best.device_suffix.size() != assignment_partition_order.size()) {
      return false;
    }

    for (std::size_t position = 0; position < assignment_partition_order.size(); ++position) {
      const auto partition_index = assignment_partition_order[position];
      const auto device_index = static_cast<std::size_t>(best.device_suffix[position]);
      partition_devices[partition_index] = eligible_devices_buffer[device_index];
    }
    return true;
  }

  void emit_partition_actions(const SchedulerState &state) {
    std::vector<std::size_t> partition_order(partitions.size(), 0);
    std::iota(partition_order.begin(), partition_order.end(), std::size_t{0});
    std::sort(partition_order.begin(), partition_order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                if (partition_devices[lhs] != partition_devices[rhs]) {
                  return partition_devices[lhs] < partition_devices[rhs];
                }
                return partitions[lhs].min_task_id < partitions[rhs].min_task_id;
              });

    for (const auto partition_index : partition_order) {
      auto &partition = partitions[partition_index];
      std::sort(partition.candidate_indices.begin(), partition.candidate_indices.end(),
                [&](int32_t lhs, int32_t rhs) {
                  const auto &lhs_candidate = candidate_records[static_cast<std::size_t>(lhs)];
                  const auto &rhs_candidate = candidate_records[static_cast<std::size_t>(rhs)];
                  if (lhs_candidate.priority != rhs_candidate.priority) {
                    return lhs_candidate.priority > rhs_candidate.priority;
                  }
                  return lhs_candidate.task_id < rhs_candidate.task_id;
                });

      const auto device_id = partition_devices[partition_index];
      for (const auto candidate_index : partition.candidate_indices) {
        const auto &candidate = candidate_records[static_cast<std::size_t>(candidate_index)];
        action_buffer.push_back(
            Action{candidate.input_pos, device_id, candidate.priority, candidate.priority});
        record_assignment(candidate.task_id, device_id, state);
      }
    }
  }

  ActionList &plan_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    action_buffer.clear();
    action_buffer.reserve(task_ids.size());
    if (task_ids.empty()) {
      last_graph.clear();
      clear_candidate_maps();
      return action_buffer;
    }

    ensure_task_buffers_size(static_cast<std::size_t>(state.get_tasks().get_n_compute_tasks()));
    if (should_reset_for_new_run(state)) {
      reset_task_buffers();
    }

    prepare_candidates(task_ids, state);
    build_metis_graph(state);
    reset_batch_device_times(state);

    std::vector<int32_t> fallback_candidate_indices;
    fallback_candidate_indices.reserve(candidate_records.size());
    for (std::size_t i = 0; i < candidate_records.size(); ++i) {
      if (!candidate_records[i].gpu_eligible) {
        fallback_candidate_indices.push_back(static_cast<int32_t>(i));
      }
    }

    const auto nparts = static_cast<int32_t>(
        std::min(eligible_devices_buffer.size(), vertex_to_candidate_index.size()));
    const bool should_fallback_all =
        eligible_devices_buffer.empty() || last_graph.num_vertices <= 1 ||
        last_graph.num_edges() == 0 || nparts <= 1;
    if (should_fallback_all) {
      fallback_candidate_indices.clear();
      fallback_candidate_indices.reserve(candidate_records.size());
      for (std::size_t i = 0; i < candidate_records.size(); ++i) {
        fallback_candidate_indices.push_back(static_cast<int32_t>(i));
      }
      emit_fallback_actions(fallback_candidate_indices, state);
      return action_buffer;
    }

    partition_labels.assign(vertex_to_candidate_index.size(), 0);
    const bool partitioned = metis.call_metis_partition(last_graph, nparts, partition_labels);
    if (!partitioned) {
      SPDLOG_WARN("METIS call failed, falling back to EFT-style mapping for this batch");
      fallback_candidate_indices.clear();
      fallback_candidate_indices.reserve(candidate_records.size());
      for (std::size_t i = 0; i < candidate_records.size(); ++i) {
        fallback_candidate_indices.push_back(static_cast<int32_t>(i));
      }
      emit_fallback_actions(fallback_candidate_indices, state);
      return action_buffer;
    }

    build_partitions(state);
    if (!assign_partition_devices(state)) {
      SPDLOG_WARN(
          "METIS partition-device assignment failed, falling back to EFT-style mapping for this batch");
      fallback_candidate_indices.clear();
      fallback_candidate_indices.reserve(candidate_records.size());
      for (std::size_t i = 0; i < candidate_records.size(); ++i) {
        fallback_candidate_indices.push_back(static_cast<int32_t>(i));
      }
      emit_fallback_actions(fallback_candidate_indices, state);
      return action_buffer;
    }

    emit_partition_actions(state);
    emit_fallback_actions(fallback_candidate_indices, state);
    return action_buffer;
  }

public:
  METISMapper() {
#ifndef ENABLE_METIS
    throw std::runtime_error("METISMapper requires ENABLE_METIS at build time");
#endif
  }

  METISMapper(const METISMapper &other) = default;

  METISMapper(std::size_t n_tasks, std::size_t n_devices) {
#ifndef ENABLE_METIS
    MONUnusedParameter(n_tasks);
    MONUnusedParameter(n_devices);
    throw std::runtime_error("METISMapper requires ENABLE_METIS at build time");
#else
    candidate_records.reserve(n_tasks);
    touched_candidate_tasks.reserve(n_tasks);
    touched_gpu_tasks.reserve(n_tasks);
    vertex_to_candidate_index.reserve(n_tasks);
    partition_labels.reserve(n_tasks);
    partitions.reserve(std::min(n_tasks, n_devices));
    eligible_devices_buffer.reserve(n_devices);
    batch_device_available_time.reserve(n_devices);
    assignment_partition_order.reserve(n_tasks);
#endif
  }

  [[nodiscard]] const MetisGraph &get_last_graph() const {
    return last_graph;
  }

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    auto &actions = plan_tasks(std::span<const taskid_t>(&task_id, 1), state);
    T4F_INVARIANT(actions.size() == 1);
    return actions.front();
  }

  ActionList &map_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) override {
    return plan_tasks(task_ids, state);
  }
};

class DARTSMapper : public Mapper {
private:
  struct CandidateTask {
    taskid_t task_id = -1;
    std::size_t input_pos = 0;
    priority_t priority = 0;
    devicemask_t supported_device_mask = 0;
    timecount_t canonical_duration = 0;
    bool active = true;
  };

  struct DeviceCandidateRecord {
    bool compatible = false;
    std::size_t missing_begin = 0;
    std::size_t missing_size = 0;
    int32_t missing_count = 0;
  };

  struct BlockAccum {
    bool transfer_feasible = false;
    dataid_t data_id = -1;
    timecount_t transfer_time = MAX_TIME;
    timecount_t c0_compute = 0;
    timecount_t c1_compute = 0;
    timecount_t c2_compute = 0;
    timecount_t c3_compute = 0;
    timecount_t r_compute = 0;
    int32_t s0_count = 0;
    int32_t s1_count = 0;
    int32_t s2_count = 0;
    int32_t s3_count = 0;
    int32_t best_s0_task_index = -1;
    int32_t best_s1_task_index = -1;
    int32_t best_s2_task_index = -1;
    int32_t best_s3_task_index = -1;
  };

  enum class DecisionReason : uint8_t {
    FALLBACK = 0,
    S0 = 1,
    S1 = 2,
    EXTENDED_S2 = 3,
    EXTENDED_S3 = 4,
  };

  std::vector<CandidateTask> candidate_tasks;
  std::vector<DeviceCandidateRecord> device_candidate_records;
  DataIDList missing_data_buffer;
  DataIDList unique_read_buffer;
  std::vector<BlockAccum> frontier_blocks;
  std::vector<int32_t> frontier_slot_by_data;
  DataIDList touched_frontier_data;
  std::vector<devid_t> selected_devices_buffer;
  std::vector<int32_t> emit_task_indices_buffer;
  std::vector<taskid_t> trace_emitted_tasks_buffer;

  [[nodiscard]] static bool device_in_mask(devicemask_t device_mask, devid_t device_id) {
    using UMask = std::make_unsigned_t<devicemask_t>;
    constexpr std::size_t mask_bits = std::numeric_limits<UMask>::digits;
    if (device_id < 0 || static_cast<std::size_t>(device_id) >= mask_bits) {
      return false;
    }
    const auto mask = static_cast<UMask>(device_mask);
    const auto bit = static_cast<UMask>(UMask{1} << static_cast<std::size_t>(device_id));
    return (mask & bit) != 0;
  }

  void ensure_scratch_sizes(std::size_t n_tasks, std::size_t n_data) {
    if (candidate_tasks.size() < n_tasks) {
      candidate_tasks.resize(n_tasks);
    }
    if (device_candidate_records.size() < n_tasks) {
      device_candidate_records.resize(n_tasks);
    }
    if (frontier_slot_by_data.size() < n_data) {
      frontier_slot_by_data.resize(n_data, -1);
    }
  }

  [[nodiscard]] bool better_task_choice(const CandidateTask &lhs, const CandidateTask &rhs) const {
    if (lhs.priority != rhs.priority) {
      return lhs.priority > rhs.priority;
    }
    return lhs.task_id < rhs.task_id;
  }

  [[nodiscard]] bool better_task_index(int32_t lhs_index, int32_t rhs_index,
                                       std::span<const CandidateTask> task_records) const {
    if (rhs_index < 0) {
      return lhs_index >= 0;
    }
    if (lhs_index < 0) {
      return false;
    }
    return better_task_choice(task_records[static_cast<std::size_t>(lhs_index)],
                              task_records[static_cast<std::size_t>(rhs_index)]);
  }

  [[nodiscard]] bool better_priority(priority_t lhs, priority_t rhs) const {
    if (lhs != rhs) {
      return lhs > rhs;
    }
    return false;
  }

  [[nodiscard]] bool task_supports_device(const CandidateTask &task_rec, devid_t device_id) const {
    return device_in_mask(task_rec.supported_device_mask, device_id);
  }

  [[nodiscard]] bool try_get_transfer_time(dataid_t data_id, devid_t device_id,
                                           const SchedulerState &state,
                                           timecount_t &transfer_time) const {
    const auto &data_manager = state.get_data_manager();
    if (data_manager.check_valid_mapped(data_id, device_id)) {
      transfer_time = 0;
      return true;
    }

    const auto &communication_manager = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto flags = data_manager.get_mapped_location_flags(data_id);
    const auto req = communication_manager.get_best_source(topology, device_id, flags);
    if (!req.found) {
      transfer_time = MAX_TIME;
      return false;
    }

    transfer_time = communication_manager.ideal_time_to_transfer(
        topology, state.get_data().get_size(data_id), req.source, device_id);
    return true;
  }

  [[nodiscard]] timecount_t duration_for_device(taskid_t task_id, devid_t device_id,
                                                const SchedulerState &state) const {
    const auto arch = state.get_devices().get_type(device_id);
    return state.get_tasks().get_mean_duration(task_id, arch);
  }

  [[nodiscard]] timecount_t canonical_duration(taskid_t task_id,
                                               const SchedulerState &state) const {
    const auto &tasks = state.get_tasks();
    if (tasks.is_architecture_supported(task_id, DeviceType::GPU)) {
      return tasks.get_mean_duration(task_id, DeviceType::GPU);
    }
    T4F_INVARIANT(tasks.is_architecture_supported(task_id, DeviceType::CPU));
    return tasks.get_mean_duration(task_id, DeviceType::CPU);
  }

  [[nodiscard]] priority_t task_priority(int32_t task_index) const {
    if (task_index < 0) {
      return std::numeric_limits<priority_t>::min();
    }
    return candidate_tasks[static_cast<std::size_t>(task_index)].priority;
  }

  [[nodiscard]] priority_t classic_block_priority(const BlockAccum &block) const {
    if (block.best_s0_task_index >= 0) {
      return task_priority(block.best_s0_task_index);
    }
    if (block.best_s1_task_index >= 0) {
      return task_priority(block.best_s1_task_index);
    }
    return std::numeric_limits<priority_t>::min();
  }

  [[nodiscard]] priority_t extended_block_priority(const BlockAccum &block) const {
    if (block.best_s1_task_index >= 0) {
      return task_priority(block.best_s1_task_index);
    }
    if (block.best_s2_task_index >= 0) {
      return task_priority(block.best_s2_task_index);
    }
    if (block.best_s3_task_index >= 0) {
      return task_priority(block.best_s3_task_index);
    }
    return std::numeric_limits<priority_t>::min();
  }

  [[nodiscard]] bool block_is_valid(const BlockAccum &block) const {
    return block.transfer_feasible;
  }

  void reset_frontier() {
    for (const auto data_id : touched_frontier_data) {
      frontier_slot_by_data[static_cast<std::size_t>(data_id)] = -1;
    }
    touched_frontier_data.clear();
    frontier_blocks.clear();
  }

  BlockAccum &get_or_create_frontier_block(dataid_t data_id, devid_t device_id,
                                           const SchedulerState &state) {
    const auto data_idx = static_cast<std::size_t>(data_id);
    auto &slot = frontier_slot_by_data[data_idx];
    if (slot >= 0) {
      return frontier_blocks[static_cast<std::size_t>(slot)];
    }

    slot = static_cast<int32_t>(frontier_blocks.size());
    touched_frontier_data.push_back(data_id);
    frontier_blocks.push_back(BlockAccum{});
    auto &block = frontier_blocks.back();
    block.data_id = data_id;
    block.transfer_feasible = try_get_transfer_time(data_id, device_id, state, block.transfer_time);
    return block;
  }

  [[nodiscard]] bool better_block(const BlockAccum &lhs, const BlockAccum &rhs) const {
    const bool lhs_valid = block_is_valid(lhs);
    const bool rhs_valid = block_is_valid(rhs);
    if (!rhs_valid) {
      return lhs_valid;
    }
    if (!lhs_valid) {
      return false;
    }

    const bool lhs_has_c0 = lhs.c0_compute > 0;
    const bool rhs_has_c0 = rhs.c0_compute > 0;
    if (lhs_has_c0 != rhs_has_c0) {
      return lhs_has_c0;
    }

    if (lhs_has_c0) {
      const __int128 lhs_ratio =
          static_cast<__int128>(lhs.transfer_time) * static_cast<__int128>(rhs.c0_compute);
      const __int128 rhs_ratio =
          static_cast<__int128>(rhs.transfer_time) * static_cast<__int128>(lhs.c0_compute);
      if (lhs_ratio != rhs_ratio) {
        return lhs_ratio < rhs_ratio;
      }
      if (lhs.s0_count != rhs.s0_count) {
        return lhs.s0_count > rhs.s0_count;
      }
      const auto lhs_priority = classic_block_priority(lhs);
      const auto rhs_priority = classic_block_priority(rhs);
      if (lhs_priority != rhs_priority) {
        return better_priority(lhs_priority, rhs_priority);
      }
      if (lhs.s1_count != rhs.s1_count) {
        return lhs.s1_count > rhs.s1_count;
      }
      if (lhs.r_compute != rhs.r_compute) {
        return lhs.r_compute > rhs.r_compute;
      }
      return lhs.data_id < rhs.data_id;
    }

    // c0 == 0 means no immediate unlocks after loading this block. In this branch the
    // classic DARTS policy falls back to S1, then p_k(D), then r(D).
    if (lhs.s1_count != rhs.s1_count) {
      return lhs.s1_count > rhs.s1_count;
    }
    const auto lhs_priority = classic_block_priority(lhs);
    const auto rhs_priority = classic_block_priority(rhs);
    if (lhs_priority != rhs_priority) {
      return better_priority(lhs_priority, rhs_priority);
    }
    if (lhs.r_compute != rhs.r_compute) {
      return lhs.r_compute > rhs.r_compute;
    }
    return lhs.data_id < rhs.data_id;
  }

  [[nodiscard]] bool better_extended_block(const BlockAccum &lhs, const BlockAccum &rhs) const {
    const bool lhs_valid = block_is_valid(lhs);
    const bool rhs_valid = block_is_valid(rhs);
    if (!rhs_valid) {
      return lhs_valid;
    }
    if (!lhs_valid) {
      return false;
    }

    const auto lhs_weighted_compute =
        static_cast<__int128>(4) * static_cast<__int128>(lhs.c1_compute) +
        static_cast<__int128>(2) * static_cast<__int128>(lhs.c2_compute) +
        static_cast<__int128>(lhs.c3_compute);
    const auto rhs_weighted_compute =
        static_cast<__int128>(4) * static_cast<__int128>(rhs.c1_compute) +
        static_cast<__int128>(2) * static_cast<__int128>(rhs.c2_compute) +
        static_cast<__int128>(rhs.c3_compute);
    if (lhs_weighted_compute != rhs_weighted_compute) {
      return lhs_weighted_compute > rhs_weighted_compute;
    }

    const int32_t lhs_weighted_count =
        4 * lhs.s1_count + 2 * lhs.s2_count + lhs.s3_count;
    const int32_t rhs_weighted_count =
        4 * rhs.s1_count + 2 * rhs.s2_count + rhs.s3_count;
    if (lhs_weighted_count != rhs_weighted_count) {
      return lhs_weighted_count > rhs_weighted_count;
    }

    const auto lhs_priority = extended_block_priority(lhs);
    const auto rhs_priority = extended_block_priority(rhs);
    if (lhs_priority != rhs_priority) {
      return better_priority(lhs_priority, rhs_priority);
    }
    if (lhs.r_compute != rhs.r_compute) {
      return lhs.r_compute > rhs.r_compute;
    }
    return lhs.data_id < rhs.data_id;
  }

  void append_action(int32_t task_index, devid_t device_id) {
    auto &task_rec = candidate_tasks[static_cast<std::size_t>(task_index)];
    if (!task_rec.active) {
      return;
    }
    task_rec.active = false;
    action_buffer.push_back(Action{task_rec.input_pos, device_id, task_rec.priority,
                                   task_rec.priority});
    trace_emitted_tasks_buffer.push_back(task_rec.task_id);
  }

  [[nodiscard]] int32_t fallback_task_for_device(devid_t device_id) const {
    int32_t best_task_index = -1;
    for (std::size_t task_index = 0; task_index < candidate_tasks.size(); ++task_index) {
      const auto &task_rec = candidate_tasks[task_index];
      if (!task_rec.active || !task_supports_device(task_rec, device_id)) {
        continue;
      }
      if (better_task_index(static_cast<int32_t>(task_index), best_task_index, candidate_tasks)) {
        best_task_index = static_cast<int32_t>(task_index);
      }
    }
    return best_task_index;
  }

  [[nodiscard]] Action fallback_action_for_task(taskid_t task_id, const SchedulerState &state) {
    fill_device_targets(task_id, state);
    T4F_INVARIANT(!device_buffer.empty());
    const auto mp = state.get_mapping_priority(task_id);
    return Action{0, device_buffer.front(), mp, mp};
  }

  void collect_selected_devices(const SchedulerState &state) {
    selected_devices_buffer.clear();
    thresholds.append_under_threshold_gpu_devices(state, selected_devices_buffer);
  }

  void prepare_candidate_records(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    const auto &tasks = state.get_tasks();
    candidate_tasks.resize(task_ids.size());
    for (std::size_t input_pos = 0; input_pos < task_ids.size(); ++input_pos) {
      const auto task_id = task_ids[input_pos];
      auto &rec = candidate_tasks[input_pos];
      rec.task_id = task_id;
      rec.input_pos = input_pos;
      rec.priority = state.get_mapping_priority(task_id);
      rec.supported_device_mask = tasks.get_supported_devices_mask(task_id);
      rec.canonical_duration = canonical_duration(task_id, state);
      rec.active = true;
    }
  }

  static bool contains_data_id(std::span<const dataid_t> values, dataid_t data_id) {
    return std::find(values.begin(), values.end(), data_id) != values.end();
  }

  void update_block_bucket(BlockAccum &block, int32_t missing_count, timecount_t device_duration,
                           int32_t task_index) {
    if (missing_count == 1) {
      block.s0_count += 1;
      block.c0_compute += device_duration;
      if (better_task_index(task_index, block.best_s0_task_index, candidate_tasks)) {
        block.best_s0_task_index = task_index;
      }
      return;
    }

    if (missing_count == 2) {
      block.s1_count += 1;
      block.c1_compute += device_duration;
      if (better_task_index(task_index, block.best_s1_task_index, candidate_tasks)) {
        block.best_s1_task_index = task_index;
      }
      return;
    }

    if (!extended_frontier_enabled) {
      return;
    }

    if (missing_count == 3) {
      block.s2_count += 1;
      block.c2_compute += device_duration;
      if (better_task_index(task_index, block.best_s2_task_index, candidate_tasks)) {
        block.best_s2_task_index = task_index;
      }
      return;
    }

    if (missing_count == 4) {
      block.s3_count += 1;
      block.c3_compute += device_duration;
      if (better_task_index(task_index, block.best_s3_task_index, candidate_tasks)) {
        block.best_s3_task_index = task_index;
      }
    }
  }

  int32_t build_device_block_stats(devid_t device_id, const SchedulerState &state) {
    const auto &tasks = state.get_tasks();
    const auto &data_manager = state.get_data_manager();

    reset_frontier();
    missing_data_buffer.clear();

    for (auto &record : device_candidate_records) {
      record = DeviceCandidateRecord{};
    }

    int32_t best_fallback_task = -1;

    for (std::size_t task_index = 0; task_index < candidate_tasks.size(); ++task_index) {
      const auto &task_rec = candidate_tasks[task_index];
      auto &device_record = device_candidate_records[task_index];
      device_record.missing_begin = missing_data_buffer.size();

      if (!task_rec.active || !task_supports_device(task_rec, device_id)) {
        continue;
      }

      device_record.compatible = true;

      if (better_task_index(static_cast<int32_t>(task_index), best_fallback_task,
                            candidate_tasks)) {
        best_fallback_task = static_cast<int32_t>(task_index);
      }

      unique_read_buffer.clear();
      const auto device_duration = duration_for_device(task_rec.task_id, device_id, state);
      for (const auto data_id : tasks.get_read(task_rec.task_id)) {
        if (contains_data_id(unique_read_buffer, data_id)) {
          continue;
        }
        unique_read_buffer.push_back(data_id);
        if (!data_manager.check_valid_mapped(data_id, device_id)) {
          missing_data_buffer.push_back(data_id);
          get_or_create_frontier_block(data_id, device_id, state);
        }
      }

      device_record.missing_size = missing_data_buffer.size() - device_record.missing_begin;
      device_record.missing_count = static_cast<int32_t>(device_record.missing_size);

      if (device_record.missing_size == 0) {
        continue;
      }

      for (std::size_t missing_offset = 0; missing_offset < device_record.missing_size;
           ++missing_offset) {
        const auto data_id =
            missing_data_buffer[device_record.missing_begin + missing_offset];
        auto &block = get_or_create_frontier_block(data_id, device_id, state);
        update_block_bucket(block, device_record.missing_count, device_duration,
                            static_cast<int32_t>(task_index));
      }
    }

    if (frontier_blocks.empty()) {
      return best_fallback_task;
    }

    for (std::size_t task_index = 0; task_index < candidate_tasks.size(); ++task_index) {
      const auto &task_rec = candidate_tasks[task_index];
      const auto &device_record = device_candidate_records[task_index];
      if (!task_rec.active || !device_record.compatible) {
        continue;
      }

      unique_read_buffer.clear();
      for (const auto data_id : tasks.get_read(task_rec.task_id)) {
        if (contains_data_id(unique_read_buffer, data_id)) {
          continue;
        }
        unique_read_buffer.push_back(data_id);
        const auto slot = frontier_slot_by_data[static_cast<std::size_t>(data_id)];
        if (slot >= 0) {
          frontier_blocks[static_cast<std::size_t>(slot)].r_compute += task_rec.canonical_duration;
        }
      }
    }

    return best_fallback_task;
  }

  [[nodiscard]] int32_t choose_best_block_for_device() const {
    int32_t best_block_index = -1;
    const bool use_extended_no_s0 =
        extended_frontier_enabled &&
        std::none_of(frontier_blocks.begin(), frontier_blocks.end(),
                     [](const BlockAccum &block) { return block.c0_compute > 0; });

    for (std::size_t block_index = 0; block_index < frontier_blocks.size(); ++block_index) {
      const auto &block = frontier_blocks[block_index];
      if (!block.transfer_feasible) {
        continue;
      }

      if (best_block_index < 0) {
        best_block_index = static_cast<int32_t>(block_index);
        continue;
      }

      const auto &best_block = frontier_blocks[static_cast<std::size_t>(best_block_index)];
      const bool better = use_extended_no_s0 ? better_extended_block(block, best_block)
                                             : better_block(block, best_block);
      if (better) {
        best_block_index = static_cast<int32_t>(block_index);
      }
    }
    return best_block_index;
  }

  [[nodiscard]] std::size_t active_candidate_count() const {
    return static_cast<std::size_t>(
        std::count_if(candidate_tasks.begin(), candidate_tasks.end(),
                      [](const CandidateTask &candidate) { return candidate.active; }));
  }

  [[nodiscard]] std::string emitted_tasks_string() const {
    std::string result = "[";
    for (std::size_t index = 0; index < trace_emitted_tasks_buffer.size(); ++index) {
      if (index > 0) {
        result += ",";
      }
      result += std::to_string(trace_emitted_tasks_buffer[index]);
    }
    result += "]";
    return result;
  }

  [[nodiscard]] int32_t normalized_extended_batch_emission_cap() const {
    return std::max<int32_t>(1, extended_batch_emission_cap);
  }

  [[nodiscard]] bool task_missing_set_contains(const DeviceCandidateRecord &device_record,
                                               dataid_t data_id) const {
    for (std::size_t missing_offset = 0; missing_offset < device_record.missing_size;
         ++missing_offset) {
      if (missing_data_buffer[device_record.missing_begin + missing_offset] == data_id) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool append_matching_bucket_actions(const BlockAccum &best_block,
                                                    int32_t missing_count, devid_t device_id) {
    emit_task_indices_buffer.clear();
    for (std::size_t task_index = 0; task_index < candidate_tasks.size(); ++task_index) {
      const auto &task_rec = candidate_tasks[task_index];
      const auto &device_record = device_candidate_records[task_index];
      if (!task_rec.active || !device_record.compatible || device_record.missing_count != missing_count) {
        continue;
      }
      if (task_missing_set_contains(device_record, best_block.data_id)) {
        emit_task_indices_buffer.push_back(static_cast<int32_t>(task_index));
      }
    }

    std::sort(emit_task_indices_buffer.begin(), emit_task_indices_buffer.end(),
              [&](int32_t lhs, int32_t rhs) {
                return better_task_index(lhs, rhs, candidate_tasks);
              });

    if (emit_task_indices_buffer.empty()) {
      return false;
    }

    const auto cap = normalized_extended_batch_emission_cap();
    const auto emit_count = std::min<std::size_t>(emit_task_indices_buffer.size(), cap);
    for (std::size_t emit_index = 0; emit_index < emit_count; ++emit_index) {
      append_action(emit_task_indices_buffer[emit_index], device_id);
    }
    return emit_count > 0;
  }

  void log_device_decision(devid_t device_id, int32_t best_block_index, DecisionReason reason,
                           std::size_t active_candidates, const SchedulerState &state) const {
    if (!trace_decisions) {
      return;
    }

    if (best_block_index < 0) {
      SPDLOG_DEBUG("Time:{} DARTS device={} candidates={} block=none t=0 c0=0 s0=0 s1=0 r=0 "
                   "s2=0 s3=0 proximity_compute=0 proximity_count=0 reason={} emitted_count={} "
                   "emitted={}",
                   state.get_global_time(), device_id, active_candidates,
                   decision_reason_name(reason), trace_emitted_tasks_buffer.size(),
                   emitted_tasks_string());
      return;
    }

    const auto &block = frontier_blocks[static_cast<std::size_t>(best_block_index)];
    const auto weighted_compute =
        static_cast<long long>(4 * block.c1_compute + 2 * block.c2_compute + block.c3_compute);
    const auto weighted_count = 4 * block.s1_count + 2 * block.s2_count + block.s3_count;
    SPDLOG_DEBUG(
        "Time:{} DARTS device={} candidates={} block={} t={} c0={} s0={} s1={} r={} s2={} s3={} "
        "proximity_compute={} proximity_count={} reason={} emitted_count={} emitted={}",
        state.get_global_time(), device_id, active_candidates, block.data_id, block.transfer_time,
        block.c0_compute, block.s0_count, block.s1_count, block.r_compute, block.s2_count,
        block.s3_count, weighted_compute, weighted_count, decision_reason_name(reason),
        trace_emitted_tasks_buffer.size(),
        emitted_tasks_string());
  }

  static const char *decision_reason_name(DecisionReason reason) {
    switch (reason) {
    case DecisionReason::S0:
      return "s0";
    case DecisionReason::S1:
      return "s1";
    case DecisionReason::EXTENDED_S2:
      return "extended_s2";
    case DecisionReason::EXTENDED_S3:
      return "extended_s3";
    case DecisionReason::FALLBACK:
    default:
      return "fallback";
    }
  }

  DecisionReason emit_actions_for_device(devid_t device_id, int32_t best_block_index,
                                         int32_t fallback_task_index) {
    trace_emitted_tasks_buffer.clear();

    if (best_block_index >= 0) {
      const auto &best_block = frontier_blocks[static_cast<std::size_t>(best_block_index)];
      if (best_block.s0_count > 0) {
        emit_task_indices_buffer.clear();
        for (std::size_t task_index = 0; task_index < candidate_tasks.size(); ++task_index) {
          const auto &task_rec = candidate_tasks[task_index];
          const auto &device_record = device_candidate_records[task_index];
          if (!task_rec.active || !device_record.compatible || device_record.missing_count != 1) {
            continue;
          }

          const auto data_offset = device_record.missing_begin;
          if (missing_data_buffer[data_offset] == best_block.data_id) {
            emit_task_indices_buffer.push_back(static_cast<int32_t>(task_index));
          }
        }

        std::sort(emit_task_indices_buffer.begin(), emit_task_indices_buffer.end(),
                  [&](int32_t lhs, int32_t rhs) {
                    return better_task_index(lhs, rhs, candidate_tasks);
                  });
        for (const auto task_index : emit_task_indices_buffer) {
          append_action(task_index, device_id);
        }
        return DecisionReason::S0;
      }

      if (best_block.best_s1_task_index >= 0) {
        if (extended_frontier_enabled && extended_batch_emission_enabled &&
            append_matching_bucket_actions(best_block, 2, device_id)) {
          return DecisionReason::S1;
        } else {
          append_action(best_block.best_s1_task_index, device_id);
        }
        return DecisionReason::S1;
      }

      if (extended_frontier_enabled && best_block.best_s2_task_index >= 0) {
        if (extended_batch_emission_enabled &&
            append_matching_bucket_actions(best_block, 3, device_id)) {
          return DecisionReason::EXTENDED_S2;
        } else {
          append_action(best_block.best_s2_task_index, device_id);
        }
        return DecisionReason::EXTENDED_S2;
      }

      if (extended_frontier_enabled && best_block.best_s3_task_index >= 0) {
        if (extended_batch_emission_enabled &&
            append_matching_bucket_actions(best_block, 4, device_id)) {
          return DecisionReason::EXTENDED_S3;
        } else {
          append_action(best_block.best_s3_task_index, device_id);
        }
        return DecisionReason::EXTENDED_S3;
      }
    }

    if (fallback_task_index >= 0) {
      append_action(fallback_task_index, device_id);
    }
    return DecisionReason::FALLBACK;
  }

  ActionList &plan_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    action_buffer.clear();
    action_buffer.reserve(task_ids.size());
    if (task_ids.empty()) {
      return action_buffer;
    }

    ensure_scratch_sizes(task_ids.size(), state.get_data().size());
    prepare_candidate_records(task_ids, state);
    collect_selected_devices(state);

    for (const auto device_id : selected_devices_buffer) {
      const auto active_candidates = active_candidate_count();
      const int32_t best_fallback_task = build_device_block_stats(device_id, state);
      const int32_t best_block_index = choose_best_block_for_device();
      const auto reason = emit_actions_for_device(device_id, best_block_index, best_fallback_task);
      log_device_decision(device_id, best_block_index, reason, active_candidates, state);
    }

    reset_frontier();
    return action_buffer;
  }

public:
  DeviceThresholdState thresholds;
  bool extended_frontier_enabled = false;
  bool extended_batch_emission_enabled = false;
  bool trace_decisions = false;
  int32_t extended_batch_emission_cap = 4;

  DARTSMapper() = default;

  DARTSMapper(const DARTSMapper &other) = default;

  DARTSMapper(std::size_t n_tasks, std::size_t n_devices) {
    candidate_tasks.reserve(n_tasks);
    device_candidate_records.reserve(n_tasks);
    missing_data_buffer.reserve(n_tasks);
    unique_read_buffer.reserve(n_tasks);
    frontier_blocks.reserve(n_tasks);
    frontier_slot_by_data.reserve(n_devices);
    touched_frontier_data.reserve(n_tasks);
    selected_devices_buffer.reserve(n_devices);
    emit_task_indices_buffer.reserve(n_tasks);
    trace_emitted_tasks_buffer.reserve(n_tasks);
  }

  [[nodiscard]] int32_t get_extended_batch_emission_cap() const {
    return extended_batch_emission_cap;
  }

  void set_extended_batch_emission_cap(int32_t cap) {
    extended_batch_emission_cap = std::max<int32_t>(1, cap);
  }

  [[nodiscard]] int32_t get_mapped_threshold() const {
    return thresholds.get_mapped_threshold();
  }

  [[nodiscard]] int32_t get_reserved_threshold() const {
    return thresholds.get_reserved_threshold();
  }

  void set_mapped_threshold(int32_t mapped_threshold) {
    thresholds.set_mapped_threshold(mapped_threshold);
  }

  void set_reserved_threshold(int32_t reserved_threshold) {
    thresholds.set_reserved_threshold(reserved_threshold);
  }

  void set_thresholds(int32_t mapped_threshold, int32_t reserved_threshold) {
    thresholds.set_thresholds(mapped_threshold, reserved_threshold);
  }

  void use_mapped_threshold(int32_t mapped_threshold) {
    thresholds.use_mapped_threshold(mapped_threshold);
  }

  void use_reserved_threshold(int32_t reserved_threshold) {
    thresholds.use_reserved_threshold(reserved_threshold);
  }

  void disable_thresholds() {
    thresholds.disable_thresholds();
  }

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    auto &actions = plan_tasks(std::span<const taskid_t>(&task_id, 1), state);
    if (!actions.empty()) {
      return actions.front();
    }
    return fallback_action_for_task(task_id, state);
  }

  ActionList &map_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) override {
    return plan_tasks(task_ids, state);
  }
};
