#pragma once

#include "action.hpp"
#include "breakpoints.hpp"
#include "communication.hpp"
#include "data.hpp"
#include "devices.hpp"
#include "events.hpp"
#include "iterator.hpp"
#include "macros.hpp"
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
#include <random>
#include <set>
#include <span>
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

using TaskIDTimeList = std::pair<TaskIDList, std::vector<timecount_t>>;

class Mapper;
class TransitionConditionBase;
using TransitionConditions = TransitionConditionBase;
template <typename TransitionConditionT> class SchedulerT;

enum class ExecutionState : int8_t {
  NONE = 0,
  RUNNING = 1,
  COMPLETE = 2,
  BREAKPOINT = 3,
  EXTERNAL_MAPPING = 4,
  ERROR = 5,
};
constexpr std::size_t num_execution_states = 6;

enum class MemoryAwareLocationState : int8_t {
  LAUNCHED = 0,
  RESERVED = 1,
  MAPPED = 2,
};

enum class MemoryAwareOverflowState : int8_t {
  RESERVED = 0,
  MAPPED = 1,
  LAUNCHED = 2,
};

enum class MemoryAwareOverflowMode : int8_t {
  FULL_SPILL = 0,
  INCOMING_ONLY = 1,
};

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
  DeviceQueue data_launchable;
  DeviceQueue eviction_launchable;

  // static TaskType id_to_type(taskid_t id, const Tasks &tasks);

  // void id_to_queue(taskid_t id, const TaskStateInfo &state);

public:
  SchedulerQueues() = default;
  priority_t data_queue_count = 0;

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

  void push_launchable_data(taskid_t id, priority_t p, devid_t device) {
    // TODO: change this to normal queue if needed keeping priority queue semantics for now
    data_launchable.push_priority_at(device, id, data_queue_count++);
    SPDLOG_DEBUG("Pushing launchable data task {} with priority {} on device {} data_queue_count "
                 "{} current_top {}",
                 id, p, device, data_queue_count - 1, data_launchable[device].top());
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

  template <typename> friend class SchedulerT;
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
      : n_devices(n_devices), per_device_counts(n_devices * n_per_device_counts),
        non_host_mapped_counts(n_devices, 0) {
    if (n_devices > 1) {
      min_non_host_mapped = 0;
    }
  };

  void count_mapped(taskid_t task_id, devid_t device_id) {
    update_non_host_mapped_count(device_id, 1);
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
    update_non_host_mapped_count(device_id, -1);
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

  [[nodiscard]] bool any_non_host_mapped_below(precision_t threshold) const {
    if (threshold <= 0 || n_devices <= 1) {
      return false;
    }
    return min_non_host_mapped < threshold;
  }

  [[nodiscard]] auto n_reserved(devid_t device_id) const {
    return per_device_counts[reserved_offset * n_devices + device_id];
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
  std::vector<precision_t> non_host_mapped_counts{};
  precision_t min_non_host_mapped{std::numeric_limits<precision_t>::max()};

  void recompute_min_non_host_mapped() {
    if (n_devices <= 1) {
      min_non_host_mapped = std::numeric_limits<precision_t>::max();
      return;
    }

    precision_t min_val = non_host_mapped_counts[1];
    for (precision_t device_id = 2; device_id < n_devices; ++device_id) {
      min_val = std::min(min_val, non_host_mapped_counts[device_id]);
    }
    min_non_host_mapped = min_val;
  }

  void update_non_host_mapped_count(devid_t device_id, precision_t delta) {
    if (device_id <= 0) {
      return;
    }

    const precision_t offset = mapped_offset * n_devices + device_id;
    const precision_t old_count = per_device_counts[offset];
    const precision_t new_count = old_count + delta;
    non_host_mapped_counts[device_id] = new_count;

    if (new_count < min_non_host_mapped) {
      min_non_host_mapped = new_count;
      return;
    }
    if (old_count == min_non_host_mapped && new_count > old_count) {
      recompute_min_non_host_mapped();
    }
  }
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

template <typename TransitionConditionT = TransitionConditions>
struct SchedulerInputT {
  std::reference_wrapper<Graph> graph;
  std::reference_wrapper<StaticTaskInfo> tasks;
  std::reference_wrapper<Data> data;
  std::reference_wrapper<Devices> devices;
  std::reference_wrapper<Topology> topology;
  std::reference_wrapper<TaskNoise> task_noise;
  std::shared_ptr<TransitionConditionT> conditions;
  int32_t top_k_candidates = 0;

  SchedulerInputT(Graph &graph, StaticTaskInfo &tasks, Data &data, Devices &devices,
                  Topology &topology, TaskNoise &task_noise, int32_t top_k_candidates = 1)
      : graph(graph), tasks(tasks), data(data), devices(devices), topology(topology),
        task_noise(task_noise), conditions(std::make_shared<TransitionConditionT>()),
        top_k_candidates(top_k_candidates) {
  }

  SchedulerInputT(Graph &graph, StaticTaskInfo &tasks, Data &data, Devices &devices,
                  Topology &topology, TaskNoise &task_noise,
                  const TransitionConditionT &conditions, int32_t top_k_candidates = 1)
      : graph(graph), tasks(tasks), data(data), devices(devices), topology(topology),
        task_noise(task_noise),
        conditions(std::static_pointer_cast<TransitionConditionT>(conditions.clone())),
        top_k_candidates(top_k_candidates) {
  }

  template <typename DerivedTransitionConditionT>
    requires(std::is_base_of_v<TransitionConditionT, DerivedTransitionConditionT>)
  SchedulerInputT(Graph &graph, StaticTaskInfo &tasks, Data &data, Devices &devices,
                  Topology &topology, TaskNoise &task_noise,
                  const DerivedTransitionConditionT &conditions, int32_t top_k_candidates = 1)
      : graph(graph), tasks(tasks), data(data), devices(devices), topology(topology),
        task_noise(task_noise),
        conditions(std::static_pointer_cast<TransitionConditionT>(conditions.clone())),
        top_k_candidates(top_k_candidates) {
  }

  SchedulerInputT(const SchedulerInputT &other) = default;

  SchedulerInputT &operator=(const SchedulerInputT &other) = default;
  SchedulerInputT(SchedulerInputT &&other) noexcept = default;
  SchedulerInputT &operator=(SchedulerInputT &&other) noexcept = default;
};

using SchedulerInput = SchedulerInputT<TransitionConditions>;

class SchedulerState {
protected:
  timecount_t global_time = 0;
  RuntimeTaskInfo task_runtime;
  DeviceManager device_manager;
  CommunicationManager communication_manager;
  DataManager data_manager;
  // ankerl::unordered_dense::set<taskid_t> mapped_but_not_reserved_tasks;
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

  template <typename TransitionConditionT>
  SchedulerState(SchedulerInputT<TransitionConditionT> &input)
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
        data_manager(other.data_manager),
        // mapped_but_not_reserved_tasks(other.mapped_but_not_reserved_tasks),
        graph(other.graph),
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
    device_manager.start_record();
  }

  void stop_record() {
    flags &= ~RECORD_FLAG;
    device_manager.stop_record();
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
    DeviceType arch = get_devices().get_type(device_id);
    timecount_t time = get_tasks().get_mean_duration(compute_task_id, arch);
    costs.count_mapped(device_id, time);
    counts.count_mapped(compute_task_id, device_id);
  }

  void update_reserved_cost(taskid_t compute_task_id, devid_t device_id) {
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

  template <typename> friend class SchedulerT;
  friend class TransitionConstraints;
};
