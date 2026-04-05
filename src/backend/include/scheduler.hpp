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

template <typename T>
concept TransitionConditionConcept = requires(T t, SchedulerState &state, SchedulerQueues &queues) {
  { t.should_map(state, queues) } -> std::convertible_to<bool>;
  { t.update_map(state, queues) } -> std::convertible_to<bool>;
  { t.should_reserve(state, queues) } -> std::convertible_to<bool>;
  { t.should_launch(state, queues) } -> std::convertible_to<bool>;
  { t.should_launch_data(state, queues) } -> std::convertible_to<bool>;
};

class TransitionConditionBase {
public:
  virtual ~TransitionConditionBase() = default;

  virtual std::shared_ptr<TransitionConditionBase> clone() const {
    return std::make_shared<TransitionConditionBase>(*this);
  }

  virtual bool should_map(SchedulerState &state, SchedulerQueues &queues) {
    MONUnusedParameter(state);
    MONUnusedParameter(queues);
    return true;
  }

  virtual bool update_map(SchedulerState &state, SchedulerQueues &queues) {
    return should_map(state, queues);
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

class DefaultTransitionConditions : public TransitionConditionBase {
public:
  std::shared_ptr<TransitionConditionBase> clone() const override {
    return std::make_shared<DefaultTransitionConditions>(*this);
  }
};

class RangeTransitionConditions : public TransitionConditionBase {
public:
  int32_t mapped_reserved_gap = 1;
  int32_t reserved_launched_gap = 1;
  int32_t total_in_flight = 1;

  RangeTransitionConditions(int32_t mapped_reserved_gap_, int32_t reserved_launched_gap_,
                            int32_t total_in_flight_)
      : mapped_reserved_gap(mapped_reserved_gap_), reserved_launched_gap(reserved_launched_gap_),
        total_in_flight(total_in_flight_) {
  }

  std::shared_ptr<TransitionConditionBase> clone() const override {
    return std::make_shared<RangeTransitionConditions>(*this);
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

  bool update_map(SchedulerState &state, SchedulerQueues &queues) override {
    return should_map(state, queues);
  }
};

class HysteresisTransitionConditions : public TransitionConditionBase {
public:
  HysteresisTransitionConditions() = default;
  timecount_t last_window_opened = 0;
  int32_t open_in_flight = 16;
  int32_t close_in_flight = 36;
  int32_t starvation_threshold = 2;
  bool window_open = false;

  HysteresisTransitionConditions(int32_t open_in_flight_, int32_t close_in_flight_,
                                 int32_t starvation_threshold_)
      : open_in_flight(open_in_flight_),
        close_in_flight(std::max(close_in_flight_, open_in_flight_)),
        starvation_threshold(starvation_threshold_) {
  }

  std::shared_ptr<TransitionConditionBase> clone() const override {
    return std::make_shared<HysteresisTransitionConditions>(*this);
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto &counts = state.counts;
    const auto n_mapped = counts.n_mapped();
    const bool starved = counts.any_non_host_mapped_below(starvation_threshold);

    if (!window_open) {
      return (n_mapped <= open_in_flight) || starved;
    }

    return !(n_mapped >= close_in_flight && !starved);
  }

  bool update_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto &counts = state.counts;
    const auto n_mapped = counts.n_mapped();
    const bool starved = counts.any_non_host_mapped_below(starvation_threshold);
    const bool open_condition = (n_mapped <= open_in_flight) || starved;
    const bool close_condition = (n_mapped >= close_in_flight) && !starved;

    if (!window_open && open_condition) {
      window_open = true;
      last_window_opened = state.get_global_time();
    } else if (window_open && close_condition) {
      window_open = false;
    }

    return should_map(state, queues);
  }
};

static_assert(TransitionConditionConcept<TransitionConditions>);

class BatchTransitionConditions : public TransitionConditionBase {
public:
  timecount_t last_accessed = 0;
  int32_t batch_size = 20;
  int32_t queue_threshold = 2;
  int32_t max_in_flight = 16;
  int32_t active_batch = 0;

  BatchTransitionConditions() = default;

  BatchTransitionConditions(int32_t batch_size_, int32_t queue_threshold_, int32_t max_in_flight_)
      : batch_size(batch_size_), queue_threshold(queue_threshold_),
        max_in_flight(max_in_flight_) {
  }

  std::shared_ptr<TransitionConditionBase> clone() const override {
    return std::make_shared<BatchTransitionConditions>(*this);
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto &counts = state.counts;
    const auto n_mapped = counts.n_mapped();
    const bool space_flag = (n_mapped <= max_in_flight + active_batch);
    bool workqueue_flag = false;
    const devid_t n_devices = state.get_devices().size();
    for (devid_t i = 1; i < n_devices; ++i) {
      if (counts.n_mapped(i) < queue_threshold) {
        workqueue_flag = true;
        break;
      }
    }

    const bool flag = space_flag || workqueue_flag;
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
      return (state.counts.n_mapped(device_id) - state.counts.n_reserved(device_id)) <=
             mapped_threshold_;
    }
    if (has_reserved_threshold()) {
      return state.counts.n_reserved(device_id) <= reserved_threshold_;
    }
    return false;
  }

  [[nodiscard]] bool any_device_under_threshold(const SchedulerState &state) const {
    if (!has_mapped_threshold() && !has_reserved_threshold()) {
      return false;
    }

    const auto &devices = state.get_devices();
    const devid_t n_devices = devices.size();
    for (devid_t device_id = 1; device_id < n_devices; ++device_id) {
      if (devices.get_type(device_id) != DeviceType::GPU) {
        continue;
      }
      if (is_device_under_threshold(state, device_id)) {
        return true;
      }
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

class DeviceThresholdTransitionConditions : public TransitionConditionBase {
public:
  DeviceThresholdState thresholds;

  DeviceThresholdTransitionConditions() = default;

  DeviceThresholdTransitionConditions(int32_t mapped_threshold_, int32_t reserved_threshold_)
      : thresholds(mapped_threshold_, reserved_threshold_) {
  }

  std::shared_ptr<TransitionConditionBase> clone() const override {
    return std::make_shared<DeviceThresholdTransitionConditions>(*this);
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
    return thresholds.any_device_under_threshold(state);
  }
};

class DARTSAdaptiveTransitionConditions : public TransitionConditionBase {
public:
  int32_t reserved_threshold = 0;
  int32_t max_mapped = 64;
  int32_t starvation_threshold = 1;

  DARTSAdaptiveTransitionConditions() = default;

  DARTSAdaptiveTransitionConditions(int32_t reserved_threshold_, int32_t max_mapped_,
                                    int32_t starvation_threshold_ = 1)
      : reserved_threshold(reserved_threshold_), max_mapped(max_mapped_),
        starvation_threshold(starvation_threshold_) {
  }

  std::shared_ptr<TransitionConditionBase> clone() const override {
    return std::make_shared<DARTSAdaptiveTransitionConditions>(*this);
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto &counts = state.counts;
    if (counts.any_non_host_mapped_below(starvation_threshold)) {
      return true;
    }
    if (counts.n_mapped() >= max_mapped) {
      return false;
    }

    const auto &devices = state.get_devices();
    const devid_t n_devs = devices.size();
    for (devid_t d = 1; d < n_devs; ++d) {
      if (devices.get_type(d) != DeviceType::GPU) {
        continue;
      }
      if (counts.n_reserved(d) <= reserved_threshold) {
        return true;
      }
    }
    return false;
  }
};

class DARTSPipelineTransitionConditions : public TransitionConditionBase {
public:
  int32_t pipeline_depth = 4;
  int32_t max_in_flight = 64;
  int32_t starvation_threshold = 1;

  DARTSPipelineTransitionConditions() = default;

  DARTSPipelineTransitionConditions(int32_t pipeline_depth_, int32_t max_in_flight_,
                                    int32_t starvation_threshold_ = 1)
      : pipeline_depth(pipeline_depth_), max_in_flight(max_in_flight_),
        starvation_threshold(starvation_threshold_) {
  }

  std::shared_ptr<TransitionConditionBase> clone() const override {
    return std::make_shared<DARTSPipelineTransitionConditions>(*this);
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto &counts = state.counts;
    if (counts.any_non_host_mapped_below(starvation_threshold)) {
      return true;
    }
    if (counts.n_mapped() >= max_in_flight) {
      return false;
    }

    const auto &devices = state.get_devices();
    const devid_t n_devs = devices.size();
    for (devid_t d = 1; d < n_devs; ++d) {
      if (devices.get_type(d) != DeviceType::GPU) {
        continue;
      }
      if (counts.n_mapped(d) < pipeline_depth) {
        return true;
      }
    }
    return false;
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

template <typename TransitionConditionT = TransitionConditions>
class SchedulerT {
  static_assert(TransitionConditionConcept<TransitionConditionT>);

protected:
  struct EvictionInvalidationInfo {
    bool future_usage = false;
    bool write_after_read = false;
  };

  SchedulerState state;
  SchedulerQueues queues;
  TaskDeviceList tasks_requesting_eviction;
  int64_t success_count = 0;
  int64_t eviction_count = 0;
  EvictionState eviction_state = EvictionState::NONE;
  ankerl::unordered_dense::map<uint64_t, uint8_t> eviction_invalidation_cache;
  ankerl::unordered_dense::set<uint64_t> eviction_planned_victim_keys;

  void enqueue_data_tasks(taskid_t task_id);
  [[nodiscard]] EvictionInvalidationInfo
  get_eviction_invalidation_info(const StaticTaskInfo &static_graph,
                                 const RuntimeTaskInfo &task_runtime, dataid_t data_id,
                                 devid_t device_id);
  void clear_eviction_invalidation_cache() {
    eviction_invalidation_cache.clear();
  }

public:
  BreakpointManager breakpoints;
  TaskIDList compute_task_buffer;
  TaskIDList data_task_buffer;
  TaskIDList python_mapper_buffer;
  std::shared_ptr<TransitionConditionT> conditions;
  int64_t scheduler_event_count = 1;
  bool initialized = false;

  SchedulerT(SchedulerInputT<TransitionConditionT> &input)
      : state(input), queues(input.devices), conditions(input.conditions) {
    compute_task_buffer.reserve(INITIAL_TASK_BUFFER_SIZE);
    data_task_buffer.reserve(INITIAL_TASK_BUFFER_SIZE);
    tasks_requesting_eviction.reserve(INITIAL_TASK_BUFFER_SIZE);
    eviction_invalidation_cache.reserve(INITIAL_TASK_BUFFER_SIZE * 8);
    eviction_planned_victim_keys.reserve(INITIAL_TASK_BUFFER_SIZE * 8);
    if (input.top_k_candidates > 0) {
      queues.mappable.set_k(static_cast<int>(input.top_k_candidates));
    }
  }

  SchedulerT(const SchedulerT &other) = default;

  void set_transition_conditions(const TransitionConditionT &conditions_) {
    conditions = std::static_pointer_cast<TransitionConditionT>(conditions_.clone());
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

  taskid_t map_task(taskid_t task_id, Action &action);
  void skip_map_tasks(MapperEvent &map_event, EventManager &event_manager);
  void map_tasks(MapperEvent &map_event, EventManager &event_manager, Mapper &mapper,
                 bool prechecked = false);
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

    queues.push_launchable_data(data_task_id, p, device);
  }

  void push_launchable_data(const std::span<const taskid_t> data_task_ids) {
    for (auto data_task_id : data_task_ids) {
      const priority_t p = state.task_runtime.get_data_task_launch_priority(data_task_id);
      const devid_t device = state.task_runtime.get_data_task_mapped_device(data_task_id);
      queues.push_launchable_data(data_task_id, p, device);
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

using Scheduler = SchedulerT<TransitionConditions>;

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

  virtual ActionList &map_tasks(const TaskIDList &task_ids, const SchedulerState &state) {
    action_buffer.clear();
    action_buffer.reserve(task_ids.size());
    for (auto task_id : task_ids) {
      action_buffer.emplace_back(map_task(task_id, state));
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

  DeviceTime get_best_device(taskid_t task_id, const SchedulerState &state) {
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
  std::vector<devid_t> claimed_for_device;

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
    if (claimed_for_device.size() < n_data) {
      claimed_for_device.resize(n_data, static_cast<devid_t>(-1));
    }
  }

  void reset_claimed_for_window(std::size_t n_data) {
    if (!intra_window_coordination) {
      return;
    }
    std::fill(claimed_for_device.begin(),
              claimed_for_device.begin() + static_cast<std::ptrdiff_t>(n_data),
              static_cast<devid_t>(-1));
  }

  void claim_blocks_for_device(dataid_t best_block_id, devid_t device_id,
                               const SchedulerState &state) {
    if (!intra_window_coordination) {
      return;
    }
    const auto n_data = claimed_for_device.size();
    const auto mark = [&](dataid_t data_id) {
      const auto idx = static_cast<std::size_t>(data_id);
      if (idx < n_data && claimed_for_device[idx] == static_cast<devid_t>(-1)) {
        claimed_for_device[idx] = device_id;
      }
    };
    mark(best_block_id);
    const auto &tasks = state.get_tasks();
    for (const auto task_id : trace_emitted_tasks_buffer) {
      for (const auto data_id : tasks.get_read(task_id)) {
        mark(data_id);
      }
    }
  }

  [[nodiscard]] bool is_claimed_by_other(dataid_t data_id, devid_t device_id) const {
    if (!intra_window_coordination) {
      return false;
    }
    const auto idx = static_cast<std::size_t>(data_id);
    if (idx >= claimed_for_device.size()) {
      return false;
    }
    const auto owner = claimed_for_device[idx];
    return owner != static_cast<devid_t>(-1) && owner != device_id;
  }

  [[nodiscard]] bool check_claimed_local(dataid_t data_id, devid_t device_id,
                                         const SchedulerState &state) const {
    if (state.get_data_manager().check_valid_mapped(data_id, device_id)) {
      return true;
    }
    if (!intra_window_coordination) {
      return false;
    }
    const auto idx = static_cast<std::size_t>(data_id);
    return idx < claimed_for_device.size() && claimed_for_device[idx] == device_id;
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
    if (check_claimed_local(data_id, device_id, state)) {
      transfer_time = 0;
      return true;
    }

    const auto &communication_manager = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto flags = state.get_data_manager().get_mapped_location_flags(data_id);
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
    if (is_claimed_by_other(data_id, device_id)) {
      block.transfer_feasible = false;
    } else {
      block.transfer_feasible = try_get_transfer_time(data_id, device_id, state, block.transfer_time);
    }
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

    const int32_t lhs_weighted_count = 4 * lhs.s1_count + 2 * lhs.s2_count + lhs.s3_count;
    const int32_t rhs_weighted_count = 4 * rhs.s1_count + 2 * rhs.s2_count + rhs.s3_count;
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
    action_buffer.push_back(
        Action{task_rec.input_pos, device_id, task_rec.priority, task_rec.priority});
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

  [[nodiscard]] int32_t fallback_task_for_device_eft(devid_t device_id,
                                                     const SchedulerState &state) const {
    const timecount_t device_load = state.costs.get_mapped_time(device_id);
    const auto &data_manager = state.get_data_manager();
    const auto &comm = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto &data = state.get_data();

    int32_t best_task_index = -1;
    timecount_t best_finish_time = std::numeric_limits<timecount_t>::max();

    for (std::size_t task_index = 0; task_index < candidate_tasks.size(); ++task_index) {
      const auto &task_rec = candidate_tasks[task_index];
      if (!task_rec.active || !task_supports_device(task_rec, device_id)) {
        continue;
      }

      const auto &dev_rec = device_candidate_records[task_index];
      timecount_t transfer_time = 0;
      if (dev_rec.compatible && dev_rec.missing_count > 0) {
        for (int32_t i = 0; i < dev_rec.missing_count; ++i) {
          const dataid_t did = missing_data_buffer[dev_rec.missing_begin + i];
          const mem_t dsize = data.get_size(did);
          const auto loc_flags = data_manager.get_mapped_location_flags(did);
          const auto req = comm.get_best_source(topology, device_id, loc_flags);
          if (req.found) {
            transfer_time += comm.ideal_time_to_transfer(topology, dsize, req.source, device_id);
          }
        }
      }

      const timecount_t finish_time =
          device_load + transfer_time + task_rec.canonical_duration;
      if (finish_time < best_finish_time) {
        best_finish_time = finish_time;
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
    if (pipeline_depth > 0) {
      const auto &devices = state.get_devices();
      const auto &counts = state.counts;
      const devid_t n_devs = devices.size();
      const bool cap_hit = (max_in_flight > 0) && (counts.n_mapped() >= max_in_flight);
      for (devid_t d = 1; d < n_devs; ++d) {
        if (devices.get_type(d) != DeviceType::GPU) {
          continue;
        }
        const auto mapped = counts.n_mapped(d);
        if (cap_hit) {
          if (mapped < starvation_threshold) {
            selected_devices_buffer.push_back(d);
          }
        } else if (mapped < pipeline_depth) {
          selected_devices_buffer.push_back(d);
        }
      }
    } else {
      thresholds.append_under_threshold_gpu_devices(state, selected_devices_buffer);
    }
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
        if (!check_claimed_local(data_id, device_id, state)) {
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
        const auto data_id = missing_data_buffer[device_record.missing_begin + missing_offset];
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
      const bool better =
          use_extended_no_s0 ? better_extended_block(block, best_block)
                             : better_block(block, best_block);
      if (better) {
        best_block_index = static_cast<int32_t>(block_index);
      }
    }
    return best_block_index;
  }

  [[nodiscard]] std::size_t active_candidate_count() const {
    return static_cast<std::size_t>(std::count_if(
        candidate_tasks.begin(), candidate_tasks.end(),
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
      if (!task_rec.active || !device_record.compatible ||
          device_record.missing_count != missing_count) {
        continue;
      }
      if (task_missing_set_contains(device_record, best_block.data_id)) {
        emit_task_indices_buffer.push_back(static_cast<int32_t>(task_index));
      }
    }

    std::sort(emit_task_indices_buffer.begin(), emit_task_indices_buffer.end(),
              [&](int32_t lhs, int32_t rhs) { return better_task_index(lhs, rhs, candidate_tasks); });

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
        static_cast<long long>(static_cast<__int128>(4) * block.c1_compute +
                               static_cast<__int128>(2) * block.c2_compute + block.c3_compute);
    const auto weighted_count = 4 * block.s1_count + 2 * block.s2_count + block.s3_count;
    SPDLOG_DEBUG(
        "Time:{} DARTS device={} candidates={} block={} t={} c0={} s0={} s1={} r={} s2={} s3={} "
        "proximity_compute={} proximity_count={} reason={} emitted_count={} emitted={}",
        state.get_global_time(), device_id, active_candidates, block.data_id, block.transfer_time,
        block.c0_compute, block.s0_count, block.s1_count, block.r_compute, block.s2_count,
        block.s3_count, weighted_compute, weighted_count, decision_reason_name(reason),
        trace_emitted_tasks_buffer.size(), emitted_tasks_string());
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
                  [&](int32_t lhs, int32_t rhs) { return better_task_index(lhs, rhs, candidate_tasks); });
        for (const auto task_index : emit_task_indices_buffer) {
          append_action(task_index, device_id);
        }
        return DecisionReason::S0;
      }

      if (best_block.best_s1_task_index >= 0) {
        if (extended_frontier_enabled && extended_batch_emission_enabled &&
            append_matching_bucket_actions(best_block, 2, device_id)) {
          return DecisionReason::S1;
        }
        append_action(best_block.best_s1_task_index, device_id);
        return DecisionReason::S1;
      }

      if (extended_frontier_enabled && best_block.best_s2_task_index >= 0) {
        if (extended_batch_emission_enabled &&
            append_matching_bucket_actions(best_block, 3, device_id)) {
          return DecisionReason::EXTENDED_S2;
        }
        append_action(best_block.best_s2_task_index, device_id);
        return DecisionReason::EXTENDED_S2;
      }

      if (extended_frontier_enabled && best_block.best_s3_task_index >= 0) {
        if (extended_batch_emission_enabled &&
            append_matching_bucket_actions(best_block, 4, device_id)) {
          return DecisionReason::EXTENDED_S3;
        }
        append_action(best_block.best_s3_task_index, device_id);
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

    const std::size_t n_data = state.get_data().size();
    ensure_scratch_sizes(task_ids.size(), n_data);
    prepare_candidate_records(task_ids, state);
    collect_selected_devices(state);

    if (finish_time_aware && selected_devices_buffer.size() > 1) {
      std::sort(selected_devices_buffer.begin(), selected_devices_buffer.end(),
                [&](devid_t a, devid_t b) {
                  return state.costs.get_mapped_time(a) < state.costs.get_mapped_time(b);
                });
    }

    reset_claimed_for_window(n_data);

    for (const auto device_id : selected_devices_buffer) {
      const int max_cascade = intra_window_coordination ? cascade_passes : 1;
      for (int pass = 0; pass < max_cascade; ++pass) {
        MONUnusedParameter(pass);
        const auto active_candidates = active_candidate_count();
        if (active_candidates == 0) {
          break;
        }
        int32_t best_fallback_task = build_device_block_stats(device_id, state);
        const int32_t best_block_index = choose_best_block_for_device();

        if (finish_time_aware && best_block_index < 0 && best_fallback_task >= 0) {
          best_fallback_task = fallback_task_for_device_eft(device_id, state);
        }

        const std::size_t actions_before = action_buffer.size();
        const auto reason =
            emit_actions_for_device(device_id, best_block_index, best_fallback_task);
        log_device_decision(device_id, best_block_index, reason, active_candidates, state);

        if (best_block_index >= 0) {
          claim_blocks_for_device(
              frontier_blocks[static_cast<std::size_t>(best_block_index)].data_id, device_id,
              state);
        }
        reset_frontier();

        if (action_buffer.size() == actions_before || reason == DecisionReason::FALLBACK) {
          break;
        }
      }
      reset_frontier();
    }

    return action_buffer;
  }

public:
  DeviceThresholdState thresholds;
  bool extended_frontier_enabled = false;
  bool extended_batch_emission_enabled = false;
  bool trace_decisions = false;
  int32_t extended_batch_emission_cap = 2;
  bool intra_window_coordination = false;
  int32_t cascade_passes = 3;
  bool finish_time_aware = false;
  int32_t pipeline_depth = 0;
  int32_t starvation_threshold = 1;
  int32_t max_in_flight = 0;

  DARTSMapper() {
    thresholds.use_mapped_threshold(0);
  }

  DARTSMapper(const DARTSMapper &other) = default;

  DARTSMapper(std::size_t n_tasks, std::size_t n_devices) : DARTSMapper() {
    candidate_tasks.reserve(n_tasks);
    device_candidate_records.reserve(n_tasks);
    missing_data_buffer.reserve(n_tasks);
    unique_read_buffer.reserve(n_tasks);
    frontier_blocks.reserve(n_tasks);
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
    const TaskIDList task_ids{task_id};
    auto &actions = plan_tasks(std::span<const taskid_t>(task_ids.data(), task_ids.size()), state);
    if (!actions.empty()) {
      return actions.front();
    }
    return fallback_action_for_task(task_id, state);
  }

  ActionList &map_tasks(const TaskIDList &task_ids, const SchedulerState &state) override {
    return plan_tasks(std::span<const taskid_t>(task_ids.data(), task_ids.size()), state);
  }
};
