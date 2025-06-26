#pragma once

#include "action.hpp"
#include "breakpoints.hpp"
#include "communication_manager.hpp"
#include "data_manager.hpp"
#include "device_manager.hpp"
#include "devices.hpp"
#include "event_manager.hpp"
#include "events.hpp"
#include "graph.hpp"
#include "iterator.hpp"
#include "macros.hpp"
#include "noise.hpp"
#include "queues.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include "task_manager.hpp"
#include "tasks.hpp"
#include <cassert>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#define TIME_TO_MAP 0
#define TIME_TO_RESERVE 0
#define TIME_TO_LAUNCH 0
#define SCHEDULER_TIME_GAP 0

using TaskQueue3 = ContainerQueue<taskid_t, TopKQueueHelper<1>::queue_type>;
using TaskQueue = ContainerQueue<taskid_t, std::priority_queue>;
using DeviceQueue = ActiveQueueIterator<TaskQueue>;

using TaskIDTimeList = std::pair<TaskIDList, std::vector<timecount_t>>;

class TransitionConditions;
class Scheduler;
class Mapper;

enum class ExecutionState {
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
  TaskQueue3 mappable;
  DeviceQueue reservable;
  DeviceQueue launchable;
  DeviceQueue data_launchable;
  DeviceQueue eviction_launchable;

  // static TaskType id_to_type(taskid_t id, const Tasks &tasks);

  // void id_to_queue(taskid_t id, const TaskStateInfo &state);

public:
  SchedulerQueues(Devices &devices)
      : reservable(devices.size()), launchable(devices.size()), data_launchable(devices.size()),
        eviction_launchable(devices.size()) {
  }

  SchedulerQueues(const SchedulerQueues &other) = default;
  SchedulerQueues &operator=(const SchedulerQueues &other) = default;

  void push_mappable(taskid_t id, priority_t p);
  void push_mappable(const TaskIDList &ids, const PriorityList &ps);

  void push_reservable(taskid_t id, priority_t p, devid_t device);
  void push_reservable(const TaskIDList &ids, const PriorityList &ps, devid_t device);
  void push_launchable(taskid_t id, priority_t p, devid_t device);
  void push_launchable(const TaskIDList &ids, const PriorityList &ps, devid_t device);

  void push_launchable_data(taskid_t id, priority_t p, devid_t device);
  void push_launchable_data(const TaskIDList &ids, const PriorityList &ps, devid_t device);

  void push_launchable_eviction(taskid_t id, priority_t p, devid_t device);
  void push_launchable_eviction(const TaskIDList &ids, const PriorityList &ps, devid_t device);

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
    return reservable.total_active_size() > 0;
  }

  [[nodiscard]] std::size_t n_launchable(devid_t device) const {
    const auto &device_queue = launchable.at(device);
    return device_queue.size();
  }
  [[nodiscard]] bool has_launchable(devid_t device) const {
    const auto &device_queue = launchable.at(device);
    return !device_queue.empty();
  }

  [[nodiscard]] bool has_active_launchable() const {
    return launchable.total_active_size() > 0;
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
    return data_launchable.total_active_size() > 0;
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
    return eviction_launchable.total_active_size() > 0;
  }

  // void populate(const TaskManager &task_manager);
  // void populate(const TaskIDList &ids, const TaskManager &task_manager);

  friend class Scheduler;
};

class TaskCountInfo {
public:
  std::unordered_set<taskid_t> active_tasks;

  TaskCountInfo(std::size_t n_devices);
  void count_mapped(taskid_t task_id, devid_t device_id);
  void count_reserved(taskid_t task_id, devid_t device_id);
  void count_launched(taskid_t task_id, devid_t device_id);

  void count_completed(taskid_t task_id, devid_t device_id);
  void count_data_completed(taskid_t task_id, devid_t device_id);

  auto get_active_task_list() const {
    return TaskIDList(active_tasks.begin(), active_tasks.end());
  }

  auto get_active_tasks() const {
    return active_tasks;
  }

  [[nodiscard]] std::size_t n_active() const {
    return n_active_tasks;
  }
  [[nodiscard]] std::size_t n_mapped() const {
    return n_mapped_tasks;
  }
  [[nodiscard]] std::size_t n_reserved() const {
    return n_reserved_tasks;
  }
  [[nodiscard]] std::size_t n_launched() const {
    return n_launched_tasks;
  }
  [[nodiscard]] std::size_t n_completed() const {
    return n_completed_tasks;
  }
  [[nodiscard]] std::size_t n_data_completed() const {
    return n_data_completed_tasks;
  }

  [[nodiscard]] std::size_t n_active(devid_t device_id) const {
    return per_device_mapped_tasks.at(device_id);
  }

  [[nodiscard]] std::size_t n_mapped(devid_t device_id) const {
    return per_device_mapped_tasks.at(device_id);
  }

  [[nodiscard]] std::size_t n_reserved(devid_t device_id) const {
    return per_device_reserved_tasks.at(device_id);
  }

  [[nodiscard]] std::size_t n_launched(devid_t device_id) const {
    return per_device_launched_tasks.at(device_id);
  }

  [[nodiscard]] std::size_t n_completed(devid_t device_id) const {
    return per_device_completed_tasks.at(device_id);
  }

  [[nodiscard]] std::size_t n_data_completed(devid_t device_id) const {
    return per_device_data_completed_tasks.at(device_id);
  }

protected:
  std::size_t n_active_tasks = 0;
  std::size_t n_mapped_tasks = 0;
  std::vector<std::size_t> per_device_mapped_tasks;
  std::size_t n_reserved_tasks = 0;
  std::vector<std::size_t> per_device_reserved_tasks;
  std::size_t n_launched_tasks = 0;
  std::vector<std::size_t> per_device_launched_tasks;
  std::size_t n_completed_tasks = 0;
  std::vector<std::size_t> per_device_completed_tasks;
  std::size_t n_data_completed_tasks = 0;
  std::vector<std::size_t> per_device_data_completed_tasks;
};

class TaskCostInfo {
public:
  TaskCostInfo(std::size_t n_tasks, std::size_t n_devices);
  void count_mapped(devid_t device_id, timecount_t time);
  void count_reserved(devid_t device_id, timecount_t time);
  void count_launched(devid_t device_id, timecount_t time);

  void count_completed(devid_t device_id, timecount_t time);

  void count_data_completed(devid_t device_id, timecount_t time);

  [[nodiscard]] timecount_t get_mapped_time(devid_t device_id) const {
    return per_device_mapped_time.at(device_id);
  }
  [[nodiscard]] timecount_t get_reserved_time(devid_t device_id) const {
    return per_device_reserved_time.at(device_id);
  }
  [[nodiscard]] timecount_t get_launched_time(devid_t device_id) const {
    return per_device_launched_time.at(device_id);
  }

  [[nodiscard]] timecount_t get_completed_time(devid_t device_id) const {
    return per_device_completed_time.at(device_id);
  }
  [[nodiscard]] timecount_t get_data_completed_time(devid_t device_id) const {
    return per_device_data_completed_time.at(device_id);
  }

protected:
  std::vector<timecount_t> per_device_mapped_time;
  std::vector<timecount_t> per_device_reserved_time;
  std::vector<timecount_t> per_device_launched_time;
  std::vector<timecount_t> per_device_completed_time;
  std::vector<timecount_t> per_device_data_completed_time;
};

struct ResourceRequest {
  Resources requested{0, 0};
  Resources missing{0, 0};
};

struct SchedulerInput {
  std::reference_wrapper<Tasks> tasks;
  std::reference_wrapper<Data> data;
  std::reference_wrapper<Devices> devices;
  std::reference_wrapper<Topology> topology;
  std::reference_wrapper<TaskNoise> task_noise;
  std::reference_wrapper<CommunicationNoise> comm_noise;
  std::reference_wrapper<TransitionConditions> conditions;

  SchedulerInput(Tasks &tasks, Data &data, Devices &devices, Topology &topology,
                 TaskNoise &task_noise, CommunicationNoise &comm_noise,
                 TransitionConditions &conditions)
      : tasks(tasks), data(data), devices(devices), topology(topology), task_noise(task_noise),
        comm_noise(comm_noise), conditions(conditions) {
  }

  SchedulerInput(const SchedulerInput &other) = default;

  SchedulerInput &operator=(const SchedulerInput &other) = default;

  // Shallow copy constructor
  SchedulerInput(SchedulerInput &&other) noexcept
      : tasks(other.tasks), data(other.data), devices(other.devices), topology(other.topology),
        task_noise(other.task_noise), comm_noise(other.comm_noise), conditions(other.conditions) {
  }

  SchedulerInput &operator=(SchedulerInput &&other) noexcept {
    if (this != &other) {
      tasks = other.tasks;
      data = other.data;
      devices = other.devices;
      topology = other.topology;
      task_noise = other.task_noise;
      comm_noise = other.comm_noise;
      conditions = other.conditions;
    }
    return *this;
  }
};

struct GraphStats {
  StatsBundle<timecount_t> duration_stats;
  StatsBundle<mem_t> block_size_stats;
  StatsBundle<mem_t> io_stats;
  StatsBundle<mem_t> total_read_stats;
  StatsBundle<taskid_t> indegree_stats;
  StatsBundle<taskid_t> outdegree_stats;
  StatsBundle<taskid_t> depth_stats;
  StatsBundle<taskid_t> tag_stats;
};

class SchedulerState {
protected:
  timecount_t global_time = 0;
  TaskManager task_manager;
  DeviceManager device_manager;
  CommunicationManager communication_manager;
  DataManager data_manager;
  TaskIDList unlaunched_compute_tasks;
  std::unordered_set<taskid_t> mapped_but_not_reserved_tasks;

  [[nodiscard]] ResourceRequest request_map_resources(taskid_t task_id, devid_t device_id) const;
  [[nodiscard]] ResourceRequest request_reserve_resources(taskid_t task_id,
                                                          devid_t device_id) const;
  [[nodiscard]] ResourceRequest request_launch_resources(taskid_t task_id, devid_t device_id) const;

  void map_resources(taskid_t task_id, devid_t device_id, const Resources &requested);

  void reserve_resources(taskid_t task_id, devid_t device_id, const Resources &requested);

  void launch_resources(taskid_t task_id, devid_t device_id, const Resources &requested);

  void free_resources(taskid_t task_id);

  const TaskIDList &notify_mapped(taskid_t task_id);
  const TaskIDList &notify_reserved(taskid_t task_id);
  void notify_launched(taskid_t task_id);
  const TaskIDList &notify_completed(taskid_t task_id);
  const TaskIDList &notify_data_completed(taskid_t task_id);

public:
  TaskCountInfo counts;
  TaskCostInfo costs;
  GraphStats stats;
  taskid_t current_candidate = 0;

  SchedulerState(SchedulerInput &input)
      : task_manager(TaskManager(input.tasks, input.task_noise)),
        device_manager(DeviceManager(input.devices)),
        communication_manager(input.topology, input.devices, input.comm_noise),
        data_manager(input.data, device_manager, communication_manager),
        counts(input.devices.get().size()),
        costs(input.tasks.get().size(), input.devices.get().size()) {
  }

  SchedulerState(const SchedulerState &other)
      : global_time(other.global_time), task_manager(other.task_manager),
        device_manager(other.device_manager), communication_manager(other.communication_manager),
        data_manager(other.data_manager, device_manager, communication_manager),
        counts(other.counts), costs(other.costs),
        unlaunched_compute_tasks(other.unlaunched_compute_tasks),
        mapped_but_not_reserved_tasks(other.mapped_but_not_reserved_tasks) {
  }

  void update_time(timecount_t time) {
    global_time = std::max(global_time, time);
    assert(time >= global_time);
    assert(global_time >= 0);
  }

  void initialize(bool create_data_tasks = false, bool initialize_data_manager = true) {
    task_manager.initialize(create_data_tasks);
    device_manager.initialize();
    communication_manager.initialize();
    if (initialize_data_manager) {
      data_manager.initialize();
    }
  }

  void initialize_data_manager() {
    data_manager.initialize();
  }

  [[nodiscard]] bool is_complete() const {
    const auto &tasks = task_manager.get_tasks();
    bool data_complete = counts.n_data_completed() ==
                         (tasks.data_size() + task_manager.eviction_tasks.get_n_eviction_tasks());
    bool compute_complete = counts.n_completed() == tasks.compute_size();
    return data_complete and compute_complete;
  }

  [[nodiscard]] const Resources &get_task_resources(taskid_t task_id, devid_t device_id) const;

  [[nodiscard]] const Resources &get_task_resources(taskid_t task_id) const;

  [[nodiscard]] const std::string &get_task_name(taskid_t task_id) const {
    if (task_id < task_manager.n_non_eviction_tasks) {
      return task_manager.get_tasks().get_name(task_id);
    } else {
      return task_manager.eviction_tasks.get_name(task_id);
    }
  }
  [[nodiscard]] const std::string &get_device_name(devid_t device_id) const {
    return device_manager.devices.get().get_name(device_id);
  }

  [[nodiscard]] bool is_compute_task(taskid_t task_id) const;
  [[nodiscard]] bool is_data_task(taskid_t task_id) const;
  [[nodiscard]] bool is_eviction_task(taskid_t task_id) const;

  [[nodiscard]] bool is_mapped(taskid_t task_id) const;
  [[nodiscard]] bool is_reserved(taskid_t task_id) const;
  [[nodiscard]] bool is_launched(taskid_t task_id) const;

  [[nodiscard]] bool is_mappable(taskid_t task_id) const;
  [[nodiscard]] bool is_reservable(taskid_t task_id) const;
  [[nodiscard]] bool is_launchable(taskid_t task_id) const;

  void set_mapping(taskid_t task_id, devid_t device_id);
  [[nodiscard]] devid_t get_mapping(taskid_t task_id) const;
  [[nodiscard]] std::vector<devid_t> &get_mappings();

  [[nodiscard]] timecount_t get_mapped_time(taskid_t task_id) const;
  [[nodiscard]] timecount_t get_reserved_time(taskid_t task_id) const;
  [[nodiscard]] timecount_t get_launched_time(taskid_t task_id) const;
  [[nodiscard]] timecount_t get_completed_time(taskid_t task_id) const;

  bool track_resource_guard() const;
  bool track_location_guard() const;

  [[nodiscard]] vcu_t get_mapped_vcu_at(devid_t device_id, timecount_t time) const;
  [[nodiscard]] vcu_t get_reserved_vcu_at(devid_t device_id, timecount_t time) const;
  [[nodiscard]] vcu_t get_launched_vcu_at(devid_t device_id, timecount_t time) const;

  [[nodiscard]] mem_t get_mapped_mem_at(devid_t device_id, timecount_t time) const;
  [[nodiscard]] mem_t get_reserved_mem_at(devid_t device_id, timecount_t time) const;
  [[nodiscard]] mem_t get_launched_mem_at(devid_t device_id, timecount_t time) const;

  [[nodiscard]] ResourceEventArray<vcu_t> get_mapped_vcu_events(devid_t device_id) const;
  [[nodiscard]] ResourceEventArray<vcu_t> get_reserved_vcu_events(devid_t device_id) const;
  [[nodiscard]] ResourceEventArray<vcu_t> get_launched_vcu_events(devid_t device_id) const;

  [[nodiscard]] ResourceEventArray<mem_t> get_mapped_mem_events(devid_t device_id) const;
  [[nodiscard]] ResourceEventArray<mem_t> get_reserved_mem_events(devid_t device_id) const;
  [[nodiscard]] ResourceEventArray<mem_t> get_launched_mem_events(devid_t device_id) const;

  [[nodiscard]] TaskState get_state_at(taskid_t task_id, timecount_t time) const;

  [[nodiscard]] ValidEventArray get_valid_intervals_mapped(dataid_t data_id,
                                                           devid_t device_id) const;
  [[nodiscard]] ValidEventArray get_valid_intervals_reserved(dataid_t data_id,
                                                             devid_t device_id) const;
  [[nodiscard]] ValidEventArray get_valid_intervals_launched(dataid_t data_id,
                                                             devid_t device_id) const;

  [[nodiscard]] bool check_valid_mapped_at(dataid_t data_id, devid_t device_id,
                                           timecount_t query_time) const;
  [[nodiscard]] bool check_valid_reserved_at(dataid_t data_id, devid_t device_id,
                                             timecount_t query_time) const;
  [[nodiscard]] bool check_valid_launched_at(dataid_t data_id, devid_t device_id,
                                             timecount_t query_time) const;

  // [[nodiscard]] check_valid_mapped(dataid_t data_id, devid_t device_id) const;
  // [[nodiscard]] check_valid_reserved(dataid_t data_id, devid_t device_id) const;
  // [[nodiscard]] check_valid_launched(dataid_t data_id, devid_t device_id) const;

  [[nodiscard]] const PriorityList &get_mapping_priorities() const;
  [[nodiscard]] const PriorityList &get_reserving_priorities() const;
  [[nodiscard]] const PriorityList &get_launching_priorities() const;

  [[nodiscard]] priority_t get_mapping_priority(taskid_t task_id) const;
  [[nodiscard]] priority_t get_reserving_priority(taskid_t task_id) const;
  [[nodiscard]] priority_t get_launching_priority(taskid_t task_id) const;

  template <typename precision_t>
  void fill_supported_devices(taskid_t task_id, std::span<precision_t> valid_devices) const;

  void set_mapping_priority(taskid_t task_id, priority_t priority);
  void set_reserving_priority(taskid_t task_id, priority_t priority);
  void set_launching_priority(taskid_t task_id, priority_t priority);

  void update_mapped_cost(taskid_t task_id, devid_t device_id);
  void update_reserved_cost(taskid_t task_id, devid_t device_id);
  void update_launched_cost(taskid_t task_id, devid_t device_id);
  void update_completed_cost(taskid_t task_id, devid_t device_id);

  devid_t get_data_task_source(taskid_t task_id) const;
  bool is_data_task_virtual(taskid_t task_id) const;

  [[nodiscard]] timecount_t get_execution_time(taskid_t task_id) const;

  StatsBundle<mem_t> get_io_statistics(std::vector<DeviceType> &device_types) const {
    const auto &tasks = task_manager.get_tasks();
    const auto &data = data_manager.get_data();
    std::vector<mem_t> sizes;
    sizes.reserve(tasks.size() + 10 * data.size());

    for (const auto &task : tasks.get_compute_tasks()) {
      const auto &read = task.get_read();
      const auto &write = task.get_write();
      mem_t input_size = data.get_total_size(read);
      mem_t output_size = data.get_total_size(write);
      mem_t internal_size = 0;
      for (const auto &device_type : device_types) {
        const auto &variant = task.get_variant(device_type);
        internal_size += variant.get_mem();
      }
      internal_size = internal_size / device_types.size();

      sizes.push_back(input_size);
      sizes.push_back(output_size);
      sizes.push_back(internal_size);
    }

    return StatsBundle<mem_t>(sizes);
  }

  // StatsBundle<timecount_t> get_total_statistics(std::vector<DeviceType> &device_types) const {
  //    Sum up all in terms of time
  //   const auto &tasks = task_manager.get_tasks();
  //   const auto &data = data_manager.get_data();
  //   std::vector<timecount_t> durations;
  //   durations.reserve(tasks.size());

  // }

  void get_degree_statistics(std::vector<DeviceType> &device_types) {
    const auto &tasks = task_manager.get_tasks();

    std::vector<taskid_t> in_degrees;
    std::vector<taskid_t> out_degrees;

    for (const auto &task : tasks.get_compute_tasks()) {
      taskid_t in_degree = task.get_dependencies().size();
      taskid_t out_degree = task.get_dependents().size();

      in_degrees.push_back(in_degree);
      out_degrees.push_back(out_degree);
    }

    this->stats.indegree_stats = StatsBundle<taskid_t>(in_degrees);
    this->stats.outdegree_stats = StatsBundle<taskid_t>(out_degrees);
  }

  StatsBundle<taskid_t> get_depth_statistics() const {
    const auto &tasks = task_manager.get_tasks();
    std::vector<taskid_t> depths;

    for (const auto &task : tasks.get_compute_tasks()) {
      taskid_t depth = task.get_depth();
      depths.push_back(depth);
    }

    return StatsBundle<taskid_t>(depths);
  }

  StatsBundle<mem_t> get_total_read_statistics() const {
    const auto &tasks = task_manager.get_tasks();
    std::vector<mem_t> sizes;

    for (const auto &task : tasks.get_compute_tasks()) {
      const auto &read = task.get_read();
      mem_t input_size = data_manager.get_data().get_total_size(read);
      sizes.push_back(input_size);
    }

    return StatsBundle<mem_t>(sizes);
  }

  StatsBundle<taskid_t> get_tag_statistics() const {

    const auto &tasks = task_manager.get_tasks();
    std::vector<taskid_t> tags;

    for (const auto &task : tasks.get_compute_tasks()) {
      taskid_t tag = task.get_tag();
      tags.push_back(tag);
    }

    return StatsBundle<taskid_t>(tags);
  }

  void gather_graph_statistics(std::vector<DeviceType> &device_types) {
    const auto &tasks = task_manager.get_tasks();
    const auto &data = data_manager.get_data();
    const auto &devices = device_manager.devices.get();

    this->stats.duration_stats = tasks.get_duration_statistics(device_types);
    this->stats.block_size_stats = data.get_block_statistics(device_types);
    this->stats.io_stats = get_io_statistics(device_types);
    this->get_degree_statistics(device_types);
    this->stats.tag_stats = get_tag_statistics();
    this->stats.depth_stats = get_depth_statistics();
    this->stats.total_read_stats = get_total_read_statistics();
  }

  [[nodiscard]] timecount_t get_global_time() const {
    assert(global_time >= 0);
    return global_time;
  }

  [[nodiscard]] const TaskManager &get_task_manager() const {
    return task_manager;
  }

  [[nodiscard]] const DeviceManager &get_device_manager() const {
    return device_manager;
  }

  [[nodiscard]] const CommunicationManager &get_communication_manager() const {
    return communication_manager;
  }

  [[nodiscard]] const DataManager &get_data_manager() const {
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

// TODO(wlr): Define these (restruct number of mapped and reservable tasks)
class DefaultTransitionConditions : public TransitionConditions {};

class RangeTransitionConditions : public TransitionConditions {
public:
  std::size_t mapped_reserved_gap = 1;
  std::size_t reserved_launched_gap = 1;
  std::size_t total_in_flight = 1;

  RangeTransitionConditions(std::size_t mapped_reserved_gap_, std::size_t reserved_launched_gap_,
                            std::size_t total_in_flight_)
      : mapped_reserved_gap(mapped_reserved_gap_), reserved_launched_gap(reserved_launched_gap_),
        total_in_flight(total_in_flight_) {
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);

    auto n_mapped = state.counts.n_mapped();
    auto n_reserved = state.counts.n_reserved();
    assert(n_mapped >= n_reserved);
    bool flag = (n_mapped - n_reserved) <= mapped_reserved_gap;
    flag = flag && (n_mapped <= total_in_flight);
    return flag;
  }

  bool should_reserve(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto n_reserved = state.counts.n_reserved();
    auto n_launched = state.counts.n_launched();
    assert(n_reserved >= n_launched);

    bool flag = (n_reserved - n_launched) <= reserved_launched_gap;
    return flag;
  }
};

class BatchTransitionConditions : public TransitionConditions {
public:
  std::size_t batch_size = 20;
  std::size_t active_batch = 0;
  std::size_t queue_threshold = 2;
  std::size_t max_in_flight = 16;
  timecount_t last_accessed = 0;

  BatchTransitionConditions(std::size_t batch_size_, std::size_t queue_threshold_,
                            std::size_t max_in_flight_)
      : batch_size(batch_size_), queue_threshold(queue_threshold_), max_in_flight(max_in_flight_) {
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto &counts = state.counts;
    auto n_mapped = counts.n_mapped();
    bool space_flag = (n_mapped <= max_in_flight + active_batch);
    // std::cout << "n_mapped: " << n_mapped << "\n";
    // std::cout << "Max in flight: " << max_in_flight << "\n";
    // std::cout << "Active batch: " << active_batch << "\n";
    // if (space_flag) {
    //   std::cout << "Remaining space: " << (max_in_flight + active_batch - n_mapped) << "\n";
    // } else {
    //   std::cout << "Overfill: " << (n_mapped - max_in_flight - active_batch) << "\n";
    // }

    bool workqueue_flag = false;

    const devid_t n_devices = state.get_device_manager().get_devices().size();

    for (int i = 1; i < n_devices; i++) {
      if (counts.n_mapped(i) < queue_threshold) {
        workqueue_flag = true;
        // std::cout << "Device " << i << " only has " << counts.n_mapped(i) << " mapped tasks.\n";
        break;
      }
    }

    bool flag = space_flag || workqueue_flag;

    if (flag) {
      if (active_batch == 0) {
        // std::cout << "Setting active batch to " << batch_size << "at time "
        //           << state.get_global_time() << "\n";
        last_accessed = state.get_global_time();
        active_batch = batch_size;
      }
    } else {
      // std::cout << "Resetting active batch to 0 at time " << state.get_global_time() << "\n";
      active_batch = 0;
    }

    return flag;
  }
};

#define INITIAL_TASK_BUFFER_SIZE 10
#define INITIAL_DEVICE_BUFFER_SIZE 10

struct SuccessPair {
  bool success = false;
  const TaskIDList *task_list = nullptr;
};

enum class EvictionState {
  NONE = 0,
  WAITING_FOR_COMPLETION = 1,
  RUNNING = 2,
};

class Scheduler {

protected:
  SchedulerState state;
  SchedulerQueues queues;

  std::size_t scheduler_event_count = 1;
  std::size_t success_count = 0;
  std::size_t eviction_count = 0;
  std::size_t compute_on_fly = 0;

  bool can_map = true;
  bool can_reserve = true;
  bool can_launch = true;

  EvictionState eviction_state = EvictionState::NONE;

  TaskIDList task_buffer;
  DeviceIDList device_buffer;
  TaskDeviceList tasks_requesting_eviction;

  void enqueue_data_tasks(taskid_t task_id);

public:
  bool initialized = false;
  BreakpointManager breakpoints;
  std::reference_wrapper<TransitionConditions> conditions;

  Scheduler(SchedulerInput &input)
      : state(input), queues(input.devices), conditions(input.conditions) {
    task_buffer.reserve(INITIAL_TASK_BUFFER_SIZE);
    device_buffer.reserve(INITIAL_DEVICE_BUFFER_SIZE);
    tasks_requesting_eviction.reserve(INITIAL_TASK_BUFFER_SIZE);
  }

  Scheduler(const Scheduler &other) = default;

  void set_transition_conditions(TransitionConditions &conditions_) {
    conditions = conditions_;
  }

  TaskIDList initially_mappable_tasks() {
    const auto &compute_tasks = state.task_manager.get_tasks().get_compute_tasks();
    return GraphManager::initial_tasks(compute_tasks);
  }

  void initialize(bool create_data_tasks = false, bool initialize_data_manager = false) {
    state.initialize(create_data_tasks, initialize_data_manager);
    auto initial_tasks = initially_mappable_tasks();
    queues.push_mappable(initial_tasks, state.get_mapping_priorities());
    initialized = true;
  }

  void initialize_data_manager() {
    state.initialize_data_manager();
  }

  size_t get_mappable_candidates(TorchInt64Arr1D &output_tensor);
  TaskIDList &get_mappable_candidates();

  const TaskIDList &map_task(taskid_t task_id, Action &action);
  void remove_mapped_tasks(ActionList &action_list);

  void map_tasks(MapperEvent &map_event, EventManager &event_manager, Mapper &mapper);
  ExecutionState map_tasks_from_python(ActionList &action_list, EventManager &event_manager);

  SuccessPair reserve_task(taskid_t task_id, devid_t device_id,
                           TaskDeviceList &tasks_requesting_eviction);
  void reserve_tasks(ReserverEvent &reserve_event, EventManager &event_manager);
  bool launch_compute_task(taskid_t task_id, devid_t device_id, EventManager &event_manager);
  bool launch_data_task(taskid_t task_id, devid_t device_id, EventManager &event_manager);
  bool launch_eviction_task(taskid_t task_id, devid_t device_id, EventManager &event_manager);
  void launch_tasks(LauncherEvent &launch_event, EventManager &event_manager);
  void evict(EvictorEvent &eviction_event, EventManager &event_manager);

  void complete_compute_task(taskid_t task_id, devid_t device_id);
  void complete_data_task(taskid_t task_id, devid_t device_id);
  void complete_task(CompleterEvent &complete_event, EventManager &event_manager);
  void complete_eviction_task(taskid_t task_id, devid_t destination_id);

  [[nodiscard]] const TaskIDList &get_task_buffer() const {
    return task_buffer;
  }

  void clear_task_buffer() {
    task_buffer.clear();
    assert(task_buffer.empty());
  }

  TaskIDList &get_clear_task_buffer() {
    task_buffer.clear();
    return task_buffer;
  }

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

  void push_mappable(taskid_t id) {
    priority_t p = state.task_manager.noise.get().get_priority(id);
    queues.push_mappable(id, p);
  }

  void push_mappable(const TaskIDList &ids) {
    const auto &ps = state.task_manager.noise.get().get_priorities();
    queues.push_mappable(ids, ps);
  }

  void push_reservable(taskid_t id, devid_t device) {
    priority_t p = state.task_manager.state.get_reserving_priority(id);
    queues.push_reservable(id, p, device);
  }

  void push_reservable(const TaskIDList &ids) {
    const auto &ps = state.task_manager.state.get_reserving_priorities();
    for (auto id : ids) {
      assert(ps.size() > id);
      queues.push_reservable(id, ps.at(id), state.task_manager.state.get_mapping(id));
    }
  }

  void push_launchable(taskid_t id, devid_t device) {
    priority_t p = state.task_manager.state.get_launching_priority(id);
    queues.push_launchable(id, p, device);
  }

  void push_launchable(const TaskIDList &ids) {
    const auto &ps = state.task_manager.state.get_launching_priorities();
    for (auto id : ids) {
      assert(ps.size() > id);
      queues.push_launchable(id, ps.at(id), state.task_manager.state.get_mapping(id));
    }
  }

  void push_launchable_data(taskid_t id) {
    const auto &data_task = state.task_manager.get_tasks().get_data_task(id);
    taskid_t associated_compute_task = data_task.get_compute_task();
    priority_t p = state.task_manager.state.get_launching_priority(associated_compute_task);
    devid_t device = state.task_manager.state.get_mapping(associated_compute_task);
    queues.push_launchable_data(id, p, device);
  }

  void push_launchable_eviction(taskid_t id) {
    const auto &evict_task = state.task_manager.get_eviction_task(id);
    taskid_t associated_compute_task = evict_task.get_compute_task();
    priority_t p = state.task_manager.state.get_launching_priority(associated_compute_task);
    devid_t device = evict_task.get_device_id();
    queues.push_launchable_eviction(id, p, device);
  }

  void push_launchable_data(const TaskIDList &ids) {
    for (auto id : ids) {
      push_launchable_data(id);
    }
  }

  [[nodiscard]] bool is_complete() const {
    return state.is_complete();
  }

  [[nodiscard]] bool is_breakpoint() const {
    bool breakpoint_status = breakpoints.check_breakpoint();
    return breakpoint_status;
  }

  void check_time_breakpoint() {
    breakpoints.check_time_breakpoint(state.get_global_time());
  }

  friend class SchedulerState;
  friend class SchedulerQueues;
};

class Mapper {

protected:
  DeviceIDList device_buffer;
  std::vector<DeviceType> arch_buffer;

  void fill_arch_targets(taskid_t task_id, const SchedulerState &state) {
    arch_buffer.clear();
    const auto &tasks = state.get_task_manager().get_tasks();
    arch_buffer = tasks.get_supported_architectures(task_id);
    assert(!arch_buffer.empty());
  }

  void fill_device_targets(taskid_t task_id, const SchedulerState &state) {
    device_buffer.clear();
    fill_arch_targets(task_id, state);
    const auto &supported_architectures = arch_buffer;
    const auto &devices = state.get_device_manager().get_devices();

    for (auto arch : supported_architectures) {
      const auto &device_ids = devices.get_devices(arch);
      device_buffer.insert(device_buffer.end(), device_ids.begin(), device_ids.end());
    }
    assert(!device_buffer.empty());
  }

  static const DeviceIDList &get_devices_from_arch(DeviceType arch, SchedulerState &state) {
    const auto &devices = state.get_device_manager().get_devices();
    return devices.get_devices(arch);
  }

public:
  Mapper() = default;

  Mapper(const Mapper &other) = default;

  void initialize() {
    device_buffer.reserve(INITIAL_DEVICE_BUFFER_SIZE);
    arch_buffer.reserve(INITIAL_DEVICE_BUFFER_SIZE);
  }

  virtual Action map_task(taskid_t task_id, const SchedulerState &state) {
    MONUnusedParameter(state);
    SPDLOG_WARN("Mapping task {} with unset mapper", task_id);
    return Action(0, 0);
  }

  virtual ActionList map_tasks(const TaskIDList &task_ids, const SchedulerState &state) {
    ActionList actions;
    for (auto task_id : task_ids) {
      actions.push_back(map_task(task_id, state));
    }
    return actions;
  }
};

class RandomMapper : public Mapper {
protected:
  std::random_device rd;
  std::mt19937 gen;

  DeviceType choose_random_architecture(std::vector<DeviceType> &arch_buffer) {
    std::uniform_int_distribution<std::size_t> dist(0, arch_buffer.size() - 1);
    return arch_buffer.at(dist(gen));
  }

  devid_t choose_random_device(DeviceIDList &device_buffer) {
    std::uniform_int_distribution<std::size_t> dist(0, device_buffer.size() - 1);
    return device_buffer.at(dist(gen));
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
  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    fill_device_targets(task_id, state);
    auto mp = state.get_mapping_priorities()[task_id];
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

  static bool check_supported_architecture(devid_t device_id, taskid_t task_id,
                                           const SchedulerState &state) {
    const auto &tasks = state.get_task_manager().get_tasks();
    const auto &supported_architectures = tasks.get_supported_architectures(task_id);
    const auto &devices = state.get_device_manager().get_devices();
    const auto &device = devices.get_device(device_id);
    return std::find(supported_architectures.begin(), supported_architectures.end(), device.arch) !=
           supported_architectures.end();
  }

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

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    devid_t device_id = 0;
    auto mp = state.get_mapping_priorities()[task_id];
    priority_t rp = mp;
    priority_t lp = mp;

    if (!mapping.empty()) {
      device_id = mapping.at(task_id % mapping.size());
    }
    if (!reserving_priorities.empty()) {
      rp = reserving_priorities.at(task_id % reserving_priorities.size());
    }
    if (!launching_priorities.empty()) {
      lp = launching_priorities.at(task_id % launching_priorities.size());
    }

    assert(check_supported_architecture(device_id, task_id, state));
    assert(device_id < state.get_device_manager().size());

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
  // Stores the temporary EFT values for each device
  std::vector<DeviceTime> finish_time_buffer;

public:
  void record_finish_time(taskid_t task_id, timecount_t time) {
    finish_time_record.at(task_id) = time;
  }

  timecount_t time_for_transfer(dataid_t data_id, devid_t destination,
                                const SchedulerState &state) {
    // Get valid locations in mapped
    const auto &data_manager = state.get_data_manager();
    const auto &communication_manager = state.get_communication_manager();
    const DeviceIDList valid_sources = data_manager.get_valid_mapped_locations(data_id);
    assert(!valid_sources.empty());
    const mem_t data_size = data_manager.get_data().get_size(data_id);

    SPDLOG_DEBUG("Data {} has size {}", data_manager.get_data().get_name(data_id), data_size);

    SPDLOG_DEBUG("Data is located on devices:");
    for (auto source : valid_sources) {
      SPDLOG_DEBUG("{}", state.get_device_name(source));
    }

    SourceRequest req = communication_manager.get_best_source(destination, valid_sources);
    assert(req.found);

    SPDLOG_DEBUG("Best source to transfer data {} to device {} is device {}",
                 data_manager.get_data().get_name(data_id), destination, req.source);
    SPDLOG_DEBUG("It has bandwidth {}",
                 communication_manager.get_bandwidth(req.source, destination));

    return communication_manager.ideal_time_to_transfer(data_size, req.source, destination);
  }

  timecount_t get_finish_time(taskid_t task_id, devid_t device_id, timecount_t start_t,
                              const SchedulerState &state) {
    const auto &read_set = state.get_task_manager().get_tasks().get_read(task_id);

    const DeviceType arch = state.get_device_manager().get_devices().get_device(device_id).arch;

    timecount_t duration =
        state.get_task_manager().get_tasks().get_variant(task_id, arch).get_observed_time();

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

  timecount_t get_dependency_finish_time(taskid_t task_id, const SchedulerState &state) {
    const auto &tasks = state.get_task_manager().get_tasks();
    const auto &dependencies = tasks.get_dependencies(task_id);

    timecount_t max_time = 0;
    for (auto dep_id : dependencies) {
      timecount_t dep_time = finish_time_record.at(dep_id);
      max_time = std::max(max_time, dep_time);
    }

    return max_time;
  }

  void fill_finish_time_buffer(taskid_t task_id, const SchedulerState &state) {
    const auto &task_manager = state.get_task_manager();
    const auto &device_manager = state.get_device_manager();
    const auto &devices = device_manager.get_devices();
    finish_time_buffer.clear();

    timecount_t dep_time = get_dependency_finish_time(task_id, state);

    SPDLOG_DEBUG("Computing EFT for task {}", task_manager.get_tasks().get_name(task_id));
    SPDLOG_DEBUG("Dependency finish time is {}", dep_time);

    for (auto device_id : device_buffer) {
      timecount_t start_time = get_device_available_time(device_id, state);
      SPDLOG_DEBUG("Device {} is available at {}", device_id, start_time);
      start_time = std::max(start_time, dep_time);
      timecount_t finish_time = get_finish_time(task_id, device_id, start_time, state);
      finish_time_buffer.emplace_back(device_id, finish_time);
      SPDLOG_DEBUG("EFT for task {} on device {} is {}", task_id, device_id, finish_time);
    }
  }

  DeviceTime get_best_device(taskid_t task_id, const SchedulerState &state) {
    fill_finish_time_buffer(task_id, state);
    auto min_time = std::numeric_limits<timecount_t>::max();
    devid_t best_device = 0;

    for (std::size_t i = 0; i < finish_time_buffer.size(); i++) {
      devid_t device_id = finish_time_buffer.at(i).device_id;
      timecount_t finish_time = finish_time_buffer.at(i).time;
      SPDLOG_DEBUG("EFT for task {} on device {} is {}", task_id, device_id, finish_time);

      if (finish_time < min_time) {
        min_time = finish_time;
        best_device = device_id;
      }
    }

    return {best_device, min_time};
  }

  EFTMapper() = default;

  EFTMapper(const EFTMapper &other) = default;

  EFTMapper(std::size_t n_tasks, std::size_t n_devices) : finish_time_record(n_tasks, 0) {
    finish_time_buffer.reserve(n_devices);
  }

  void initialize(std::size_t n_tasks, std::size_t n_devices) {
    finish_time_record = std::vector<timecount_t>(n_tasks, 0);
  }

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    finish_time_record.resize(state.get_task_manager().get_tasks().size());
    finish_time_buffer.reserve(state.get_device_manager().size());

    fill_device_targets(task_id, state);
    auto [best_device, min_time] = get_best_device(task_id, state);
    record_finish_time(task_id, min_time);
    auto mp = state.get_mapping_priorities()[task_id];
    return Action(0, best_device, mp, mp);
  }
};

class DequeueEFTMapper : public EFTMapper {

  std::vector<timecount_t> device_available_time_buffer;

public:
  timecount_t get_device_available_time(devid_t device_id, const SchedulerState &state) override {
    return device_available_time_buffer.at(device_id);
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

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    finish_time_record.resize(state.get_task_manager().get_tasks().size());
    finish_time_buffer.reserve(state.get_device_manager().size());
    device_available_time_buffer.resize(state.get_device_manager().size());

    fill_device_targets(task_id, state);
    auto [best_device, min_time] = get_best_device(task_id, state);
    record_finish_time(task_id, min_time);
    set_device_available_time(best_device, min_time);
    auto mp = state.get_mapping_priorities()[task_id];
    auto task_name = state.get_task_manager().get_tasks().get_name(task_id);
    return Action(0, best_device, mp, mp);
  }
};

template <typename precision_t>
void SchedulerState::fill_supported_devices(taskid_t task_id,
                                            std::span<precision_t> valid_devices) const {
  const auto &devices = device_manager.devices.get();
  const auto &tasks = task_manager.get_tasks();

  const auto &task = tasks.get_compute_task(task_id);

  auto supported_architectures = task.get_supported_architectures();

  for (auto arch : supported_architectures) {
    auto &device_ids = devices.get_devices(arch);
    for (auto device_id : device_ids) {
      valid_devices[device_id] = true;
    }
  }
}