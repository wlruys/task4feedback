#pragma once

#include "action.hpp"
#include "breakpoints.hpp"
#include "communication_manager.hpp"
#include "data_manager.hpp"
#include "device_manager.hpp"
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
#include <random>
#include <utility>

#define TIME_TO_MAP 0
#define TIME_TO_RESERVE 0
#define TIME_TO_LAUNCH 0
#define SCHEDULER_TIME_GAP 0

using TaskQueue3 = ContainerQueue<taskid_t, TopKQueueHelper<3>::queue_type>;
using TaskQueue = ContainerQueue<taskid_t, std::priority_queue>;
using DeviceQueue = ActiveQueueIterator<TaskQueue>;

using TaskIDTimeList = std::pair<TaskIDList, std::vector<timecount_t>>;

class TransitionConditions;
class Scheduler;
class Mapper;

class SchedulerQueues {
protected:
  TaskQueue3 mappable;
  DeviceQueue reservable;
  DeviceQueue launchable;
  DeviceQueue data_launchable;

  static TaskType id_to_type(taskid_t id, const Tasks &tasks);

  void id_to_queue(taskid_t id, const TaskStateInfo &state);

public:
  SchedulerQueues(Devices &devices)
      : reservable(devices.size()), launchable(devices.size()),
        data_launchable(devices.size()) {}

  SchedulerQueues(const SchedulerQueues &other) = default;
  SchedulerQueues &operator=(const SchedulerQueues &other) = default;

  void push_mappable(taskid_t id, priority_t p);
  void push_mappable(const TaskIDList &ids, const PriorityList &ps);

  void push_reservable(taskid_t id, priority_t p, devid_t device);
  void push_reservable(const TaskIDList &ids, const PriorityList &ps,
                       devid_t device);
  void push_launchable(taskid_t id, priority_t p, devid_t device);
  void push_launchable(const TaskIDList &ids, const PriorityList &ps,
                       devid_t device);

  void push_launchable_data(taskid_t id, priority_t p, devid_t device);
  void push_launchable_data(const TaskIDList &ids, const PriorityList &ps,
                            devid_t device);

  [[nodiscard]] std::size_t n_mappable() const { return mappable.size(); }
  [[nodiscard]] bool has_mappable() const { return !mappable.empty(); }

  [[nodiscard]] std::size_t n_reservable(devid_t device) const {
    const auto &device_queue = reservable[device];
    return device_queue.size();
  }

  [[nodiscard]] bool has_reservable(devid_t device) const {
    const auto &device_queue = reservable[device];
    return !device_queue.empty();
  }

  [[nodiscard]] bool has_active_reservable() const {
    return reservable.total_active_size() > 0;
  }

  [[nodiscard]] std::size_t n_launchable(devid_t device) const {
    const auto &device_queue = launchable[device];
    return device_queue.size();
  }
  [[nodiscard]] bool has_launchable(devid_t device) const {
    const auto &device_queue = launchable[device];
    return !device_queue.empty();
  }

  [[nodiscard]] bool has_active_launchable() const {
    return launchable.total_active_size() > 0;
  }

  [[nodiscard]] std::size_t n_data_launchable(devid_t device) const {
    const auto &device_queue = data_launchable[device];
    return device_queue.size();
  }
  [[nodiscard]] bool has_data_launchable(devid_t device) const {
    const auto &device_queue = data_launchable[device];
    return !device_queue.empty();
  }

  [[nodiscard]] bool has_active_data_launchable() const {
    return data_launchable.total_active_size() > 0;
  }

  void populate(const TaskManager &task_manager);
  void populate(const TaskIDList &ids, const TaskManager &task_manager);

  friend class Scheduler;
};

class TaskCountInfo {
public:
  TaskCountInfo(std::size_t n_devices);
  void count_mapped(devid_t device_id);
  void count_reserved(devid_t device_id);
  void count_launched(devid_t device_id);

  void count_completed(devid_t device_id);
  void count_data_completed(devid_t device_id);

  [[nodiscard]] std::size_t n_active() const { return n_active_tasks; }
  [[nodiscard]] std::size_t n_mapped() const { return n_mapped_tasks; }
  [[nodiscard]] std::size_t n_reserved() const { return n_reserved_tasks; }
  [[nodiscard]] std::size_t n_launched() const { return n_launched_tasks; }
  [[nodiscard]] std::size_t n_completed() const { return n_completed_tasks; }
  [[nodiscard]] std::size_t n_data_completed() const {
    return n_data_completed_tasks;
  }

  [[nodiscard]] std::size_t n_active(devid_t device_id) const {
    return per_device_mapped_tasks[device_id];
  }

  [[nodiscard]] std::size_t n_mapped(devid_t device_id) const {
    return per_device_mapped_tasks[device_id];
  }

  [[nodiscard]] std::size_t n_reserved(devid_t device_id) const {
    return per_device_reserved_tasks[device_id];
  }

  [[nodiscard]] std::size_t n_launched(devid_t device_id) const {
    return per_device_launched_tasks[device_id];
  }

  [[nodiscard]] std::size_t n_completed(devid_t device_id) const {
    return per_device_completed_tasks[device_id];
  }

  [[nodiscard]] std::size_t n_data_completed(devid_t device_id) const {
    return per_device_data_completed_tasks[device_id];
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

protected:
  std::vector<timecount_t> per_device_mapped_time;
  std::vector<timecount_t> per_device_reserved_time;
  std::vector<timecount_t> per_device_launched_time;
  std::vector<timecount_t> per_device_completed_time;
  std::vector<timecount_t> per_device_data_completed_time;
};

struct ResourceRequest {
  Resources requested;
  Resources missing;
};

class Mapper;

struct SchedulerInput {
  std::reference_wrapper<Tasks> tasks;
  std::reference_wrapper<Data> data;
  std::reference_wrapper<Devices> devices;
  std::reference_wrapper<Topology> topology;
  std::reference_wrapper<Mapper> mapper;
  std::reference_wrapper<TaskNoise> task_noise;
  std::reference_wrapper<CommunicationNoise> comm_noise;

  SchedulerInput(Tasks &tasks, Data &data, Devices &devices, Topology &topology,
                 Mapper &mapper, TaskNoise &task_noise,
                 CommunicationNoise &comm_noise)
      : tasks(tasks), data(data), devices(devices), topology(topology),
        mapper(mapper), task_noise(task_noise), comm_noise(comm_noise) {}

  SchedulerInput(const SchedulerInput &other) = default;

  SchedulerInput &operator=(const SchedulerInput &other) = default;

  // Shallow copy constructor
  SchedulerInput(SchedulerInput &&other) noexcept
      : tasks(other.tasks), data(other.data), devices(other.devices),
        topology(other.topology), mapper(other.mapper),
        task_noise(other.task_noise), comm_noise(other.comm_noise) {}

  SchedulerInput &operator=(SchedulerInput &&other) noexcept {
    if (this != &other) {
      tasks = other.tasks;
      data = other.data;
      devices = other.devices;
      topology = other.topology;
      mapper = other.mapper;
      task_noise = other.task_noise;
      comm_noise = other.comm_noise;
    }
    return *this;
  }
};

class SchedulerState {
protected:
  timecount_t global_time = 0;
  TaskManager task_manager;
  DeviceManager device_manager;
  CommunicationManager communication_manager;
  DataManager data_manager;

  [[nodiscard]] ResourceRequest request_map_resources(taskid_t task_id,
                                                      devid_t device_id) const;
  [[nodiscard]] ResourceRequest
  request_reserve_resources(taskid_t task_id, devid_t device_id) const;
  [[nodiscard]] ResourceRequest
  request_launch_resources(taskid_t task_id, devid_t device_id) const;

  void map_resources(taskid_t task_id, devid_t device_id,
                     const Resources &requested);

  void reserve_resources(taskid_t task_id, devid_t device_id,
                         const Resources &requested);

  void launch_resources(taskid_t task_id, devid_t device_id,
                        const Resources &requested);

  void free_resources(taskid_t task_id);

  const TaskIDList &notify_mapped(taskid_t task_id);
  const TaskIDList &notify_reserved(taskid_t task_id);
  void notify_launched(taskid_t task_id);
  const TaskIDList &notify_completed(taskid_t task_id);
  const TaskIDList &notify_data_completed(taskid_t task_id);

public:
  TaskCountInfo counts;
  TaskCostInfo costs;
  SchedulerState(SchedulerInput &input)
      : task_manager(TaskManager(input.tasks, input.task_noise)),
        device_manager(DeviceManager(input.devices)),
        communication_manager(input.topology, input.devices, input.comm_noise),
        data_manager(input.data, device_manager, communication_manager),
        counts(input.devices.get().size()),
        costs(input.tasks.get().size(), input.devices.get().size()) {}

  void update_time(timecount_t time) {
    global_time = std::max(global_time, time);
    assert(time >= global_time);
    assert(global_time >= 0);
  }

  void initialize(bool create_data_tasks = false) {
    task_manager.initialize(create_data_tasks);
    device_manager.initialize();
    communication_manager.initialize();
    data_manager.initialize();
  }

  [[nodiscard]] bool is_complete() const {
    const auto &tasks = task_manager.get_tasks();
    bool data_complete = counts.n_data_completed() == tasks.data_size();
    bool compute_complete = counts.n_completed() == tasks.compute_size();
    return data_complete and compute_complete;
  }

  [[nodiscard]] const Resources &get_task_resources(taskid_t task_id,
                                                    devid_t device_id) const;

  [[nodiscard]] const Resources &get_task_resources(taskid_t task_id) const;

  [[nodiscard]] const std::string &get_task_name(taskid_t task_id) const {
    const auto &tasks = task_manager.get_tasks();
    return tasks.get_name(task_id);
  }
  [[nodiscard]] const std::string &get_device_name(devid_t device_id) const {
    return device_manager.devices.get().get_name(device_id);
  }

  [[nodiscard]] bool is_compute_task(taskid_t task_id) const;
  [[nodiscard]] bool is_data_task(taskid_t task_id) const;

  [[nodiscard]] bool is_mapped(taskid_t task_id) const;
  [[nodiscard]] bool is_reserved(taskid_t task_id) const;
  [[nodiscard]] bool is_launched(taskid_t task_id) const;

  [[nodiscard]] bool is_mappable(taskid_t task_id) const;
  [[nodiscard]] bool is_reservable(taskid_t task_id) const;
  [[nodiscard]] bool is_launchable(taskid_t task_id) const;

  void set_mapping(taskid_t task_id, devid_t device_id);
  [[nodiscard]] devid_t get_mapping(taskid_t task_id) const;

  [[nodiscard]] const PriorityList &get_mapping_priorities() const;
  [[nodiscard]] const PriorityList &get_reserving_priorities() const;
  [[nodiscard]] const PriorityList &get_launching_priorities() const;

  [[nodiscard]] priority_t get_reserving_priority(taskid_t task_id) const;
  [[nodiscard]] priority_t get_launching_priority(taskid_t task_id) const;

  void set_reserving_priority(taskid_t task_id, priority_t priority);
  void set_launching_priority(taskid_t task_id, priority_t priority);

  [[nodiscard]] timecount_t get_execution_time(taskid_t task_id) const;

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
concept TransitionConditionConcept = requires(T t, SchedulerState &state,
                                              SchedulerQueues &queues) {
  { t.should_map(state, queues) } -> std::convertible_to<bool>;
  { t.should_reserve(state, queues) } -> std::convertible_to<bool>;
  { t.should_launch(state, queues) } -> std::convertible_to<bool>;
  { t.should_launch_data(state, queues) } -> std::convertible_to<bool>;
};

class TransitionConditions {
public:
  bool should_map(SchedulerState &state, SchedulerQueues &queues) {
    MONUnusedParameter(state);
    MONUnusedParameter(queues);
    return true;
  }

  bool should_reserve(SchedulerState &state, SchedulerQueues &queues) {
    MONUnusedParameter(state);
    MONUnusedParameter(queues);
    return true;
  }

  bool should_launch(SchedulerState &state, SchedulerQueues &queues) {
    MONUnusedParameter(state);
    MONUnusedParameter(queues);
    return true;
  }

  bool should_launch_data(SchedulerState &state, SchedulerQueues &queues) {
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

  RangeTransitionConditions(std::size_t mapped_launched_gap_,
                            std::size_t reserved_launched_gap_)
      : mapped_reserved_gap(mapped_launched_gap_),
        reserved_launched_gap(reserved_launched_gap_) {}

  bool should_map(SchedulerState &state, SchedulerQueues &queues) const {
    MONUnusedParameter(queues);
    auto n_mapped = state.counts.n_mapped();
    auto n_reserved = state.counts.n_reserved();
    assert(n_mapped >= n_reserved);

    const bool flag = (n_mapped - n_reserved) < mapped_reserved_gap;
    return flag;
  }

  bool should_reserve(SchedulerState &state, SchedulerQueues &queues) const {
    MONUnusedParameter(queues);
    auto n_reserved = state.counts.n_reserved();
    auto n_launched = state.counts.n_launched();
    assert(n_reserved >= n_launched);

    const bool flag = (n_reserved - n_launched) < reserved_launched_gap;
    return flag;
  }
};

#define INITIAL_TASK_BUFFER_SIZE 10
#define INITIAL_DEVICE_BUFFER_SIZE 10
#define INITIAL_EVENT_BUFFER_SIZE 10

struct SuccessPair {
  bool success = false;
  const TaskIDList *task_list = nullptr;
};

class Scheduler {

protected:
  SchedulerState state;
  SchedulerQueues queues;
  TransitionConditions conditions;

  std::size_t scheduler_event_count = 1;
  std::size_t success_count = 0;

  bool can_map = true;
  bool can_reserve = true;
  bool can_launch = true;

  TaskIDList task_buffer;
  DeviceIDList device_buffer;
  EventList event_buffer;

  void enqueue_data_tasks(taskid_t task_id);

public:
  bool initialized = false;
  BreakpointManager breakpoints;

  Scheduler(SchedulerInput &input) : state(input), queues(input.devices) {
    task_buffer.reserve(INITIAL_TASK_BUFFER_SIZE);
    device_buffer.reserve(INITIAL_DEVICE_BUFFER_SIZE);
    event_buffer.reserve(INITIAL_EVENT_BUFFER_SIZE);
  }

  Scheduler(const Scheduler &other) = default;
  Scheduler &operator=(Scheduler &&other) = default;

  void set_transition_conditions(TransitionConditions &conditions_) {
    this->conditions = conditions_;
  }

  TaskIDList initially_mappable_tasks() {
    const auto &compute_tasks =
        state.task_manager.get_tasks().get_compute_tasks();
    return GraphManager::initial_tasks(compute_tasks);
  }

  void initialize(bool create_data_tasks = false) {
    state.initialize(create_data_tasks);
    const auto &task_states = state.task_manager.state;
    auto initial_tasks = initially_mappable_tasks();
    queues.push_mappable(initial_tasks, task_states.get_mapping_priorities());
    initialized = true;
  }

  TaskIDList &get_mappable_candidates();

  const TaskIDList &map_task(Action &action);
  void remove_mapped_tasks(ActionList &action_list);

  void map_tasks(Event &map_event, EventManager &event_manager, Mapper &mapper);
  void map_tasks_from_python(ActionList &action_list,
                             EventManager &event_manager);

  SuccessPair reserve_task(taskid_t task_id, devid_t device_id);
  void reserve_tasks(Event &reserve_event, EventManager &event_manager);
  bool launch_compute_task(taskid_t task_id, devid_t device_id,
                           EventManager &event_manager);
  bool launch_data_task(taskid_t task_id, devid_t device_id,
                        EventManager &event_manager);
  void launch_tasks(Event &launch_event, EventManager &event_manager);
  void evict(Event &eviction_event, EventManager &event_manager);

  void complete_compute_task(taskid_t task_id, devid_t device_id);
  void complete_data_task(taskid_t task_id, devid_t device_id);
  void complete_task(Event &complete_event, EventManager &event_manager);

  [[nodiscard]] const EventList &get_event_buffer() const {
    return event_buffer;
  }
  [[nodiscard]] const TaskIDList &get_task_buffer() const {
    return task_buffer;
  }

  void clear_event_buffer() {
    event_buffer.clear();
    assert(event_buffer.empty());
  }
  void clear_task_buffer() {
    task_buffer.clear();
    assert(task_buffer.empty());
  }

  TaskIDList &get_clear_task_buffer() {
    task_buffer.clear();
    return task_buffer;
  }

  EventList &get_clear_event_buffer() {
    clear_event_buffer();
    return event_buffer;
  }

  void update_time(timecount_t time) { state.update_time(time); }

  [[nodiscard]] const SchedulerState &get_state() const { return state; }
  [[nodiscard]] const SchedulerQueues &get_queues() const { return queues; }

  void push_mappable(taskid_t id) {
    priority_t p = state.task_manager.state.get_mapping_priority(id);
    queues.push_mappable(id, p);
  }

  void push_mappable(const TaskIDList &ids) {
    const auto &ps = state.task_manager.state.get_mapping_priorities();
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
      queues.push_reservable(id, ps[id],
                             state.task_manager.state.get_mapping(id));
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
      queues.push_launchable(id, ps[id],
                             state.task_manager.state.get_mapping(id));
    }
  }

  void push_launchable_data(taskid_t id) {
    const auto &data_task = state.task_manager.get_tasks().get_data_task(id);
    taskid_t associated_compute_task = data_task.get_compute_task();
    priority_t p = state.task_manager.state.get_launching_priority(
        associated_compute_task);
    devid_t device =
        state.task_manager.state.get_mapping(associated_compute_task);
    queues.push_launchable_data(id, p, device);
  }

  void push_launchable_data(const TaskIDList &ids) {
    for (auto id : ids) {
      push_launchable_data(id);
    }
  }

  [[nodiscard]] bool is_complete() const { return state.is_complete(); }

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
      device_buffer.insert(device_buffer.end(), device_ids.begin(),
                           device_ids.end());
    }
    assert(!device_buffer.empty());
  }

  static const DeviceIDList &get_devices_from_arch(DeviceType arch,
                                                   SchedulerState &state) {
    const auto &devices = state.get_device_manager().get_devices();
    return devices.get_devices(arch);
  }

public:
  Mapper() = default;
  virtual Action map_task(taskid_t task_id, const SchedulerState &state) {
    MONUnusedParameter(state);
    return Action(task_id, 0);
  }

  virtual ActionList map_tasks(const TaskIDList &task_ids,
                               const SchedulerState &state) {
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
    return arch_buffer[dist(gen)];
  }

  devid_t choose_random_device(DeviceIDList &device_buffer) {
    std::uniform_int_distribution<std::size_t> dist(0,
                                                    device_buffer.size() - 1);
    return device_buffer[dist(gen)];
  }

public:
  RandomMapper() = default;
  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    fill_device_targets(task_id, state);
    devid_t device_id = choose_random_device(device_buffer);
    return Action(task_id, 0, device_id);
  }
};

class RoundRobinMapper : public Mapper {
protected:
  std::size_t device_index = 0;

public:
  RoundRobinMapper() = default;
  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    fill_device_targets(task_id, state);
    devid_t device_id = device_buffer[device_index];
    device_index = (device_index + 1) % device_buffer.size();
    return Action(task_id, 0, device_id);
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
    const auto &supported_architectures =
        tasks.get_supported_architectures(task_id);
    const auto &devices = state.get_device_manager().get_devices();
    const auto &device = devices.get_device(device_id);
    return std::find(supported_architectures.begin(),
                     supported_architectures.end(),
                     device.arch) != supported_architectures.end();
  }

public:
  StaticMapper() = default;

  StaticMapper(DeviceIDList device_ids_) : mapping(std::move(device_ids_)) {}

  StaticMapper(DeviceIDList device_ids_, PriorityList reserving_priorities_,
               PriorityList launching_priorities_)
      : mapping(std::move(device_ids_)),
        reserving_priorities(std::move(reserving_priorities_)),
        launching_priorities(std::move(launching_priorities_)) {}

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
    priority_t rp = 0;
    priority_t lp = 0;

    if (!mapping.empty()) {
      device_id = mapping[task_id % mapping.size()];
    }
    if (!reserving_priorities.empty()) {
      rp = reserving_priorities[task_id % reserving_priorities.size()];
    }
    if (!launching_priorities.empty()) {
      lp = launching_priorities[task_id % launching_priorities.size()];
    }

    assert(check_supported_architecture(device_id, task_id, state));
    assert(device_id < state.get_device_manager().size());
    assert(rp >= 0);
    assert(lp >= 0);

    return Action(task_id, 0, device_id, rp, lp);
  }
};

class StaticActionMapper : public Mapper {
protected:
  ActionList actions;

public:
  StaticActionMapper(ActionList actions_) : actions(std::move(actions_)) {}

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    MONUnusedParameter(state);
    return actions[task_id];
  }
};