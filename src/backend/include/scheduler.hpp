#pragma once

#include "scheduler_state.hpp"
#include "transition_conditions.hpp"
#include "eviction_policy.hpp"
#include "mappers.hpp"

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
