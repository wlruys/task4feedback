#pragma once

#include "scheduler_state.hpp"
#include "kahypar_wrapper.hpp"
#include "metis_wrapper.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stack>
#include <type_traits>

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
                                const SchedulerState &state) const {
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

class MemoryAwareEFTMapper : public DequeueEFTMapper {

protected:
  mutable DataIDList sorted_unique_buf_;

  [[nodiscard]] devicemask_t get_eviction_cost_location_flags(dataid_t data_id,
                                                              const SchedulerState &state) const {
    switch (eviction_cost_location_state) {
    case MemoryAwareLocationState::LAUNCHED:
      return state.get_data_manager().get_launched_location_flags(data_id);
    case MemoryAwareLocationState::RESERVED:
      return state.get_data_manager().get_reserved_location_flags(data_id);
    case MemoryAwareLocationState::MAPPED:
      return state.get_data_manager().get_mapped_location_flags(data_id);
    }
    T4F_INVARIANT(false && "Unsupported MemoryAwareLocationState");
  }

  [[nodiscard]] mem_t get_non_local_task_memory(std::span<const dataid_t> unique, devid_t device_id,
                                                const SchedulerState &state) const {
    switch (overflow_state) {
    case MemoryAwareOverflowState::RESERVED:
      return state.get_data_manager().non_local_size_reserved(state.get_data(), unique, device_id);
    case MemoryAwareOverflowState::MAPPED:
      return state.get_data_manager().non_local_size_mapped(state.get_data(), unique, device_id);
    case MemoryAwareOverflowState::LAUNCHED:
      return state.get_data_manager().non_local_size_launched(state.get_data(), unique, device_id);
    }
    T4F_INVARIANT(false && "Unsupported MemoryAwareOverflowState");
  }

  [[nodiscard]] mem_t get_overflow_bytes(devid_t device_id, mem_t task_mem,
                                         const SchedulerState &state) const {
    const auto &dm = state.get_device_manager();
    const mem_t capacity = state.get_devices().get_max_resources(device_id).mem;

    mem_t current_mem = 0;
    switch (overflow_state) {
    case MemoryAwareOverflowState::RESERVED:
      current_mem = dm.get_mem<TaskState::RESERVED>(device_id);
      break;
    case MemoryAwareOverflowState::MAPPED:
      current_mem = dm.get_mem<TaskState::MAPPED>(device_id);
      break;
    case MemoryAwareOverflowState::LAUNCHED:
      current_mem = dm.get_mem<TaskState::LAUNCHED>(device_id);
      break;
    }

    switch (overflow_mode) {
    case MemoryAwareOverflowMode::FULL_SPILL:
      return std::max<mem_t>(0, current_mem + task_mem - capacity);
    case MemoryAwareOverflowMode::INCOMING_ONLY: {
      const mem_t free_mem = std::max<mem_t>(0, capacity - current_mem);
      return std::max<mem_t>(0, task_mem - free_mem);
    }
    }
    T4F_INVARIANT(false && "Unsupported MemoryAwareOverflowMode");
  }

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
                             auto flags = get_eviction_cost_location_flags(victim_id, state);
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
  MemoryAwareLocationState eviction_cost_location_state = MemoryAwareLocationState::LAUNCHED;
  MemoryAwareOverflowState overflow_state = MemoryAwareOverflowState::LAUNCHED;
  MemoryAwareOverflowMode overflow_mode = MemoryAwareOverflowMode::FULL_SPILL;

  MemoryAwareEFTMapper() = default;
  MemoryAwareEFTMapper(const MemoryAwareEFTMapper &) = default;

  MemoryAwareEFTMapper(std::size_t n_tasks, std::size_t n_devices, double alpha_val = 1.0)
      : DequeueEFTMapper(n_tasks, n_devices), alpha(alpha_val) {}

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
    auto best_finish_time = MAX_TIME;
    auto best_device = device_buffer[0];

    for (auto dev : device_buffer) {
      const timecount_t dev_avail = get_device_available_time(dev, state);
      const timecount_t start_t = std::max(dev_avail, dep_time);
      const timecount_t ft = get_finish_time(task_id, dev, start_t, state);

      const DeviceType arch = state.get_devices().get_type(dev);
      const Resources &res = tasks.get_compute_task_resources(task_id, arch);
      mem_t non_local = get_non_local_task_memory(unique, dev, state);
      mem_t task_mem = res.mem + non_local;
      mem_t overflow = get_overflow_bytes(dev, task_mem, state);

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
        best_finish_time = ft;
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
      best_finish_time =
          get_finish_time(task_id, best_device, std::max(dev_avail, dep_time), state);
    }

    return {best_device, best_finish_time};
  }
};


template <typename Derived>
class PartitioningMapperBase : public EFTMapper {
protected:
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

  static constexpr timecount_t saturating_add(timecount_t lhs, timecount_t rhs) {
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

  void ensure_scratch_sizes(std::size_t n_candidates, std::size_t n_compute_tasks,
                            std::size_t n_devices) {
    candidate_records.reserve(n_candidates);
    touched_candidate_tasks.reserve(n_candidates);
    touched_gpu_tasks.reserve(n_candidates);
    vertex_to_candidate_index.reserve(n_candidates);
    partition_labels.reserve(n_candidates);
    partitions.reserve(n_candidates);
    eligible_devices_buffer.reserve(n_devices);

    if (task_to_candidate_index.size() < n_compute_tasks) {
      task_to_candidate_index.resize(n_compute_tasks, -1);
    }
    if (task_to_gpu_vertex.size() < n_compute_tasks) {
      task_to_gpu_vertex.resize(n_compute_tasks, -1);
    }
    if (batch_device_available_time.size() < n_devices) {
      batch_device_available_time.resize(n_devices, 0);
    }

    // Let the derived class reserve its own extra buffers.
    static_cast<Derived *>(this)->ensure_scratch_sizes_extra(n_candidates);
  }

  void prepare_candidates_common(std::span<const taskid_t> task_ids,
                                 const SchedulerState &state) {
    clear_candidate_maps();
    candidate_records.clear();
    vertex_to_candidate_index.clear();
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
      this->fill_device_targets(task_id, state);
      candidate.supported_devices = this->device_buffer;
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
    const auto dep_time = this->get_dependency_finish_time(candidate.task_id, state);
    timecount_t best_finish = MAX_TIME;
    devid_t best_device = candidate.supported_devices.front();

    for (const auto device_id : candidate.supported_devices) {
      const auto start_time =
          std::max(batch_device_available_time[static_cast<std::size_t>(device_id)], dep_time);
      const auto finish_time = this->get_finish_time(candidate.task_id, device_id, start_time, state);
      if (finish_time < best_finish ||
          (finish_time == best_finish && device_id < best_device)) {
        best_finish = finish_time;
        best_device = device_id;
      }
    }

    return {best_device, best_finish};
  }

  void record_assignment(taskid_t task_id, devid_t device_id, const SchedulerState &state) {
    const auto dep_time = this->get_dependency_finish_time(task_id, state);
    const auto start_time =
        std::max(batch_device_available_time[static_cast<std::size_t>(device_id)], dep_time);
    const auto finish_time = this->get_finish_time(task_id, device_id, start_time, state);
    this->record_finish_time(task_id, finish_time, state);
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
      this->action_buffer.push_back(
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
        this->action_buffer.push_back(
            Action{candidate.input_pos, device_id, candidate.priority, candidate.priority});
        record_assignment(candidate.task_id, device_id, state);
      }
    }
  }

  void collect_fallback_indices(std::vector<int32_t> &fallback_candidate_indices) const {
    fallback_candidate_indices.clear();
    fallback_candidate_indices.reserve(candidate_records.size());
    for (std::size_t i = 0; i < candidate_records.size(); ++i) {
      if (!candidate_records[i].gpu_eligible) {
        fallback_candidate_indices.push_back(static_cast<int32_t>(i));
      }
    }
  }

  void collect_all_fallback_indices(std::vector<int32_t> &fallback_candidate_indices) const {
    fallback_candidate_indices.clear();
    fallback_candidate_indices.reserve(candidate_records.size());
    for (std::size_t i = 0; i < candidate_records.size(); ++i) {
      fallback_candidate_indices.push_back(static_cast<int32_t>(i));
    }
  }

  void reserve_common_buffers(std::size_t n_tasks, std::size_t n_devices) {
    candidate_records.reserve(n_tasks);
    touched_candidate_tasks.reserve(n_tasks);
    touched_gpu_tasks.reserve(n_tasks);
    vertex_to_candidate_index.reserve(n_tasks);
    partition_labels.reserve(n_tasks);
    partitions.reserve(std::min(n_tasks, n_devices));
    eligible_devices_buffer.reserve(n_devices);
    batch_device_available_time.reserve(n_devices);
  }
};

class KaHyParMapper : public PartitioningMapperBase<KaHyParMapper> {
  friend class PartitioningMapperBase<KaHyParMapper>;

private:
  KaHyPar_wrapper kahypar;
  KaHyParHypergraph last_hypergraph;

  void ensure_scratch_sizes_extra(std::size_t n_candidates) {
    average_transfer_cost_cache.reserve(n_candidates);
  }

  void prepare_candidates(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    last_hypergraph.clear();
    prepare_candidates_common(task_ids, state);

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

  void build_hypergraph(const SchedulerState &state) {
    if (last_hypergraph.num_vertices <= 0) {
      return;
    }

    const auto &tasks = state.get_tasks();

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
    if (lhs_device_id != rhs_device_id) {
      return lhs_device_id < rhs_device_id;
    }

    return partitions[lhs_partition].min_task_id < partitions[rhs_partition].min_task_id;
  }

  void assign_partition_devices(const SchedulerState &state) {
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

  ActionList &plan_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    this->action_buffer.clear();
    this->action_buffer.reserve(task_ids.size());
    if (task_ids.empty()) {
      last_hypergraph.clear();
      clear_candidate_maps();
      return this->action_buffer;
    }

    this->ensure_task_buffers_size(static_cast<std::size_t>(state.get_tasks().get_n_compute_tasks()));
    if (this->should_reset_for_new_run(state)) {
      this->reset_task_buffers();
    }

    prepare_candidates(task_ids, state);
    build_hypergraph(state);
    reset_batch_device_times(state);

    std::vector<int32_t> fallback_candidate_indices;
    collect_fallback_indices(fallback_candidate_indices);

    const auto nparts = static_cast<int32_t>(
        std::min(eligible_devices_buffer.size(), vertex_to_candidate_index.size()));
    const bool should_fallback_all =
        eligible_devices_buffer.empty() || last_hypergraph.num_vertices <= 1 ||
        last_hypergraph.num_hyperedges() == 0 || nparts <= 1;
    if (should_fallback_all) {
      collect_all_fallback_indices(fallback_candidate_indices);
      emit_fallback_actions(fallback_candidate_indices, state);
      return this->action_buffer;
    }

    partition_labels.assign(vertex_to_candidate_index.size(), 0);
    const bool partitioned =
        kahypar.call_kahypar_partition(last_hypergraph, nparts, partition_labels);
    if (!partitioned) {
      SPDLOG_WARN("KaHyPar call failed, falling back to EFT-style mapping for this batch");
      collect_all_fallback_indices(fallback_candidate_indices);
      emit_fallback_actions(fallback_candidate_indices, state);
      return this->action_buffer;
    }

    build_partitions(state);
    assign_partition_devices(state);
    emit_partition_actions(state);
    emit_fallback_actions(fallback_candidate_indices, state);
    return this->action_buffer;
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
    reserve_common_buffers(n_tasks, n_devices);
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

class METISMapper : public PartitioningMapperBase<METISMapper> {
  friend class PartitioningMapperBase<METISMapper>;

private:
  struct AssignmentSolution {
    bool feasible = false;
    timecount_t total_cost = MAX_TIME;
    std::vector<uint8_t> device_suffix;
  };

  METIS_wrapper metis;
  MetisGraph last_graph;
  std::vector<std::size_t> assignment_partition_order;
  std::vector<AssignmentSolution> assignment_memo;
  std::vector<uint8_t> assignment_memo_ready;

  void ensure_scratch_sizes_extra(std::size_t n_candidates) {
    assignment_partition_order.reserve(n_candidates);
  }

  void prepare_candidates(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    last_graph.clear();
    prepare_candidates_common(task_ids, state);

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
          const auto avg_transfer_cost = average_transfer_cost_for_data(data_id, state);
          // Normalize by (k-1) so that cutting one vertex away from the clique
          // costs avg_transfer_cost regardless of group size, matching the true
          // hyperedge cut objective.
          const auto divisor = static_cast<timecount_t>(group_vertices.size() - 1);
          const auto per_edge_cost =
              std::max<timecount_t>(avg_transfer_cost / divisor, 1);
          const auto group_weight =
              METIS_wrapper::clamp_weight(static_cast<uint64_t>(per_edge_cost));

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

  ActionList &plan_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    this->action_buffer.clear();
    this->action_buffer.reserve(task_ids.size());
    if (task_ids.empty()) {
      last_graph.clear();
      clear_candidate_maps();
      return this->action_buffer;
    }

    this->ensure_task_buffers_size(static_cast<std::size_t>(state.get_tasks().get_n_compute_tasks()));
    if (this->should_reset_for_new_run(state)) {
      this->reset_task_buffers();
    }

    prepare_candidates(task_ids, state);
    build_metis_graph(state);
    reset_batch_device_times(state);

    std::vector<int32_t> fallback_candidate_indices;
    collect_fallback_indices(fallback_candidate_indices);

    const auto nparts = static_cast<int32_t>(
        std::min(eligible_devices_buffer.size(), vertex_to_candidate_index.size()));
    const bool should_fallback_all =
        eligible_devices_buffer.empty() || last_graph.num_vertices <= 1 ||
        last_graph.num_edges() == 0 || nparts <= 1;
    if (should_fallback_all) {
      collect_all_fallback_indices(fallback_candidate_indices);
      emit_fallback_actions(fallback_candidate_indices, state);
      return this->action_buffer;
    }

    partition_labels.assign(vertex_to_candidate_index.size(), 0);
    const bool partitioned = metis.call_metis_partition(last_graph, nparts, partition_labels);
    if (!partitioned) {
      SPDLOG_WARN("METIS call failed, falling back to EFT-style mapping for this batch");
      collect_all_fallback_indices(fallback_candidate_indices);
      emit_fallback_actions(fallback_candidate_indices, state);
      return this->action_buffer;
    }

    build_partitions(state);
    if (!assign_partition_devices(state)) {
      SPDLOG_WARN(
          "METIS partition-device assignment failed, falling back to EFT-style mapping for this batch");
      collect_all_fallback_indices(fallback_candidate_indices);
      emit_fallback_actions(fallback_candidate_indices, state);
      return this->action_buffer;
    }

    emit_partition_actions(state);
    emit_fallback_actions(fallback_candidate_indices, state);
    return this->action_buffer;
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
    reserve_common_buffers(n_tasks, n_devices);
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
    LOCAL_DATA = 5,  // all read data already present on device (zero transfer cost)
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

  // Cross-device claim tracking
  // `claimed_for_device[data_id]` = device_id that has committed to loading data_id
  // in the current plan_tasks call (-1 = unclaimed).

  // Used for two purposes:
  //   (cross-device): when device D2 evaluates block B already claimed by D1,
  //     mark B infeasible for D2 → prevents double-loading.
  //   (cascade): when device D re-evaluates its frontier in a later pass,
  //     blocks it already claimed are treated as locally available (transfer_time=0)
  //     → tasks that were S1(B+E) appear as S0(E) in the next pass.
  std::vector<devid_t> claimed_for_device; // indexed by data_id, -1 = unclaimed

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

  // Claim the best block and all missing blocks of emitted tasks for device_id.
  // True when claimed_for_device[] should be maintained and consulted.
  // Active when either:
  //   - intra_window_coordination: classical IWC flag (enables cascade + cross-device blocking)
  //   - push_pipeline_depth > 0: pipeline-fill mode implicitly needs claimed tracking
  [[nodiscard]] bool use_claimed_tracking() const {
    return intra_window_coordination || push_pipeline_depth > 0;
  }

  void claim_blocks_for_device(dataid_t best_block_id, devid_t device_id,
                               const SchedulerState &state) {
    if (!use_claimed_tracking()) {
      return;
    }
    const auto n_data = claimed_for_device.size();
    const auto mark = [&](dataid_t data_id) {
      const auto idx = static_cast<std::size_t>(data_id);
      if (idx < n_data && claimed_for_device[idx] == static_cast<devid_t>(-1)) {
        claimed_for_device[idx] = device_id;
      }
    };
    // Only claim data that is NOT already locally committed on device_id.
    // Claiming already-local data would block other GPUs from transferring it
    // as a missing block for their own tasks — an unnecessary conflict.
    // The cascade benefit only comes from "in-flight" data (needed transfers);
    // locally-present data is already free on this device and irrelevant to others.
    const auto mark_if_missing = [&](dataid_t data_id) {
      if (state.get_data_manager().check_valid_mapped(data_id, device_id)) {
        return;  // already committed on this device — don't claim
      }
      const auto idx = static_cast<std::size_t>(data_id);
      if (idx < n_data && claimed_for_device[idx] == static_cast<devid_t>(-1)) {
        claimed_for_device[idx] = device_id;
      }
    };
    mark_if_missing(best_block_id);
    const auto &tasks = state.get_tasks();
    for (const auto task_id : trace_emitted_tasks_buffer) {
      for (const auto data_id : tasks.get_read(task_id)) {
        mark_if_missing(data_id);
      }
      // simulate_memory: also mark WRITE data (output) of emitted tasks as planned.
      // This extends the cascade to producer-consumer chains (e.g. Cholesky) where
      // task A's output is input to task B in the same candidate pool.
      if (simulate_memory) {
        for (const auto data_id : tasks.get_write(task_id)) {
          mark_if_missing(data_id);
        }
      }
    }
  }

  // true if block is infeasible for device_id because another device claimed it.
  [[nodiscard]] bool is_claimed_by_other(dataid_t data_id, devid_t device_id) const {
    if (!use_claimed_tracking()) {
      return false;
    }
    const auto idx = static_cast<std::size_t>(data_id);
    if (idx >= claimed_for_device.size()) {
      return false;
    }
    const auto owner = claimed_for_device[idx];
    return owner != static_cast<devid_t>(-1) && owner != device_id;
  }

  // true if data is locally available on device_id — either genuinely mapped
  // or claimed by this device this window (cascade: will be loaded here).
  [[nodiscard]] bool check_claimed_local(dataid_t data_id, devid_t device_id,
                                         const SchedulerState &state) const {
    if (state.get_data_manager().check_valid_mapped(data_id, device_id)) {
      return true;
    }
    if (!use_claimed_tracking()) {
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
    // Item 5 (cascade): if this device claimed data_id this window, treat as local.
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
    // (cross-device): block claimed by a different device this window
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

    // c0 == 0 means no immediate unlocks after loading this block.
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

  // EFT-style fallback: among all active candidates compatible with this device,
  // pick the one with the lowest estimated finish time (device_load + transfer + compute).
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

      // Estimate transfer time for all read data
      const auto &dev_rec = device_candidate_records[task_index];
      timecount_t transfer_time = 0;
      if (dev_rec.compatible && dev_rec.missing_count > 0) {
        for (int32_t i = 0; i < dev_rec.missing_count; ++i) {
          const dataid_t did = missing_data_buffer[dev_rec.missing_begin + i];
          const mem_t dsize = data.get_size(did);
          auto loc_flags = data_manager.get_mapped_location_flags(did);
          auto req = comm.get_best_source(topology, device_id, loc_flags);
          if (req.found) {
            transfer_time += comm.ideal_time_to_transfer(topology, dsize, req.source, device_id);
          }
        }
      }

      const timecount_t finish_time = device_load + transfer_time + task_rec.canonical_duration;
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

    // Classical DARTS mode: process exactly ONE device per trigger, matching the
    // StarPU per-GPU reactive scheduling model.  When true, the under-threshold
    // device with the minimum current mapped time (most starved) is selected.
    // This prevents multiple GPUs from competing for the same data block within
    // a single plan_tasks call, eliminating the N-fold data movement that occurs
    // when all idle GPUs independently pick the same best block.
    if (single_device_per_trigger && pipeline_depth == 0) {
      thresholds.append_under_threshold_gpu_devices(state, selected_devices_buffer);
      if (selected_devices_buffer.size() > 1) {
        // Keep only the most-starved device (minimum mapped time).
        const auto best_it = std::min_element(
            selected_devices_buffer.begin(), selected_devices_buffer.end(),
            [&](devid_t a, devid_t b) {
              return state.costs.get_mapped_time(a) < state.costs.get_mapped_time(b);
            });
        const devid_t best_dev = *best_it;
        selected_devices_buffer.clear();
        selected_devices_buffer.push_back(best_dev);
      }
      return;
    }

    if (pipeline_depth > 0) {
      const auto &devices = state.get_devices();
      const auto &counts = state.counts;
      const devid_t n_devs = devices.size();
      // When max_in_flight is configured and hit, the transition condition should only
      // have fired due to a starvation override.  In that case restrict device selection
      // to truly-starved devices so we don't over-commit past the cap.
      // When max_in_flight == 0 (disabled) or cap is not hit, use the full pipeline_depth
      // criterion to select all hungry devices.
      const bool cap_hit = (max_in_flight > 0) && (counts.n_mapped() >= max_in_flight);
      for (devid_t d = 1; d < n_devs; ++d) {
        if (devices.get_type(d) != DeviceType::GPU) {
          continue;
        }
        const auto mapped = counts.n_mapped(d);
        if (cap_hit) {
          // Starvation-bypass mode: only map to devices that are critically empty.
          if (mapped < starvation_threshold) {
            selected_devices_buffer.push_back(d);
          }
        } else {
          // Normal pipeline mode: fill all hungry devices up to pipeline_depth.
          if (mapped < pipeline_depth) {
            selected_devices_buffer.push_back(d);
          }
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
        static_cast<long long>(static_cast<__int128>(4) * block.c1_compute +
                               static_cast<__int128>(2) * block.c2_compute + block.c3_compute);
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
    case DecisionReason::LOCAL_DATA:
      return "local_data";
    case DecisionReason::FALLBACK:
    default:
      return "fallback";
    }
  }

  DecisionReason emit_actions_for_device(devid_t device_id, int32_t best_block_index,
                                         int32_t fallback_task_index) {
    trace_emitted_tasks_buffer.clear();

    // Local-data priority pass: emit tasks that have ALL their read data already
    // present on this device (missing_count == 0, zero transfer cost).  These are
    // strictly free to run and should always be preferred over any block-loading
    // scheme — particularly important in abundant-memory / transfer-dominated regimes
    // where DARTS's block heuristic would otherwise skip them in favour of tasks
    // that "need" a block to be loaded, causing unnecessary large data transfers.
    emit_task_indices_buffer.clear();
    for (std::size_t task_index = 0; task_index < candidate_tasks.size(); ++task_index) {
      const auto &task_rec = candidate_tasks[task_index];
      const auto &device_record = device_candidate_records[task_index];
      if (!task_rec.active || !device_record.compatible || device_record.missing_count != 0) {
        continue;
      }
      emit_task_indices_buffer.push_back(static_cast<int32_t>(task_index));
    }
    if (!emit_task_indices_buffer.empty()) {
      std::sort(emit_task_indices_buffer.begin(), emit_task_indices_buffer.end(),
                [&](int32_t lhs, int32_t rhs) {
                  return better_task_index(lhs, rhs, candidate_tasks);
                });
      // Emit all locally-available tasks (no cap — they're free, no block loading needed).
      for (const auto task_index : emit_task_indices_buffer) {
        append_action(task_index, device_id);
      }
      return DecisionReason::LOCAL_DATA;
    }

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

    const std::size_t n_data = state.get_data().size();
    ensure_scratch_sizes(task_ids.size(), n_data);
    prepare_candidate_records(task_ids, state);
    collect_selected_devices(state);

    // When finish_time_aware, sort selected devices by ascending mapped workload
    // so that the lightest-loaded device is filled first (EFT-style load balancing).
    if (finish_time_aware && selected_devices_buffer.size() > 1) {
      std::sort(selected_devices_buffer.begin(), selected_devices_buffer.end(),
                [&](devid_t a, devid_t b) {
                  return state.costs.get_mapped_time(a) < state.costs.get_mapped_time(b);
                });
    }

    reset_claimed_for_window(n_data);

    // Global EFT batch mode: bypass per-device DARTS logic and use task-first
    // EFT with per-device planned-data tracking across all selected devices.
    // This matches DequeueEFTMapper quality (data-affinity cascades) while
    // preserving DARTS's DeviceThreshold transition semantics.
    if (global_eft_batch && !selected_devices_buffer.empty()) {
      emit_eft_global_batch(state);
      return action_buffer;
    }

    // DARTSMapper requires device thresholds to be configured (mapped or reserved).
    // When thresholds are disabled (both -1), no devices are selected and no actions
    // are emitted. Use use_mapped_threshold() or use_reserved_threshold() before mapping.
    for (const auto device_id : selected_devices_buffer) {
      // Determine how many cascade passes to run for this device.
      //
      // Classical DARTS (push_pipeline_depth == 0):
      //   - Without IWC: 1 pass (one block per device per trigger).
      //   - With IWC:    cascade_passes (default 3) — chain of data-dependency unlocks.
      //
      // Push-pipeline mode (push_pipeline_depth > 0):
      //   - Run up to push_pipeline_depth passes.  Each pass claims an independent block
      //     and emits its S0 tasks, filling the device's execution pipeline.  Claimed
      //     data is tracked cross-device to prevent N-fold duplication (IWC semantics
      //     are active regardless of the intra_window_coordination flag).
      //   - This is the key push-model adaptation: StarPU's pull model fills each GPU's
      //     planned_task queue in one shot; we replicate that by running many passes.
      const int max_cascade = (push_pipeline_depth > 0) ? push_pipeline_depth
                              : (intra_window_coordination ? cascade_passes : 1);
      for (int pass = 0; pass < max_cascade; ++pass) {
        const auto active_candidates = active_candidate_count();
        if (active_candidates == 0) {
          break;
        }
        int32_t best_fallback_task = build_device_block_stats(device_id, state);
        int32_t best_block_index = choose_best_block_for_device();

        // When finish_time_aware, use EFT-style fallback that considers device load
        // + total transfer + compute to pick the globally cheapest task:
        //   (a) When no block was found (best_block_index < 0): always use EFT.
        //   (b) When the best block has NO S0 tasks (only S1+ tasks remain): the
        //       DARTS block heuristic would pick a task that still needs 2+ more
        //       transfers after loading the best block.  EFT gives a better estimate
        //       of actual finish time and avoids misassigning tasks to devices that
        //       don't have enough of their data.  Override to EFT and suppress the
        //       block-based path by clearing best_block_index.
        //   (c) When the best block is transfer-dominated (sum of S0 compute < transfer
        //       time): the data movement cost outweighs the batch-emission benefit.
        //       EFT picks the task that minimises total finish time with proper data
        //       affinity.  This fires in transfer-heavy regimes (r_interior >> 1).
        if (finish_time_aware && best_fallback_task >= 0) {
          const bool no_block = (best_block_index < 0);
          const bool block_only_s1_plus =
              (best_block_index >= 0) &&
              (frontier_blocks[static_cast<std::size_t>(best_block_index)].s0_count == 0);
          const bool block_transfer_dominated =
              (best_block_index >= 0) &&
              (frontier_blocks[static_cast<std::size_t>(best_block_index)].c0_compute <
               frontier_blocks[static_cast<std::size_t>(best_block_index)].transfer_time);
          if (no_block || block_only_s1_plus || block_transfer_dominated) {
            best_fallback_task = fallback_task_for_device_eft(device_id, state);
            // Suppress block-based emission so that emit_actions_for_device falls
            // through to the fallback path.
            best_block_index = -1;
          }
        }

        const std::size_t actions_before = action_buffer.size();
        const auto reason =
            emit_actions_for_device(device_id, best_block_index, best_fallback_task);
        // For LOCAL_DATA decisions, log without a block reference (no block was loaded).
        const int32_t log_block_index =
            (reason == DecisionReason::LOCAL_DATA) ? -1 : best_block_index;
        log_device_decision(device_id, log_block_index, reason, active_candidates, state);

        if (best_block_index >= 0 && reason != DecisionReason::LOCAL_DATA) {
          claim_blocks_for_device(
              frontier_blocks[static_cast<std::size_t>(best_block_index)].data_id,
              device_id, state);
        }
        reset_frontier();

        // Stop cascading if no new actions were emitted, it was a fallback, or all
        // remaining tasks were locally available (LOCAL_DATA: no block was claimed,
        // so no new tasks become locally available in subsequent passes).
        if (action_buffer.size() == actions_before ||
            reason == DecisionReason::FALLBACK ||
            reason == DecisionReason::LOCAL_DATA) {
          break;
        }
      }
      // Ensure frontier is reset even if we exited early.
      reset_frontier();
    }

    return action_buffer;
  }

public:
  DeviceThresholdState thresholds;
  // Sane defaults: extended frontier + batch emission cap=2 (best in sweep),
  // legacy threshold mode with mapped_threshold=0 (map only to idle devices).
  bool extended_frontier_enabled = true;
  bool extended_batch_emission_enabled = true;
  bool trace_decisions = false;
  int32_t extended_batch_emission_cap = 2;  // cap=2 outperformed cap=4 in sweeps
  bool intra_window_coordination = false;
  int32_t cascade_passes = 3;
  bool finish_time_aware = false;
  // Push-pipeline mode: number of independent cascade passes per device per trigger.
  // 0 = classical DARTS (use cascade_passes; IWC flag controls cascade).
  // > 0 = fill the device pipeline with this many independently-chosen blocks.
  //   Each pass claims the next-best unclaimed block (IWC semantics forced on),
  //   enabling transfer-compute overlap for the next trigger cycle.
  //   Analogous to StarPU DARTS filling planned_task[] in one shot.
  //   Recommended value: ~N_candidates / N_gpus (e.g. 16 for 64-task Jacobi / 4 GPUs).
  int32_t push_pipeline_depth = 0;
  // simulate_memory: extend the claimed-data cascade to include WRITE outputs of
  // emitted tasks.  When task T is planned for device D and writes data O, subsequent
  // tasks that read O see it as "virtually present" on D (zero transfer cost).
  // Replicates StarPU's STARPU_DARTS_SIMULATE_MEMORY=1 behaviour.
  // Useful for producer-consumer chains (e.g. Cholesky) where dependent tasks share
  // the candidate pool.  For iterative workloads (Jacobi) the benefit is minimal
  // since consumers become eligible only after producers complete.
  bool simulate_memory = false;
  // Global EFT batch mode: when true, plan_tasks replaces the per-device DARTS
  // loop with emit_eft_global_batch() — task-first EFT across all selected
  // devices simultaneously, with in-batch planned-data locality tracking.
  // This closely matches DequeueEFTMapper quality while keeping DeviceThreshold
  // transition semantics.  Ideal for abundant-memory / transfer-dominated
  // workloads where DARTS block heuristics underperform.  Orthogonal to
  // finish_time_aware (both can be enabled; global_eft_batch takes priority).
  bool global_eft_batch = false;
  // Per-device task cap for global_eft_batch (selected-devices mode only).
  // 1 = one task per GPU per trigger (safe default).
  // Ignored when global_eft_all_devices=true (all candidates are processed).
  int32_t global_eft_batch_cap = 1;
  // All-devices mode for global_eft_batch: when true, consider ALL devices for
  // each task (not just idle selected_devices_buffer).  Also persists dev_eft
  // across triggers via global_eft_eft_buf, exactly matching DequeueEFTMapper's
  // device_available_time_buffer cascade.  Processes all candidates per trigger.
  // This is the correct adaptation of EFT scoring to the push-model scheduler.
  bool global_eft_all_devices = false;
  // Persistent per-device EFT buffer (absolute times) used when global_eft_all_devices=true.
  // Mirrors DequeueEFTMapper::device_available_time_buffer.  Reset to 0 at run start.
  std::vector<timecount_t> global_eft_eft_buf;
  // Classical StarPU DARTS mode: process exactly one device per plan_tasks call.
  // In StarPU, DARTS is invoked reactively per-GPU (one GPU pulls work at a
  // time).  Our multi-device framework calls plan_tasks for all idle GPUs
  // simultaneously, which causes N-fold data movement when multiple GPUs
  // independently select the same best block.  This flag restores the
  // single-device-per-trigger semantics by selecting only the most-starved
  // GPU per plan_tasks call, preventing cross-GPU block competition.
  // Pair with intra_window_coordination=true for cascade within the device.
  bool single_device_per_trigger = false;
  // Pipeline-depth mode (active when pipeline_depth > 0):
  //   - pipeline_depth: normal pipelining target — keep this many tasks per GPU.
  //   - starvation_threshold: emergency lower bound — when the global cap is hit,
  //     only devices below this are selected (matches DARTSPipelineTransitionConditions).
  //     Must be <= pipeline_depth.  Default 1 (select only idle devices under cap).
  //   - max_in_flight: global cap mirroring the transition condition.  0 = disabled
  //     (no cap enforcement in device selection).
  // Legacy threshold mode (active when pipeline_depth == 0):
  //   - Uses thresholds (mapped/reserved threshold) for device selection.
  //   - Default: mapped_threshold=0 (map only to devices with 0 in-flight tasks).
  int32_t pipeline_depth = 0;
  int32_t starvation_threshold = 1;
  int32_t max_in_flight = 0;

  // ---------------------------------------------------------------------------
  // Global EFT batch: task-first EFT across devices at once.
  //
  // Two modes controlled by global_eft_all_devices:
  //
  //   false (default): selected-devices mode.
  //     Only considers idle devices in selected_devices_buffer.  Per-device
  //     cap = global_eft_batch_cap.  dev_eft reset from committed state each
  //     trigger.  Good for memory-constrained workloads where mapping to busy
  //     devices would cause premature evictions.
  //
  //   true: all-devices mode — matches DequeueEFTMapper quality.
  //     Considers ALL devices for each task (same as EFT's fill_device_targets).
  //     Uses persistent global_eft_eft_buf across triggers (same as EFT's
  //     device_available_time_buffer).  Processes all candidates per trigger
  //     (no per-device cap).  This is the correct push-model EFT adaptation:
  //     tasks with data already on a busy device are assigned there rather than
  //     forced onto an idle device that then duplicates the transfer.
  // ---------------------------------------------------------------------------
  void emit_eft_global_batch(const SchedulerState &state) {
    const auto &tasks_static = state.get_tasks();
    const auto &data_manager = state.get_data_manager();
    const auto &comm = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto &data = state.get_data();
    const auto n_devs = static_cast<std::size_t>(state.get_devices().size());
    const timecount_t global_time = state.get_global_time();

    // Initialize device EFT time estimates.
    std::vector<timecount_t> dev_eft(n_devs, 0);

    if (global_eft_all_devices) {
      // All-devices mode: ensure persistent buffer is large enough.
      if (global_eft_eft_buf.size() < n_devs) {
        global_eft_eft_buf.assign(n_devs, 0);
      }
      // Reset at the start of a new simulation run.
      if (global_time == 0) {
        std::fill(global_eft_eft_buf.begin(), global_eft_eft_buf.end(), 0);
      }
      // dev_eft[d] = max(committed remaining work, planned remaining work from last trigger).
      // committed remaining work = state.costs.get_mapped_time(d) (remaining task durations).
      // planned remaining work = global_eft_eft_buf[d] - global_time (clipped to 0).
      for (std::size_t i = 0; i < n_devs; ++i) {
        const devid_t d = static_cast<devid_t>(i);
        const timecount_t committed = state.costs.get_mapped_time(d);
        const timecount_t planned_remaining =
            (global_eft_eft_buf[i] > global_time) ? (global_eft_eft_buf[i] - global_time) : 0;
        dev_eft[i] = std::max(committed, planned_remaining);
      }
    } else {
      // Selected-devices mode: initialize only selected devices from committed state.
      for (const devid_t d : selected_devices_buffer) {
        dev_eft[static_cast<std::size_t>(d)] = state.costs.get_mapped_time(d);
      }
    }

    // Per-device: data_id → already committed to arrive here in this batch
    std::vector<ankerl::unordered_dense::set<dataid_t>> planned_data(n_devs);

    // Helper: compute transfer time for task on device_id, accounting for
    // both committed state AND planned-in-batch data.
    auto compute_xfer_for_task = [&](const CandidateTask &task, devid_t d) -> timecount_t {
      const auto dev_idx = static_cast<std::size_t>(d);
      timecount_t xfer = 0;
      for (const dataid_t did : tasks_static.get_read(task.task_id)) {
        if (device_in_mask(data_manager.get_mapped_location_flags(did), d)) {
          continue;  // already in committed state
        }
        if (planned_data[dev_idx].count(did)) {
          continue;  // will arrive from an earlier in-batch assignment
        }
        const auto flags = data_manager.get_mapped_location_flags(did);
        const auto req = comm.get_best_source(topology, d, flags);
        if (req.found) {
          xfer += comm.ideal_time_to_transfer(topology, data.get_size(did), req.source, d);
        }
      }
      return xfer;
    };

    // Task-first EFT: repeatedly pick globally cheapest (task, device) pair.
    // Stopping criterion depends on mode:
    //   all-devices: stop when no candidates remain (process everything).
    //   selected: stop when all selected devices hit their per-trigger cap.
    const int32_t tasks_per_device_cap = global_eft_batch_cap;
    std::vector<int32_t> tasks_emitted(n_devs, 0);

    while (true) {
      if (!global_eft_all_devices) {
        // Check if all selected devices have hit their cap.
        bool all_capped = true;
        for (const devid_t d : selected_devices_buffer) {
          if (tasks_emitted[static_cast<std::size_t>(d)] < tasks_per_device_cap) {
            all_capped = false;
            break;
          }
        }
        if (all_capped) break;
      }

      int32_t best_task = -1;
      devid_t best_dev = -1;
      timecount_t best_score = std::numeric_limits<timecount_t>::max();

      // Device iteration: all-devices mode scores every device; selected mode
      // only scores idle selected devices that haven't hit their cap.
      const std::size_t n_score_devs = global_eft_all_devices ? n_devs : selected_devices_buffer.size();
      for (std::size_t di = 0; di < n_score_devs; ++di) {
        const devid_t d = global_eft_all_devices ? static_cast<devid_t>(di) : selected_devices_buffer[di];
        if (!global_eft_all_devices &&
            tasks_emitted[static_cast<std::size_t>(d)] >= tasks_per_device_cap) {
          continue;
        }
        const timecount_t dev_time = dev_eft[static_cast<std::size_t>(d)];

        for (std::size_t ti = 0; ti < candidate_tasks.size(); ++ti) {
          const auto &task = candidate_tasks[ti];
          if (!task.active || !task_supports_device(task, d)) {
            continue;
          }

          const timecount_t xfer = compute_xfer_for_task(task, d);
          const timecount_t score = dev_time + xfer + task.canonical_duration;
          if (score < best_score ||
              (score == best_score &&
               (best_dev > d || (best_dev == d && static_cast<int32_t>(ti) < best_task)))) {
            best_score = score;
            best_task = static_cast<int32_t>(ti);
            best_dev = d;
          }
        }
      }

      if (best_task < 0) {
        break;
      }

      // Commit assignment
      append_action(best_task, best_dev);
      dev_eft[static_cast<std::size_t>(best_dev)] = best_score;
      tasks_emitted[static_cast<std::size_t>(best_dev)]++;
      candidate_tasks[static_cast<std::size_t>(best_task)].active = false;

      // Mark read/write data as planned-locally-present on best_dev.
      const auto &task_rec = candidate_tasks[static_cast<std::size_t>(best_task)];
      const auto best_dev_idx = static_cast<std::size_t>(best_dev);
      for (const dataid_t did : tasks_static.get_read(task_rec.task_id)) {
        planned_data[best_dev_idx].insert(did);
      }
      for (const dataid_t did : tasks_static.get_write(task_rec.task_id)) {
        planned_data[best_dev_idx].insert(did);
      }
    }

    // In all-devices mode: persist the planned end times (absolute) for next trigger.
    if (global_eft_all_devices) {
      for (std::size_t i = 0; i < n_devs; ++i) {
        global_eft_eft_buf[i] = global_time + dev_eft[i];
      }
    }
  }

  DARTSMapper() = default;

  DARTSMapper(const DARTSMapper &other) = default;

  DARTSMapper(std::size_t n_tasks, std::size_t n_devices) {
    candidate_tasks.reserve(n_tasks);
    device_candidate_records.reserve(n_tasks);
    missing_data_buffer.reserve(n_tasks);
    unique_read_buffer.reserve(n_tasks);
    frontier_blocks.reserve(n_tasks);
    // frontier_slot_by_data is indexed by data_id; n_data is unknown here so
    // ensure_scratch_sizes will resize it on the first plan_tasks call.
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
