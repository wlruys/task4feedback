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
  double alpha = 1.0;
  MemoryAwareLocationState eviction_cost_location_state = MemoryAwareLocationState::RESERVED;
  MemoryAwareOverflowState overflow_state = MemoryAwareOverflowState::RESERVED;
  MemoryAwareOverflowMode overflow_mode = MemoryAwareOverflowMode::INCOMING_ONLY;

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
          continue;
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
  using TaskIndex = int32_t;
  static constexpr TaskIndex kNoTask = -1;

  struct Config {
    uint32_t short_horizon_threshold;
    uint32_t medium_horizon_threshold;
    bool emit_short_horizon = true;
    bool emit_medium_horizon = true;
    uint32_t short_horizon_k = 4;
    uint32_t medium_horizon_k = 4;

    constexpr Config(uint32_t short_horizon_threshold_ = 4,
                     uint32_t medium_horizon_threshold_ = 8)
        : short_horizon_threshold(short_horizon_threshold_),
          medium_horizon_threshold(medium_horizon_threshold_) {}
  };

  struct TaskCandidate {
    taskid_t task_id = -1;
    std::size_t original_input_index = 0;
    priority_t priority = 0;
    devicemask_t supported_devices = 0;
    timecount_t predicted_duration = 0;
  };


  struct TaskAnalysis {
    bool compatible = false;
    std::vector<dataid_t> unique_reads;
    std::vector<dataid_t> missing_reads;
  };

  enum class MissingBand : uint8_t {
    ReadyAfterOne,
    ShortHorizon,
    MediumHorizon,
    LongHorizon
  };

  struct FrontierEntry {
    dataid_t data_id = -1;
    bool transfer_possible = false;
    timecount_t transfer_time = MAX_TIME;

    // ready_after_one (remaining==0): task becomes free when D is loaded
    uint32_t ready_after_one_count = 0;
    timecount_t ready_after_one_compute = 0;
    TaskIndex best_ready_after_one_task = kNoTask;

    // short_horizon (1 <= remaining <= short_horizon_threshold): "1-from-free" etc.
    uint32_t short_horizon_count = 0;
    timecount_t short_horizon_compute = 0;
    TaskIndex best_short_horizon_task = kNoTask;

    // medium_horizon:
    // short_horizon_threshold < remaining <= medium_horizon_threshold
    uint32_t medium_horizon_count = 0;
    timecount_t medium_horizon_compute = 0;
    TaskIndex best_medium_horizon_task = kNoTask;

    // long_horizon: remaining > medium_horizon_threshold
    uint32_t long_horizon_count = 0;
    timecount_t long_horizon_compute = 0;
    TaskIndex best_long_horizon_task = kNoTask;

    // Compute over all compatible tasks that use D.
    // "sum_remaining_task_expected_length". 
    timecount_t remaining_expected_length = 0;
  };

  struct Scratch {
    std::vector<TaskCandidate> candidates;
    std::vector<TaskAnalysis> task_analysis;
    ankerl::unordered_dense::map<dataid_t, FrontierEntry> frontier_by_data;
    std::vector<dataid_t> frontier_keys;
    std::vector<TaskIndex> emission_tasks;
  };

  Scratch scratch_;
  Config config_;
  mutable std::mt19937 rng_{std::random_device{}()};
  mutable std::vector<devid_t> device_order_;

  void validate_config() const {
    T4F_INVARIANT(config_.short_horizon_threshold >= 1);
    T4F_INVARIANT(config_.short_horizon_threshold < config_.medium_horizon_threshold);
  }

  [[nodiscard]] static bool device_in_mask(devicemask_t mask, devid_t device_id) {
    using UMask = std::make_unsigned_t<devicemask_t>;
    constexpr std::size_t kBits = std::numeric_limits<UMask>::digits;

    if (device_id < 0 || static_cast<std::size_t>(device_id) >= kBits) {
      return false;
    }

    const auto unsigned_mask = static_cast<UMask>(mask);
    const auto bit = static_cast<UMask>(UMask{1} << static_cast<std::size_t>(device_id));
    return (unsigned_mask & bit) != 0;
  }

  [[nodiscard]] timecount_t estimate_task_duration(taskid_t task_id,
                                                   const SchedulerState& state) const {
    const auto& tasks = state.get_tasks();
    if (tasks.is_architecture_supported(task_id, DeviceType::GPU)) {
      return tasks.get_mean_duration(task_id, DeviceType::GPU);
    }
    T4F_INVARIANT(tasks.is_architecture_supported(task_id, DeviceType::CPU));
    return tasks.get_mean_duration(task_id, DeviceType::CPU);
  }

  [[nodiscard]] bool estimate_transfer_time(dataid_t data_id,
                                            devid_t device_id,
                                            const SchedulerState& state,
                                            timecount_t& out_time) const {
    if (state.get_data_manager().check_valid_mapped(data_id, device_id)) {
      out_time = 0;
      return true;
    }

    const auto& comm = state.get_communication_manager();
    const auto& topology = state.get_topology();
    const auto flags = state.get_data_manager().get_mapped_location_flags(data_id);
    const auto source = comm.get_best_source(topology, device_id, flags);

    if (!source.found) {
      out_time = MAX_TIME;
      return false;
    }

    out_time = comm.ideal_time_to_transfer(
        topology,
        state.get_data().get_size(data_id),
        source.source,
        device_id);
    return true;
  }

  void ensure_capacity(std::size_t num_tasks) {
    if (scratch_.candidates.size() < num_tasks) {
      scratch_.candidates.resize(num_tasks);
    }
    if (scratch_.task_analysis.size() < num_tasks) {
      scratch_.task_analysis.resize(num_tasks);
    }
  }

  void clear_scratch() {
    scratch_.frontier_by_data.clear();
    scratch_.frontier_keys.clear();
    scratch_.emission_tasks.clear();

    for (auto& analysis : scratch_.task_analysis) {
      analysis.compatible = false;
      analysis.unique_reads.clear();
      analysis.missing_reads.clear();
    }
  }

  [[nodiscard]] bool better_task(const TaskCandidate& lhs,
                                 const TaskCandidate& rhs) const {
    if (lhs.priority != rhs.priority) {
      return lhs.priority > rhs.priority;
    }
    return lhs.task_id < rhs.task_id;
  }

  [[nodiscard]] bool better_task_index(TaskIndex lhs, TaskIndex rhs) const {
    if (rhs == kNoTask) {
      return lhs != kNoTask;
    }
    if (lhs == kNoTask) {
      return false;
    }
    return better_task(
        scratch_.candidates[static_cast<std::size_t>(lhs)],
        scratch_.candidates[static_cast<std::size_t>(rhs)]);
  }

  [[nodiscard]] priority_t task_priority(TaskIndex idx) const {
    if (idx == kNoTask) {
      return std::numeric_limits<priority_t>::min();
    }
    return scratch_.candidates[static_cast<std::size_t>(idx)].priority;
  }

  // minimize transfer_time / ready_after_one_compute,
  // treating ready_after_one_compute == 0 as infinity. 
  [[nodiscard]] bool better_frontier_ratio(const FrontierEntry& lhs,
                                           const FrontierEntry& rhs) const {
    const bool lhs_has_ready_after_one = lhs.ready_after_one_compute > 0;
    const bool rhs_has_ready_after_one = rhs.ready_after_one_compute > 0;

    if (lhs_has_ready_after_one != rhs_has_ready_after_one) {
      return lhs_has_ready_after_one;
    }
    if (!lhs_has_ready_after_one) {
      return false;
    }

    const __int128 lhs_cross =
        static_cast<__int128>(lhs.transfer_time) *
        static_cast<__int128>(rhs.ready_after_one_compute);
    const __int128 rhs_cross =
        static_cast<__int128>(rhs.transfer_time) *
        static_cast<__int128>(lhs.ready_after_one_compute);

    return lhs_cross < rhs_cross;
  }

  // Default simplified DOPT order used here:
  //   1) feasible transfer
  //   2) min transfer_time / ready_after_one_compute
  //   3) max ready_after_one_count
  //   4) max priority among ready_after_one tasks
  //   5) max short_horizon_count (& max medium_horizon_count)
  //   6) max remaining_expected_length
  //   7) stable tie-break by data_id
  //
  // This mirrors StarPU order 7 in structure, with short_horizon_count as "one_from_free_task_count". 
  [[nodiscard]] bool better_frontier_entry(const FrontierEntry& lhs,
                                           const FrontierEntry& rhs) const {
    if (!rhs.transfer_possible) {
      return lhs.transfer_possible;
    }
    if (!lhs.transfer_possible) {
      return false;
    }

    if (better_frontier_ratio(lhs, rhs)) {
      return true;
    }
    if (better_frontier_ratio(rhs, lhs)) {
      return false;
    }

    if (lhs.ready_after_one_count != rhs.ready_after_one_count) {
      return lhs.ready_after_one_count > rhs.ready_after_one_count;
    }

    const auto lhs_priority = task_priority(lhs.best_ready_after_one_task);
    const auto rhs_priority = task_priority(rhs.best_ready_after_one_task);
    if (lhs_priority != rhs_priority) {
      return lhs_priority > rhs_priority;
    }

    if (lhs.short_horizon_count != rhs.short_horizon_count) {
      return lhs.short_horizon_count > rhs.short_horizon_count;
    }

    if (lhs.medium_horizon_count != rhs.medium_horizon_count) {
      return lhs.medium_horizon_count > rhs.medium_horizon_count;
    }

    if (lhs.remaining_expected_length != rhs.remaining_expected_length) {
      return lhs.remaining_expected_length > rhs.remaining_expected_length;
    }

    return lhs.data_id < rhs.data_id;
  }

  // Classify by the number of *other* missing data besides the candidate
  // block D.  missing_count is the total number of missing reads for a task;
  // subtracting 1 gives the count relative to D, matching StarPU semantics:
  //   remaining == 0  →  task becomes free when D is loaded ("free").
  //   remaining == 1  →  one more data needed after D ("1-from-free").
  [[nodiscard]] MissingBand classify_missing_count(uint32_t missing_count) const {
    T4F_INVARIANT(missing_count >= 1);
    const uint32_t remaining = missing_count - 1;
    if (remaining == 0) {
      return MissingBand::ReadyAfterOne;
    }
    if (remaining <= config_.short_horizon_threshold) {
      return MissingBand::ShortHorizon;
    }
    if (remaining <= config_.medium_horizon_threshold) {
      return MissingBand::MediumHorizon;
    }
    return MissingBand::LongHorizon;
  }

  void build_candidates(std::span<const taskid_t> task_ids,
                        const SchedulerState& state) {
    const auto& tasks = state.get_tasks();
    scratch_.candidates.resize(task_ids.size());

    for (std::size_t i = 0; i < task_ids.size(); ++i) {
      const auto task_id = task_ids[i];
      auto& c = scratch_.candidates[i];
      c.task_id = task_id;
      c.original_input_index = i;
      c.priority = state.get_mapping_priority(task_id);
      c.supported_devices = tasks.get_supported_devices_mask(task_id);
      c.predicted_duration = estimate_task_duration(task_id, state);
    }
  }

  [[nodiscard]] devid_t choose_target_device(const SchedulerState& state) const {
    const auto& topology = state.get_topology();
    const auto& devices = state.get_devices();
    const auto n_devices = static_cast<devid_t>(topology.num_devices);

    // Build GPU device list once, then shuffle to break ties fairly.
    device_order_.clear();
    for (devid_t d = 0; d < n_devices; ++d) {
      if (devices.get_type(d) == DeviceType::GPU) {
        device_order_.push_back(d);
      }
    }
    std::shuffle(device_order_.begin(), device_order_.end(), rng_);

    devid_t best_device = -1;
    timecount_t best_load = MAX_TIME;

    for (const devid_t device_id : device_order_) {
      const timecount_t load = state.costs.get_mapped_time(device_id);
      if (best_device < 0 || load < best_load) {
        best_device = device_id;
        best_load = load;
      }
    }

    return best_device;
  }

  FrontierEntry& get_or_create_frontier_entry(dataid_t data_id,
                                              devid_t device_id,
                                              const SchedulerState& state) {
    auto it = scratch_.frontier_by_data.find(data_id);
    if (it != scratch_.frontier_by_data.end()) {
      return it->second;
    }

    timecount_t transfer_time = MAX_TIME;
    const bool transfer_possible =
        estimate_transfer_time(data_id, device_id, state, transfer_time);

    FrontierEntry entry;
    entry.data_id = data_id;
    entry.transfer_possible = transfer_possible;
    entry.transfer_time = transfer_time;

    auto [inserted_it, inserted] =
        scratch_.frontier_by_data.emplace(data_id, std::move(entry));
    (void)inserted;

    scratch_.frontier_keys.push_back(data_id);
    return inserted_it->second;
  }

  [[nodiscard]] TaskIndex analyze_device_and_build_frontier(devid_t device_id,
                                                            const SchedulerState& state) {
    const auto& tasks = state.get_tasks();

    TaskIndex best_priority_fallback = kNoTask;

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& candidate = scratch_.candidates[i];
      auto& analysis = scratch_.task_analysis[i];

      if (!device_in_mask(candidate.supported_devices, device_id)) {
        continue;
      }

      analysis.compatible = true;
      if (better_task_index(task_index, best_priority_fallback)) {
        best_priority_fallback = task_index;
      }

      ankerl::unordered_dense::set<dataid_t> unique_read_set;
      unique_read_set.reserve(14);

      for (const auto data_id : tasks.get_read(candidate.task_id)) {
        if (!unique_read_set.emplace(data_id).second) {
          continue;
        }
        analysis.unique_reads.push_back(data_id);
      }

      for (const auto data_id : analysis.unique_reads) {
        if (!state.get_data_manager().check_valid_mapped(data_id, device_id)) {
          analysis.missing_reads.push_back(data_id);
          auto& entry = get_or_create_frontier_entry(data_id, device_id, state);
          entry.remaining_expected_length += candidate.predicted_duration;
        }
      }
    }

    return best_priority_fallback;
  }

  void accumulate_frontier_metrics() {
    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& candidate = scratch_.candidates[i];
      const auto& analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }
      if (analysis.missing_reads.empty()) {
        continue;
      }

      const uint32_t missing_count =
          static_cast<uint32_t>(analysis.missing_reads.size());
      const auto band = classify_missing_count(missing_count);

      for (const auto data_id : analysis.missing_reads) {
        auto& entry = scratch_.frontier_by_data.at(data_id);

        switch (band) {
          case MissingBand::ReadyAfterOne:
            entry.ready_after_one_count += 1;
            entry.ready_after_one_compute += candidate.predicted_duration;
            if (better_task_index(task_index, entry.best_ready_after_one_task)) {
              entry.best_ready_after_one_task = task_index;
            }
            break;

          case MissingBand::ShortHorizon:
            entry.short_horizon_count += 1;
            entry.short_horizon_compute += candidate.predicted_duration;
            if (better_task_index(task_index, entry.best_short_horizon_task)) {
              entry.best_short_horizon_task = task_index;
            }
            break;

          case MissingBand::MediumHorizon:
            entry.medium_horizon_count += 1;
            entry.medium_horizon_compute += candidate.predicted_duration;
            if (better_task_index(task_index, entry.best_medium_horizon_task)) {
              entry.best_medium_horizon_task = task_index;
            }
            break;

          case MissingBand::LongHorizon:
            entry.long_horizon_count += 1;
            entry.long_horizon_compute += candidate.predicted_duration;
            if (better_task_index(task_index, entry.best_long_horizon_task)) {
              entry.best_long_horizon_task = task_index;
            }
            break;
        }
      }
    }
  }

  [[nodiscard]] FrontierEntry* choose_best_frontier_entry() {
    FrontierEntry* best = nullptr;

    for (const auto data_id : scratch_.frontier_keys) {
      auto& entry = scratch_.frontier_by_data.at(data_id);
      if (!entry.transfer_possible) {
        continue;
      }

      if (best == nullptr || better_frontier_entry(entry, *best)) {
        best = &entry;
      }
    }

    return best;
  }

  void append_action(TaskIndex task_index, devid_t device_id) {
    const auto& candidate = scratch_.candidates[static_cast<std::size_t>(task_index)];
    action_buffer.push_back(Action{
        candidate.original_input_index,
        device_id,
        candidate.priority,
        candidate.priority
    });
  }

  // Emit exactly all tasks that are ready after loading the selected data block:
  // missing_count == 1 and missing selected.data_id.
  void emit_top_k_band_tasks_for_data(const FrontierEntry& selected,
                                      devid_t device_id,
                                      MissingBand band,
                                      uint32_t k) {
    if (k == 0) {
      return;
    }

    scratch_.emission_tasks.clear();

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }

      const uint32_t missing_count = static_cast<uint32_t>(analysis.missing_reads.size());
      if (classify_missing_count(missing_count) != band) {
        continue;
      }

      bool uses_selected = false;
      for (const auto data_id : analysis.missing_reads) {
        if (data_id == selected.data_id) {
          uses_selected = true;
          break;
        }
      }
      if (!uses_selected) {
        continue;
      }

      scratch_.emission_tasks.push_back(task_index);
    }

    if (scratch_.emission_tasks.empty()) {
      return;
    }

    const std::size_t emit_count =
        std::min(static_cast<std::size_t>(k), scratch_.emission_tasks.size());

    std::partial_sort(
        scratch_.emission_tasks.begin(),
        scratch_.emission_tasks.begin() + static_cast<std::ptrdiff_t>(emit_count),
        scratch_.emission_tasks.end(),
        [&](TaskIndex lhs, TaskIndex rhs) {
          return better_task_index(lhs, rhs);
        });

    for (std::size_t j = 0; j < emit_count; ++j) {
      append_action(scratch_.emission_tasks[j], device_id);
    }
  }

  void emit_ready_after_one_tasks_for_data(const FrontierEntry& selected,
                                           devid_t device_id) {
    scratch_.emission_tasks.clear();

    if (selected.ready_after_one_count == 0) {
      return;
    }

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }
      if (analysis.missing_reads.size() != 1) {
        continue;
      }
      if (analysis.missing_reads.front() != selected.data_id) {
        continue;
      }

      scratch_.emission_tasks.push_back(task_index);
    }

    std::sort(
        scratch_.emission_tasks.begin(),
        scratch_.emission_tasks.end(),
        [&](TaskIndex lhs, TaskIndex rhs) {
          return better_task_index(lhs, rhs);
        });

    for (const auto task_index : scratch_.emission_tasks) {
      append_action(task_index, device_id);
    }
  }

  ActionList& plan_tasks(std::span<const taskid_t> task_ids,
                         const SchedulerState& state) {
    action_buffer.clear();
    action_buffer.reserve(task_ids.size());

    if (task_ids.empty()) {
      return action_buffer;
    }

    validate_config();
    ensure_capacity(task_ids.size());
    clear_scratch();
    build_candidates(task_ids, state);

    const devid_t device_id = choose_target_device(state);
    if (device_id < 0) {
      return action_buffer;
    }

    const TaskIndex best_priority_fallback =
        analyze_device_and_build_frontier(device_id, state);

    accumulate_frontier_metrics();

    auto* best_frontier = choose_best_frontier_entry();

    if (best_frontier != nullptr) {
      emit_ready_after_one_tasks_for_data(*best_frontier, device_id);

      // StarPU-like cascaded fallback: only try the next horizon band
      // when no tasks were emitted from the previous band.
      if (action_buffer.empty() && config_.emit_short_horizon) {
        emit_top_k_band_tasks_for_data(*best_frontier, device_id,
                                       MissingBand::ShortHorizon, config_.short_horizon_k);
      }
      if (action_buffer.empty() && config_.emit_medium_horizon) {
        emit_top_k_band_tasks_for_data(*best_frontier, device_id,
                                       MissingBand::MediumHorizon, config_.medium_horizon_k);
      }
    }


    // StarPU-like fallback: if chosen-data logic yields no ready tasks,
    // map the highest-priority compatible task. 
    // Better policy might be to do EFT-style fallback
    if (action_buffer.empty() && best_priority_fallback != kNoTask) {
      append_action(best_priority_fallback, device_id);
    }

    if (action_buffer.empty()) {
      SPDLOG_CRITICAL(
          "DARTSMapper produced no actions for non-empty batch: batch_size={}, chosen_device={}"
          ", compatible_fallback_found={}",
          task_ids.size(), device_id, best_priority_fallback != kNoTask);
    }

    SPDLOG_INFO("DARTSMapping round for device {}: emitting {} actions, n_candidates: {}", device_id, action_buffer.size(), scratch_.candidates.size());
    for (devid_t d = 1; d < static_cast<devid_t>(state.get_topology().num_devices); ++d) {
      SPDLOG_INFO("  [device {}] N_mapped / n_reserved / n_launchable: {} / {} / {}",
                  d,
                  state.counts.n_mapped(d),
                  state.counts.n_reserved(d),
                  state.counts.n_launched(d));
    }
    SPDLOG_INFO("Frontier states: ready_after_one_count={}, short_horizon_count={}, medium_horizon_count={}, long_horizon_count={}",
                best_frontier != nullptr ? best_frontier->ready_after_one_count : 0,
                best_frontier != nullptr ? best_frontier->short_horizon_count : 0,
                best_frontier != nullptr ? best_frontier->medium_horizon_count : 0,
                best_frontier != nullptr ? best_frontier->long_horizon_count : 0);
    SPDLOG_INFO("Best frontier entry: data_id={}, transfer_possible={}, transfer_time={}, best_ready_after_one_task={}, best_short_horizon_task={}, best_medium_horizon_task={}, best_long_horizon_task={}",
                best_frontier != nullptr ? best_frontier->data_id : -1,
                best_frontier != nullptr ? best_frontier->transfer_possible : false,
                best_frontier != nullptr ? best_frontier->transfer_time : MAX_TIME,
                best_frontier != nullptr ? best_frontier->best_ready_after_one_task : kNoTask,
                best_frontier != nullptr ? best_frontier->best_short_horizon_task : kNoTask,
                best_frontier != nullptr ? best_frontier->best_medium_horizon_task : kNoTask,
                best_frontier != nullptr ? best_frontier->best_long_horizon_task : kNoTask);
    for (const auto& action : action_buffer) {
      if (action.pos >= scratch_.candidates.size()) {
        SPDLOG_WARN("DARTSMapper emitted action with out-of-range pos: pos={}, candidates_size={}"
                    ", device={}",
                    action.pos, scratch_.candidates.size(), action.device);
        continue;
      }

      SPDLOG_INFO("Emitting action: task_id={}, pos={}, device={}, reservable_priority={}"
                  ", launchable_priority={}",
                  scratch_.candidates[action.pos].task_id,
                  action.pos,
                  action.device,
                  action.reservable_priority,
                  action.launchable_priority);
    }

    return action_buffer;
  }

public:
  DARTSMapper()
      : config_{} {
    validate_config();
  }

  explicit DARTSMapper(Config config)
      : config_(config) {
    validate_config();
  }

  DARTSMapper(std::size_t num_tasks,
              std::size_t /*num_devices*/,
              Config config = {})
      : config_(config) {
    validate_config();
    scratch_.candidates.reserve(num_tasks);
    scratch_.task_analysis.reserve(num_tasks);
    scratch_.emission_tasks.reserve(num_tasks);
    scratch_.frontier_keys.reserve(num_tasks);
  }

  void set_short_horizon_threshold(uint32_t value) {
    config_.short_horizon_threshold = value;
    validate_config();
  }

  void set_medium_horizon_threshold(uint32_t value) {
    config_.medium_horizon_threshold = value;
    validate_config();
  }

  void set_emit_short_horizon(bool value) { config_.emit_short_horizon = value; }
  void set_emit_medium_horizon(bool value) { config_.emit_medium_horizon = value; }
  void set_short_horizon_k(uint32_t value) { config_.short_horizon_k = value; }
  void set_medium_horizon_k(uint32_t value) { config_.medium_horizon_k = value; }

  [[nodiscard]] uint32_t short_horizon_threshold() const {
    return config_.short_horizon_threshold;
  }

  [[nodiscard]] uint32_t medium_horizon_threshold() const {
    return config_.medium_horizon_threshold;
  }

  [[nodiscard]] bool emit_short_horizon() const { return config_.emit_short_horizon; }
  [[nodiscard]] bool emit_medium_horizon() const { return config_.emit_medium_horizon; }
  [[nodiscard]] uint32_t short_horizon_k() const { return config_.short_horizon_k; }
  [[nodiscard]] uint32_t medium_horizon_k() const { return config_.medium_horizon_k; }

  Action map_task(taskid_t task_id, const SchedulerState& state) override {
    (void)task_id;
    (void)state;
    throw std::runtime_error(
        "DARTSMapper does not support map_task; use map_tasks with a batch of task IDs instead.");
  }

  ActionList& map_tasks(std::span<const taskid_t> task_ids,
                        const SchedulerState& state) override {
    return plan_tasks(task_ids, state);
  }
};

class EnhancedDARTSMapper : public Mapper {
private:
  using TaskIndex = int32_t;
  static constexpr TaskIndex kNoTask = -1;

  struct Config {
    // Horizon bands, interpreted relative to selected block D:
    // remaining = missing_count - 1
    uint32_t short_horizon_threshold;
    uint32_t medium_horizon_threshold;

    // Optional non-classic emission for deeper bands if ready_after_one is empty.
    bool emit_short_horizon;
    bool emit_medium_horizon;
    uint32_t short_horizon_k;
    uint32_t medium_horizon_k;

    bool finish_time_aware;
    bool local_data_first;

    Config() noexcept
        : short_horizon_threshold(4),
          medium_horizon_threshold(8),
          emit_short_horizon(true),
          emit_medium_horizon(true),
          short_horizon_k(4),
          medium_horizon_k(4),
          finish_time_aware(true),
          local_data_first(true) {}
  };

  struct TaskCandidate {
    taskid_t task_id = -1;
    std::size_t original_input_index = 0;
    priority_t priority = 0;
    devicemask_t supported_devices = 0;
    timecount_t predicted_duration = 0;
  };

  struct TaskAnalysis {
    bool compatible = false;
    std::vector<dataid_t> unique_reads;
    std::vector<dataid_t> missing_reads;
  };

  enum class MissingBand : uint8_t {
    ReadyAfterOne,   // remaining == 0
    ShortHorizon,    // 1 <= remaining <= short_horizon_threshold
    MediumHorizon,   // short_horizon_threshold < remaining <= medium_horizon_threshold
    LongHorizon      // remaining > medium_horizon_threshold
  };

  struct FrontierEntry {
    dataid_t data_id = -1;
    bool transfer_possible = false;
    timecount_t transfer_time = MAX_TIME;

    uint32_t ready_after_one_count = 0;
    timecount_t ready_after_one_compute = 0;
    TaskIndex best_ready_after_one_task = kNoTask;

    uint32_t short_horizon_count = 0;
    timecount_t short_horizon_compute = 0;
    TaskIndex best_short_horizon_task = kNoTask;

    uint32_t medium_horizon_count = 0;
    timecount_t medium_horizon_compute = 0;
    TaskIndex best_medium_horizon_task = kNoTask;

    uint32_t long_horizon_count = 0;
    timecount_t long_horizon_compute = 0;
    TaskIndex best_long_horizon_task = kNoTask;

    // Sum over all compatible tasks that use D.
    timecount_t remaining_expected_length = 0;
  };

  struct Scratch {
    std::vector<TaskCandidate> candidates;
    std::vector<TaskAnalysis> task_analysis;
    ankerl::unordered_dense::map<dataid_t, FrontierEntry> frontier_by_data;
    std::vector<dataid_t> frontier_keys;
    std::vector<TaskIndex> emission_tasks;
  };

  Scratch scratch_;
  Config config_;
  mutable std::mt19937 rng_{std::random_device{}()};
  mutable std::vector<devid_t> device_order_;

private:
  void validate_config() const {
    T4F_INVARIANT(config_.short_horizon_threshold >= 1);
    T4F_INVARIANT(config_.short_horizon_threshold < config_.medium_horizon_threshold);
  }

  [[nodiscard]] static bool device_in_mask(devicemask_t mask, devid_t device_id) {
    using UMask = std::make_unsigned_t<devicemask_t>;
    constexpr std::size_t kBits = std::numeric_limits<UMask>::digits;

    if (device_id < 0 || static_cast<std::size_t>(device_id) >= kBits) {
      return false;
    }

    const auto unsigned_mask = static_cast<UMask>(mask);
    const auto bit = static_cast<UMask>(UMask{1} << static_cast<std::size_t>(device_id));
    return (unsigned_mask & bit) != 0;
  }

  [[nodiscard]] timecount_t estimate_task_duration(taskid_t task_id,
                                               const SchedulerState& state) const {
    const auto& tasks = state.get_tasks();
    if (tasks.is_architecture_supported(task_id, DeviceType::GPU)) {
      return tasks.get_mean_duration(task_id, DeviceType::GPU);
    }
    T4F_INVARIANT(tasks.is_architecture_supported(task_id, DeviceType::CPU));
    return tasks.get_mean_duration(task_id, DeviceType::CPU);
  }

  [[nodiscard]] timecount_t duration_for_device(taskid_t task_id,
                                                devid_t device_id,
                                                const SchedulerState& state) const {
    const auto arch = state.get_devices().get_type(device_id);
    return state.get_tasks().get_mean_duration(task_id, arch);
  }

  void ensure_capacity(std::size_t num_tasks) {
    if (scratch_.candidates.size() < num_tasks) {
      scratch_.candidates.resize(num_tasks);
    }
    if (scratch_.task_analysis.size() < num_tasks) {
      scratch_.task_analysis.resize(num_tasks);
    }
  }

  void clear_scratch() {
    scratch_.frontier_by_data.clear();
    scratch_.frontier_keys.clear();
    scratch_.emission_tasks.clear();

    for (auto& analysis : scratch_.task_analysis) {
      analysis.compatible = false;
      analysis.unique_reads.clear();
      analysis.missing_reads.clear();
    }
  }

  [[nodiscard]] bool better_task(const TaskCandidate& lhs,
                                 const TaskCandidate& rhs) const {
    if (lhs.priority != rhs.priority) {
      return lhs.priority > rhs.priority;
    }
    return lhs.task_id < rhs.task_id;
  }

  [[nodiscard]] bool better_task_index(TaskIndex lhs, TaskIndex rhs) const {
    if (rhs == kNoTask) {
      return lhs != kNoTask;
    }
    if (lhs == kNoTask) {
      return false;
    }
    return better_task(
        scratch_.candidates[static_cast<std::size_t>(lhs)],
        scratch_.candidates[static_cast<std::size_t>(rhs)]);
  }

  [[nodiscard]] priority_t task_priority(TaskIndex idx) const {
    if (idx == kNoTask) {
      return std::numeric_limits<priority_t>::min();
    }
    return scratch_.candidates[static_cast<std::size_t>(idx)].priority;
  }

  [[nodiscard]] bool better_frontier_ratio(const FrontierEntry& lhs,
                                           const FrontierEntry& rhs) const {
    const bool lhs_has_ready_after_one = lhs.ready_after_one_compute > 0;
    const bool rhs_has_ready_after_one = rhs.ready_after_one_compute > 0;

    if (lhs_has_ready_after_one != rhs_has_ready_after_one) {
      return lhs_has_ready_after_one;
    }
    if (!lhs_has_ready_after_one) {
      return false;
    }

    const __int128 lhs_cross =
        static_cast<__int128>(lhs.transfer_time) *
        static_cast<__int128>(rhs.ready_after_one_compute);
    const __int128 rhs_cross =
        static_cast<__int128>(rhs.transfer_time) *
        static_cast<__int128>(lhs.ready_after_one_compute);

    return lhs_cross < rhs_cross;
  }

  // DOPT-style structure:
  //   1) feasible transfer
  //   2) min transfer_time / ready_after_one_compute
  //   3) max ready_after_one_count
  //   4) max priority among ready_after_one tasks
  //   5) max short_horizon_count
  //   6) max remaining_expected_length
  //   7) stable tie-break by data_id
  [[nodiscard]] bool better_frontier_entry(const FrontierEntry& lhs,
                                           const FrontierEntry& rhs) const {
    if (!rhs.transfer_possible) {
      return lhs.transfer_possible;
    }
    if (!lhs.transfer_possible) {
      return false;
    }

    if (better_frontier_ratio(lhs, rhs)) {
      return true;
    }
    if (better_frontier_ratio(rhs, lhs)) {
      return false;
    }

    if (lhs.ready_after_one_count != rhs.ready_after_one_count) {
      return lhs.ready_after_one_count > rhs.ready_after_one_count;
    }

    const auto lhs_priority = task_priority(lhs.best_ready_after_one_task);
    const auto rhs_priority = task_priority(rhs.best_ready_after_one_task);
    if (lhs_priority != rhs_priority) {
      return lhs_priority > rhs_priority;
    }

    if (lhs.short_horizon_count != rhs.short_horizon_count) {
      return lhs.short_horizon_count > rhs.short_horizon_count;
    }

    if (lhs.medium_horizon_count != rhs.medium_horizon_count) {
      return lhs.medium_horizon_count > rhs.medium_horizon_count;
    }

    if (lhs.remaining_expected_length != rhs.remaining_expected_length) {
      return lhs.remaining_expected_length > rhs.remaining_expected_length;
    }

    return lhs.data_id < rhs.data_id;
  }

  [[nodiscard]] bool estimate_transfer_time(dataid_t data_id,
                                            devid_t device_id,
                                            const SchedulerState& state,
                                            timecount_t& out_time) const {
    if (state.get_data_manager().check_valid_mapped(data_id, device_id)) {
      out_time = 0;
      return true;
    }

    const auto& comm = state.get_communication_manager();
    const auto& topology = state.get_topology();
    const auto flags = state.get_data_manager().get_mapped_location_flags(data_id);
    const auto source = comm.get_best_source(topology, device_id, flags);

    if (!source.found) {
      out_time = MAX_TIME;
      return false;
    }

    out_time = comm.ideal_time_to_transfer(
        topology,
        state.get_data().get_size(data_id),
        source.source,
        device_id);
    return true;
  }

  [[nodiscard]] MissingBand classify_missing_count(uint32_t missing_count) const {
    T4F_INVARIANT(missing_count >= 1);
    const uint32_t remaining = missing_count - 1;
    if (remaining == 0) {
      return MissingBand::ReadyAfterOne;
    }
    if (remaining <= config_.short_horizon_threshold) {
      return MissingBand::ShortHorizon;
    }
    if (remaining <= config_.medium_horizon_threshold) {
      return MissingBand::MediumHorizon;
    }
    return MissingBand::LongHorizon;
  }

  void build_candidates(std::span<const taskid_t> task_ids,
                        const SchedulerState& state) {
    const auto& tasks = state.get_tasks();
    scratch_.candidates.resize(task_ids.size());

    for (std::size_t i = 0; i < task_ids.size(); ++i) {
      const auto task_id = task_ids[i];
      auto& c = scratch_.candidates[i];
      c.task_id = task_id;
      c.original_input_index = i;
      c.priority = state.get_mapping_priority(task_id);
      c.supported_devices = tasks.get_supported_devices_mask(task_id);
      c.predicted_duration = estimate_task_duration(task_id, state);
    }
  }

  [[nodiscard]] devid_t choose_target_device(const SchedulerState& state) const {
    const auto& devices = state.get_devices();
    const auto n_devices = static_cast<devid_t>(devices.size());

    // Build GPU device list once, then shuffle to break ties fairly.
    device_order_.clear();
    for (devid_t d = 0; d < n_devices; ++d) {
      if (devices.get_type(d) == DeviceType::GPU) {
        device_order_.push_back(d);
      }
    }
    std::shuffle(device_order_.begin(), device_order_.end(), rng_);

    devid_t best_device = -1;
    timecount_t best_load = MAX_TIME;

    for (const devid_t device_id : device_order_) {
      const timecount_t load = state.costs.get_mapped_time(device_id);
      if (best_device < 0 || load < best_load) {
        best_device = device_id;
        best_load = load;
      }
    }

    return best_device;
  }

  FrontierEntry& get_or_create_frontier_entry(dataid_t data_id,
                                              devid_t device_id,
                                              const SchedulerState& state) {
    auto it = scratch_.frontier_by_data.find(data_id);
    if (it != scratch_.frontier_by_data.end()) {
      return it->second;
    }

    timecount_t transfer_time = MAX_TIME;
    const bool transfer_possible =
        estimate_transfer_time(data_id, device_id, state, transfer_time);

    FrontierEntry entry;
    entry.data_id = data_id;
    entry.transfer_possible = transfer_possible;
    entry.transfer_time = transfer_time;

    auto [inserted_it, inserted] =
        scratch_.frontier_by_data.emplace(data_id, std::move(entry));
    (void)inserted;
    scratch_.frontier_keys.push_back(data_id);
    return inserted_it->second;
  }

  // Returns highest-priority compatible task for fallback.
  [[nodiscard]] TaskIndex analyze_device_and_build_frontier(devid_t device_id,
                                                            const SchedulerState& state) {
    const auto& tasks = state.get_tasks();
    TaskIndex best_priority_fallback = kNoTask;

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& candidate = scratch_.candidates[i];
      auto& analysis = scratch_.task_analysis[i];

      if (!device_in_mask(candidate.supported_devices, device_id)) {
        continue;
      }

      analysis.compatible = true;
      if (better_task_index(task_index, best_priority_fallback)) {
        best_priority_fallback = task_index;
      }

      ankerl::unordered_dense::set<dataid_t> unique_read_set;
      unique_read_set.reserve(16);

      for (const auto data_id : tasks.get_read(candidate.task_id)) {
        if (!unique_read_set.emplace(data_id).second) {
          continue;
        }
        analysis.unique_reads.push_back(data_id);
      }

      for (const auto data_id : analysis.unique_reads) {
        if (!state.get_data_manager().check_valid_mapped(data_id, device_id)) {
          analysis.missing_reads.push_back(data_id);
          auto& entry = get_or_create_frontier_entry(data_id, device_id, state);
          entry.remaining_expected_length += candidate.predicted_duration;
        }
      }
    }

    return best_priority_fallback;
  }

  void accumulate_frontier_metrics(devid_t device_id,
                                   const SchedulerState& state) {
    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& candidate = scratch_.candidates[i];
      const auto& analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }
      if (analysis.missing_reads.empty()) {
        continue;
      }

      const uint32_t missing_count =
          static_cast<uint32_t>(analysis.missing_reads.size());
      const auto band = classify_missing_count(missing_count);

      // Use device-specific predicted time for bucket compute accumulation.
      const timecount_t device_duration =
          duration_for_device(candidate.task_id, device_id, state);

      for (const auto data_id : analysis.missing_reads) {
        auto& entry = scratch_.frontier_by_data.at(data_id);

        switch (band) {
          case MissingBand::ReadyAfterOne:
            entry.ready_after_one_count += 1;
            entry.ready_after_one_compute += device_duration;
            if (better_task_index(task_index, entry.best_ready_after_one_task)) {
              entry.best_ready_after_one_task = task_index;
            }
            break;

          case MissingBand::ShortHorizon:
            entry.short_horizon_count += 1;
            entry.short_horizon_compute += device_duration;
            if (better_task_index(task_index, entry.best_short_horizon_task)) {
              entry.best_short_horizon_task = task_index;
            }
            break;

          case MissingBand::MediumHorizon:
            entry.medium_horizon_count += 1;
            entry.medium_horizon_compute += device_duration;
            if (better_task_index(task_index, entry.best_medium_horizon_task)) {
              entry.best_medium_horizon_task = task_index;
            }
            break;

          case MissingBand::LongHorizon:
            entry.long_horizon_count += 1;
            entry.long_horizon_compute += device_duration;
            if (better_task_index(task_index, entry.best_long_horizon_task)) {
              entry.best_long_horizon_task = task_index;
            }
            break;
        }
      }
    }
  }

  [[nodiscard]] FrontierEntry* choose_best_frontier_entry() {
    FrontierEntry* best = nullptr;

    for (const auto data_id : scratch_.frontier_keys) {
      auto& entry = scratch_.frontier_by_data.at(data_id);
      if (!entry.transfer_possible) {
        continue;
      }

      if (best == nullptr || better_frontier_entry(entry, *best)) {
        best = &entry;
      }
    }

    return best;
  }

  void append_action(TaskIndex task_index, devid_t device_id) {
    const auto& candidate = scratch_.candidates[static_cast<std::size_t>(task_index)];
    action_buffer.push_back(Action{
        candidate.original_input_index,
        device_id,
        candidate.priority,
        candidate.priority
    });
  }

  [[nodiscard]] bool task_uses_selected_missing_data(TaskIndex task_index,
                                                     dataid_t selected_data_id) const {
    const auto& analysis = scratch_.task_analysis[static_cast<std::size_t>(task_index)];
    for (const auto data_id : analysis.missing_reads) {
      if (data_id == selected_data_id) {
        return true;
      }
    }
    return false;
  }

  void emit_local_data_tasks(devid_t device_id) {
    scratch_.emission_tasks.clear();

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }
      if (!analysis.missing_reads.empty()) {
        continue;
      }

      scratch_.emission_tasks.push_back(task_index);
    }

    if (scratch_.emission_tasks.empty()) {
      return;
    }

    std::sort(scratch_.emission_tasks.begin(),
              scratch_.emission_tasks.end(),
              [&](TaskIndex lhs, TaskIndex rhs) {
                return better_task_index(lhs, rhs);
              });

    for (const auto task_index : scratch_.emission_tasks) {
      append_action(task_index, device_id);
    }
  }

  void emit_ready_after_one_tasks_for_data(const FrontierEntry& selected,
                                           devid_t device_id) {
    scratch_.emission_tasks.clear();

    if (selected.ready_after_one_count == 0) {
      return;
    }

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }
      if (analysis.missing_reads.size() != 1) {
        continue;
      }
      if (analysis.missing_reads.front() != selected.data_id) {
        continue;
      }

      scratch_.emission_tasks.push_back(task_index);
    }

    std::sort(scratch_.emission_tasks.begin(),
              scratch_.emission_tasks.end(),
              [&](TaskIndex lhs, TaskIndex rhs) {
                return better_task_index(lhs, rhs);
              });

    for (const auto task_index : scratch_.emission_tasks) {
      append_action(task_index, device_id);
    }
  }

  void emit_top_k_band_tasks_for_data(const FrontierEntry& selected,
                                      devid_t device_id,
                                      MissingBand band,
                                      uint32_t k) {
    if (k == 0) {
      return;
    }

    scratch_.emission_tasks.clear();

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }

      const uint32_t missing_count =
          static_cast<uint32_t>(analysis.missing_reads.size());
      if (classify_missing_count(missing_count) != band) {
        continue;
      }
      if (!task_uses_selected_missing_data(task_index, selected.data_id)) {
        continue;
      }

      scratch_.emission_tasks.push_back(task_index);
    }

    if (scratch_.emission_tasks.empty()) {
      return;
    }

    const std::size_t emit_count =
        std::min(static_cast<std::size_t>(k), scratch_.emission_tasks.size());

    std::partial_sort(scratch_.emission_tasks.begin(),
                      scratch_.emission_tasks.begin() +
                          static_cast<std::ptrdiff_t>(emit_count),
                      scratch_.emission_tasks.end(),
                      [&](TaskIndex lhs, TaskIndex rhs) {
                        return better_task_index(lhs, rhs);
                      });

    for (std::size_t j = 0; j < emit_count; ++j) {
      append_action(scratch_.emission_tasks[j], device_id);
    }
  }

  [[nodiscard]] TaskIndex fallback_task_for_device_eft(devid_t device_id,
                                                       const SchedulerState& state) const {
    const timecount_t device_load = state.costs.get_mapped_time(device_id);
    const auto& data_manager = state.get_data_manager();
    const auto& comm = state.get_communication_manager();
    const auto& topology = state.get_topology();
    const auto& data = state.get_data();

    TaskIndex best_task = kNoTask;
    timecount_t best_finish_time = std::numeric_limits<timecount_t>::max();

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto& candidate = scratch_.candidates[i];
      const auto& analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }

      timecount_t transfer_time = 0;
      bool feasible = true;

      for (const auto data_id : analysis.missing_reads) {
        if (data_manager.check_valid_mapped(data_id, device_id)) {
          continue;
        }

        const auto flags = data_manager.get_mapped_location_flags(data_id);
        const auto req = comm.get_best_source(topology, device_id, flags);
        if (!req.found) {
          feasible = false;
          break;
        }

        transfer_time += comm.ideal_time_to_transfer(
            topology, data.get_size(data_id), req.source, device_id);
      }

      if (!feasible) {
        continue;
      }

      const timecount_t finish_time =
          device_load + transfer_time +
          duration_for_device(candidate.task_id, device_id, state);

      if (finish_time < best_finish_time ||
          (finish_time == best_finish_time &&
           better_task_index(task_index, best_task))) {
        best_finish_time = finish_time;
        best_task = task_index;
      }
    }

    return best_task;
  }

  ActionList& plan_tasks(std::span<const taskid_t> task_ids,
                         const SchedulerState& state) {
    action_buffer.clear();
    action_buffer.reserve(task_ids.size());

    if (task_ids.empty()) {
      return action_buffer;
    }

    validate_config();
    ensure_capacity(task_ids.size());
    clear_scratch();
    build_candidates(task_ids, state);

    const devid_t device_id = choose_target_device(state);
    if (device_id < 0) {
      return action_buffer;
    }

    const TaskIndex best_priority_fallback =
        analyze_device_and_build_frontier(device_id, state);

    accumulate_frontier_metrics(device_id, state);
    auto* best_frontier = choose_best_frontier_entry();

    // Enhanced: emit tasks whose data is already fully local.
    if (config_.local_data_first) {
      emit_local_data_tasks(device_id);
      if (!action_buffer.empty()) {
        return action_buffer;
      }
    }

    // Enhanced: finish-time-aware override — when the best frontier block has
    // no ready_after_one payoff or transfer dominates unlocked compute, fall
    // back to EFT selection instead of block-based emission.
    TaskIndex selected_fallback = best_priority_fallback;
    if (config_.finish_time_aware && best_priority_fallback != kNoTask) {
      const bool no_block = (best_frontier == nullptr);
      const bool block_only_deeper =
          (best_frontier != nullptr && best_frontier->ready_after_one_count == 0);
      const bool transfer_dominated =
          (best_frontier != nullptr &&
           best_frontier->ready_after_one_compute < best_frontier->transfer_time);

      if (no_block || block_only_deeper || transfer_dominated) {
        selected_fallback = fallback_task_for_device_eft(device_id, state);
        best_frontier = nullptr;
      }
    }

    if (best_frontier != nullptr) {
      emit_ready_after_one_tasks_for_data(*best_frontier, device_id);

      // Cascaded fallback: only try the next horizon band when no tasks were
      // emitted from the previous band (matches DARTSMapper order).
      if (action_buffer.empty() && config_.emit_short_horizon) {
        emit_top_k_band_tasks_for_data(*best_frontier, device_id,
                                       MissingBand::ShortHorizon,
                                       config_.short_horizon_k);
      }
      if (action_buffer.empty() && config_.emit_medium_horizon) {
        emit_top_k_band_tasks_for_data(*best_frontier, device_id,
                                       MissingBand::MediumHorizon,
                                       config_.medium_horizon_k);
      }
    }

    // Fallback: if block-based logic yields nothing, emit the fallback task.
    if (action_buffer.empty() && selected_fallback != kNoTask) {
      append_action(selected_fallback, device_id);
    }

    return action_buffer;
  }

public:
  EnhancedDARTSMapper()
      : config_{} {
    validate_config();
  }

  explicit EnhancedDARTSMapper(Config config)
      : config_(config) {
    validate_config();
  }

  EnhancedDARTSMapper(std::size_t num_tasks,
                      std::size_t /*num_devices*/,
                      Config config = {})
      : config_(config) {
    validate_config();
    scratch_.candidates.reserve(num_tasks);
    scratch_.task_analysis.reserve(num_tasks);
    scratch_.emission_tasks.reserve(num_tasks);
    scratch_.frontier_keys.reserve(num_tasks);
  }

  void set_short_horizon_threshold(uint32_t value) {
    config_.short_horizon_threshold = value;
    validate_config();
  }

  void set_medium_horizon_threshold(uint32_t value) {
    config_.medium_horizon_threshold = value;
    validate_config();
  }

  void set_emit_short_horizon(bool value) { config_.emit_short_horizon = value; }
  void set_emit_medium_horizon(bool value) { config_.emit_medium_horizon = value; }
  void set_short_horizon_k(uint32_t value) { config_.short_horizon_k = value; }
  void set_medium_horizon_k(uint32_t value) { config_.medium_horizon_k = value; }

  void set_finish_time_aware(bool value) { config_.finish_time_aware = value; }
  void set_local_data_first(bool value) { config_.local_data_first = value; }

  [[nodiscard]] uint32_t short_horizon_threshold() const {
    return config_.short_horizon_threshold;
  }

  [[nodiscard]] uint32_t medium_horizon_threshold() const {
    return config_.medium_horizon_threshold;
  }

  [[nodiscard]] bool emit_short_horizon() const { return config_.emit_short_horizon; }
  [[nodiscard]] bool emit_medium_horizon() const { return config_.emit_medium_horizon; }
  [[nodiscard]] uint32_t short_horizon_k() const { return config_.short_horizon_k; }
  [[nodiscard]] uint32_t medium_horizon_k() const { return config_.medium_horizon_k; }

  [[nodiscard]] bool finish_time_aware() const { return config_.finish_time_aware; }
  [[nodiscard]] bool local_data_first() const { return config_.local_data_first; }

  Action map_task(taskid_t task_id, const SchedulerState& state) override {
    auto& actions = plan_tasks(std::span<const taskid_t>(&task_id, 1), state);
    if (!actions.empty()) {
      return actions.front();
    }

    // Conservative fallback if planning produced nothing.
    fill_device_targets(task_id, state);
    T4F_INVARIANT(!device_buffer.empty());
    const auto mp = state.get_mapping_priority(task_id);
    return Action{0, device_buffer.front(), mp, mp};
  }

  ActionList& map_tasks(std::span<const taskid_t> task_ids,
                        const SchedulerState& state) override {
    return plan_tasks(task_ids, state);
  }
};