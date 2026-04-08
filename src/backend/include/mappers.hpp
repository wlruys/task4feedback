#pragma once

#include "scheduler_state.hpp"

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

  [[nodiscard]] devicemask_t get_eviction_cost_location_flags(
      dataid_t data_id, const SchedulerState &state) const {
    switch (eviction_cost_location_state) {
    case MemoryAwareLocationState::LAUNCHED:
      return state.get_data_manager().get_launched_location_flags(data_id);
    case MemoryAwareLocationState::RESERVED:
      return state.get_data_manager().get_reserved_location_flags(data_id);
    case MemoryAwareLocationState::MAPPED:
      return state.get_data_manager().get_mapped_location_flags(data_id);
    }
    T4F_INVARIANT(false && "Unsupported MemoryAwareLocationState");
    return 0;
  }

  [[nodiscard]] mem_t get_non_local_task_memory(std::span<const dataid_t> unique,
                                                devid_t device_id,
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
    return 0;
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
    return 0;
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

    const auto [victim_ids, accumulated] =
        lru.getLRUidsIfAvailable(device_id, static_cast<std::size_t>(bytes_to_evict), protected_ids);
    if (accumulated < static_cast<std::size_t>(bytes_to_evict)) {
      return MAX_TIME;
    }

    timecount_t cost = 0;
    mem_t selected_bytes = 0;
    for (auto victim_id : victim_ids) {
      if (selected_bytes >= bytes_to_evict) {
        break;
      }
      const mem_t victim_bytes = state.get_data().get_size(victim_id);
      selected_bytes += victim_bytes;
      auto flags = get_eviction_cost_location_flags(victim_id, state);
      if (__builtin_popcount(flags) <= 1) {
        cost += comm.ideal_time_to_transfer(topo, victim_bytes, device_id, host);
      }
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
      : DequeueEFTMapper(n_tasks, n_devices), alpha(alpha_val) {
  }

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
      const mem_t non_local = get_non_local_task_memory(unique, dev, state);
      const mem_t task_mem = res.mem + non_local;
      const mem_t overflow = get_overflow_bytes(dev, task_mem, state);

      timecount_t evict_cost = 0;
      if (overflow > 0) {
        evict_cost = estimate_eviction_cost(dev, overflow, sorted_unique_buf_, state);
        if (evict_cost >= MAX_TIME) {
          SPDLOG_DEBUG("Task {} on device {}: INFEASIBLE (cannot evict enough memory)", task_id,
                       dev);
          continue;
        }
      }

      const timecount_t score =
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
        const mem_t usage = dm.get_mem<TaskState::MAPPED>(dev);
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

class DARTSMapper : public Mapper {
private:
  using TaskIndex = int32_t;
  static constexpr TaskIndex kNoTask = -1;

  struct Config {
    uint32_t short_horizon_threshold;
    uint32_t medium_horizon_threshold;
    bool emit_short_horizon = true;
    bool emit_medium_horizon = true;
    uint32_t short_horizon_k = 1;
    uint32_t medium_horizon_k = 1;

    constexpr Config(uint32_t short_horizon_threshold_ = 2,
                     uint32_t medium_horizon_threshold_ = 4)
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

  enum class MissingBand : uint8_t { ReadyAfterOne, ShortHorizon, MediumHorizon, LongHorizon };

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
                                                   const SchedulerState &state) const {
    const auto &tasks = state.get_tasks();
    if (tasks.is_architecture_supported(task_id, DeviceType::GPU)) {
      return tasks.get_mean_duration(task_id, DeviceType::GPU);
    }
    T4F_INVARIANT(tasks.is_architecture_supported(task_id, DeviceType::CPU));
    return tasks.get_mean_duration(task_id, DeviceType::CPU);
  }

  [[nodiscard]] bool estimate_transfer_time(dataid_t data_id, devid_t device_id,
                                            const SchedulerState &state,
                                            timecount_t &out_time) const {
    if (state.get_data_manager().check_valid_mapped(data_id, device_id)) {
      out_time = 0;
      return true;
    }

    const auto &comm = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto flags = state.get_data_manager().get_mapped_location_flags(data_id);
    const auto source = comm.get_best_source(topology, device_id, flags);

    if (!source.found) {
      out_time = MAX_TIME;
      return false;
    }

    out_time = comm.ideal_time_to_transfer(topology, state.get_data().get_size(data_id),
                                           source.source, device_id);
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

    for (auto &analysis : scratch_.task_analysis) {
      analysis.compatible = false;
      analysis.unique_reads.clear();
      analysis.missing_reads.clear();
    }
  }

  [[nodiscard]] bool better_task(const TaskCandidate &lhs, const TaskCandidate &rhs) const {
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
    return better_task(scratch_.candidates[static_cast<std::size_t>(lhs)],
                       scratch_.candidates[static_cast<std::size_t>(rhs)]);
  }

  [[nodiscard]] priority_t task_priority(TaskIndex idx) const {
    if (idx == kNoTask) {
      return std::numeric_limits<priority_t>::min();
    }
    return scratch_.candidates[static_cast<std::size_t>(idx)].priority;
  }

  [[nodiscard]] bool better_frontier_ratio(const FrontierEntry &lhs,
                                           const FrontierEntry &rhs) const {
    const bool lhs_has_ready_after_one = lhs.ready_after_one_compute > 0;
    const bool rhs_has_ready_after_one = rhs.ready_after_one_compute > 0;

    if (lhs_has_ready_after_one != rhs_has_ready_after_one) {
      return lhs_has_ready_after_one;
    }
    if (!lhs_has_ready_after_one) {
      return false;
    }

    const __int128 lhs_cross = static_cast<__int128>(lhs.transfer_time) *
                               static_cast<__int128>(rhs.ready_after_one_compute);
    const __int128 rhs_cross = static_cast<__int128>(rhs.transfer_time) *
                               static_cast<__int128>(lhs.ready_after_one_compute);

    return lhs_cross < rhs_cross;
  }

  [[nodiscard]] bool better_frontier_entry(const FrontierEntry &lhs,
                                           const FrontierEntry &rhs) const {
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

  void build_candidates(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    const auto &tasks = state.get_tasks();
    scratch_.candidates.resize(task_ids.size());

    for (std::size_t i = 0; i < task_ids.size(); ++i) {
      const auto task_id = task_ids[i];
      auto &candidate = scratch_.candidates[i];
      candidate.task_id = task_id;
      candidate.original_input_index = i;
      candidate.priority = state.get_mapping_priority(task_id);
      candidate.supported_devices = tasks.get_supported_devices_mask(task_id);
      candidate.predicted_duration = estimate_task_duration(task_id, state);
    }
  }

  [[nodiscard]] devid_t choose_target_device(const SchedulerState &state) const {
    const auto &devices = state.get_devices();
    const auto n_devices = static_cast<devid_t>(devices.size());

    device_order_.clear();
    for (devid_t device_id = 0; device_id < n_devices; ++device_id) {
      if (devices.get_type(device_id) == DeviceType::GPU) {
        device_order_.push_back(device_id);
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

  FrontierEntry &get_or_create_frontier_entry(dataid_t data_id, devid_t device_id,
                                              const SchedulerState &state) {
    auto it = scratch_.frontier_by_data.find(data_id);
    if (it != scratch_.frontier_by_data.end()) {
      return it->second;
    }

    timecount_t transfer_time = MAX_TIME;
    const bool transfer_possible = estimate_transfer_time(data_id, device_id, state, transfer_time);

    FrontierEntry entry;
    entry.data_id = data_id;
    entry.transfer_possible = transfer_possible;
    entry.transfer_time = transfer_time;

    auto [inserted_it, inserted] = scratch_.frontier_by_data.emplace(data_id, std::move(entry));
    MONUnusedParameter(inserted);
    scratch_.frontier_keys.push_back(data_id);
    return inserted_it->second;
  }

  [[nodiscard]] TaskIndex analyze_device_and_build_frontier(devid_t device_id,
                                                            const SchedulerState &state) {
    const auto &tasks = state.get_tasks();
    TaskIndex best_priority_fallback = kNoTask;

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto &candidate = scratch_.candidates[i];
      auto &analysis = scratch_.task_analysis[i];

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
          auto &entry = get_or_create_frontier_entry(data_id, device_id, state);
          entry.remaining_expected_length += candidate.predicted_duration;
        }
      }
    }

    return best_priority_fallback;
  }

  void accumulate_frontier_metrics() {
    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto &candidate = scratch_.candidates[i];
      const auto &analysis = scratch_.task_analysis[i];

      if (!analysis.compatible || analysis.missing_reads.empty()) {
        continue;
      }

      const uint32_t missing_count = static_cast<uint32_t>(analysis.missing_reads.size());
      const auto band = classify_missing_count(missing_count);

      for (const auto data_id : analysis.missing_reads) {
        auto &entry = scratch_.frontier_by_data.at(data_id);

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

  [[nodiscard]] FrontierEntry *choose_best_frontier_entry() {
    FrontierEntry *best = nullptr;

    for (const auto data_id : scratch_.frontier_keys) {
      auto &entry = scratch_.frontier_by_data.at(data_id);
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
    const auto &candidate = scratch_.candidates[static_cast<std::size_t>(task_index)];
    action_buffer.push_back(Action{candidate.original_input_index, device_id, candidate.priority,
                                   candidate.priority});
  }

  void emit_top_k_band_tasks_for_data(const FrontierEntry &selected, devid_t device_id,
                                      MissingBand band, uint32_t k) {
    if (k == 0) {
      return;
    }

    scratch_.emission_tasks.clear();

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto &analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }

      const uint32_t missing_count = static_cast<uint32_t>(analysis.missing_reads.size());
      if (missing_count == 0 || classify_missing_count(missing_count) != band) {
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

    std::partial_sort(scratch_.emission_tasks.begin(),
                      scratch_.emission_tasks.begin() +
                          static_cast<std::ptrdiff_t>(emit_count),
                      scratch_.emission_tasks.end(),
                      [&](TaskIndex lhs, TaskIndex rhs) { return better_task_index(lhs, rhs); });

    for (std::size_t index = 0; index < emit_count; ++index) {
      append_action(scratch_.emission_tasks[index], device_id);
    }
  }

  void emit_ready_after_one_tasks_for_data(const FrontierEntry &selected, devid_t device_id) {
    scratch_.emission_tasks.clear();

    if (selected.ready_after_one_count == 0) {
      return;
    }

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto &analysis = scratch_.task_analysis[i];

      if (!analysis.compatible || analysis.missing_reads.size() != 1) {
        continue;
      }
      if (analysis.missing_reads.front() != selected.data_id) {
        continue;
      }

      scratch_.emission_tasks.push_back(task_index);
    }

    std::sort(scratch_.emission_tasks.begin(), scratch_.emission_tasks.end(),
              [&](TaskIndex lhs, TaskIndex rhs) { return better_task_index(lhs, rhs); });

    for (const auto task_index : scratch_.emission_tasks) {
      append_action(task_index, device_id);
    }
  }

  ActionList &plan_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
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

    const TaskIndex best_priority_fallback = analyze_device_and_build_frontier(device_id, state);

    accumulate_frontier_metrics();
    auto *best_frontier = choose_best_frontier_entry();

    if (best_frontier != nullptr) {
      emit_ready_after_one_tasks_for_data(*best_frontier, device_id);

      if (action_buffer.empty() && config_.emit_short_horizon) {
        emit_top_k_band_tasks_for_data(*best_frontier, device_id, MissingBand::ShortHorizon,
                                       config_.short_horizon_k);
      }
      if (action_buffer.empty() && config_.emit_medium_horizon) {
        emit_top_k_band_tasks_for_data(*best_frontier, device_id, MissingBand::MediumHorizon,
                                       config_.medium_horizon_k);
      }
    }

    if (action_buffer.empty() && best_priority_fallback != kNoTask) {
      append_action(best_priority_fallback, device_id);
    }

    if (action_buffer.empty()) {
      SPDLOG_CRITICAL(
          "DARTSMapper produced no actions for non-empty batch: batch_size={}, chosen_device={}",
          task_ids.size(), device_id);
    }

    for (const auto &action : action_buffer) {
      if (action.pos >= scratch_.candidates.size()) {
        SPDLOG_WARN("DARTSMapper emitted action with out-of-range pos: pos={}, candidates_size={}"
                    ", device={}",
                    action.pos, scratch_.candidates.size(), action.device);
      }
    }

    return action_buffer;
  }

public:
  DARTSMapper() : config_{} {
    validate_config();
  }

  explicit DARTSMapper(Config config) : config_(config) {
    validate_config();
  }

  DARTSMapper(std::size_t num_tasks, std::size_t /*num_devices*/, Config config = {})
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

  void set_emit_short_horizon(bool value) {
    config_.emit_short_horizon = value;
  }

  void set_emit_medium_horizon(bool value) {
    config_.emit_medium_horizon = value;
  }

  void set_short_horizon_k(uint32_t value) {
    config_.short_horizon_k = value;
  }

  void set_medium_horizon_k(uint32_t value) {
    config_.medium_horizon_k = value;
  }

  [[nodiscard]] uint32_t short_horizon_threshold() const {
    return config_.short_horizon_threshold;
  }

  [[nodiscard]] uint32_t medium_horizon_threshold() const {
    return config_.medium_horizon_threshold;
  }

  [[nodiscard]] bool emit_short_horizon() const {
    return config_.emit_short_horizon;
  }

  [[nodiscard]] bool emit_medium_horizon() const {
    return config_.emit_medium_horizon;
  }

  [[nodiscard]] uint32_t short_horizon_k() const {
    return config_.short_horizon_k;
  }

  [[nodiscard]] uint32_t medium_horizon_k() const {
    return config_.medium_horizon_k;
  }

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    const TaskIDList task_ids{task_id};
    auto &actions = plan_tasks(std::span<const taskid_t>(task_ids.data(), task_ids.size()), state);
    if (!actions.empty()) {
      return actions.front();
    }

    fill_device_targets(task_id, state);
    T4F_INVARIANT(!device_buffer.empty());
    const auto mp = state.get_mapping_priority(task_id);
    return Action{0, device_buffer.front(), mp, mp};
  }

  ActionList &map_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    return plan_tasks(task_ids, state);
  }

  ActionList &map_tasks(const TaskIDList &task_ids, const SchedulerState &state) override {
    return plan_tasks(std::span<const taskid_t>(task_ids.data(), task_ids.size()), state);
  }
};

class EnhancedDARTSMapper : public Mapper {
private:
  using TaskIndex = int32_t;
  static constexpr TaskIndex kNoTask = -1;

  struct Config {
    uint32_t short_horizon_threshold;
    uint32_t medium_horizon_threshold;
    bool emit_short_horizon;
    bool emit_medium_horizon;
    uint32_t short_horizon_k;
    uint32_t medium_horizon_k;
    bool finish_time_aware;
    bool local_data_first;

    Config() noexcept
        : short_horizon_threshold(4), medium_horizon_threshold(8), emit_short_horizon(true),
          emit_medium_horizon(true), short_horizon_k(2), medium_horizon_k(4),
          finish_time_aware(true), local_data_first(true) {
    }
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
                                                   const SchedulerState &state) const {
    const auto &tasks = state.get_tasks();
    if (tasks.is_architecture_supported(task_id, DeviceType::GPU)) {
      return tasks.get_mean_duration(task_id, DeviceType::GPU);
    }
    T4F_INVARIANT(tasks.is_architecture_supported(task_id, DeviceType::CPU));
    return tasks.get_mean_duration(task_id, DeviceType::CPU);
  }

  [[nodiscard]] timecount_t duration_for_device(taskid_t task_id, devid_t device_id,
                                                const SchedulerState &state) const {
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

    for (auto &analysis : scratch_.task_analysis) {
      analysis.compatible = false;
      analysis.unique_reads.clear();
      analysis.missing_reads.clear();
    }
  }

  [[nodiscard]] bool better_task(const TaskCandidate &lhs, const TaskCandidate &rhs) const {
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
    return better_task(scratch_.candidates[static_cast<std::size_t>(lhs)],
                       scratch_.candidates[static_cast<std::size_t>(rhs)]);
  }

  [[nodiscard]] priority_t task_priority(TaskIndex idx) const {
    if (idx == kNoTask) {
      return std::numeric_limits<priority_t>::min();
    }
    return scratch_.candidates[static_cast<std::size_t>(idx)].priority;
  }

  [[nodiscard]] bool better_frontier_ratio(const FrontierEntry &lhs,
                                           const FrontierEntry &rhs) const {
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

  [[nodiscard]] bool better_frontier_entry(const FrontierEntry &lhs,
                                           const FrontierEntry &rhs) const {
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

  [[nodiscard]] bool estimate_transfer_time(dataid_t data_id, devid_t device_id,
                                            const SchedulerState &state,
                                            timecount_t &out_time) const {
    if (state.get_data_manager().check_valid_mapped(data_id, device_id)) {
      out_time = 0;
      return true;
    }

    const auto &comm = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto flags = state.get_data_manager().get_mapped_location_flags(data_id);
    const auto source = comm.get_best_source(topology, device_id, flags);

    if (!source.found) {
      out_time = MAX_TIME;
      return false;
    }

    out_time = comm.ideal_time_to_transfer(topology, state.get_data().get_size(data_id),
                                           source.source, device_id);
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

  void build_candidates(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    const auto &tasks = state.get_tasks();
    scratch_.candidates.resize(task_ids.size());

    for (std::size_t i = 0; i < task_ids.size(); ++i) {
      const auto task_id = task_ids[i];
      auto &candidate = scratch_.candidates[i];
      candidate.task_id = task_id;
      candidate.original_input_index = i;
      candidate.priority = state.get_mapping_priority(task_id);
      candidate.supported_devices = tasks.get_supported_devices_mask(task_id);
      candidate.predicted_duration = estimate_task_duration(task_id, state);
    }
  }

  [[nodiscard]] devid_t choose_target_device(const SchedulerState &state) const {
    const auto &devices = state.get_devices();
    const auto n_devices = static_cast<devid_t>(devices.size());

    device_order_.clear();
    for (devid_t device_id = 0; device_id < n_devices; ++device_id) {
      if (devices.get_type(device_id) == DeviceType::GPU) {
        device_order_.push_back(device_id);
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

  FrontierEntry &get_or_create_frontier_entry(dataid_t data_id, devid_t device_id,
                                              const SchedulerState &state) {
    auto it = scratch_.frontier_by_data.find(data_id);
    if (it != scratch_.frontier_by_data.end()) {
      return it->second;
    }

    timecount_t transfer_time = MAX_TIME;
    const bool transfer_possible = estimate_transfer_time(data_id, device_id, state, transfer_time);

    FrontierEntry entry;
    entry.data_id = data_id;
    entry.transfer_possible = transfer_possible;
    entry.transfer_time = transfer_time;

    auto [inserted_it, inserted] = scratch_.frontier_by_data.emplace(data_id, std::move(entry));
    MONUnusedParameter(inserted);
    scratch_.frontier_keys.push_back(data_id);
    return inserted_it->second;
  }

  [[nodiscard]] TaskIndex analyze_device_and_build_frontier(devid_t device_id,
                                                            const SchedulerState &state) {
    const auto &tasks = state.get_tasks();
    TaskIndex best_priority_fallback = kNoTask;

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto &candidate = scratch_.candidates[i];
      auto &analysis = scratch_.task_analysis[i];

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
          auto &entry = get_or_create_frontier_entry(data_id, device_id, state);
          entry.remaining_expected_length += candidate.predicted_duration;
        }
      }
    }

    return best_priority_fallback;
  }

  void accumulate_frontier_metrics(devid_t device_id, const SchedulerState &state) {
    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto &candidate = scratch_.candidates[i];
      const auto &analysis = scratch_.task_analysis[i];

      if (!analysis.compatible || analysis.missing_reads.empty()) {
        continue;
      }

      const uint32_t missing_count = static_cast<uint32_t>(analysis.missing_reads.size());
      const auto band = classify_missing_count(missing_count);
      const timecount_t device_duration = duration_for_device(candidate.task_id, device_id, state);

      for (const auto data_id : analysis.missing_reads) {
        auto &entry = scratch_.frontier_by_data.at(data_id);
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

  [[nodiscard]] FrontierEntry *choose_best_frontier_entry() {
    FrontierEntry *best = nullptr;

    for (const auto data_id : scratch_.frontier_keys) {
      auto &entry = scratch_.frontier_by_data.at(data_id);
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
    const auto &candidate = scratch_.candidates[static_cast<std::size_t>(task_index)];
    action_buffer.push_back(Action{candidate.original_input_index, device_id, candidate.priority,
                                   candidate.priority});
  }

  [[nodiscard]] bool task_uses_selected_missing_data(TaskIndex task_index,
                                                     dataid_t selected_data_id) const {
    const auto &analysis = scratch_.task_analysis[static_cast<std::size_t>(task_index)];
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
      const auto &analysis = scratch_.task_analysis[i];
      if (!analysis.compatible || !analysis.missing_reads.empty()) {
        continue;
      }
      scratch_.emission_tasks.push_back(task_index);
    }

    if (scratch_.emission_tasks.empty()) {
      return;
    }

    std::sort(scratch_.emission_tasks.begin(), scratch_.emission_tasks.end(),
              [&](TaskIndex lhs, TaskIndex rhs) { return better_task_index(lhs, rhs); });

    for (const auto task_index : scratch_.emission_tasks) {
      append_action(task_index, device_id);
    }
  }

  void emit_ready_after_one_tasks_for_data(const FrontierEntry &selected, devid_t device_id) {
    scratch_.emission_tasks.clear();

    if (selected.ready_after_one_count == 0) {
      return;
    }

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto &analysis = scratch_.task_analysis[i];

      if (!analysis.compatible || analysis.missing_reads.size() != 1) {
        continue;
      }
      if (analysis.missing_reads.front() != selected.data_id) {
        continue;
      }
      scratch_.emission_tasks.push_back(task_index);
    }

    std::sort(scratch_.emission_tasks.begin(), scratch_.emission_tasks.end(),
              [&](TaskIndex lhs, TaskIndex rhs) { return better_task_index(lhs, rhs); });

    for (const auto task_index : scratch_.emission_tasks) {
      append_action(task_index, device_id);
    }
  }

  void emit_top_k_band_tasks_for_data(const FrontierEntry &selected, devid_t device_id,
                                      MissingBand band, uint32_t k) {
    if (k == 0) {
      return;
    }

    scratch_.emission_tasks.clear();

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto &analysis = scratch_.task_analysis[i];

      if (!analysis.compatible) {
        continue;
      }

      const uint32_t missing_count = static_cast<uint32_t>(analysis.missing_reads.size());
      if (missing_count == 0 || classify_missing_count(missing_count) != band) {
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
                      [&](TaskIndex lhs, TaskIndex rhs) { return better_task_index(lhs, rhs); });

    for (std::size_t index = 0; index < emit_count; ++index) {
      append_action(scratch_.emission_tasks[index], device_id);
    }
  }

  [[nodiscard]] TaskIndex fallback_task_for_device_eft(devid_t device_id,
                                                       const SchedulerState &state) const {
    const timecount_t device_load = state.costs.get_mapped_time(device_id);
    const auto &data_manager = state.get_data_manager();
    const auto &comm = state.get_communication_manager();
    const auto &topology = state.get_topology();
    const auto &data = state.get_data();

    TaskIndex best_task = kNoTask;
    timecount_t best_finish_time = std::numeric_limits<timecount_t>::max();

    for (std::size_t i = 0; i < scratch_.candidates.size(); ++i) {
      const TaskIndex task_index = static_cast<TaskIndex>(i);
      const auto &candidate = scratch_.candidates[i];
      const auto &analysis = scratch_.task_analysis[i];

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

        transfer_time +=
            comm.ideal_time_to_transfer(topology, data.get_size(data_id), req.source, device_id);
      }

      if (!feasible) {
        continue;
      }

      const timecount_t finish_time =
          device_load + transfer_time + duration_for_device(candidate.task_id, device_id, state);

      if (finish_time < best_finish_time ||
          (finish_time == best_finish_time && better_task_index(task_index, best_task))) {
        best_finish_time = finish_time;
        best_task = task_index;
      }
    }

    return best_task;
  }

public:
  EnhancedDARTSMapper() : config_{} {
    validate_config();
  }

  explicit EnhancedDARTSMapper(Config config) : config_(config) {
    validate_config();
  }

  EnhancedDARTSMapper(std::size_t num_tasks, std::size_t /*num_devices*/, Config config = {})
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

  void set_emit_short_horizon(bool value) {
    config_.emit_short_horizon = value;
  }

  void set_emit_medium_horizon(bool value) {
    config_.emit_medium_horizon = value;
  }

  void set_short_horizon_k(uint32_t value) {
    config_.short_horizon_k = value;
  }

  void set_medium_horizon_k(uint32_t value) {
    config_.medium_horizon_k = value;
  }

  void set_finish_time_aware(bool value) {
    config_.finish_time_aware = value;
  }

  void set_local_data_first(bool value) {
    config_.local_data_first = value;
  }

  [[nodiscard]] uint32_t short_horizon_threshold() const {
    return config_.short_horizon_threshold;
  }

  [[nodiscard]] uint32_t medium_horizon_threshold() const {
    return config_.medium_horizon_threshold;
  }

  [[nodiscard]] bool emit_short_horizon() const {
    return config_.emit_short_horizon;
  }

  [[nodiscard]] bool emit_medium_horizon() const {
    return config_.emit_medium_horizon;
  }

  [[nodiscard]] uint32_t short_horizon_k() const {
    return config_.short_horizon_k;
  }

  [[nodiscard]] uint32_t medium_horizon_k() const {
    return config_.medium_horizon_k;
  }

  [[nodiscard]] bool finish_time_aware() const {
    return config_.finish_time_aware;
  }

  [[nodiscard]] bool local_data_first() const {
    return config_.local_data_first;
  }

  ActionList &plan_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
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

    const TaskIndex best_priority_fallback = analyze_device_and_build_frontier(device_id, state);
    accumulate_frontier_metrics(device_id, state);
    auto *best_frontier = choose_best_frontier_entry();

    if (config_.local_data_first) {
      emit_local_data_tasks(device_id);
      if (!action_buffer.empty()) {
        return action_buffer;
      }
    }

    TaskIndex selected_fallback = best_priority_fallback;
    if (config_.finish_time_aware && best_priority_fallback != kNoTask) {
      const bool no_block = (best_frontier == nullptr);
      const bool block_only_deeper =
          (best_frontier != nullptr && best_frontier->ready_after_one_count == 0);
      const bool transfer_dominated = (best_frontier != nullptr &&
                                       best_frontier->ready_after_one_compute <
                                           best_frontier->transfer_time);

      if (no_block || block_only_deeper || transfer_dominated) {
        selected_fallback = fallback_task_for_device_eft(device_id, state);
        best_frontier = nullptr;
      }
    }

    if (best_frontier != nullptr) {
      emit_ready_after_one_tasks_for_data(*best_frontier, device_id);

      if (action_buffer.empty() && config_.emit_short_horizon) {
        emit_top_k_band_tasks_for_data(*best_frontier, device_id, MissingBand::ShortHorizon,
                                       config_.short_horizon_k);
      }
      if (action_buffer.empty() && config_.emit_medium_horizon) {
        emit_top_k_band_tasks_for_data(*best_frontier, device_id, MissingBand::MediumHorizon,
                                       config_.medium_horizon_k);
      }
    }

    if (action_buffer.empty() && selected_fallback != kNoTask) {
      append_action(selected_fallback, device_id);
    }

    return action_buffer;
  }

  Action map_task(taskid_t task_id, const SchedulerState &state) override {
    const TaskIDList task_ids{task_id};
    auto &actions = plan_tasks(std::span<const taskid_t>(task_ids.data(), task_ids.size()), state);
    if (!actions.empty()) {
      return actions.front();
    }

    fill_device_targets(task_id, state);
    T4F_INVARIANT(!device_buffer.empty());
    const auto mp = state.get_mapping_priority(task_id);
    return Action{0, device_buffer.front(), mp, mp};
  }

  ActionList &map_tasks(std::span<const taskid_t> task_ids, const SchedulerState &state) {
    return plan_tasks(task_ids, state);
  }

  ActionList &map_tasks(const TaskIDList &task_ids, const SchedulerState &state) override {
    return plan_tasks(std::span<const taskid_t>(task_ids.data(), task_ids.size()), state);
  }
};
