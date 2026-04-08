#pragma once

#include "scheduler_state.hpp"
#include <concepts>

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

class PlannedThresholdTransitionConditions : public TransitionConditionBase {
public:
  int64_t planned_threshold = 1;
  int64_t max_reserved_threshold = 16;

  PlannedThresholdTransitionConditions() = default;

  PlannedThresholdTransitionConditions(int32_t planned_threshold_,
                                       int64_t max_reserved_threshold_ = 16)
      : planned_threshold(planned_threshold_),
        max_reserved_threshold(max_reserved_threshold_) {
  }

  std::shared_ptr<TransitionConditionBase> clone() const override {
    return std::make_shared<PlannedThresholdTransitionConditions>(*this);
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto &counts = state.counts;
    const auto &devices = state.get_devices();
    const devid_t n_devices = devices.size();

    bool has_gpu_device = false;
    bool all_over_reserved_threshold = true;
    for (devid_t device_id = 1; device_id < n_devices; ++device_id) {
      if (devices.get_type(device_id) != DeviceType::GPU) {
        continue;
      }
      has_gpu_device = true;
      if (static_cast<int64_t>(counts.n_reserved(device_id)) < max_reserved_threshold) {
        all_over_reserved_threshold = false;
        break;
      }
    }

    if (!has_gpu_device || all_over_reserved_threshold) {
      return false;
    }

    for (devid_t device_id = 1; device_id < n_devices; ++device_id) {
      if (devices.get_type(device_id) != DeviceType::GPU) {
        continue;
      }
      const int64_t mapped = static_cast<int64_t>(counts.n_mapped(device_id));
      const int64_t reserved = static_cast<int64_t>(counts.n_reserved(device_id));
      if ((mapped - reserved) < planned_threshold) {
        return true;
      }
    }
    return false;
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
