#pragma once

#include "scheduler_state.hpp"
#include <stdexcept>

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


class PlannedThresholdTransitionConditions : public TransitionConditions {
public:
  int64_t planned_threshold = 1; // Starvation threshold for planned tasks
  int64_t max_reserved_threshold = 16; // Maximum (per-device) allowed reserved tasks to prevent over-reservation (in memory-abundant scenarios)

  PlannedThresholdTransitionConditions(int32_t planned_threshold_,
                                       int64_t max_reserved_threshold_ = 16)
      : planned_threshold(planned_threshold_),
        max_reserved_threshold(max_reserved_threshold_) {
  }

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
    MONUnusedParameter(queues);
    auto &counts = state.counts;

    const devid_t n_devices = state.get_devices().size();

    if (n_devices <= 1) {
      // No non-host devices to evaluate.
      // NOTE: Fails on CPU-only code. Only used for GPU-only apps. 
      return false;
    }

    // If all non-host devices have reserved tasks >= max_reserved_threshold, don't map more.
    if (n_devices > 1) {
      bool all_over = true;
      for (devid_t i = 1; i < n_devices; ++i) {
        if (static_cast<int64_t>(counts.n_reserved(i)) < max_reserved_threshold) {
          all_over = false;
          break;
        }
      }
      if (all_over) {
        return false;
      }
    }

    const int64_t threshold = static_cast<int64_t>(planned_threshold);
    for (devid_t i = 1; i < n_devices; ++i) {
      const int64_t mapped = static_cast<int64_t>(counts.n_mapped(i));
      const int64_t reserved = static_cast<int64_t>(counts.n_reserved(i));
      if ((mapped - reserved) < threshold) {
        return true;
      }
    }
    return false;
  }
};

class HysteresisTransitionConditions : public TransitionConditions {
public:
  HysteresisTransitionConditions() = default;

  HysteresisTransitionConditions(int32_t open_in_flight_, int32_t close_in_flight_,
                                 int32_t starvation_threshold_)
      : open_in_flight(open_in_flight_),
        close_in_flight(std::max(close_in_flight_, open_in_flight_)),
        starvation_threshold(starvation_threshold_) {
  }

  timecount_t last_window_opened = 0;
  int32_t open_in_flight = 16;
  int32_t close_in_flight = 36;
  int32_t starvation_threshold = 2;
  bool window_open = false;

  bool should_map(SchedulerState &state, SchedulerQueues &queues) override {
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
    return window_open || open_condition;
  }

  bool update_map(SchedulerState &state, SchedulerQueues &queues) override {
    return should_map(state, queues);
  }
};