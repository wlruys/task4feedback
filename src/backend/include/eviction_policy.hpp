#pragma once

#include "scheduler_state.hpp"
#include <cstdint>
#include <span>

enum class EvictionState : int8_t {
  NONE = 0,
  WAITING_FOR_COMPLETION = 2,
  RUNNING = 4,
};


class LRUEvictionPolicy {
public:
  [[nodiscard]] std::span<const dataid_t>
  select_victims(const LRU_manager &lru_manager, devid_t device_id, mem_t missing_memory,
                 std::span<const dataid_t> used_ids) const {
    return lru_manager.getLRUids(device_id, static_cast<std::size_t>(missing_memory), used_ids);
  }
};

class LeastUsedMappedEvictionPolicy {
public:
  [[nodiscard]] std::span<const dataid_t>
  select_victims(const SchedulerState &state, const Data &data, const LRU_manager &lru_manager,
                 devid_t device_id, mem_t missing_memory, std::span<const dataid_t> used_ids,
                 std::vector<DataIDList> &candidate_buckets,
                 std::vector<taskid_t> &nonempty_buckets, DataIDList &victim_buffer) const;
};
