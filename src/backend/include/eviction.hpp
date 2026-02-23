#pragma once

#include "devices.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cassert>
#include <cstdint>
#include <list>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

namespace eviction {

[[nodiscard]] inline uint64_t pack_key(dataid_t data_id, devid_t device_id) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(device_id)) << 32) |
         static_cast<uint32_t>(data_id);
}

struct InvalidationInfo {
  enum class MappedCleanupDecision : uint8_t {
    KEEP_MAPPED = 0,
    REMOVE_MAPPED_BYTES_ONLY = 1,
    REMOVE_MAPPED_AND_LOCATION = 2,
  };

  bool pending_local_users = false;
  MappedCleanupDecision mapped_cleanup = MappedCleanupDecision::KEEP_MAPPED;
};

enum class VictimAction : uint8_t {
  MOVE_TO_HOST = 0,
  INVALIDATE_ONLY = 1,
};

struct VictimPlan {
  taskid_t compute_task_id = -1;
  devid_t device_id = -1;
  dataid_t data_id = -1;
  VictimAction action = VictimAction::INVALIDATE_ONLY;
  bool has_invalidation_info = false;
  InvalidationInfo invalidation{};
};

enum class State : int8_t {
  NONE = 0,
  WAITING_FOR_COMPLETION = 2,
  RUNNING = 4,
};

template <class Derived> class ResidencyManagerCRTP {
public:
  void read(devid_t device_id, dataid_t data_id, mem_t mem_size) {
    static_cast<Derived *>(this)->read_impl(device_id, data_id, mem_size);
  }

  void invalidate(devid_t device_id, dataid_t data_id, bool evict = false) {
    static_cast<Derived *>(this)->invalidate_impl(device_id, data_id, evict);
  }

  [[nodiscard]] mem_t get_evictable_size(devid_t device_id,
                                         std::span<const dataid_t> used_ids) const {
    return static_cast<const Derived *>(this)->get_evictable_size_impl(device_id, used_ids);
  }

  [[nodiscard]] std::span<const dataid_t>
  select_victim_ids(devid_t device_id, std::size_t mem_size,
                    std::span<const dataid_t> used_ids) const {
    return static_cast<const Derived *>(this)->select_victim_ids_impl(device_id, mem_size,
                                                                      used_ids);
  }

  [[nodiscard]] mem_t get_mem(devid_t device_id) const {
    return static_cast<const Derived *>(this)->get_mem_impl(device_id);
  }

  [[nodiscard]] mem_t get_evicted_memory_size() const {
    return static_cast<const Derived *>(this)->get_evicted_memory_size_impl();
  }

  [[nodiscard]] mem_t get_max_memory_usage() const {
    return static_cast<const Derived *>(this)->get_max_memory_usage_impl();
  }
};

class ResidencyManager final : public ResidencyManagerCRTP<ResidencyManager> {
private:
  struct LRUNode {
    dataid_t id;
    mem_t size;
  };

  mem_t evicted_size = 0;
  mem_t max_usage = 0;
  uint32_t n_devices_{0};
  std::vector<std::list<LRUNode>> lru_lists_;
  std::vector<ankerl::unordered_dense::map<dataid_t, typename std::list<LRUNode>::iterator>>
      position_maps_;
  std::vector<mem_t> sizes_;
  std::vector<mem_t> max_sizes_;
  mutable DataIDList id_buffer;
  mutable ankerl::unordered_dense::set<dataid_t> used_id_scratch;
  static constexpr std::size_t k_small_used_id_threshold = 8;

  [[nodiscard]] bool initialize_used_id_membership(std::span<const dataid_t> used_ids) const {
    const bool use_hash_membership = used_ids.size() > k_small_used_id_threshold;
    auto &used_set = used_id_scratch;
    used_set.clear();
    if (use_hash_membership) {
      used_set.reserve(used_ids.size());
      for (const auto used_id : used_ids) {
        used_set.insert(used_id);
      }
    }
    return use_hash_membership;
  }

  [[nodiscard]] bool is_used_id(dataid_t data_id, std::span<const dataid_t> used_ids,
                                bool use_hash_membership) const {
    if (!use_hash_membership) {
      return std::find(used_ids.begin(), used_ids.end(), data_id) != used_ids.end();
    }
    return used_id_scratch.contains(data_id);
  }

public:
  ResidencyManager() = default;

  explicit ResidencyManager(const Devices &devices)
      : n_devices_(devices.size()), lru_lists_(devices.size()), position_maps_(devices.size()),
        sizes_(devices.size()), max_sizes_(devices.size()) {
    initialize_from_devices(devices);
  }

  ResidencyManager(const ResidencyManager &other)
      : n_devices_(other.n_devices_), lru_lists_(other.lru_lists_),
        position_maps_(other.n_devices_), sizes_(other.sizes_),
        max_sizes_(other.max_sizes_), evicted_size(other.evicted_size), max_usage(other.max_usage) {
    for (devid_t dev = 0; dev < n_devices_; ++dev) {
      position_maps_[dev].reserve(other.position_maps_[dev].size());
      for (auto it = lru_lists_[dev].begin(); it != lru_lists_[dev].end(); ++it) {
        position_maps_[dev][it->id] = it;
      }
    }
    id_buffer.reserve(other.id_buffer.capacity());
    used_id_scratch.reserve(other.used_id_scratch.size());
  }

  void initialize_from_devices(const Devices &devices) {
    n_devices_ = devices.size();
    lru_lists_.assign(devices.size(), {});
    position_maps_.assign(devices.size(), {});
    sizes_.assign(devices.size(), 0);
    max_sizes_.assign(devices.size(), 0);
    evicted_size = 0;
    max_usage = 0;
    for (devid_t i = 0; i < devices.size(); ++i) {
      max_sizes_[i] = devices.get_max_resources(i).mem;
    }
    id_buffer.clear();
    used_id_scratch.clear();
    id_buffer.reserve(std::max<std::size_t>(id_buffer.capacity(), 20));
    used_id_scratch.reserve(std::max<std::size_t>(used_id_scratch.capacity(), 64));
  }

  void read_impl(devid_t device_id, dataid_t data_id, mem_t mem_size) {
    assert(device_id >= 0 && device_id < n_devices_);

    auto &lst = lru_lists_[device_id];
    auto &pos = position_maps_[device_id];
    auto &size = sizes_[device_id];
    auto &max_size = max_sizes_[device_id];
    auto it = pos.find(data_id);
    if (it != pos.end()) {
      it->second->size = mem_size;
      lst.splice(lst.end(), lst, it->second);
      return;
    }
    size += mem_size;
    if (size > max_usage && device_id > 0) {
      max_usage = size;
    }
    if (size > max_size) {
      SPDLOG_DEBUG("ResidencyManager::read_impl(): Device {}: Adding data_id {} with size {}",
                   device_id,
                   data_id, mem_size);
      assert(size <= max_size && "ResidencyManager::read_impl(): size exceeds max size");
    }
    lst.push_back(LRUNode{data_id, mem_size});
    pos[data_id] = std::prev(lst.end());
  }

  void invalidate_impl(devid_t device_id, dataid_t data_id, bool evict = false) {
    assert(device_id >= 0 && device_id < n_devices_);

    auto &lst = lru_lists_[device_id];
    auto &pos = position_maps_[device_id];
    auto &size = sizes_[device_id];

    auto it = pos.find(data_id);
    assert(it != pos.end() && "invalidate_impl(): data_id not present");

    auto node_it = it->second;
    const mem_t node_size = node_it->size;
    size -= node_size;
    if (evict) {
      evicted_size += node_size;
    }
    lst.erase(node_it);
    pos.erase(it);
  }

  [[nodiscard]] mem_t get_evictable_size_impl(devid_t device_id,
                                              std::span<const dataid_t> used_ids) const {
    assert(device_id >= 0 && device_id < n_devices_);

    const auto &lst = lru_lists_[device_id];
    const bool use_hash_membership = initialize_used_id_membership(used_ids);

    mem_t evictable = 0;
    for (const auto &node : lst) {
      if (is_used_id(node.id, used_ids, use_hash_membership)) {
        continue;
      }
      evictable += node.size;
    }
    return evictable;
  }

  [[nodiscard]] std::span<const dataid_t>
  select_victim_ids_impl(devid_t device_id, std::size_t mem_size,
                         std::span<const dataid_t> used_ids) const {
    assert(device_id >= 0 && device_id < n_devices_);

    const auto &lst = lru_lists_[device_id];
    id_buffer.clear();
    if (mem_size == 0) {
      return id_buffer;
    }

    const bool use_hash_membership = initialize_used_id_membership(used_ids);

    mem_t accumulated = 0;
    for (const auto &node : lst) {
      if (is_used_id(node.id, used_ids, use_hash_membership)) {
        continue;
      }
      accumulated += node.size;
      id_buffer.push_back(node.id);
      if (accumulated >= static_cast<mem_t>(mem_size)) {
        return id_buffer;
      }
    }

    assert(accumulated >= static_cast<mem_t>(mem_size) &&
           "select_victim_ids_impl(): evictable memory size is smaller than requested size");
    if (accumulated < static_cast<mem_t>(mem_size)) {
      id_buffer.clear();
    }
    return id_buffer;
  }

  [[nodiscard]] mem_t get_mem_impl(devid_t device_id) const {
    assert(device_id >= 0 && device_id < n_devices_);
    return sizes_[device_id];
  }

  [[nodiscard]] mem_t get_evicted_memory_size_impl() const {
    return evicted_size;
  }

  [[nodiscard]] mem_t get_max_memory_usage_impl() const {
    mem_t total_usage = 0;
    for (devid_t device_id = 1; device_id < n_devices_; ++device_id) {
      total_usage += sizes_[device_id];
    }
    return total_usage;
  }
};

enum class PolicyKind : uint8_t {
  LRU = 0,
};

enum class UsagePhase : uint8_t {
  MAPPED = 0,
  RESERVED = 1,
  MOVED = 2,
  LAUNCHED = 4,
};

enum class UsageTransition : uint8_t {
  SET = 0,
  RELEASE = 1,
};

class EvictionRequestBuffer {
public:
  void reserve(std::size_t n) {
    compute_task_ids_.reserve(n);
    device_ids_.reserve(n);
    missing_mem_.reserve(n);
  }

  void clear() {
    compute_task_ids_.clear();
    device_ids_.clear();
    missing_mem_.clear();
  }

  [[nodiscard]] bool empty() const {
    return compute_task_ids_.empty();
  }

  [[nodiscard]] std::size_t size() const {
    return compute_task_ids_.size();
  }

  void push(taskid_t compute_task_id, devid_t device_id, mem_t missing_mem) {
    compute_task_ids_.push_back(compute_task_id);
    device_ids_.push_back(device_id);
    missing_mem_.push_back(missing_mem);
  }

  [[nodiscard]] taskid_t compute_task_id(std::size_t i) const {
    return compute_task_ids_[i];
  }
  [[nodiscard]] devid_t device_id(std::size_t i) const {
    return device_ids_[i];
  }
  [[nodiscard]] mem_t missing_mem(std::size_t i) const {
    return missing_mem_[i];
  }

private:
  std::vector<taskid_t> compute_task_ids_;
  std::vector<devid_t> device_ids_;
  std::vector<mem_t> missing_mem_;
};

class EvictionPlanSoA {
public:
  void reserve(std::size_t n) {
    compute_task_ids_.reserve(n);
    device_ids_.reserve(n);
    data_ids_.reserve(n);
    actions_.reserve(n);
    has_invalidation_info_.reserve(n);
    invalidations_.reserve(n);
  }

  void clear() {
    compute_task_ids_.clear();
    device_ids_.clear();
    data_ids_.clear();
    actions_.clear();
    has_invalidation_info_.clear();
    invalidations_.clear();
  }

  [[nodiscard]] std::size_t size() const {
    return data_ids_.size();
  }

  void push(const VictimPlan &plan) {
    compute_task_ids_.push_back(plan.compute_task_id);
    device_ids_.push_back(plan.device_id);
    data_ids_.push_back(plan.data_id);
    actions_.push_back(plan.action);
    has_invalidation_info_.push_back(plan.has_invalidation_info ? 1U : 0U);
    invalidations_.push_back(plan.invalidation);
  }

  [[nodiscard]] taskid_t compute_task_id(std::size_t i) const {
    return compute_task_ids_[i];
  }
  [[nodiscard]] devid_t device_id(std::size_t i) const {
    return device_ids_[i];
  }
  [[nodiscard]] dataid_t data_id(std::size_t i) const {
    return data_ids_[i];
  }
  [[nodiscard]] VictimAction action(std::size_t i) const {
    return actions_[i];
  }
  [[nodiscard]] bool has_invalidation_info(std::size_t i) const {
    return has_invalidation_info_[i] != 0;
  }
  [[nodiscard]] const InvalidationInfo &invalidation(std::size_t i) const {
    return invalidations_[i];
  }

private:
  std::vector<taskid_t> compute_task_ids_;
  std::vector<devid_t> device_ids_;
  std::vector<dataid_t> data_ids_;
  std::vector<VictimAction> actions_;
  std::vector<uint8_t> has_invalidation_info_;
  std::vector<InvalidationInfo> invalidations_;
};

template <class Derived> class EvictionPolicy {
public:
  void initialize_residency(const Devices &devices) {
    static_cast<Derived *>(this)->initialize_residency_impl(devices);
  }

  void initialize(std::size_t n_devices, std::size_t reserve_hint = 0) {
    static_cast<Derived *>(this)->initialize_impl(n_devices, reserve_hint);
  }

  void reserve(std::size_t reserve_hint) {
    static_cast<Derived *>(this)->reserve_impl(reserve_hint);
  }

  void clear_cycle_state() {
    static_cast<Derived *>(this)->clear_cycle_state_impl();
  }

  template <UsagePhase Phase, UsageTransition Transition>
  void update_used(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    static_cast<Derived *>(this)->template update_used_impl<Phase, Transition>(data_id, device_id,
                                                                               current_time);
  }

  template <class StateT>
  void plan(const StateT &state, const EvictionRequestBuffer &requests) {
    static_cast<Derived *>(this)->plan_impl(state, requests);
  }

  template <class StateT>
  [[nodiscard]] InvalidationInfo get_invalidation_info(const StateT &state, dataid_t data_id,
                                                       devid_t invalidate_device) const {
    return static_cast<const Derived *>(this)->get_invalidation_info_impl(state, data_id,
                                                                          invalidate_device);
  }

  [[nodiscard]] const EvictionPlanSoA &plan_results() const {
    return static_cast<const Derived *>(this)->plan_results_impl();
  }
};

[[nodiscard]] inline VictimAction classify_victim_action(devicemask_t launched_location_flags) {
  const int32_t n_sources = __builtin_popcount(launched_location_flags);
  assert(n_sources > 0);
  return (n_sources == 1) ? VictimAction::MOVE_TO_HOST : VictimAction::INVALIDATE_ONLY;
}

class LRUEvictionPolicy final : public EvictionPolicy<LRUEvictionPolicy> {
public:
  static constexpr PolicyKind kKind = PolicyKind::LRU;
  static constexpr const char *kName = "LRU";

  LRUEvictionPolicy() = default;

  explicit LRUEvictionPolicy(const Devices &devices) : residency_manager_(devices) {
  }

  void initialize_residency_impl(const Devices &devices) {
    residency_manager_.initialize_from_devices(devices);
  }

  void initialize_impl(std::size_t n_devices, std::size_t reserve_hint) {
    n_devices_ = static_cast<devid_t>(n_devices);
    mapped_use_counts_by_device_.assign(n_devices, {});
    reserved_use_counts_by_device_.assign(n_devices, {});
    launched_use_counts_by_device_.assign(n_devices, {});
    reserve_impl(reserve_hint);
  }

  void reserve_impl(std::size_t reserve_hint) {
    if (reserve_hint == 0) {
      return;
    }
    for (auto &m : mapped_use_counts_by_device_) {
      m.reserve(reserve_hint * 8);
    }
    for (auto &m : reserved_use_counts_by_device_) {
      m.reserve(reserve_hint * 8);
    }
    for (auto &m : launched_use_counts_by_device_) {
      m.reserve(reserve_hint * 8);
    }
    planned_keys_.reserve(reserve_hint * 8);
    plan_results_.reserve(reserve_hint * 4);
  }

  void clear_cycle_state_impl() {
    planned_keys_.clear();
    plan_results_.clear();
  }

  template <UsagePhase Phase, UsageTransition Transition>
  void update_used_impl(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    (void)current_time;
    assert(device_id >= 0 && device_id < n_devices_);
    auto &counts = use_counts_by_device<Phase>()[device_id];
    if constexpr (Transition == UsageTransition::SET) {
      counts[data_id] += 1;
    } else {
      decrement_count_map(counts, data_id);
    }
  }

  template <class StateT>
  void plan_impl(const StateT &state, const EvictionRequestBuffer &requests) {
    clear_cycle_state_impl();

    for (std::size_t i = 0; i < requests.size(); ++i) {
      const taskid_t compute_task_id = requests.compute_task_id(i);
      const devid_t device_id = requests.device_id(i);
      const mem_t missing_mem = requests.missing_mem(i);
      if (missing_mem <= 0) {
        continue;
      }

      const auto protected_data = state.get_task_unique_data(compute_task_id);
      const auto victim_ids = residency_manager_.select_victim_ids(
          device_id, static_cast<std::size_t>(missing_mem), protected_data);

      for (const auto data_id : victim_ids) {
        if (!planned_keys_.emplace(pack_key(data_id, device_id)).second) {
          continue;
        }

        VictimPlan plan{};
        plan.compute_task_id = compute_task_id;
        plan.device_id = device_id;
        plan.data_id = data_id;
        plan.action = classify_victim_action(state.get_launched_location_flags(data_id));
        if (plan.action == VictimAction::INVALIDATE_ONLY) {
          plan.invalidation = get_invalidation_info_impl(state, data_id, device_id);
          plan.has_invalidation_info = true;
        }
        plan_results_.push(plan);
      }
    }
  }

  template <class StateT>
  [[nodiscard]] InvalidationInfo get_invalidation_info_impl(const StateT &state, dataid_t data_id,
                                                            devid_t invalidate_device) const {
    (void)state;
    InvalidationInfo info{};
    const auto &mapped_counts = mapped_use_counts_by_device_[invalidate_device];
    info.pending_local_users = mapped_counts.find(data_id) != mapped_counts.end();
    info.mapped_cleanup = info.pending_local_users
                              ? InvalidationInfo::MappedCleanupDecision::KEEP_MAPPED
                              : InvalidationInfo::MappedCleanupDecision::REMOVE_MAPPED_AND_LOCATION;
    return info;
  }

  [[nodiscard]] const EvictionPlanSoA &plan_results_impl() const {
    return plan_results_;
  }

  [[nodiscard]] ResidencyManager &residency() {
    return residency_manager_;
  }

  [[nodiscard]] const ResidencyManager &residency() const {
    return residency_manager_;
  }

private:
  template <UsagePhase Phase>
  auto &use_counts_by_device() {
    if constexpr (Phase == UsagePhase::MAPPED) {
      return mapped_use_counts_by_device_;
    } else if constexpr (Phase == UsagePhase::RESERVED) {
      return reserved_use_counts_by_device_;
    } else {
      static_assert(Phase == UsagePhase::LAUNCHED);
      return launched_use_counts_by_device_;
    }
  }

  static void decrement_count_map(ankerl::unordered_dense::map<dataid_t, int32_t> &m,
                                  dataid_t data_id) {
    auto it = m.find(data_id);
    if (it == m.end()) {
      return;
    }
    if (--(it->second) <= 0) {
      m.erase(it);
    }
  }

  devid_t n_devices_ = 0;
  std::vector<ankerl::unordered_dense::map<dataid_t, int32_t>> mapped_use_counts_by_device_;
  std::vector<ankerl::unordered_dense::map<dataid_t, int32_t>> reserved_use_counts_by_device_;
  std::vector<ankerl::unordered_dense::map<dataid_t, int32_t>> launched_use_counts_by_device_;
  ankerl::unordered_dense::set<uint64_t> planned_keys_;
  EvictionPlanSoA plan_results_;
  ResidencyManager residency_manager_;
};

} // namespace eviction
