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
    for (auto &size : sizes_) {
      size = 0;
    }
    for (int i = 0; i < devices.size(); i++) {
      max_sizes_[i] = devices.get_max_resources(i).mem;
    }
    id_buffer.reserve(20);
    used_id_scratch.reserve(64);
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

// Backward-compatible alias while call sites migrate.
using LRUManager = ResidencyManager;

enum class PolicyKind : uint8_t {
  LRU = 0,
};

template <class Derived> class PolicyCRTP {
public:
  template <class ResidencyManagerT>
  [[nodiscard]] std::span<const dataid_t>
  select_victims(const ResidencyManagerT &residency_manager, devid_t device_id, mem_t missing_mem,
                 std::span<const dataid_t> protected_data) const {
    return static_cast<const Derived *>(this)->select_victims_impl(residency_manager, device_id,
                                                                   missing_mem, protected_data);
  }

  [[nodiscard]] static constexpr PolicyKind kind() {
    return Derived::kKind;
  }

  [[nodiscard]] static constexpr const char *name() {
    return Derived::kName;
  }
};

class LRUPolicy final : public PolicyCRTP<LRUPolicy> {
public:
  static constexpr PolicyKind kKind = PolicyKind::LRU;
  static constexpr const char *kName = "LRU";

  template <class ResidencyManagerT>
  [[nodiscard]] std::span<const dataid_t>
  select_victims_impl(const ResidencyManagerT &residency_manager, devid_t device_id,
                      mem_t missing_mem, std::span<const dataid_t> protected_data) const {
    return residency_manager.select_victim_ids(device_id, missing_mem, protected_data);
  }
};

template <class WriterSetT, class DepthFnT>
[[nodiscard]] inline taskid_t select_top_writer_by_depth(const WriterSetT &writer_tasks,
                                                         DepthFnT &&depth_of) {
  assert(!writer_tasks.empty());
  if (writer_tasks.size() == 1) {
    return *writer_tasks.begin();
  }

  taskid_t top = *writer_tasks.begin();
  int32_t top_depth = depth_of(top);
  for (auto tid : writer_tasks) {
    const int32_t d = depth_of(tid);
    if (d < top_depth) {
      top_depth = d;
      top = tid;
    }
  }
  return top;
}

template <class WriterSetT, class ContextT>
[[nodiscard]] inline taskid_t select_top_writer_by_depth(const WriterSetT &writer_tasks,
                                                         const ContextT &ctx) {
  assert(!writer_tasks.empty());
  if (writer_tasks.size() == 1) {
    return *writer_tasks.begin();
  }

  taskid_t top = *writer_tasks.begin();
  int32_t top_depth = ctx.get_task_depth(top);
  for (auto tid : writer_tasks) {
    const int32_t d = ctx.get_task_depth(tid);
    if (d < top_depth) {
      top_depth = d;
      top = tid;
    }
  }
  return top;
}

class InvalidationCache {
public:
  void reserve(std::size_t n) {
    encoded_.reserve(n);
  }

  void clear() {
    encoded_.clear();
  }

  [[nodiscard]] bool get(dataid_t data_id, devid_t invalidate_device,
                         InvalidationInfo &out_info) const {
    const auto key = pack_key(data_id, invalidate_device);
    auto it = encoded_.find(key);
    if (it == encoded_.end()) {
      return false;
    }

    const uint8_t encoded = it->second;
    const uint8_t cleanup_bits = (encoded >> 1) & 0x03;
    InvalidationInfo::MappedCleanupDecision mapped_cleanup =
        InvalidationInfo::MappedCleanupDecision::KEEP_MAPPED;
    switch (cleanup_bits) {
    case 0:
      mapped_cleanup = InvalidationInfo::MappedCleanupDecision::KEEP_MAPPED;
      break;
    case 1:
      mapped_cleanup = InvalidationInfo::MappedCleanupDecision::REMOVE_MAPPED_BYTES_ONLY;
      break;
    case 2:
      mapped_cleanup = InvalidationInfo::MappedCleanupDecision::REMOVE_MAPPED_AND_LOCATION;
      break;
    default:
      mapped_cleanup = InvalidationInfo::MappedCleanupDecision::KEEP_MAPPED;
      break;
    }

    out_info = {
        .pending_local_users = (encoded & 0x01) != 0,
        .mapped_cleanup = mapped_cleanup,
    };
    return true;
  }

  void put(dataid_t data_id, devid_t invalidate_device, const InvalidationInfo &info) {
    uint8_t encoded = 0;
    encoded |= static_cast<uint8_t>(info.pending_local_users ? 0x01 : 0);
    encoded |= static_cast<uint8_t>(static_cast<uint8_t>(info.mapped_cleanup) << 1);
    encoded_[pack_key(data_id, invalidate_device)] = encoded;
  }

private:
  ankerl::unordered_dense::map<uint64_t, uint8_t> encoded_;
};

class PlannedVictimSet {
public:
  void reserve(std::size_t n) {
    keys_.reserve(n);
  }

  void clear() {
    keys_.clear();
  }

  [[nodiscard]] bool mark(dataid_t data_id, devid_t device_id) {
    return keys_.emplace(pack_key(data_id, device_id)).second;
  }

private:
  ankerl::unordered_dense::set<uint64_t> keys_;
};

struct PredecessorReaderCacheEntry {
  taskid_t top_writer = -1;
  uint32_t write_generation = 0;
  uint32_t read_generation = 0;
  bool has_local_predecessor_reader = false;
};

class PredecessorReaderCache {
public:
  void reserve(std::size_t n) {
    entries_.reserve(n);
  }

  void clear() {
    entries_.clear();
  }

  [[nodiscard]] bool get(dataid_t data_id, devid_t invalidate_device, taskid_t top_writer,
                         uint32_t write_generation, uint32_t read_generation,
                         bool &has_local_predecessor_reader) const {
    auto it = entries_.find(pack_key(data_id, invalidate_device));
    if (it == entries_.end()) {
      return false;
    }

    const auto &entry = it->second;
    if (entry.top_writer != top_writer || entry.write_generation != write_generation ||
        entry.read_generation != read_generation) {
      return false;
    }

    has_local_predecessor_reader = entry.has_local_predecessor_reader;
    return true;
  }

  void put(dataid_t data_id, devid_t invalidate_device, taskid_t top_writer,
           uint32_t write_generation, uint32_t read_generation,
           bool has_local_predecessor_reader) {
    entries_[pack_key(data_id, invalidate_device)] = {
        .top_writer = top_writer,
        .write_generation = write_generation,
        .read_generation = read_generation,
        .has_local_predecessor_reader = has_local_predecessor_reader,
    };
  }

private:
  ankerl::unordered_dense::map<uint64_t, PredecessorReaderCacheEntry> entries_;
};

template <class PolicyT, class StateT, class PredecessorQueryT, class RequestListT>
class RequestPlanner;

class RuntimeStack {
public:
  void initialize(std::size_t n_devices, std::size_t n_compute_tasks,
                  std::size_t reserve_hint = 0) {
    n_devices_ = static_cast<devid_t>(n_devices);
    is_mapped_not_reserved_.assign(n_compute_tasks, 0);
    mapped_unique_count_.assign(n_devices, {});
    mapped_read_count_by_data_device_.assign(n_devices, {});
    mapped_read_tasks_by_data_device_.assign(n_devices, {});
    mapped_read_generation_by_data_device_.assign(n_devices, {});

    if (reserve_hint > 0) {
      reserve(reserve_hint);
    }
  }

  void reserve(std::size_t reserve_hint) {
    invalidation_cache_.reserve(reserve_hint * 8);
    predecessor_reader_cache_.reserve(reserve_hint * 16);
    planned_victims_.reserve(reserve_hint * 8);
    plan_buffer_.reserve(reserve_hint * 4);
    dfs_stack_scratch_.reserve(64);
    dfs_visited_scratch_.reserve(128);
  }

  template <class StaticGraphT>
  void build_compute_task_ancestor_index(const StaticGraphT &static_graph) {
    constexpr std::size_t kMaxIndexedComputeTasks = 16384;
    const auto n_compute_tasks = static_cast<std::size_t>(static_graph.get_n_compute_tasks());
    compute_task_ancestor_words_ = (n_compute_tasks + 63) / 64;

    if (n_compute_tasks == 0 || n_compute_tasks > kMaxIndexedComputeTasks) {
      has_precomputed_ancestors_ = false;
      compute_task_ancestor_bits_.clear();
      return;
    }

    compute_task_ancestor_bits_.assign(n_compute_tasks * compute_task_ancestor_words_, 0);
    std::vector<taskid_t> topo_order(n_compute_tasks);
    std::iota(topo_order.begin(), topo_order.end(), 0);
    std::sort(topo_order.begin(), topo_order.end(),
              [&static_graph](taskid_t lhs, taskid_t rhs) {
                return static_graph.get_depth(lhs) < static_graph.get_depth(rhs);
              });

    for (auto task_id : topo_order) {
      auto *row = compute_task_ancestor_bits_.data() +
                  static_cast<std::size_t>(task_id) * compute_task_ancestor_words_;
      for (auto dep : static_graph.get_compute_task_dependencies(task_id)) {
        const auto *dep_row = compute_task_ancestor_bits_.data() +
                              static_cast<std::size_t>(dep) * compute_task_ancestor_words_;
        for (std::size_t w = 0; w < compute_task_ancestor_words_; ++w) {
          row[w] |= dep_row[w];
        }
        const auto dep_idx = static_cast<std::size_t>(dep);
        row[dep_idx / 64] |= (1ULL << (dep_idx % 64));
      }
    }

    has_precomputed_ancestors_ = true;
  }

  [[nodiscard]] bool has_precomputed_ancestors() const {
    return has_precomputed_ancestors_;
  }

  void clear_cycle_state() {
    invalidation_cache_.clear();
    predecessor_reader_cache_.clear();
    planned_victims_.clear();
    plan_buffer_.clear();
  }

  void clear_invalidation_caches() {
    invalidation_cache_.clear();
    predecessor_reader_cache_.clear();
  }

  template <class StaticGraphT>
  void on_compute_mapped(const StaticGraphT &static_graph, taskid_t compute_task_id,
                         devid_t device_id) {
    mapped_but_not_reserved_tasks_.insert(compute_task_id);
    is_mapped_not_reserved_[static_cast<std::size_t>(compute_task_id)] = 1;

    auto &unique_count_map = mapped_unique_count_[device_id];
    for (auto data_id : static_graph.get_unique(compute_task_id)) {
      unique_count_map[data_id]++;
    }

    for (auto data_id : static_graph.get_write(compute_task_id)) {
      mapped_write_by_data_[data_id].insert(compute_task_id);
      mapped_write_generation_by_data_[data_id] += 1;
    }

    auto &read_count_map = mapped_read_count_by_data_device_[device_id];
    auto &read_tasks_map = mapped_read_tasks_by_data_device_[device_id];
    auto &read_generation_map = mapped_read_generation_by_data_device_[device_id];
    for (auto data_id : static_graph.get_read(compute_task_id)) {
      read_count_map[data_id]++;
      read_tasks_map[data_id].insert(compute_task_id);
      read_generation_map[data_id] += 1;
    }
  }

  template <class StaticGraphT>
  void on_compute_reserved(const StaticGraphT &static_graph, taskid_t compute_task_id,
                           devid_t device_id) {
    mapped_but_not_reserved_tasks_.erase(compute_task_id);
    is_mapped_not_reserved_[static_cast<std::size_t>(compute_task_id)] = 0;

    auto &unique_count_map = mapped_unique_count_[device_id];
    for (auto data_id : static_graph.get_unique(compute_task_id)) {
      auto unique_it = unique_count_map.find(data_id);
      assert(unique_it != unique_count_map.end());
      if (--(unique_it->second) == 0) {
        unique_count_map.erase(unique_it);
      }
    }

    for (auto data_id : static_graph.get_write(compute_task_id)) {
      auto writers_it = mapped_write_by_data_.find(data_id);
      if (writers_it != mapped_write_by_data_.end()) {
        writers_it->second.erase(compute_task_id);
        if (writers_it->second.empty()) {
          mapped_write_by_data_.erase(writers_it);
        }
      }
      mapped_write_generation_by_data_[data_id] += 1;
    }

    auto &read_count_map = mapped_read_count_by_data_device_[device_id];
    auto &read_tasks_map = mapped_read_tasks_by_data_device_[device_id];
    auto &read_generation_map = mapped_read_generation_by_data_device_[device_id];
    for (auto data_id : static_graph.get_read(compute_task_id)) {
      auto read_count_it = read_count_map.find(data_id);
      assert(read_count_it != read_count_map.end());
      if (--(read_count_it->second) == 0) {
        read_count_map.erase(read_count_it);
      }

      auto tasks_it = read_tasks_map.find(data_id);
      if (tasks_it != read_tasks_map.end()) {
        tasks_it->second.erase(compute_task_id);
        if (tasks_it->second.empty()) {
          read_tasks_map.erase(tasks_it);
        }
      }
      read_generation_map[data_id] += 1;
    }
  }

  [[nodiscard]] bool has_pending_local_users(dataid_t data_id, devid_t invalidate_device) const {
    const auto &unique_count = mapped_unique_count_[invalidate_device];
    return unique_count.find(data_id) != unique_count.end();
  }

  [[nodiscard]] const ankerl::unordered_dense::set<taskid_t> &
  get_mapped_writers(dataid_t data_id) const {
    static const ankerl::unordered_dense::set<taskid_t> kEmptyWriters;
    auto it = mapped_write_by_data_.find(data_id);
    if (it == mapped_write_by_data_.end()) {
      return kEmptyWriters;
    }
    return it->second;
  }

  [[nodiscard]] bool has_local_mapped_readers(dataid_t data_id, devid_t invalidate_device) const {
    const auto &read_count = mapped_read_count_by_data_device_[invalidate_device];
    return read_count.find(data_id) != read_count.end();
  }

  [[nodiscard]] uint32_t get_mapped_write_generation(dataid_t data_id) const {
    auto it = mapped_write_generation_by_data_.find(data_id);
    return it == mapped_write_generation_by_data_.end() ? 0 : it->second;
  }

  [[nodiscard]] uint32_t get_mapped_read_generation(dataid_t data_id,
                                                    devid_t invalidate_device) const {
    const auto &read_generation = mapped_read_generation_by_data_device_[invalidate_device];
    auto it = read_generation.find(data_id);
    return it == read_generation.end() ? 0 : it->second;
  }

  [[nodiscard]] bool try_get_cached_invalidation_info(
      dataid_t data_id, devid_t invalidate_device, InvalidationInfo &out_info) const {
    return invalidation_cache_.get(data_id, invalidate_device, out_info);
  }

  void cache_invalidation_info(dataid_t data_id, devid_t invalidate_device,
                               const InvalidationInfo &info) {
    invalidation_cache_.put(data_id, invalidate_device, info);
  }

  template <class StateT>
  [[nodiscard]] InvalidationInfo get_invalidation_info(const StateT &state, dataid_t data_id,
                                                       devid_t invalidate_device) {
    InvalidationInfo info{};
    if (invalidation_cache_.get(data_id, invalidate_device, info)) {
      return info;
    }

    info = compute_invalidation_info(state, *this, predecessor_reader_cache_, data_id,
                                     invalidate_device);
    invalidation_cache_.put(data_id, invalidate_device, info);
    return info;
  }

  template <class PolicyT, class StateT, class RequestListT>
  void plan_with_policy(const StateT &state, const RequestListT &requests) {
    static const PolicyT kPolicy{};
    static const RequestPlanner<PolicyT, StateT, RuntimeStack, RequestListT> kPlanner{};
    kPlanner.plan(kPolicy, state, *this, requests, planned_victims_, invalidation_cache_,
                  predecessor_reader_cache_, plan_buffer_);
  }

  [[nodiscard]] const std::vector<VictimPlan> &plan_buffer() const {
    return plan_buffer_;
  }

  [[nodiscard]] std::vector<VictimPlan> &plan_buffer() {
    return plan_buffer_;
  }

  template <class StateT>
  [[nodiscard]] bool has_local_reader_predecessor_for_eviction(
      const StateT &state, taskid_t top_writer_task_id, dataid_t data_id,
      devid_t invalidate_device) {
    bool has_local_predecessor_reader = false;
    const auto &read_tasks_map = mapped_read_tasks_by_data_device_[invalidate_device];
    auto readers_it = read_tasks_map.find(data_id);
    if (readers_it == read_tasks_map.end()) {
      return false;
    }

    if (has_precomputed_ancestors_) {
      for (auto reader_task_id : readers_it->second) {
        if (is_compute_task_ancestor(top_writer_task_id, reader_task_id,
                                     state.get_tasks().get_n_compute_tasks())) {
          has_local_predecessor_reader = true;
          break;
        }
      }
      return has_local_predecessor_reader;
    }

    const auto &static_graph = state.get_tasks();
    auto &stack = dfs_stack_scratch_;
    auto &visited = dfs_visited_scratch_;
    stack.clear();
    visited.clear();

    for (auto dep0 : static_graph.get_compute_task_dependencies(top_writer_task_id)) {
      if (!is_mapped_not_reserved_[static_cast<std::size_t>(dep0)]) {
        continue;
      }
      if (!visited.emplace(dep0).second) {
        continue;
      }
      if (readers_it->second.contains(dep0)) {
        return true;
      }
      stack.push_back(dep0);
    }

    while (!stack.empty()) {
      const auto curr = stack.back();
      stack.pop_back();
      for (auto dep : static_graph.get_compute_task_dependencies(curr)) {
        if (!is_mapped_not_reserved_[static_cast<std::size_t>(dep)]) {
          continue;
        }
        if (!visited.emplace(dep).second) {
          continue;
        }
        if (readers_it->second.contains(dep)) {
          return true;
        }
        stack.push_back(dep);
      }
    }

    return false;
  }

private:
  [[nodiscard]] bool is_compute_task_ancestor(taskid_t descendant, taskid_t ancestor,
                                              std::size_t n_compute_tasks) const {
    if (!has_precomputed_ancestors_) {
      return false;
    }

    const auto desc_idx = static_cast<std::size_t>(descendant);
    const auto anc_idx = static_cast<std::size_t>(ancestor);
    if (desc_idx >= n_compute_tasks || anc_idx >= n_compute_tasks) {
      return false;
    }

    const auto word_idx = anc_idx / 64;
    const auto bit = (1ULL << (anc_idx % 64));
    const auto *row = compute_task_ancestor_bits_.data() + desc_idx * compute_task_ancestor_words_;
    return (row[word_idx] & bit) != 0;
  }

  devid_t n_devices_ = 0;
  ankerl::unordered_dense::set<taskid_t> mapped_but_not_reserved_tasks_;
  std::vector<uint8_t> is_mapped_not_reserved_;
  std::vector<ankerl::unordered_dense::map<dataid_t, int32_t>> mapped_unique_count_;
  ankerl::unordered_dense::map<dataid_t, ankerl::unordered_dense::set<taskid_t>>
      mapped_write_by_data_;
  std::vector<ankerl::unordered_dense::map<dataid_t, int32_t>> mapped_read_count_by_data_device_;
  std::vector<ankerl::unordered_dense::map<dataid_t, ankerl::unordered_dense::set<taskid_t>>>
      mapped_read_tasks_by_data_device_;
  ankerl::unordered_dense::map<dataid_t, uint32_t> mapped_write_generation_by_data_;
  std::vector<ankerl::unordered_dense::map<dataid_t, uint32_t>>
      mapped_read_generation_by_data_device_;

  InvalidationCache invalidation_cache_;
  PlannedVictimSet planned_victims_;
  PredecessorReaderCache predecessor_reader_cache_;
  std::vector<VictimPlan> plan_buffer_;

  std::vector<taskid_t> dfs_stack_scratch_;
  ankerl::unordered_dense::set<taskid_t> dfs_visited_scratch_;

  std::vector<uint64_t> compute_task_ancestor_bits_;
  std::size_t compute_task_ancestor_words_ = 0;
  bool has_precomputed_ancestors_ = false;
};

[[nodiscard]] inline VictimAction classify_victim_action(devicemask_t launched_location_flags) {
  const int32_t n_sources = __builtin_popcount(launched_location_flags);
  assert(n_sources > 0);
  return (n_sources == 1) ? VictimAction::MOVE_TO_HOST : VictimAction::INVALIDATE_ONLY;
}

template <class StateT, class PredecessorQueryT>
[[nodiscard]] inline bool query_predecessor_reader_oracle(
    const StateT &state, PredecessorQueryT &predecessor_query, PredecessorReaderCache &cache,
    taskid_t top_writer_task_id, dataid_t data_id, devid_t invalidate_device) {
  const auto write_generation = state.get_mapped_write_generation(data_id);
  const auto read_generation = state.get_mapped_read_generation(data_id, invalidate_device);

  bool cached_has_local_reader = false;
  if (cache.get(data_id, invalidate_device, top_writer_task_id, write_generation, read_generation,
                cached_has_local_reader)) {
    return cached_has_local_reader;
  }

  const bool has_local_predecessor_reader =
      predecessor_query.has_local_reader_predecessor_for_eviction(state, top_writer_task_id,
                                                                  data_id, invalidate_device);
  cache.put(data_id, invalidate_device, top_writer_task_id, write_generation, read_generation,
            has_local_predecessor_reader);
  return has_local_predecessor_reader;
}

template <class StateT, class PredecessorQueryT>
[[nodiscard]] inline InvalidationInfo compute_invalidation_info(
    const StateT &state, PredecessorQueryT &predecessor_query,
    PredecessorReaderCache &predecessor_reader_cache, dataid_t data_id, devid_t invalidate_device) {
  // Contract:
  // - pending_local_users: mapped-not-reserved tasks on invalidate_device may still consume data_id.
  // - mapped_cleanup:
  //   KEEP_MAPPED: keep mapped accounting/location untouched on invalidate_device.
  //   REMOVE_MAPPED_BYTES_ONLY: remove mapped bytes but keep mapped validity.
  //   REMOVE_MAPPED_AND_LOCATION: remove mapped bytes and mapped validity.
  InvalidationInfo info{};
  info.pending_local_users = state.has_pending_local_users(data_id, invalidate_device);
  info.mapped_cleanup = info.pending_local_users
                            ? InvalidationInfo::MappedCleanupDecision::KEEP_MAPPED
                            : InvalidationInfo::MappedCleanupDecision::REMOVE_MAPPED_AND_LOCATION;

  const auto &writers = state.get_mapped_writers(data_id);
  if (writers.empty()) {
    return info;
  }

  const taskid_t top_writer = select_top_writer_by_depth(writers, state);
  if (state.get_task_mapped_device(top_writer) == invalidate_device) {
    return info;
  }

  if (!state.has_local_mapped_readers(data_id, invalidate_device)) {
    if (info.pending_local_users) {
      info.mapped_cleanup = InvalidationInfo::MappedCleanupDecision::REMOVE_MAPPED_BYTES_ONLY;
    }
    return info;
  }

  const bool has_local_predecessor_reader = query_predecessor_reader_oracle(
      state, predecessor_query, predecessor_reader_cache, top_writer, data_id, invalidate_device);
  if (!has_local_predecessor_reader && info.pending_local_users) {
    info.mapped_cleanup = InvalidationInfo::MappedCleanupDecision::REMOVE_MAPPED_BYTES_ONLY;
  }
  return info;
}

template <class PolicyT, class StateT, class PredecessorQueryT, class RequestListT>
class RequestPlanner {
public:
  void plan(const PolicyT &policy, const StateT &state, PredecessorQueryT &predecessor_query,
            const RequestListT &requests, PlannedVictimSet &planned_victims,
            InvalidationCache &invalidation_cache,
            PredecessorReaderCache &predecessor_reader_cache,
            std::vector<VictimPlan> &out_plans) const {
    out_plans.clear();

    for (const auto &[compute_task_id, device_id] : requests) {
      // Planner only creates candidates for requests that still overflow reserve memory.
      const mem_t missing_mem = state.get_missing_reserve_memory(compute_task_id, device_id);
      if (missing_mem <= 0) {
        continue;
      }

      const auto protected_data = state.get_task_unique_data(compute_task_id);
      const auto victim_data_ids =
          policy.select_victims(state.eviction_residency(), device_id, missing_mem, protected_data);
      for (const auto data_id : victim_data_ids) {
        if (!planned_victims.mark(data_id, device_id)) {
          continue;
        }

        VictimPlan plan{};
        plan.compute_task_id = compute_task_id;
        plan.device_id = device_id;
        plan.data_id = data_id;
        plan.action = classify_victim_action(state.get_launched_location_flags(data_id));

        if (plan.action == VictimAction::INVALIDATE_ONLY) {
          // Invalidation decisions are shared across requesters in a cycle through cache.
          if (!invalidation_cache.get(data_id, device_id, plan.invalidation)) {
            plan.invalidation = compute_invalidation_info(state, predecessor_query,
                                                          predecessor_reader_cache, data_id,
                                                          device_id);
            invalidation_cache.put(data_id, device_id, plan.invalidation);
          }
          plan.has_invalidation_info = true;
        }

        out_plans.push_back(plan);
      }
    }
  }
};

} // namespace eviction
