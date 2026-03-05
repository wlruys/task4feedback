#pragma once
#include "communication.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include "tasks.hpp"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <span>
#include <string>
#include <tracy/Tracy.hpp>
#include <unordered_map>

struct XYPosition {
  float x = 0.0f;
  float y = 0.0f;

  XYPosition() = default;
  XYPosition(float x, float y) : x(x), y(y) {
  }

  bool operator==(const XYPosition &other) const {
    return x == other.x && y == other.y;
  }
};

class Data {
protected:
  std::vector<mem_t> sizes;
  std::vector<XYPosition> xy_positions;
  std::vector<int32_t> data_types;
  std::vector<int32_t> data_tags;
  std::vector<devid_t> initial_location;
  std::vector<std::string> data_names;
  std::unordered_map<std::string, dataid_t> name_to_id;

public:
  Data() = default;
  Data(std::size_t num_data)
      : sizes(num_data), xy_positions(num_data), initial_location(num_data, 0),
        data_names(num_data), data_types(num_data, 0), data_tags(num_data, 0) {
  }

  [[nodiscard]] bool empty() const {
    return size() == 0;
  }

  void set_size(dataid_t id, mem_t size) {
    T4F_INVARIANT(id < sizes.size());
    sizes[id] = size;
  }

  void set_tag(dataid_t id, int tag) {
    T4F_INVARIANT(id < data_tags.size());
    data_tags[id] = tag;
  }

  void set_x_pos(dataid_t id, float x) {
    T4F_INVARIANT(id < xy_positions.size());
    xy_positions[id].x = x;
  }

  void set_y_pos(dataid_t id, float y) {
    T4F_INVARIANT(id < xy_positions.size());
    xy_positions[id].y = y;
  }

  [[nodiscard]] float get_x_pos(dataid_t id) const {
    T4F_INVARIANT(id < xy_positions.size());
    return xy_positions[id].x;
  }

  [[nodiscard]] float get_y_pos(dataid_t id) const {
    T4F_INVARIANT(id < xy_positions.size());
    return xy_positions[id].y;
  }

  int get_tag(dataid_t id) const {
    T4F_INVARIANT(id < data_tags.size());
    return data_tags[id];
  }

  void set_type(dataid_t id, int type) {
    T4F_INVARIANT(id < data_types.size());
    data_types[id] = type;
  }

  int get_type(dataid_t id) const {
    T4F_INVARIANT(id < data_types.size());
    return data_types[id];
  }

  void set_location(dataid_t id, devid_t location) {
    T4F_INVARIANT(id < initial_location.size());

    initial_location[id] = location;
  }
  void set_name(dataid_t id, std::string name) {
    T4F_INVARIANT(id < data_names.size());
    data_names[id] = std::move(name);
    name_to_id[data_names[id]] = id;
  }

  dataid_t get_id(const std::string &name) const {
    return name_to_id.at(name);
  }

  void create_block(dataid_t id, mem_t size, devid_t location, std::string name) {
    // extend the vectors if necessary
    if (id >= sizes.size()) {
      sizes.resize(id + 1);
      initial_location.resize(id + 1);
      data_names.resize(id + 1);
      data_types.resize(id + 1);
      data_tags.resize(id + 1);
      xy_positions.resize(id + 1);
    }

    T4F_INVARIANT(id < sizes.size());
    set_size(id, size);
    set_location(id, location);
    set_name(id, std::move(name));
    set_type(id, 0);
    set_tag(id, 0);
  }

  [[nodiscard]] dataid_t size() const {
    return sizes.size();
  }

  dataid_t append_block(mem_t size, devid_t location, std::string name) {
    create_block(sizes.size(), size, location, std::move(name));
    return sizes.size() - 1;
  }

  [[nodiscard]] mem_t get_size(dataid_t id) const {
    return sizes[id];
  }

  [[nodiscard]] mem_t get_total_size(const std::span<const dataid_t> ids) const {
    mem_t total_size = 0;
    for (const auto &id : ids) {
      total_size += sizes[id];
    }
    return total_size;
  }

  [[nodiscard]] mem_t get_total_size() const {
    return std::accumulate(sizes.begin(), sizes.end(), static_cast<mem_t>(0));
  }

  [[nodiscard]] devid_t get_location(dataid_t id) const { 
      T4F_INVARIANT(id < initial_location.size()); 
      return initial_location[id]; 
  }
  [[nodiscard]] const std::string &get_name(dataid_t id) const { 
      T4F_INVARIANT(id < data_names.size()); 
      return data_names[id]; 
  }

  [[nodiscard]] auto get_sizes() const {
    return sizes;
  }
  [[nodiscard]] auto get_locations() const {
    return initial_location;
  }
  [[nodiscard]] auto get_names() const {
    return data_names;
  }

  friend class DataManager;
};

struct ValidEventArray {
  std::vector<timecount_t> starts;
  std::vector<timecount_t> stops;
  std::size_t size = 0;
};

class LocationManager {
protected:
  uint8_t num_devices{0};
  dataid_t num_data{0};
  std::vector<devicemask_t> locations;
  std::vector<ValidEventArray> valid_intervals;
  bool record{false};

public:
  LocationManager() = default;

  LocationManager(dataid_t num_data, devid_t num_devices)
      : num_devices(num_devices), num_data(num_data), locations(num_data, 0) {

#ifdef SIM_RECORD
    constexpr size_t buffer_initial_size = 100;
    valid_intervals.resize(num_data * num_devices);
    for (auto &intervals : valid_intervals) {
      intervals.starts.reserve(buffer_initial_size);
      intervals.stops.reserve(buffer_initial_size);
    }
#endif
  }

  LocationManager(const LocationManager &) = default;

  [[nodiscard]] inline bool is_valid(dataid_t data_id, devid_t device_id) const {
    T4F_INVARIANT(data_id < num_data && device_id < num_devices);
    // Check if "device_id"-th bit of "data_id"-th location is set
    return locations[data_id] & (1 << device_id);
  }

  [[nodiscard]] inline bool is_invalid(dataid_t data_id, devid_t device_id) const {
    return !is_valid(data_id, device_id);
  }

  inline devicemask_t set_valid(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    const devicemask_t mask = (1 << device_id);
    auto old_status = locations[data_id] & mask;

#ifdef SIM_RECORD
    if (record) {
      if (old_status) {
        // start new interval
        valid_intervals[data_id * num_devices + device_id].starts.push_back(current_time);
        valid_intervals[data_id * num_devices + device_id].stops.push_back(MAX_TIME);
      }
    }
#endif

    locations[data_id] |= mask;
    return old_status;
  }

  inline devicemask_t set_invalid(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    const devicemask_t mask = (1 << device_id);
    auto old_status = locations[data_id] & mask;
#ifdef SIM_RECORD
    if (record) {
      if (is_valid(data_id, device_id)) {
        // close the current interval
        valid_intervals[data_id * num_devices + device_id].stops.back() = current_time;
      }
    }
#endif

    locations[data_id] &= ~mask;
    return old_status;
  }

  [[nodiscard]] inline std::size_t count_valid(dataid_t data_id) const {
    return std::popcount(static_cast<std::make_unsigned_t<devicemask_t>>(locations[data_id]));
  }

  [[nodiscard]] inline devicemask_t get_location_flags(dataid_t data_id) const {
    return locations[data_id];
  }

  [[nodiscard]] inline devicemask_t get_location_flags(dataid_t data_id) {
    return locations[data_id];
  }

  void populate_valid_locations(dataid_t data_id, std::vector<devid_t> &valid_locations) const {
    auto mask = locations[data_id];
    while (mask) {
      auto bit = std::countr_zero(mask);
      valid_locations.push_back(static_cast<devid_t>(bit));
      mask &= (mask - 1);
    }
  }

  bool inline validate(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    return set_valid(data_id, device_id, current_time) == 0;
  }

  inline devicemask_t invalidate_except(dataid_t data_id, devid_t device_id,
                                        timecount_t current_time) {
    T4F_INVARIANT(data_id < num_data && device_id < num_devices);
    devicemask_t old_status = locations[data_id];
    devicemask_t keep_mask = (1 << device_id);
    // Keep only the specified device, invalidate all others
    locations[data_id] &= keep_mask;

    // which bits changed from 1 to 0
    // assumes that data was already valid on device_id
    devicemask_t changed_bits = old_status ^ locations[data_id];

    return changed_bits;
  }

  inline devicemask_t invalidate_all(dataid_t data_id, timecount_t current_time) {

    devicemask_t old_status = locations[data_id];

    // Invalidate all devices
    locations[data_id] = 0;

    // which bits changed from 1 to 0
    devicemask_t changed_bits = old_status ^ locations[data_id];

    return changed_bits;
  }

  inline devicemask_t invalidate_on(dataid_t data_id, devid_t device_id, timecount_t current_time) {

    devicemask_t old_status = locations[data_id];

    locations[data_id] &= ~(1 << device_id); // invalidate the specified device
    devicemask_t changed_bits = old_status ^ locations[data_id];

    return changed_bits;
  }

  void finalize(timecount_t current_time) {
// tie off any open/hanging interval at the end of the simulation
#ifdef SIM_RECORD
    for (dataid_t i = 0; i < num_data; i++) {
      for (devicemask_t j = 0; j < num_devices; j++) {
        if (is_valid(i, j)) {
          valid_intervals[i * num_devices + j].stops.back() = current_time;
        }
      }
    }
#endif
  }

  ValidEventArray &get_valid_intervals(dataid_t data_id, devid_t device_id) {
    return valid_intervals[data_id * num_devices + device_id];
  }
};

[[nodiscard]] inline uint64_t pack_movement_key(dataid_t data_id, devid_t destination) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(destination)) << 32U) |
         static_cast<uint32_t>(data_id);
}

class MovementManager {
protected:
  ankerl::unordered_dense::map<uint64_t, timecount_t> movement_times;

public:
  MovementManager() = default;

  bool is_moving(dataid_t data_id, devid_t destination) const {
    return movement_times.find(pack_movement_key(data_id, destination)) != movement_times.end();
  }

  [[nodiscard]] inline timecount_t get_time(dataid_t data_id, devid_t destination) const {
    auto it = movement_times.find(pack_movement_key(data_id, destination));
    return it == movement_times.end() ? 0 : it->second;
  }

  [[nodiscard]] inline bool try_get_time(dataid_t data_id, devid_t destination,
                                         timecount_t &completion_time) const {
    auto it = movement_times.find(pack_movement_key(data_id, destination));
    if (it == movement_times.end()) {
      return false;
    }
    completion_time = it->second;
    return true;
  }

  inline void set_completion(dataid_t data_id, devid_t destination,
                             timecount_t global_completion_time) {
    movement_times[pack_movement_key(data_id, destination)] = global_completion_time;
  }

  inline void remove(dataid_t data_id, devid_t destination) {
    movement_times.erase(pack_movement_key(data_id, destination));
  }
};

struct MovementStatus {
  bool is_virtual = false;
  timecount_t duration = 0;
};

enum class InsertResult {
  Updated,
  Inserted,
  Failed
};

class LRU_manager {
private:
  mem_t evicted_size = 0;
  mem_t max_usage = 0;

  struct alignas(64) DeviceLRU {
    using node_t = int32_t;
    static constexpr node_t kNull = -1;

    struct Node {
      dataid_t id;
      mem_t bytes;
      node_t next;
      node_t prev;
    };

    std::vector<Node> nodes;
    ankerl::unordered_dense::map<dataid_t, node_t> where;
    node_t free_head = kNull;
    mem_t used_bytes = 0;
    mem_t capacity_bytes = 0;
    std::size_t hard_max_items = 0;

#ifndef NDEBUG
    uint32_t op_counter = 0;
    static constexpr uint32_t kCheckPeriod = 8192;
#endif

    void init(std::size_t initial_max_items, mem_t cap_bytes,
              std::size_t reserve_items_hint, std::size_t hard_cap_items = 0) {
      capacity_bytes = cap_bytes;
      hard_max_items = hard_cap_items;

      nodes.resize(initial_max_items + 1);
      nodes[0] = {dataid_t(-1), 0, 0, 0}; // Sentinel node

      free_head = initial_max_items > 0 ? 1 : kNull;
      for (node_t i = 1; i < static_cast<node_t>(initial_max_items); ++i) {
        nodes[i].next = i + 1;
      }
      if (initial_max_items > 0) {
        nodes[initial_max_items].next = kNull;
      }

      where.clear();
      where.reserve(reserve_items_hint);
      used_bytes = 0;

#ifndef NDEBUG
      op_counter = 0;
#endif
    }

    bool ensure_free_node_() {
      if (free_head != kNull) return true;

      const std::size_t old_user_items = nodes.size() - 1;
      std::size_t new_user_items = old_user_items == 0 ? 1024 : old_user_items * 2;

      if (hard_max_items != 0 && new_user_items > hard_max_items) {
        new_user_items = hard_max_items;
        if (new_user_items <= old_user_items) return false;
      }

      const std::size_t new_n = new_user_items + 1;
      nodes.resize(new_n);

      free_head = static_cast<node_t>(old_user_items + 1);
      for (node_t i = free_head; i < static_cast<node_t>(new_n - 1); ++i) {
        nodes[i].next = i + 1;
      }
      nodes[new_n - 1].next = kNull;

      SPDLOG_WARN("LRU_manager: expanding DeviceLRU node pool from {} -> {} user nodes",
                  old_user_items, new_user_items);
      return true;
    }

    void unlink_(node_t n) {
      const node_t p = nodes[n].prev;
      const node_t q = nodes[n].next;
      nodes[p].next = q;
      nodes[q].prev = p;
    }

    void link_front_(node_t n) {
      const node_t head = nodes[0].next;
      nodes[n].prev = 0;
      nodes[n].next = head;
      nodes[head].prev = n;
      nodes[0].next = n;
    }

    void touch_node_(node_t n) {
      if (nodes[0].next == n) return;
      unlink_(n);
      link_front_(n);
    }

#ifndef NDEBUG
    void maybe_check_invariants_() {
      if ((++op_counter % kCheckPeriod) != 0) return;
      
      const node_t head = nodes[0].next;
      const node_t tail = nodes[0].prev;
      
      if (head == 0) {
        T4F_INVARIANT(tail == 0 && where.empty());
        return;
      }

      T4F_INVARIANT(nodes[head].prev == 0 && nodes[tail].next == 0);

      std::size_t seen = 0;
      node_t cur = head;
      node_t last = 0;
      
      while (cur != 0) {
        T4F_INVARIANT(nodes[cur].id != dataid_t(-1));
        auto it = where.find(nodes[cur].id);
        T4F_INVARIANT(it != where.end() && it->second == cur);
        T4F_INVARIANT(nodes[cur].prev == last);

        last = cur;
        cur = nodes[cur].next;
        T4F_INVARIANT(++seen <= nodes.size());
      }
      T4F_INVARIANT(last == tail && seen == where.size());
    }
#endif

    InsertResult insert_or_update(dataid_t id, mem_t bytes) {
      auto [it, inserted] = where.try_emplace(id, 0);

      if (!inserted) {
        const node_t n = it->second;
        if (nodes[n].bytes != bytes) {
          used_bytes += (bytes - nodes[n].bytes);
          nodes[n].bytes = bytes;
        }
        touch_node_(n);
#ifndef NDEBUG
        maybe_check_invariants_();
#endif
        return InsertResult::Updated;
      }

      if (!ensure_free_node_()) {
        where.erase(it);
        return InsertResult::Failed;
      }

      const node_t n = free_head;
      free_head = nodes[n].next;

      it->second = n;

      nodes[n].id = id;
      nodes[n].bytes = bytes;
      link_front_(n);

      used_bytes += bytes;

#ifndef NDEBUG
      maybe_check_invariants_();
#endif
      return InsertResult::Inserted;
    }

    mem_t erase(dataid_t id) {
      auto it = where.find(id);
      if (it == where.end()) return 0;

      const node_t n = it->second;
      where.erase(it);
      unlink_(n);

      const mem_t b = nodes[n].bytes;
      used_bytes -= b;

      nodes[n].next = free_head;
      free_head = n;

#ifndef NDEBUG
      maybe_check_invariants_();
#endif
      return b;
    }

    std::size_t collect_lru_victims(std::size_t bytes_needed,
                                    std::span<const dataid_t> used_ids,
                                    DataIDList &out) const {
      out.clear();
      std::size_t acc = 0;
      node_t cur = nodes[0].prev;

      if (used_ids.empty()) {
        while (cur != 0 && acc < bytes_needed) {
          acc += static_cast<std::size_t>(nodes[cur].bytes);
          out.push_back(nodes[cur].id);
          cur = nodes[cur].prev;
        }
      } else {
        // Check that this data is not used by the requester before adding to eviction list
        while (cur != 0 && acc < bytes_needed) {
          const dataid_t id = nodes[cur].id;
          
          if (!std::binary_search(used_ids.begin(), used_ids.end(), id)) {
            acc += static_cast<std::size_t>(nodes[cur].bytes);
            out.push_back(id);
          }
          cur = nodes[cur].prev;
        }
      }
      return acc;
    }

    static DeviceLRU clone_with_headroom(const DeviceLRU &src) {
      DeviceLRU dst;
      const std::size_t active_items = src.where.size();
      if (active_items == 0) {
        dst.init(0, src.capacity_bytes, 0, src.hard_max_items);
        return dst;
      }

      const std::size_t src_user_capacity = src.nodes.size() > 0 ? src.nodes.size() - 1 : 0;
      const std::size_t active_half = active_items / 2;
      const std::size_t retained_slack_half =
          src_user_capacity > active_items
              ? std::min((src_user_capacity - active_items) / 2, active_items)
              : 0;
      const std::size_t headroom = std::max<std::size_t>({64, active_half, retained_slack_half});

      std::size_t target_items = active_items + headroom;
      target_items = std::max(target_items, active_items);

      if (src.hard_max_items != 0) {
        target_items = std::min(target_items, src.hard_max_items);
      }

      dst.init(target_items, src.capacity_bytes, target_items, src.hard_max_items);

      node_t cur = src.nodes[0].prev; // LRU -> MRU so insert_front preserves order
      while (cur != 0) {
        const auto result = dst.insert_or_update(src.nodes[cur].id, src.nodes[cur].bytes);
        T4F_INVARIANT(result != InsertResult::Failed);
        cur = src.nodes[cur].prev;
      }

      dst.used_bytes = src.used_bytes;
      return dst;
    }
  };

  std::vector<DeviceLRU> lrus_;
  mutable DataIDList id_buffer;

  bool is_valid_device(devid_t id) const {
    return id >= 0 && static_cast<std::size_t>(id) < lrus_.size();
  }

public:
  LRU_manager() = default;

  explicit LRU_manager(const Devices &devices,
                       mem_t median_block_size,
                       std::size_t hard_max_items_per_device = 0,
                       std::size_t distinct_items_hint = static_cast<std::size_t>(-1))
      : lrus_(devices.size()) {
        
    median_block_size = median_block_size > 0 ? median_block_size : 1;
    constexpr std::size_t kMinItems = 4096;
    constexpr std::size_t kMaxItems = 1'000'000;
    const bool has_distinct_items_hint = distinct_items_hint != static_cast<std::size_t>(-1);

    for (std::size_t dev = 0; dev < lrus_.size(); ++dev) {
      const mem_t cap = devices.get_max_resources(dev).mem;
      std::size_t expected = static_cast<std::size_t>(cap / median_block_size);

      if (has_distinct_items_hint) expected = std::min(expected, distinct_items_hint);
      if (hard_max_items_per_device != 0) expected = std::min(expected, hard_max_items_per_device);

      std::size_t initial_items, reserve_items;
      if (has_distinct_items_hint) {
        initial_items = expected == 0 ? 0 : 2 * expected;
        reserve_items = expected;
      } else {
        initial_items = std::clamp(2 * expected, kMinItems, kMaxItems);
        reserve_items = std::clamp(expected, kMinItems / 2, initial_items);
      }

      if (hard_max_items_per_device != 0) {
        initial_items = std::min(initial_items, hard_max_items_per_device);
        reserve_items = std::min(reserve_items, initial_items);
      }

      lrus_[dev].init(initial_items, cap, reserve_items, hard_max_items_per_device);
    }

    id_buffer.reserve(20);
  }

  LRU_manager(const LRU_manager &other)
      : evicted_size(other.evicted_size), max_usage(other.max_usage), lrus_(other.lrus_.size()) {
    for (std::size_t dev = 0; dev < lrus_.size(); ++dev) {
      lrus_[dev] = DeviceLRU::clone_with_headroom(other.lrus_[dev]);
    }
    id_buffer.reserve(other.id_buffer.capacity());
  }

  void read(devid_t device_id, dataid_t data_id, mem_t mem_size) {
    T4F_INVARIANT(is_valid_device(device_id));
    // if (mem_size <= 0) {
    //   // Zero-sized objects are intentionally not tracked in the LRU.
    //   return;
    // }
    auto &lru = lrus_[device_id];

    const InsertResult result = lru.insert_or_update(data_id, mem_size);

    if (result == InsertResult::Failed) [[unlikely]] {
      SPDLOG_ERROR("LRU_manager::read(): hard cap hit on device {}; data_id {} not tracked", device_id, data_id);
      T4F_INVARIANT(false && "LRU_manager::read(): hard cap hit; increase hard_max_items_per_device");
      return;
    }

    if (result == InsertResult::Inserted) {
      if (device_id > 0 && lru.used_bytes > max_usage) {
        max_usage = lru.used_bytes;
      }
      if (lru.used_bytes > lru.capacity_bytes) [[unlikely]] {
        SPDLOG_DEBUG("LRU_manager::read(): Device {}: Adding data_id {} with size {}", device_id, data_id, mem_size);
        T4F_INVARIANT(lru.used_bytes <= lru.capacity_bytes && "LRU_manager::read(): size exceeds max size");
      }
    }
  }

  void invalidate(devid_t device_id, dataid_t data_id, bool evict = false) {
    T4F_INVARIANT(is_valid_device(device_id));
    const mem_t removed = lrus_[device_id].erase(data_id);
    // if (removed <= 0) {
    //   // Object is not tracked by LRU (e.g., zero-sized); nothing to invalidate.
    //   return;
    // }

    if (evict) evicted_size += removed;
  }

  std::span<const dataid_t> getLRUids(devid_t device_id, std::size_t mem_size,
                                      std::span<const dataid_t> used_ids) const {
    T4F_INVARIANT(is_valid_device(device_id));
    
    const std::size_t accumulated = lrus_[device_id].collect_lru_victims(mem_size, used_ids, id_buffer);
    T4F_INVARIANT(accumulated >= mem_size && "getLRUids(): evictable memory size is smaller than requested");
    
    return id_buffer;
  }

  mem_t get_mem(devid_t device_id) const {
    T4F_INVARIANT(is_valid_device(device_id));
    return lrus_[device_id].used_bytes;
  }

  mem_t get_evicted_memory_size() const {
    return evicted_size;
  }

  mem_t get_max_memory_usage() const {
    mem_t sum = 0;
    for (const auto& lru : lrus_) sum += lru.used_bytes;
    return sum;
  }
};


// class LRU_manager {
// private:
//   mem_t evicted_size = 0;
//   mem_t max_usage = 0;
//   uint32_t n_devices_{0};
//   // For each device:
//   //  - a list maintaining LRU (front) → MRU (back)
//   //  - a map from data_id → its position in that list
//   //  - a map from data_id → its mem_size
//   std::vector<std::list<dataid_t>> lru_lists_;
//   std::vector<ankerl::unordered_dense::map<dataid_t, typename std::list<dataid_t>::iterator>>
//       position_maps_;
//   std::vector<ankerl::unordered_dense::map<dataid_t, mem_t>> size_maps_;
//   std::vector<mem_t> sizes_;
//   std::vector<mem_t> max_sizes_;
//   mutable DataIDList id_buffer;

// public:
//   LRU_manager() = default;

//   // Constructor: initialize for n_devices [0 .. n_devices-1]
//   explicit LRU_manager(const Devices &devices)
//       : n_devices_(devices.size()), lru_lists_(devices.size()), position_maps_(devices.size()),
//         size_maps_(devices.size()), sizes_(devices.size()), max_sizes_(devices.size()) {
//     for (auto &size : sizes_) {
//       size = 0;
//     }
//     for (int i = 0; i < devices.size(); i++) {
//       max_sizes_[i] = devices.get_max_resources(i).mem;
//     }
//     id_buffer.reserve(20);
//   }

//   // read: add (device_id, data_id, mem_size). If present, update MRU; else insert.
//   void read(devid_t device_id, dataid_t data_id, mem_t mem_size) {
//     T4F_INVARIANT(device_id >= 0 && device_id < n_devices_);

//     auto &lst = lru_lists_[device_id];
//     auto &pos = position_maps_[device_id];
//     auto &smap = size_maps_[device_id];
//     auto &size = sizes_[device_id];
//     auto &max_size = max_sizes_[device_id];
//     auto it = pos.find(data_id);
//     if (it != pos.end()) {
//       // already present: move to MRU
//       lst.erase(it->second);
//     } else {
//       size += mem_size;
//       if (size > max_usage && device_id > 0) {
//         max_usage = size;
//       }
//       if (size > max_size) {
//         SPDLOG_DEBUG("LRU_manager::read(): Device {}: Adding data_id {} with size {}", device_id,
//                      data_id, mem_size);
//         T4F_INVARIANT(size <= max_size && "LRU_manager::read(): size exceeds max size");
//       }
//     }
//     // insert at MRU (back)
//     lst.push_back(data_id);
//     auto new_it = std::prev(lst.end());
//     pos[data_id] = new_it;
//     smap[data_id] = mem_size; // update size
//   }

//   LRU_manager(const LRU_manager &other)
//       : n_devices_(other.n_devices_), lru_lists_(other.lru_lists_),
//         position_maps_(other.n_devices_), size_maps_(other.size_maps_), sizes_(other.sizes_),
//         max_sizes_(other.max_sizes_), evicted_size(other.evicted_size), max_usage(other.max_usage) {
//     ZoneScoped;
//     // Rebuild position_maps_
//     for (devid_t dev = 0; dev < n_devices_; ++dev) {
//       position_maps_[dev].reserve(other.position_maps_[dev].size());
//       for (auto it = lru_lists_[dev].begin(); it != lru_lists_[dev].end(); ++it) {
//         position_maps_[dev][*it] = it;
//       }
//     }
//     id_buffer.reserve(other.id_buffer.capacity());
//   }

//   // invalidate: remove (device_id, data_id); assert if missing
//   void invalidate(devid_t device_id, dataid_t data_id, bool evict = false) {
//     T4F_INVARIANT(device_id >= 0 && device_id < n_devices_);

//     auto &lst = lru_lists_[device_id];
//     auto &pos = position_maps_[device_id];
//     auto &smap = size_maps_[device_id];
//     auto &size = sizes_[device_id];

//     auto it = pos.find(data_id);
//     T4F_INVARIANT(it != pos.end() && "invalidate(): data_id not present");

//     lst.erase(it->second);
//     pos.erase(it);
//     size -= smap[data_id]; // update size
//     if (evict)
//       evicted_size += smap[data_id];
//     smap.erase(data_id);
//   }

//   // getLRUids: fill id_buffer[device_id] with the least-recently-used data_ids
//   // until their cumulative mem_size ≥ requested mem_size, and return it.
//   const std::span<const dataid_t> getLRUids(devid_t device_id, std::size_t mem_size,
//                                             std::span<const dataid_t> used_ids) const {
//     T4F_INVARIANT(device_id >= 0 && device_id < n_devices_);

//     auto &lst = lru_lists_[device_id];
//     auto &smap = size_maps_[device_id];
//     id_buffer.clear();
//     std::size_t accumulated = 0;

//     for (auto it = lst.begin(); it != lst.end() && accumulated < mem_size; ++it) {
//       dataid_t did = *it;
//       if (std::find(used_ids.begin(), used_ids.end(), did) != used_ids.end()) {
//         continue; // skip if used by the task
//       }
//       auto sz_it = smap.find(did);
//       T4F_INVARIANT(sz_it != smap.end() && "size missing for data_id");
//       accumulated += sz_it->second;
//       id_buffer.push_back(did);
//     }
//     T4F_INVARIANT(accumulated >= mem_size &&
//            "getLRUids(): evictable memory isze is smaller than the requested size");
//     return id_buffer;
//   }

//   mem_t get_mem(devid_t device_id) const {
//     T4F_INVARIANT((device_id >= 0) && (device_id < n_devices_));
//     return sizes_[device_id];
//   }

//   mem_t get_evicted_memory_size() const {
//     return evicted_size;
//   }

//   mem_t get_max_memory_usage() const {
//     mem_t total = 0;
//     for (const auto size : sizes_) {
//       total += size;
//     }
//     return total;
//   }
// };

class MovementCounter {
private:
  std::vector<mem_t> total_data_movement;
  std::vector<mem_t> eviction_data_movement;

public:
  MovementCounter() = default;

  MovementCounter(devid_t n_devices)
      : total_data_movement(n_devices, 0), eviction_data_movement(n_devices, 0) {
  }

  void add_total_movement(devid_t src, devid_t dest, mem_t size) {
    T4F_INVARIANT(src < total_data_movement.size());
    T4F_INVARIANT(dest < total_data_movement.size());
    total_data_movement[src] += size;
    total_data_movement[dest] += size;
    if (dest == 0) {
      eviction_data_movement[src] += size;
      eviction_data_movement[dest] += size;
    }
  }

  void add_eviction_movement(devid_t device_id, mem_t size) {
    T4F_INVARIANT(device_id < eviction_data_movement.size());
    eviction_data_movement[device_id] += size;
  }

  const std::vector<mem_t> &get_total_data_movement() const {
    return total_data_movement;
  }

  const mem_t get_total_data_movement(devid_t device_id) const {
    T4F_INVARIANT(device_id < total_data_movement.size());
    return total_data_movement[device_id];
  }

  const std::vector<mem_t> &get_eviction_data_movement() const {
    return eviction_data_movement;
  }

  const mem_t get_eviction_data_movement(devid_t device_id) const {
    T4F_INVARIANT(device_id < eviction_data_movement.size());
    return eviction_data_movement[device_id];
  }
};

class DataManager {
protected:
  LocationManager mapped_locations;
  LocationManager reserved_locations;
  LocationManager launched_locations;
  MovementManager movement_manager;
  LRU_manager lru_manager;
  MovementCounter movement_counter;
  bool initialized = false;

  static bool check_valid(size_t data_id, const LocationManager &locations, devid_t device_id) {
    return locations.is_valid(data_id, device_id);
  }

  static bool check_valid(std::span<const dataid_t> list, const LocationManager &locations,
                          devid_t device_id) {
    return std::ranges::all_of(
        list, [&](auto data_id) { return !locations.is_invalid(data_id, device_id); });
  }

  static bool read_update(dataid_t data_id, devid_t device_id, LocationManager &locations,
                          timecount_t current_time) {
    const bool changed = locations.validate(data_id, device_id, current_time);
    T4F_INVARIANT(locations.is_valid(data_id, device_id));
    return changed;
  }

  static auto write_update(dataid_t data_id, devid_t device_id, LocationManager &locations,
                           timecount_t current_time) {
    T4F_INVARIANT(locations.is_valid(data_id, device_id) &&
                  "write_update requires data to be valid on the writing device");
    auto updated_ids = locations.invalidate_except(data_id, device_id, current_time);
    return updated_ids;
  }

  static auto evict_on_update(dataid_t data_id, devid_t device_id, LocationManager &locations,
                              timecount_t current_time) {
    auto updated_ids = locations.invalidate_on(data_id, device_id, current_time);
    return updated_ids;
  }

  static mem_t estimate_lru_block_size(const Data &data) {
    if (data.empty()) {
      return 1;
    }

    auto sizes = data.get_sizes();
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(), [](mem_t s) { return s <= 0; }),
                sizes.end());
    if (sizes.empty()) {
      return 1;
    }

    int64_t total = 0;
    for (const mem_t size : sizes) {
      total += size;
    }

    const mem_t average = static_cast<mem_t>(total / static_cast<int64_t>(sizes.size()));
    return average > 0 ? average : 1;
  }

public:
  std::vector<devid_t> valid_location_buffer;

  DataManager() = default;

  DataManager(const Data &data, const Devices &devices)
      : mapped_locations(data.size(), devices.size()),
        reserved_locations(data.size(), devices.size()),
        launched_locations(data.size(), devices.size()),
        lru_manager(devices, estimate_lru_block_size(data), 0, data.size()),
        movement_counter(devices.size()) {
  }

  DataManager(const DataManager &o_)
      : lru_manager(o_.lru_manager), movement_counter(o_.movement_counter) {
    ZoneScopedN("Copy DataManager");
    {
      ZoneScopedN("Copy Mapped Locations");
      mapped_locations = o_.mapped_locations;
    }
    {
      ZoneScopedN("Copy Reserved Locations");
      reserved_locations = o_.reserved_locations;
    }

    {
      ZoneScopedN("Copy Launched Locations");
      launched_locations = o_.launched_locations;
    }

    {
      ZoneScopedN("Copy Movement Manager");
      movement_manager = o_.movement_manager;
    }
    initialized = o_.initialized;
  }

  void initialize(const Data &data, const Devices &devices, DeviceManager &device_manager) {
    ZoneScoped;
    if (initialized) {
      SPDLOG_WARN("DataManager already initialized. Skipping re-initialization.");
      return;
    }
    initialized = true;
    for (dataid_t i = 0; i < data.size(); i++) {
      auto initial_location = data.get_location(i);
      const auto data_size = data.get_size(i);
      const bool initial_location_in_bounds =
          initial_location >= 0 && initial_location < devices.size();

      if (initial_location > -1 && !initial_location_in_bounds) {
        SPDLOG_CRITICAL(
            "DataManager::initialize(): data_id {} has invalid initial location {} (valid "
            "device ids: 0..{})",
            i, initial_location, devices.size() - 1);
        T4F_INVARIANT(false &&
               "DataManager::initialize(): initial data location out of bounds for system "
               "devices");
      }

      if (initial_location_in_bounds && (lru_manager.get_mem(initial_location) + data_size) <=
                                       devices.get_max_resources(initial_location).mem) {
        mapped_locations.set_valid(i, initial_location, 0);
        reserved_locations.set_valid(i, initial_location, 0);
        launched_locations.set_valid(i, initial_location, 0);
        device_manager.add_mem<TaskState::MAPPED>(initial_location, data_size, 0);
        device_manager.add_mem<TaskState::RESERVED>(initial_location, data_size, 0);
        device_manager.add_mem<TaskState::LAUNCHED>(initial_location, data_size, 0);
        lru_manager.read(initial_location, i, data_size);
      } else {
        mapped_locations.set_valid(i, 0, 0);
        reserved_locations.set_valid(i, 0, 0);
        launched_locations.set_valid(i, 0, 0);
        device_manager.add_mem<TaskState::MAPPED>(0, data_size, 0);
        device_manager.add_mem<TaskState::RESERVED>(0, data_size, 0);
        device_manager.add_mem<TaskState::LAUNCHED>(0, data_size, 0);
        lru_manager.read(0, i, data_size);
      }
    }
    for (devid_t i = 0; i < devices.size(); i++) {
      SPDLOG_DEBUG("DataManager: Device {} initialized with {}/{} memory", devices.get_name(i),
                   lru_manager.get_mem(i), devices.get_max_resources(i).mem);
    }
    valid_location_buffer.reserve(devices.size());
  }

  void initialize_data_replicate(const Data &data, const Devices &devices,
                                 DeviceManager &device_manager, dataid_t data_id,
                                 devid_t device_id) {
    const auto data_size = data.get_size(data_id);
    const bool device_id_in_bounds = device_id >= 0 && device_id < devices.size();

    if (device_id > -1 && !device_id_in_bounds) {
      SPDLOG_CRITICAL("DataManager::initialize_data_replicate(): invalid device_id {} "
                      "(valid device ids: 0..{})",
                      device_id, devices.size() - 1);
      T4F_INVARIANT(false && "DataManager::initialize_data_replicate(): device_id out of bounds");
    }

    if (device_id_in_bounds &&
        (lru_manager.get_mem(device_id) + data_size) <= devices.get_max_resources(device_id).mem &&
        !mapped_locations.is_valid(data_id, device_id)) {
      mapped_locations.set_valid(data_id, device_id, 0);
      reserved_locations.set_valid(data_id, device_id, 0);
      launched_locations.set_valid(data_id, device_id, 0);
      device_manager.add_mem<TaskState::MAPPED>(device_id, data_size, 0);
      device_manager.add_mem<TaskState::RESERVED>(device_id, data_size, 0);
      device_manager.add_mem<TaskState::LAUNCHED>(device_id, data_size, 0);
      lru_manager.read(device_id, data_id, data_size);
    }
  }

  [[nodiscard]] const LRU_manager &get_lru_manager() const {
    return lru_manager;
  }

  [[nodiscard]] const MovementCounter &get_movement_counter() const {
    return movement_counter;
  }

  [[nodiscard]] const LocationManager &get_mapped_locations() const {
    return mapped_locations;
  }

  [[nodiscard]] const LocationManager &get_reserved_locations() const {
    return reserved_locations;
  }

  [[nodiscard]] const LocationManager &get_launched_locations() const {
    return launched_locations;
  }

  bool check_valid_mapped(std::span<const dataid_t> list, devid_t device_id) const {
    return check_valid(list, mapped_locations, device_id);
  }

  bool check_valid_mapped(dataid_t data_id, devid_t device_id) const {
    return check_valid(data_id, mapped_locations, device_id);
  }

  bool check_valid_reserved(std::span<const dataid_t> list, devid_t device_id) const {
    return check_valid(list, reserved_locations, device_id);
  }

  bool check_valid_reserved(dataid_t data_id, devid_t device_id) const {
    return check_valid(data_id, reserved_locations, device_id);
  }

  bool check_valid_launched(std::span<const dataid_t> list, devid_t device_id) const {
    return check_valid(list, launched_locations, device_id);
  }

  bool check_valid_launched(dataid_t data_id, devid_t device_id) const {
    return check_valid(data_id, launched_locations, device_id);
  }

  [[nodiscard]] bool is_moving(dataid_t data_id, devid_t device_id) const {
    return movement_manager.is_moving(data_id, device_id);
  }

  [[nodiscard]] mem_t total_size(const Data &data, std::span<const dataid_t> list) const {
    mem_t total_size = 0;
    for (auto data_id : list) {
      total_size += data.get_size(data_id);
    }
    return total_size;
  }

  [[nodiscard]] mem_t local_size(const Data &data, std::span<const dataid_t> list,
                                 const LocationManager &locations, devid_t device_id) const {
    mem_t local_size = 0;
    for (auto data_id : list) {
      if (locations.is_valid(data_id, device_id)) {
        local_size += data.get_size(data_id);
      }
    }
    return local_size;
  }

  mem_t local_size_mapped(const Data &data, std::span<const dataid_t> list,
                          devid_t device_id) const {
    return local_size(data, list, mapped_locations, device_id);
  }

  mem_t local_size_reserved(const Data &data, std::span<const dataid_t> list,
                            devid_t device_id) const {
    return local_size(data, list, reserved_locations, device_id);
  }

  mem_t local_size_launched(const Data &data, std::span<const dataid_t> list,
                            devid_t device_id) const {
    return local_size(data, list, launched_locations, device_id);
  }

  [[nodiscard]] mem_t non_local_size(const Data &data, std::span<const dataid_t> list,
                                     const LocationManager &locations, devid_t device_id) const {
    mem_t non_local_size = 0;
    for (auto data_id : list) {
      if (locations.is_invalid(data_id, device_id)) {
        non_local_size += data.get_size(data_id);
      }
    }
    return non_local_size;
  }

  mem_t non_local_size_mapped(const Data &data, std::span<const dataid_t> list,
                              devid_t device_id) const {
    return non_local_size(data, list, mapped_locations, device_id);
  }

  mem_t non_local_size_reserved(const Data &data, std::span<const dataid_t> list,
                                devid_t device_id) const {
    return non_local_size(data, list, reserved_locations, device_id);
  }

  mem_t non_local_size_launched(const Data &data, std::span<const dataid_t> list,
                                devid_t device_id) const {
    return non_local_size(data, list, launched_locations, device_id);
  }

  mem_t shared_size(const Data &data, std::span<const dataid_t> list1,
                    std::span<const dataid_t> list2) const {
    mem_t shared_size = 0;
    for (auto data_id : list1) {
      if (std::find(list2.begin(), list2.end(), data_id) != list2.end()) {
        shared_size += data.get_size(data_id);
      }
    }
    return shared_size;
  }

  void read_update_mapped(const Data &data, DeviceManager &device_manager,
                          std::span<const dataid_t> list, devid_t device_id,
                          timecount_t current_time) {
    for (auto data_id : list) {
      read_update(data_id, device_id, mapped_locations, current_time);
      T4F_INVARIANT(mapped_locations.is_valid(data_id, device_id));
    }
    // Memory change is handled by task request in mapper
  }

  void write_update_mapped(const Data &data, DeviceManager &device_manager,
                           std::span<const dataid_t> list, devid_t device_id,
                           timecount_t current_time) {
    for (auto data_id : list) {
      write_update(data_id, device_id, mapped_locations, current_time);
      T4F_INVARIANT(mapped_locations.is_valid(data_id, device_id));
    }
    // Memory change is handled by task complete
  }

  void read_update_reserved(const Data &data, DeviceManager &device_manager,
                            std::span<const dataid_t> list, devid_t device_id,
                            timecount_t current_time) {
    for (auto data_id : list) {
      read_update(data_id, device_id, reserved_locations, current_time);
      T4F_INVARIANT(reserved_locations.is_valid(data_id, device_id));
    }
    // Memory change is handeled by task request in reserver
  }

  void write_update_reserved(const Data &data, DeviceManager &device_manager,
                             std::span<const dataid_t> list, devid_t device_id,
                             timecount_t current_time) {
    for (auto data_id : list) {
      write_update(data_id, device_id, reserved_locations, current_time);
      T4F_INVARIANT(reserved_locations.is_valid(data_id, device_id));
    }
    // Memory change is handled by task complete
  }

  void add_memory(DeviceManager &device_manager, devid_t device_id, dataid_t data_id, mem_t size,
                  timecount_t current_time) {
    SPDLOG_DEBUG("Adding data block {} to device {} with size {}", data_id, device_id, size);
    device_manager.add_mem<TaskState::LAUNCHED>(device_id, size, current_time);
  }

  void read_update_launched(const Data &data, DeviceManager &device_manager,
                            std::span<const dataid_t> list, devid_t device_id,
                            timecount_t current_time) {
    for (auto data_id : list) {
      const auto size = data.get_size(data_id);
      lru_manager.read(device_id, data_id, size);
      bool changed = read_update(data_id, device_id, launched_locations, current_time);
      if (changed) {
        add_memory(device_manager, device_id, data_id, size, current_time);
      }
      T4F_INVARIANT(launched_locations.is_valid(data_id, device_id));
    }
  }

  void write_update_launched(const Data &data, DeviceManager &device_manager,
                             std::span<const dataid_t> list, devid_t device_id,
                             timecount_t current_time) {
    for (auto data_id : list) {
      auto changed_flags = write_update(data_id, device_id, launched_locations, current_time);
      const auto size = data.get_size(data_id);
      remove_memory(device_manager, changed_flags, data_id, size, current_time);
      T4F_INVARIANT(launched_locations.is_valid(data_id, device_id));
    }
  }

  void evict_on_update_launched(const Data &data, DeviceManager &device_manager, dataid_t data_id,
                                devid_t device_id, timecount_t current_time, bool future_usage,
                                bool write_after_read) {
    T4F_INVARIANT(launched_locations.is_valid(data_id, device_id));
    auto updated_devices_launched =
        evict_on_update(data_id, device_id, launched_locations, current_time);
    evict_on_update(data_id, device_id, reserved_locations, current_time);

    auto size = data.get_size(data_id);
    auto mask = static_cast<std::make_unsigned_t<devicemask_t>>(updated_devices_launched);
    while (mask) {
      const auto device = static_cast<devid_t>(std::countr_zero(mask));
      SPDLOG_DEBUG("Evicting data block {} from device {} with size {}", data_id, device, size);
      device_manager.remove_mem<TaskState::RESERVED>(device, size, current_time);
      device_manager.remove_mem<TaskState::LAUNCHED>(device, size, current_time);
      lru_manager.invalidate(device, data_id, true);
      mask &= (mask - 1);
    }
    if (!future_usage) {
      // If there are no further usage for the data block (in mapped but not reserved tasks).
      // Invalidate for future mapping decisions.
      device_manager.remove_mem<TaskState::MAPPED>(device_id, size, current_time);
      mapped_locations.set_invalid(data_id, device_id, current_time);
    }
    // else if (mapped_locations.is_invalid(data_id, device_id) || write_after_read) {
    else if (write_after_read) {
      // write_after_read is needed to handle a case where the next usage for the data block is
      // write from the other device. Since launched_location is invalidated by the eviction
      // this redundant mapped_memory will not be removed. (Which should be).
      // mapped_locations.is_invalid(data_id, device_id) is only checking a subset of above
      // cases since to be valid in launced_location and invalid in mapped_location there is
      // only one scenario.
      // -> the last operation to the data block in mapped_but_not_reserved_tasks is a write
      // from another device.
      //   GPU0   |   GPU1
      // ---------|----------
      // read B0  |
      // ------EVICTION------ <- Mapped and completed B0 valid in launched_location GPU0
      // read B0  |
      // ~~~~~~~~~~~~~~~~~~~~~ < other ops
      //          |  Write B0
      // ---------|---------- <- Mapped but not reserved: B0 invalid in mapped_location GPU0
      //
      // However below case is not handeled
      //
      //   GPU0   |   GPU1
      // ---------|----------
      // read B0  |
      // ------EVICTION------ <- Mapped and completed B0 valid in launched_location GPU0
      //          |  Write B0
      // ~~~~~~~~~~~~~~~~~~~~~ < other ops
      // read B0  |
      // ---------|---------- <- Mapped but not reserved: B0 valid in mapped_location GPU0
      //
      // Write B0 from the GPU should have removed the mapped memory from GPU0 since
      // launched_location is valid (without eviction).
      // After eviction launched_location has changed and the removal doesn't happen.
      device_manager.remove_mem<TaskState::MAPPED>(device_id, size, current_time);
    }
    T4F_INVARIANT(launched_locations.is_invalid(data_id, device_id));
    T4F_INVARIANT(reserved_locations.is_invalid(data_id, device_id));
    if (!future_usage) {
      T4F_INVARIANT(mapped_locations.is_invalid(data_id, device_id));
    }
  }

  devicemask_t get_mapped_location_flags(dataid_t data_id) const {
    return mapped_locations.get_location_flags(data_id);
  }

  devicemask_t get_reserved_location_flags(dataid_t data_id) const {
    return reserved_locations.get_location_flags(data_id);
  }

  devicemask_t get_launched_location_flags(dataid_t data_id) const {
    return launched_locations.get_location_flags(data_id);
  }

  SourceRequest request_source(const Topology &topology, CommunicationManager &comm_manager,
                               dataid_t data_id, devid_t destination) {
    auto location_flags = launched_locations.get_location_flags(data_id);

    SPDLOG_DEBUG("Requesting source for data block {} to device {}", data_id, destination);
    // SPDLOG_DEBUG("Number of valid locations: {}", valid_locations.size());

    SourceRequest req =
        comm_manager.get_best_available_source(topology, destination, location_flags);
    if (req.found) {
      T4F_INVARIANT(launched_locations.is_valid(data_id, req.source) &&
                    "Selected source must have a launched-valid copy");
    }

    return req;
  }

  MovementStatus start_move(const Topology &topology, CommunicationManager &comm_manager,
                            DeviceManager &device_manager, const Data &data, dataid_t data_id,
                            devid_t source, devid_t destination, timecount_t current_time) {
    T4F_INVARIANT(launched_locations.is_valid(data_id, source));
    T4F_INVARIANT(source >= 0 && source < device_manager.n_devices);
    T4F_INVARIANT(destination >= 0 && destination < device_manager.n_devices);

    timecount_t completion_time = 0;
    if (movement_manager.try_get_time(data_id, destination, completion_time)) {
      T4F_INVARIANT(completion_time >= current_time &&
                    "Outstanding move completion time cannot be in the past");
      timecount_t time_left = completion_time - current_time;
      SPDLOG_DEBUG("Data block {} already moving to device {} expected to end after {}", data_id,
                   destination, time_left);
      return {.is_virtual = true, .duration = time_left};
    }

    if (launched_locations.is_valid(data_id, destination)) {
      SPDLOG_DEBUG("Data block {} already at device {}", data_id, destination);
      return {.is_virtual = true, .duration = 0};
    }
    T4F_INVARIANT(!movement_manager.is_moving(data_id, destination));

    SPDLOG_DEBUG("Starting move of data block {} from device {} to device {}", data_id, source,
                 destination);

    const auto size = data.get_size(data_id);
    T4F_INVARIANT(device_manager.overflow_mem<TaskState::LAUNCHED>(destination, size) == 0 &&
                  "start_move would exceed LAUNCHED memory on destination");

    lru_manager.read(destination, data_id, size);
    add_memory(device_manager, destination, data_id, size, current_time);

    timecount_t duration = comm_manager.ideal_time_to_transfer(topology, size, source, destination);

    if (duration == 0) {
      T4F_INVARIANT(source != destination);
      SPDLOG_DEBUG("Block moving instantly from {} to {}. Check bandwidth settings.", source,
                   destination);
    }

    movement_manager.set_completion(data_id, destination, current_time + duration);
    T4F_INVARIANT(movement_manager.is_moving(data_id, destination));

    comm_manager.reserve_connection(source, destination);

    movement_counter.add_total_movement(source, destination, size);

    return {.is_virtual = false, .duration = duration};
  }

  void complete_move(CommunicationManager &comm_manager, dataid_t data_id, devid_t source,
                     devid_t destination, bool is_virtual, timecount_t current_time) {

    if (is_virtual) {
      SPDLOG_DEBUG("Completing virtual move of data block {} from device {} to "
                   "device {}",
                   data_id, source, destination);

      if (movement_manager.is_moving(data_id, destination)) {
        SPDLOG_DEBUG("Virtual move of data block {} from device {} to device {} "
                     "beat the real move",
                     data_id, source, destination);
        // Update will happen in the real move
        // Not valid until the real move is completed
      } else {
        // NOTE(wlr): I'm not 100% sure about the source check
        // Could something that starts at the same time as the move completes be
        // a problem?
        T4F_INVARIANT(launched_locations.is_valid(data_id, source));
        T4F_INVARIANT(launched_locations.is_valid(data_id, destination));
      }
      return;
    }

    SPDLOG_DEBUG("Completing real move of data block {} from device {} to device {}", data_id,
                 source, destination);

    T4F_INVARIANT(movement_manager.is_moving(data_id, destination));
    T4F_INVARIANT(source != destination);
    launched_locations.set_valid(data_id, destination, current_time);
    movement_manager.remove(data_id, destination);
    T4F_INVARIANT(!movement_manager.is_moving(data_id, destination));
    T4F_INVARIANT(launched_locations.is_valid(data_id, destination));

    comm_manager.release_connection(source, destination);
  }

  void complete_eviction_move(CommunicationManager &comm_manager, dataid_t data_id, devid_t source,
                              devid_t destination, bool is_virtual, timecount_t current_time) {

    if (is_virtual) {
      SPDLOG_DEBUG("Completing virtual move of data block {} from device {} to "
                   "device {}",
                   data_id, source, destination);

      if (movement_manager.is_moving(data_id, destination)) {
        SPDLOG_DEBUG("Virtual move of data block {} from device {} to device {} "
                     "beat the real move",
                     data_id, source, destination);
        // Update will happen in the real move
        // Not valid until the real move is completed
      } else {
        // NOTE(wlr): I'm not 100% sure about the source check
        // Could something that starts at the same time as the move completes be
        // a problem?
        T4F_INVARIANT(launched_locations.is_valid(data_id, source));
        T4F_INVARIANT(launched_locations.is_valid(data_id, destination));
      }
      return;
    }

    SPDLOG_DEBUG("Completing eviction move of data block {} from device {} to device {}", data_id,
                 source, destination);

    T4F_INVARIANT(movement_manager.is_moving(data_id, destination));
    T4F_INVARIANT(source != destination);
    launched_locations.set_valid(data_id, destination, current_time);
    reserved_locations.set_valid(data_id, destination, current_time);
    mapped_locations.set_valid(data_id, destination, current_time);
    movement_manager.remove(data_id, destination);
    T4F_INVARIANT(!movement_manager.is_moving(data_id, destination));
    T4F_INVARIANT(launched_locations.is_valid(data_id, destination));
    T4F_INVARIANT(reserved_locations.is_valid(data_id, destination));
    T4F_INVARIANT(mapped_locations.is_valid(data_id, destination));

    comm_manager.release_connection(source, destination);
  }

  void remove_memory(DeviceManager &device_manager, const devicemask_t changed_flags,
                     dataid_t data_id, mem_t size, timecount_t current_time) {
    auto mask = static_cast<std::make_unsigned_t<devicemask_t>>(changed_flags);
    while (mask) {
      const auto device = static_cast<devid_t>(std::countr_zero(mask));
      SPDLOG_DEBUG("Removing data block {} from device {} with size {}", data_id, device, size);
      device_manager.remove_mem<TaskState::MAPPED>(device, size, current_time);
      device_manager.remove_mem<TaskState::RESERVED>(device, size, current_time);
      device_manager.remove_mem<TaskState::LAUNCHED>(device, size, current_time);
      lru_manager.invalidate(device, data_id);
      mask &= (mask - 1);
    }
  }

  void retire_data(const Data &data, DeviceManager &device_manager, dataid_t data_id,
                   devid_t device_id, timecount_t current_time) {
    auto size = data.get_size(data_id);
    SPDLOG_DEBUG("Retiring data block {} from device {} with size {}", data_id, device_id, size);

    auto mapped_flags = mapped_locations.invalidate_all(data_id, current_time);
    auto reserved_flags = reserved_locations.invalidate_all(data_id, current_time);
    auto launched_flags = launched_locations.invalidate_all(data_id, current_time);
    auto mask = static_cast<std::make_unsigned_t<devicemask_t>>(
        mapped_flags | reserved_flags | launched_flags);
    while (mask) {
      const auto device = static_cast<devid_t>(std::countr_zero(mask));
      const devicemask_t device_mask = (1 << device);
      if (mapped_flags & device_mask) {
        device_manager.remove_mem<TaskState::MAPPED>(device, size, current_time);
      }
      if (reserved_flags & device_mask) {
        device_manager.remove_mem<TaskState::RESERVED>(device, size, current_time);
      }
      if (launched_flags & device_mask) {
        device_manager.remove_mem<TaskState::LAUNCHED>(device, size, current_time);
        lru_manager.invalidate(device, data_id);
        T4F_INVARIANT(launched_locations.is_invalid(data_id, device));
      }
      mask &= (mask - 1);
    }
  }

  void finalize(timecount_t current_time) {
    mapped_locations.finalize(current_time);
    reserved_locations.finalize(current_time);
    launched_locations.finalize(current_time);
  }

  friend class SchedulerState;
};
