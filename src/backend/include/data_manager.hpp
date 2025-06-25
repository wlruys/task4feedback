#pragma once

#include "communication_manager.hpp"
#include "device_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include "tasks.hpp"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <tracy/Tracy.hpp>
#include <unordered_map>

enum class DataState : int8_t {
  NONE = 0,
  PLANNED = 1,
  MOVING = 2,
  VALID = 3,
};
constexpr std::size_t num_data_states = 4;

class Data {
protected:
  std::vector<mem_t> sizes;
  std::vector<devid_t> initial_location;
  std::vector<std::string> data_names;
  std::vector<int> data_types;
  std::vector<int> data_tags;
  std::vector<float> x_pos;
  std::vector<float> y_pos;

  std::unordered_map<std::string, dataid_t> name_to_id;

public:
  Data() = default;
  Data(std::size_t num_data)
      : sizes(num_data), initial_location(num_data), data_names(num_data), x_pos(num_data, 0.0f),
        y_pos(num_data, 0.0f) {
  }

  [[nodiscard]] bool empty() const {
    return size() == 0;
  }

  void set_size(dataid_t id, mem_t size) {
    assert(id < sizes.size());
    sizes.at(id) = size;
  }

  void set_tag(dataid_t id, int tag) {
    assert(id < data_tags.size());
    data_tags.at(id) = tag;
  }

  void set_x_pos(dataid_t id, float x) {
    assert(id < x_pos.size());
    x_pos.at(id) = x;
  }

  void set_y_pos(dataid_t id, float y) {
    assert(id < y_pos.size());
    y_pos.at(id) = y;
  }

  [[nodiscard]] float get_x_pos(dataid_t id) const {
    return x_pos.at(id);
  }

  [[nodiscard]] float get_y_pos(dataid_t id) const {
    return y_pos.at(id);
  }

  [[nodiscard]] const std::vector<float> &get_x_pos_vec() const {
    return x_pos;
  }

  [[nodiscard]] const std::vector<float> &get_y_pos_vec() const {
    return y_pos;
  }

  int get_tag(dataid_t id) const {
    return data_tags.at(id);
  }

  void set_type(dataid_t id, int type) {
    assert(id < data_types.size());
    data_types.at(id) = type;
  }

  int get_type(dataid_t id) const {
    return data_types.at(id);
  }

  void set_location(dataid_t id, devid_t location) {
    assert(id < initial_location.size());

    initial_location.at(id) = location;
  }
  void set_name(dataid_t id, std::string name) {
    assert(id < data_names.size());
    data_names.at(id) = std::move(name);
    name_to_id[data_names.at(id)] = id;
  }

  dataid_t get_id(const std::string &name) {
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
      x_pos.resize(id + 1, 0.0f);
      y_pos.resize(id + 1, 0.0f);
    }

    assert(id < sizes.size());
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

  [[nodiscard]] mem_t get_total_size(const std::vector<dataid_t> &ids) const {
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
    return initial_location.at(id);
  }
  [[nodiscard]] const std::string &get_name(dataid_t id) const {
    return data_names.at(id);
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

  StatsBundle<mem_t> get_block_statistics(const std::vector<DeviceType> &device_types) const {
    std::vector<mem_t> block_sizes;
    block_sizes.reserve(sizes.size());

    for (const auto &device_type : device_types) {
      for (const auto &size : sizes) {
        block_sizes.push_back(size);
      }
    }

    return StatsBundle(block_sizes);
  }

  friend class DataManager;
};

struct ValidInterval {
  timecount_t start = 0;
  timecount_t stop = 0;
};

struct ValidEventArray {
  std::vector<timecount_t> starts;
  std::vector<timecount_t> stops;
  std::size_t size = 0;
};

// class BlockLocation {
// protected:
//   dataid_t data_id;
//   std::vector<int8_t> locations;
//   std::vector<std::vector<ValidInterval>> valid_intervals;
//   std::vector<timecount_t> current_start;

// public:
//   BlockLocation(dataid_t data_id, std::size_t n_devices)
//       : data_id(data_id), locations(n_devices), valid_intervals(n_devices),
//         current_start(n_devices) {

// #ifdef SIM_TRACK_LOCATION
//     for (auto &interval : valid_intervals) {
//       interval.reserve(50);
//     }
// #endif
//   }

//   bool check_valid_at_time(devid_t device_id, timecount_t query_time) const {
//     const auto &intervals = valid_intervals[device_id];

// #ifndef SIM_TRACK_LOCATION
//     return false;
// #endif

//     if (!intervals.empty()) {
//       // Use binary search to find the first interval whose start is greater than query_time.
//       auto it = std::upper_bound(
//           intervals.begin(), intervals.end(), query_time,
//           [](const timecount_t &t, const auto &interval) { return t < interval.start; });

//       // Check the interval immediately before 'it'
//       if (it != intervals.begin()) {
//         auto candidate = std::prev(it);
//         if (candidate->start <= query_time && query_time < candidate->stop) {
//           return true;
//         }
//       }
//     }

//     // Otherwise, check if the device is currently valid and has an open interval.
//     if (locations[device_id] && query_time >= current_start[device_id]) {
//       return true;
//     }

//     return false;
//   }

//   void set_valid(devid_t device_id, timecount_t current_time) {
// #ifdef SIM_TRACK_LOCATION
//     if (!locations[device_id]) {
//       locations[device_id] = true;
//       current_start[device_id] = current_time;
//     }
// #else
//     locations[device_id] = true;
// #endif
//   }

//   void set_invalid(devid_t device_id, timecount_t current_time) {
// #ifdef SIM_TRACK_LOCATION
//     if (locations[device_id]) {
//       locations[device_id] = false;
//       if (current_start[device_id] != current_time) {
//         valid_intervals[device_id].emplace_back(current_start[device_id], current_time);
//       }
//     }
// #else
//     locations[device_id] = false;
// #endif
//   }

//   [[nodiscard]] bool is_valid(devid_t device_id) const {
//     return locations.at(device_id);
//   }

//   [[nodiscard]] bool is_invalid(devid_t device_id) const {
//     return !is_valid(device_id);
//   }

//   // Return the number of valid locations
//   [[nodiscard]] std::size_t count_valid() const {
//     return static_cast<std::size_t>(std::count(locations.begin(), locations.end(), true));
//   }

//   // Return indexes of valid locations
//   [[nodiscard]] std::vector<devid_t> get_valid_locations() const {
//     std::vector<devid_t> valid_locations;
//     for (devid_t i = 0; i < locations.size(); i++) {
//       if (is_valid(i)) {
//         valid_locations.push_back(i);
//       }
//     }
//     return valid_locations;
//   }

//   void populate_valid_locations(std::vector<devid_t> &valid_locations) const {
//     for (devid_t i = 0; i < locations.size(); i++) {
//       if (is_valid(i)) {
//         valid_locations.push_back(i);
//       }
//     }
//   }

//   bool validate(devid_t device_id, timecount_t current_time) {
//     if (is_valid(device_id)) {
//       return false;
//     }
//     set_valid(device_id, current_time);
//     return true;
//   }

//   std::vector<devid_t> invalidate_except(devid_t device_id, timecount_t current_time) {
//     std::vector<devid_t> updated;
//     for (devid_t i = 0; i < locations.size(); i++) {
//       if (i != device_id && is_valid(i)) {
//         set_invalid(i, current_time);
//         updated.push_back(i);
//       }
//     }
//     return updated;
//   }

//   std::vector<devid_t> invalidate_all(timecount_t current_time) {
//     std::vector<devid_t> updated;
//     for (devid_t i = 0; i < locations.size(); i++) {
//       if (is_valid(i)) {
//         set_invalid(i, current_time);
//         updated.push_back(i);
//       }
//     }
//     return updated;
//   }

//   std::vector<devid_t> invalidate_on(devid_t device_id, timecount_t current_time) {
//     std::vector<devid_t> updated;
//     if (is_valid(device_id)) {
//       set_invalid(device_id, current_time);
//       updated.push_back(device_id);
//     }
//     return updated;
//   }

//   void finalize(timecount_t current_time) {
//     // tie off any open/hanging interval at the end of the simulation
//     for (devid_t i = 0; i < locations.size(); i++) {
//       if (is_valid(i)) {
//         valid_intervals[i].emplace_back(current_start[i], current_time);
//         current_start[i] = current_time;
//       }
//     }
//   }

//   ValidEventArray get_valid_intervals(devid_t device_id) const {
//     assert(device_id < valid_intervals.size());

//     const auto &intervals = valid_intervals[device_id];
//     ValidEventArray valid_events;

//     bool has_open_interval = is_valid(device_id);

//     valid_events.size = intervals.size() + has_open_interval;

//     if (valid_events.size == 0) {
//       // Return a single interval from 0 to 0 to indicate no valid intervals
//       valid_events.starts.resize(1);
//       valid_events.stops.resize(1);
//       valid_events.size = 1;
//       valid_events.starts[0] = 0;
//       valid_events.stops[0] = 0;
//       return valid_events;
//     }

//     valid_events.starts.resize(valid_events.size);
//     valid_events.stops.resize(valid_events.size);
//     for (std::size_t i = 0; i < intervals.size(); i++) {
//       valid_events.starts[i] = intervals[i].start;
//       valid_events.stops[i] = intervals[i].stop;
//     }
//     if (has_open_interval) {
//       valid_events.starts[intervals.size()] = current_start[device_id];
//       valid_events.stops[intervals.size()] = MAX_TIME;
//     }
//     return valid_events;
//   }

//   friend std::ostream &operator<<(std::ostream &os, const BlockLocation &bl) {
//     os << "[";
//     for (devid_t i = 0; i < bl.locations.size(); i++) {
//       os << (bl.is_valid(i) ? "1" : "0");
//     }
//     os << "]";
//     return os;
//   }
// };

class LocationManager {
protected:
  const devid_t num_devices;
  const dataid_t num_data;
  std::vector<int8_t> locations;
  std::vector<devid_t> device_id_buffer;
  std::vector<devid_t> device_id_buffer2;
  std::vector<ValidEventArray> valid_intervals;

public:
  LocationManager(dataid_t num_data, devid_t num_devices)
      : locations(num_data * num_devices, 0), num_devices(num_devices), num_data(num_data) {

    device_id_buffer.reserve(num_devices);
    device_id_buffer2.reserve(num_devices);

#ifdef SIM_TRACK_LOCATION
    constexpr size_t buffer_initial_size = 100;
    valid_intervals.resize(num_data * num_devices);
    for (auto &intervals : valid_intervals) {
      intervals.starts.reserve(buffer_initial_size);
      intervals.stops.reserve(buffer_initial_size);
    }
#endif
  }

  [[nodiscard]] inline bool is_valid(dataid_t data_id, devid_t device_id) const {
    assert(data_id < num_data && device_id < num_devices);
    return locations[data_id * num_devices + device_id] == 1;
  }

  [[nodiscard]] inline bool is_invalid(dataid_t data_id, devid_t device_id) const {
    return !is_valid(data_id, device_id);
  }

  inline void set_valid(dataid_t data_id, devid_t device_id, timecount_t current_time) {
#ifdef SIM_TRACK_LOCATION
    if (is_invalid(data_id, device_id)) {
      // start new interval
      valid_intervals[data_id * num_devices + device_id].starts.push_back(current_time);
      valid_intervals[data_id * num_devices + device_id].stops.push_back(MAX_TIME);
    }
#else
    locations[data_id * num_devices + device_id] = 1;
#endif
  }

  inline void set_invalid(dataid_t data_id, devid_t device_id, timecount_t current_time) {
#ifdef SIM_TRACK_LOCATION
    if (is_valid(data_id, device_id)) {
      locations[data_id * num_devices + device_id] = 0;
      // close the current interval
      valid_intervals[data_id * num_devices + device_id].stops.back() = current_time;
    }
#else
    locations[data_id * num_devices + device_id] = 0;
#endif
  }

  [[nodiscard]] inline std::size_t count_valid(dataid_t data_id) const {
    std::size_t count = 0;
    for (devid_t i = 0; i < num_devices; i++) {
      if (is_valid(data_id, i)) {
        count++;
      }
    }
    return count;
  }

  [[nodiscard]] inline std::span<const int8_t> get_location_flags(dataid_t data_id) const {
    return std::span<const int8_t>(locations.data() + data_id * num_devices, num_devices);
  }

  [[nodiscard]] inline std::vector<devid_t> &get_valid_locations(dataid_t data_id) {
    device_id_buffer2.clear();
    device_id_buffer2.reserve(num_devices);

    for (devid_t i = 0; i < num_devices; i++) {
      if (is_valid(data_id, i)) {
        device_id_buffer2.push_back(i);
      }
    }
    return device_id_buffer2;
  }

  void populate_valid_locations(dataid_t data_id, std::vector<devid_t> &valid_locations) const {
    for (devid_t i = 0; i < num_devices; i++) {
      if (is_valid(data_id, i)) {
        valid_locations.push_back(i);
      }
    }
  }

  bool inline validate(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    if (is_valid(data_id, device_id)) {
      return false;
    }
    set_valid(data_id, device_id, current_time);
    return true;
  }

  std::vector<devid_t> &invalidate_except(dataid_t data_id, devid_t device_id,
                                          timecount_t current_time) {
    device_id_buffer.clear();
    device_id_buffer.reserve(num_devices);

    for (devid_t i = 0; i < num_devices; i++) {
      if (i != device_id && is_valid(data_id, i)) {
        set_invalid(data_id, i, current_time);
        device_id_buffer.push_back(i);
      }
    }
    return device_id_buffer;
  }

  std::vector<devid_t> &invalidate_all(dataid_t data_id, timecount_t current_time) {
    device_id_buffer.clear();
    device_id_buffer.reserve(num_devices);

    for (devid_t i = 0; i < num_devices; i++) {
      if (is_valid(data_id, i)) {
        set_invalid(data_id, i, current_time);
        device_id_buffer.push_back(i);
      }
    }
    return device_id_buffer;
  }

  std::vector<devid_t> &invalidate_on(dataid_t data_id, devid_t device_id,
                                      timecount_t current_time) {
    device_id_buffer.clear();
    if (is_valid(data_id, device_id)) {
      set_invalid(data_id, device_id, current_time);
      device_id_buffer.push_back(device_id);
    }
    return device_id_buffer;
  }

  void finalize(timecount_t current_time) {
    // tie off any open/hanging interval at the end of the simulation
    for (dataid_t i = 0; i < num_data; i++) {
      for (devid_t j = 0; j < num_devices; j++) {
        if (is_valid(i, j)) {
          valid_intervals[i * num_devices + j].stops.back() = current_time;
        }
      }
    }
  }

  ValidEventArray &get_valid_intervals(dataid_t data_id, devid_t device_id) {
    return valid_intervals[data_id * num_devices + device_id];
  }
};

struct MovementPair {
  dataid_t data_id{};
  devid_t destination{};

  MovementPair(dataid_t data_id, devid_t destination) : data_id(data_id), destination(destination) {
  }

  auto operator==(const MovementPair &other) const -> bool {
    return data_id == other.data_id && destination == other.destination;
  }
};

struct mp_hash {
  using is_avalanching = void;

  [[nodiscard]] auto operator()(MovementPair const &f) const noexcept -> uint64_t {
    static_assert(std::has_unique_object_representations_v<MovementPair>);
    return ankerl::unordered_dense::detail::wyhash::hash(&f, sizeof(f));
  }
};

class MovementManager {
protected:
  ankerl::unordered_dense::map<MovementPair, timecount_t, mp_hash> movement_times;

public:
  MovementManager() = default;

  bool is_moving(dataid_t data_id, devid_t destination) const {
    return movement_times.find({data_id, destination}) != movement_times.end();
  }

  [[nodiscard]] inline timecount_t get_time(dataid_t data_id, devid_t destination) const {
    auto it = movement_times.find({data_id, destination});
    return it == movement_times.end() ? 0 : it->second;
  }

  inline void set_completion(dataid_t data_id, devid_t destination,
                             timecount_t global_completion_time) {
    movement_times[{data_id, destination}] = global_completion_time;
  }

  inline void remove(dataid_t data_id, devid_t destination) {
    movement_times.erase({data_id, destination});
  }
};

class DeviceDataCounts {
protected:
  using DataCount = ankerl::unordered_dense::map<dataid_t, std::size_t>;
  DataCount mapped_counts;
  DataCount reserved_counts;
  DataCount launched_counts;
  DataCount moving_counts;

  // Increment the count of a data block, return true if the block is new
  static bool increment(DataCount &counts, dataid_t data_id) {
    auto it = counts.find(data_id);
    if (it == counts.end()) {
      counts.at(data_id) = 1;
      return true;
    }
    it->second++;
    return false;
  }

  // Decrement the count of a data block, removing it if the count reaches 0 and
  // return true if the block is removed
  static bool decrement(DataCount &counts, dataid_t data_id) {
    auto it = counts.find(data_id);
    if (it == counts.end()) {
      return false;
    }
    it->second--;
    if (it->second == 0) {
      counts.erase(it);
      return true;
    }
    return false;
  }

public:
  DeviceDataCounts() = default;

  DeviceDataCounts(const DeviceDataCounts &other) {
    mapped_counts = other.mapped_counts;
    reserved_counts = other.reserved_counts;
    launched_counts = other.launched_counts;
    moving_counts = other.moving_counts;
  }

  bool increment_mapped(dataid_t data_id) {
    return increment(mapped_counts, data_id);
  }

  bool increment_reserved(dataid_t data_id) {
    return increment(reserved_counts, data_id);
  }

  bool increment_launched(dataid_t data_id) {
    return increment(launched_counts, data_id);
  }

  bool increment_moving(dataid_t data_id) {
    return increment(moving_counts, data_id);
  }

  bool decrement_mapped(dataid_t data_id) {
    return decrement(mapped_counts, data_id);
  }

  bool decrement_reserved(dataid_t data_id) {
    return decrement(reserved_counts, data_id);
  }

  bool decrement_launched(dataid_t data_id) {
    return decrement(launched_counts, data_id);
  }

  bool decrement_moving(dataid_t data_id) {
    return decrement(moving_counts, data_id);
  }

  [[nodiscard]] std::size_t count_mapped(dataid_t data_id) const {
    auto it = mapped_counts.find(data_id);
    return it == mapped_counts.end() ? 0 : it->second;
  }

  [[nodiscard]] std::size_t count_reserved(dataid_t data_id) const {
    auto it = reserved_counts.find(data_id);
    return it == reserved_counts.end() ? 0 : it->second;
  }

  [[nodiscard]] std::size_t count_launched(dataid_t data_id) const {
    auto it = launched_counts.find(data_id);
    return it == launched_counts.end() ? 0 : it->second;
  }

  [[nodiscard]] std::size_t count_moving(dataid_t data_id) const {
    auto it = moving_counts.find(data_id);
    return it == moving_counts.end() ? 0 : it->second;
  }

  void run_asserts(dataid_t data_id) const {
    assert(count_mapped(data_id) >= count_reserved(data_id));
    assert(count_reserved(data_id) >= count_launched(data_id));
    assert(count_reserved(data_id) >= count_moving(data_id));
  }

  friend class DataCounts;
};

class DataCounts {
protected:
  std::vector<DeviceDataCounts> device_counts;

public:
  DataCounts(std::size_t n_devices) : device_counts(n_devices) {
  }

  bool increment_mapped(dataid_t data_id, devid_t device_id) {
    return device_counts.at(device_id).increment_mapped(data_id);
  }

  bool increment_reserved(dataid_t data_id, devid_t device_id) {
    return device_counts.at(device_id).increment_reserved(data_id);
  }

  bool increment_launched(dataid_t data_id, devid_t device_id) {
    return device_counts.at(device_id).increment_launched(data_id);
  }

  bool increment_moving(dataid_t data_id, devid_t device_id) {
    return device_counts.at(device_id).increment_moving(data_id);
  }

  bool decrement_mapped(dataid_t data_id, devid_t device_id) {
    return device_counts.at(device_id).decrement_mapped(data_id);
  }

  bool decrement_reserved(dataid_t data_id, devid_t device_id) {
    return device_counts.at(device_id).decrement_reserved(data_id);
  }

  bool decrement_launched(dataid_t data_id, devid_t device_id) {
    return device_counts.at(device_id).decrement_launched(data_id);
  }

  bool decrement_moving(dataid_t data_id, devid_t device_id) {
    return device_counts.at(device_id).decrement_moving(data_id);
  }

  [[nodiscard]] std::size_t count_mapped(dataid_t data_id, devid_t device_id) const {
    return device_counts.at(device_id).count_mapped(data_id);
  }

  [[nodiscard]] std::size_t count_reserved(dataid_t data_id, devid_t device_id) const {
    return device_counts.at(device_id).count_reserved(data_id);
  }

  [[nodiscard]] std::size_t count_launched(dataid_t data_id, devid_t device_id) const {
    return device_counts.at(device_id).count_launched(data_id);
  }

  [[nodiscard]] std::size_t count_moving(dataid_t data_id, devid_t device_id) const {
    return device_counts.at(device_id).count_moving(data_id);
  }
};

struct MovementStatus {
  bool is_virtual = false;
  timecount_t duration = 0;
};

class LRU_manager {
public:
  // Constructor: initialize for n_devices [0 .. n_devices-1]
  explicit LRU_manager(DeviceManager &device_manager)
      : n_devices_(device_manager.size()), lru_lists_(device_manager.size()),
        position_maps_(device_manager.size()), size_maps_(device_manager.size()),
        sizes_(device_manager.size()), max_sizes_(device_manager.size()) {
    for (auto &size : sizes_) {
      size = 0;
    }
    for (int i = 0; i < device_manager.size(); i++) {
      max_sizes_[i] = device_manager.devices.get().get_max_resources(i).mem;
    }
    id_buffer.reserve(20);
  }

  // read: add (device_id, data_id, mem_size). If present, update MRU; else insert.
  void read(devid_t device_id, dataid_t data_id, mem_t mem_size) {
    assert(device_id >= 0 && device_id < n_devices_);

    auto &lst = lru_lists_[device_id];
    auto &pos = position_maps_[device_id];
    auto &smap = size_maps_[device_id];
    auto &size = sizes_[device_id];
    auto &max_size = max_sizes_[device_id];
    auto it = pos.find(data_id);
    if (it != pos.end()) {
      // already present: move to MRU
      lst.erase(it->second);
    } else {
      size += mem_size;
      if (size > max_size) {
        SPDLOG_DEBUG("LRU_manager::read(): Device {}: Adding data_id {} with size {}", device_id,
                     data_id, mem_size);
        assert(size <= max_size && "LRU_manager::read(): size exceeds max size");
      }
    }
    // insert at MRU (back)
    lst.push_back(data_id);
    auto new_it = std::prev(lst.end());
    pos[data_id] = new_it;
    smap[data_id] = mem_size; // update size
  }

  LRU_manager(const LRU_manager &other)
      : n_devices_(other.n_devices_), lru_lists_(other.lru_lists_),
        position_maps_(other.n_devices_), size_maps_(other.size_maps_), sizes_(other.sizes_),
        max_sizes_(other.max_sizes_), evicted_size(other.evicted_size) {
    // Rebuild position_maps_
    for (devid_t dev = 0; dev < n_devices_; ++dev) {
      for (auto it = lru_lists_[dev].begin(); it != lru_lists_[dev].end(); ++it) {
        position_maps_[dev][*it] = it;
      }
    }
    id_buffer.reserve(other.id_buffer.capacity());
  }
  // invalidate: remove (device_id, data_id); assert if missing
  void invalidate(devid_t device_id, dataid_t data_id, bool evict = false) {
    assert(device_id >= 0 && device_id < n_devices_);

    auto &lst = lru_lists_[device_id];
    auto &pos = position_maps_[device_id];
    auto &smap = size_maps_[device_id];
    auto &size = sizes_[device_id];

    auto it = pos.find(data_id);
    assert(it != pos.end() && "invalidate(): data_id not present");

    lst.erase(it->second);
    pos.erase(it);
    size -= smap[data_id]; // update size
    if (evict)
      evicted_size += smap[data_id];
    smap.erase(data_id);
  }

  // getLRUids: fill id_buffer[device_id] with the least-recently-used data_ids
  // until their cumulative mem_size ≥ requested mem_size, and return it.
  const DataIDList &getLRUids(devid_t device_id, std::size_t mem_size,
                              const DataIDList &used_ids) const {
    assert(device_id >= 0 && device_id < n_devices_);

    auto &lst = lru_lists_[device_id];
    auto &smap = size_maps_[device_id];
    id_buffer.clear();
    std::size_t accumulated = 0;

    for (auto it = lst.begin(); it != lst.end() && accumulated < mem_size; ++it) {
      dataid_t did = *it;
      if (std::find(used_ids.begin(), used_ids.end(), did) != used_ids.end()) {
        continue; // skip if used by the task
      }
      auto sz_it = smap.find(did);
      assert(sz_it != smap.end() && "size missing for data_id");
      accumulated += sz_it->second;
      id_buffer.push_back(did);
    }
    assert(accumulated <= mem_size && "getLRUids(): accumulated size exceeds requested size");
    return id_buffer;
  }

  mem_t get_mem(devid_t device_id) const {
    assert((device_id >= 0) && (device_id < n_devices_));
    return sizes_[device_id];
  }

  mem_t get_evicted_memory_size() const {
    return evicted_size;
  }

private:
  std::size_t n_devices_;

  // For each device:
  //  - a list maintaining LRU (front) → MRU (back)
  //  - a map from data_id → its position in that list
  //  - a map from data_id → its mem_size
  std::vector<std::list<dataid_t>> lru_lists_;
  std::vector<ankerl::unordered_dense::map<dataid_t, typename std::list<dataid_t>::iterator>>
      position_maps_;
  std::vector<ankerl::unordered_dense::map<dataid_t, mem_t>> size_maps_;
  std::vector<mem_t> sizes_;
  std::vector<mem_t> max_sizes_;
  mem_t evicted_size = 0;
  mutable DataIDList id_buffer;
};

class DataManager {
protected:
  std::reference_wrapper<Data> data;
  std::reference_wrapper<DeviceManager> device_manager;
  std::reference_wrapper<CommunicationManager> communication_manager;
  LocationManager mapped_locations;
  LocationManager reserved_locations;
  LocationManager launched_locations;
  MovementManager movement_manager;
  DataCounts counts;
  LRU_manager lru_manager;

  static bool check_valid(size_t data_id, const LocationManager &locations, devid_t device_id) {
    return locations.is_valid(data_id, device_id);
  }

  static bool check_valid(const DataIDList &list, const LocationManager &locations,
                          devid_t device_id) {
    return std::ranges::all_of(
        list, [&](auto data_id) { return !locations.is_invalid(data_id, device_id); });
  }

  static bool read_update(dataid_t data_id, devid_t device_id, LocationManager &locations,
                          timecount_t current_time) {
    return locations.validate(data_id, device_id, current_time);
  }

  static auto write_update(dataid_t data_id, devid_t device_id, LocationManager &locations,
                           timecount_t current_time) {
    auto updated_ids = locations.invalidate_except(data_id, device_id, current_time);
    return updated_ids;
  }

  static auto evict_on_update(dataid_t data_id, devid_t device_id, LocationManager &locations,
                              timecount_t current_time) {
    auto updated_ids = locations.invalidate_on(data_id, device_id, current_time);
    return updated_ids;
  }

public:
  std::vector<devid_t> valid_location_buffer;
  DataManager(Data &data_, DeviceManager &device_manager_,
              CommunicationManager &communication_manager_)
      : data(data_), device_manager(device_manager_), communication_manager(communication_manager_),
        mapped_locations(data.get().size(), device_manager_.size()),
        reserved_locations(data.get().size(), device_manager_.size()),
        launched_locations(data.get().size(), device_manager_.size()),
        counts(device_manager_.size()), lru_manager(device_manager_) {
  }

  DataManager(const DataManager &o_, DeviceManager &device_manager_,
              CommunicationManager &communication_manager_)
      : data(o_.data), device_manager(device_manager_),
        communication_manager(communication_manager_), mapped_locations(o_.mapped_locations),
        reserved_locations(o_.reserved_locations), launched_locations(o_.launched_locations),
        movement_manager(o_.movement_manager), counts(o_.counts), lru_manager(o_.lru_manager) {
  }

  DataManager(const DataManager &o_) = delete;
  DataManager &operator=(const DataManager &o_) = delete;

  void initialize() {
    ZoneScoped;
    for (dataid_t i = 0; i < data.get().size(); i++) {
      auto initial_location = data.get().get_location(i);
      if (initial_location > -1) {
        mapped_locations.set_valid(i, initial_location, 0);
        reserved_locations.set_valid(i, initial_location, 0);
        launched_locations.set_valid(i, initial_location, 0);
        const auto size = data.get().get_size(i);
        device_manager.get().add_mem<TaskState::MAPPED>(initial_location, size, 0);
        device_manager.get().add_mem<TaskState::RESERVED>(initial_location, size, 0);
        device_manager.get().add_mem<TaskState::LAUNCHED>(initial_location, size, 0);
        lru_manager.read(initial_location, i, size);
      }
    }
    valid_location_buffer.reserve(device_manager.get().size());
  }

  [[nodiscard]] const Data &get_data() const {
    return data;
  }

  [[nodiscard]] const LRU_manager &get_lru_manager() const {
    return lru_manager;
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

  [[nodiscard]] const DataCounts &get_counts() const {
    return counts;
  }

  bool check_valid_mapped(const DataIDList &list, devid_t device_id) const {
    return check_valid(list, mapped_locations, device_id);
  }

  bool check_valid_mapped(dataid_t data_id, devid_t device_id) const {
    return check_valid(data_id, mapped_locations, device_id);
  }

  bool check_valid_reserved(const DataIDList &list, devid_t device_id) const {
    return check_valid(list, reserved_locations, device_id);
  }

  bool check_valid_reserved(dataid_t data_id, devid_t device_id) const {
    return check_valid(data_id, reserved_locations, device_id);
  }

  bool check_valid_launched(const DataIDList &list, devid_t device_id) const {
    return check_valid(list, launched_locations, device_id);
  }

  bool check_valid_launched(dataid_t data_id, devid_t device_id) const {
    return check_valid(data_id, launched_locations, device_id);
  }

  [[nodiscard]] mem_t total_size(const DataIDList &list) const {
    mem_t total_size = 0;
    for (auto data_id : list) {
      total_size += data.get().get_size(data_id);
    }
    return total_size;
  }

  [[nodiscard]] mem_t local_size(const DataIDList &list, devid_t device_id) const {
    return local_size(list, mapped_locations, device_id);
  }

  [[nodiscard]] mem_t local_size(const DataIDList &list, const LocationManager &locations,
                                 devid_t device_id) const {
    mem_t local_size = 0;
    for (auto data_id : list) {
      if (locations.is_valid(data_id, device_id)) {
        local_size += data.get().get_size(data_id);
      }
    }
    return local_size;
  }

  mem_t local_size_mapped(const DataIDList &list, devid_t device_id) const {
    return local_size(list, mapped_locations, device_id);
  }

  mem_t local_size_reserved(const DataIDList &list, devid_t device_id) const {
    return local_size(list, reserved_locations, device_id);
  }

  mem_t local_size_launched(const DataIDList &list, devid_t device_id) const {
    return local_size(list, launched_locations, device_id);
  }

  [[nodiscard]] mem_t non_local_size(const DataIDList &list, const LocationManager &locations,
                                     devid_t device_id) const {
    mem_t non_local_size = 0;
    for (auto data_id : list) {
      if (locations.is_invalid(data_id, device_id)) {
        non_local_size += data.get().get_size(data_id);
      }
    }
    return non_local_size;
  }

  mem_t non_local_size_mapped(const DataIDList &list, devid_t device_id) const {
    return non_local_size(list, mapped_locations, device_id);
  }

  mem_t non_local_size_reserved(const DataIDList &list, devid_t device_id) const {
    return non_local_size(list, reserved_locations, device_id);
  }

  mem_t non_local_size_launched(const DataIDList &list, devid_t device_id) const {
    return non_local_size(list, launched_locations, device_id);
  }

  mem_t shared_size(const DataIDList &list1, const DataIDList &list2) const {
    mem_t shared_size = 0;
    for (auto data_id : list1) {
      if (std::find(list2.begin(), list2.end(), data_id) != list2.end()) {
        shared_size += data.get().get_size(data_id);
      }
    }
    return shared_size;
  }

  void read_update_mapped(const DataIDList &list, devid_t device_id, timecount_t current_time) {
    for (auto data_id : list) {
      read_update(data_id, device_id, mapped_locations, current_time);
    }
    // Memory change is handled by task request in mapper
  }

  void write_update_mapped(const DataIDList &list, devid_t device_id, timecount_t current_time) {
    for (auto data_id : list) {
      write_update(data_id, device_id, mapped_locations, current_time);
    }
    // Memory change is handled by task complete
  }

  void read_update_reserved(const DataIDList &list, devid_t device_id, timecount_t current_time) {
    for (auto data_id : list) {
      read_update(data_id, device_id, reserved_locations, current_time);
    }
    // Memory change is handeled by task request in reserver
  }

  void write_update_reserved(const DataIDList &list, devid_t device_id, timecount_t current_time) {
    for (auto data_id : list) {
      write_update(data_id, device_id, reserved_locations, current_time);
    }
    // Memory change is handled by task complete
  }

  void add_memory(dataid_t data_id, devid_t device_id, timecount_t current_time, mem_t size) {
    SPDLOG_DEBUG("Adding data block {} to device {} with size {}", data_id, device_id, size);
    device_manager.get().add_mem<TaskState::LAUNCHED>(device_id, size, current_time);
  }

  void read_update_launched(const DataIDList &list, devid_t device_id, timecount_t current_time) {
    for (auto data_id : list) {
      const auto size = data.get().get_size(data_id);
      lru_manager.read(device_id, data_id, size);
      bool changed = read_update(data_id, device_id, launched_locations, current_time);
      if (changed) {
        add_memory(data_id, device_id, current_time, size);
      }
    }
  }

  void write_update_launched(const DataIDList &list, devid_t device_id, timecount_t current_time) {
    for (auto data_id : list) {
      auto updated_devices = write_update(data_id, device_id, launched_locations, current_time);
      // auto updated_devices_mapped =
      //     write_update(data_id, device_id, mapped_locations, current_time);
      remove_memory(updated_devices, data_id, current_time);
      auto size = data.get().get_size(data_id);
      for (auto device : updated_devices) {
        // SPDLOG_DEBUG("Removing data block {} from device {} with size {}", data_id, device,
        // size); device_manager.get().remove_mem<TaskState::MAPPED>(device, size, current_time);
        // device_manager.get().remove_mem<TaskState::RESERVED>(device, size, current_time);
        // device_manager.get().remove_mem<TaskState::LAUNCHED>(device, size, current_time);
        lru_manager.invalidate(device, data_id);
      }
    }
  }

  void evict_on_update_launched(const DataIDList &list, devid_t device_id, timecount_t current_time,
                                bool future_usage, bool write_after_read) {
    for (auto data_id : list) {
      auto updated_devices_launched =
          evict_on_update(data_id, device_id, launched_locations, current_time);
      auto updated_devices_reserved =
          evict_on_update(data_id, device_id, reserved_locations, current_time);

      auto size = data.get().get_size(data_id);
      for (auto device : updated_devices_launched) {
        SPDLOG_DEBUG("Evicting data block {} from device {} with size {}", data_id, device, size);
        device_manager.get().remove_mem<TaskState::RESERVED>(device, size, current_time);
        device_manager.get().remove_mem<TaskState::LAUNCHED>(device, size, current_time);
        lru_manager.invalidate(device, data_id, true);
      }
      if (!future_usage) {
        // If there are no further usage for the data block (in mapped but not reserved tasks).
        // Invalidate for future mapping decisions.
        device_manager.get().remove_mem<TaskState::MAPPED>(device_id, size, current_time);
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
        device_manager.get().remove_mem<TaskState::MAPPED>(device_id, size, current_time);
      }
    }
  }

  auto &get_valid_mapped_locations(dataid_t data_id) {
    return mapped_locations.get_valid_locations(data_id);
  }

  auto &get_valid_reserved_locations(dataid_t data_id) {
    return reserved_locations.get_valid_locations(data_id);
  }

  auto &get_valid_launched_locations(dataid_t data_id) {
    return launched_locations.get_valid_locations(data_id);
  }

  std::span<const int8_t> get_mapped_location_flags(dataid_t data_id) const {
    return mapped_locations.get_location_flags(data_id);
  }

  std::span<const int8_t> get_reserved_location_flags(dataid_t data_id) const {
    return reserved_locations.get_location_flags(data_id);
  }

  std::span<const int8_t> get_launched_location_flags(dataid_t data_id) const {
    return launched_locations.get_location_flags(data_id);
  }

  SourceRequest request_source(dataid_t data_id, devid_t destination) {
    auto location_flags = launched_locations.get_location_flags(data_id);

    SPDLOG_DEBUG("Requesting source for data block {} to device {}", data_id, destination);
    // SPDLOG_DEBUG("Number of valid locations: {}", valid_locations.size());

    SourceRequest req =
        communication_manager.get().get_best_available_source(destination, location_flags);

    return req;
  }

  MovementStatus start_move(dataid_t data_id, devid_t source, devid_t destination,
                            timecount_t current_time) {
    assert(launched_locations.is_valid(data_id, source));

    bool is_moving = movement_manager.is_moving(data_id, destination);
    if (is_moving) {
      timecount_t time_left = movement_manager.get_time(data_id, destination) - current_time;
      SPDLOG_DEBUG("Data block {} already moving to device {} expected to end after {}", data_id,
                   destination, time_left);
      return {.is_virtual = true, .duration = time_left};
    }

    if (launched_locations.is_valid(data_id, destination)) {
      SPDLOG_DEBUG("Data block {} already at device {}", data_id, destination);
      return {.is_virtual = true, .duration = 0};
    }

    SPDLOG_DEBUG("Starting move of data block {} from device {} to device {}", data_id, source,
                 destination);

    const auto size = data.get().get_size(data_id);

    lru_manager.read(destination, data_id, size);
    add_memory(data_id, destination, current_time, size);

    timecount_t duration =
        communication_manager.get().ideal_time_to_transfer(size, source, destination);

    if (duration == 0) {
      assert(source != destination);
      SPDLOG_DEBUG("Block moving instantly from {} to {}. Check bandwidth settings.", source,
                   destination);
    }

    movement_manager.set_completion(data_id, destination, current_time + duration);

    communication_manager.get().reserve_connection(source, destination);

    return {.is_virtual = false, .duration = duration};
  }

  void complete_move(dataid_t data_id, devid_t source, devid_t destination, bool is_virtual,
                     timecount_t current_time) {

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
        assert(launched_locations.is_valid(data_id, source));
        assert(launched_locations.is_valid(data_id, destination));
      }
      return;
    }

    SPDLOG_DEBUG("Completing real move of data block {} from device {} to device {}", data_id,
                 source, destination);

    assert(movement_manager.is_moving(data_id, destination));
    launched_locations.set_valid(data_id, destination, current_time);
    // lru_manager.read(destination, data_id, data.get().get_size(data_id));
    movement_manager.remove(data_id, destination);

    communication_manager.get().release_connection(source, destination);
  }

  void complete_eviction_move(dataid_t data_id, devid_t source, devid_t destination,
                              bool is_virtual, timecount_t current_time) {

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
        assert(launched_locations.is_valid(data_id, source));
        assert(launched_locations.is_valid(data_id, destination));
      }
      return;
    }

    SPDLOG_DEBUG("Completing eviction move of data block {} from device {} to device {}", data_id,
                 source, destination);

    assert(movement_manager.is_moving(data_id, destination));
    launched_locations.set_valid(data_id, destination, current_time);
    reserved_locations.set_valid(data_id, destination, current_time);
    mapped_locations.set_valid(data_id, destination, current_time);
    // lru_manager.read(destination, data_id, data.get().get_size(data_id));
    movement_manager.remove(data_id, destination);

    communication_manager.get().release_connection(source, destination);
  }

  void remove_memory(const DeviceIDList &device_list, dataid_t data_id, timecount_t current_time) {
    for (auto device : device_list) {
      auto size = data.get().get_size(data_id);
      SPDLOG_DEBUG("Removing data block {} from device {} with size {}", data_id, device, size);
      device_manager.get().remove_mem<TaskState::MAPPED>(device, size, current_time);
      device_manager.get().remove_mem<TaskState::RESERVED>(device, size, current_time);
      device_manager.get().remove_mem<TaskState::LAUNCHED>(device, size, current_time);
    }
  }

  void retire_data(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    auto size = data.get().get_size(data_id);
    SPDLOG_DEBUG("Retiring data block {} from device {} with size {}", data_id, device_id, size);
    for (auto device : mapped_locations.invalidate_all(data_id, current_time)) {
      device_manager.get().remove_mem<TaskState::MAPPED>(device, size, current_time);
    }
    for (auto device : reserved_locations.invalidate_all(data_id, current_time)) {
      device_manager.get().remove_mem<TaskState::RESERVED>(device, size, current_time);
    }
    for (auto device : launched_locations.invalidate_all(data_id, current_time)) {
      device_manager.get().remove_mem<TaskState::LAUNCHED>(device, size, current_time);
      lru_manager.invalidate(device, data_id);
    }
  }

  void finalize(timecount_t current_time) {
    mapped_locations.finalize(current_time);
    reserved_locations.finalize(current_time);
    launched_locations.finalize(current_time);
  }

  ValidEventArray &get_valid_intervals_mapped(dataid_t data_id, devid_t device_id) {
    return mapped_locations.get_valid_intervals(data_id, device_id);
  }

  ValidEventArray &get_valid_intervals_reserved(dataid_t data_id, devid_t device_id) {
    return reserved_locations.get_valid_intervals(data_id, device_id);
  }

  ValidEventArray &get_valid_intervals_launched(dataid_t data_id, devid_t device_id) {
    return launched_locations.get_valid_intervals(data_id, device_id);
  }

  friend class SchedulerState;
};