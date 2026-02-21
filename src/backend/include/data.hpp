#pragma once
#include "communication.hpp"
#include "devices.hpp"
#include "eviction.hpp"
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
    sizes[id] = size;
  }

  void set_tag(dataid_t id, int tag) {
    assert(id < data_tags.size());
    data_tags[id] = tag;
  }

  void set_x_pos(dataid_t id, float x) {
    xy_positions[id].x = x;
  }

  void set_y_pos(dataid_t id, float y) {
    xy_positions[id].y = y;
  }

  [[nodiscard]] float get_x_pos(dataid_t id) const {
    return xy_positions[id].x;
  }

  [[nodiscard]] float get_y_pos(dataid_t id) const {
    return xy_positions[id].y;
  }

  int get_tag(dataid_t id) const {
    return data_tags[id];
  }

  void set_type(dataid_t id, int type) {
    data_types[id] = type;
  }

  int get_type(dataid_t id) const {
    return data_types[id];
  }

  void set_location(dataid_t id, devid_t location) {
    assert(id < initial_location.size());

    initial_location[id] = location;
  }
  void set_name(dataid_t id, std::string name) {
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
    assert(data_id < num_data && device_id < num_devices);
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
        locations[data_id * num_devices + device_id] = 0;
        // close the current interval
        valid_intervals[data_id * num_devices + device_id].stops.back() = current_time;
      }
    }
#endif

    locations[data_id] &= ~mask;
    return old_status;
  }

  [[nodiscard]] inline std::size_t count_valid(dataid_t data_id) const {
    return std::count(locations.data() + data_id * num_devices,
                      locations.data() + (data_id + 1) * num_devices, 1);
  }

  [[nodiscard]] inline devicemask_t get_location_flags(dataid_t data_id) const {
    return locations[data_id];
  }

  [[nodiscard]] inline devicemask_t get_location_flags(dataid_t data_id) {
    return locations[data_id];
  }

  void populate_valid_locations(dataid_t data_id, std::vector<devid_t> &valid_locations) const {
    for (devicemask_t i = 0; i < num_devices; i++) {
      if (is_valid(data_id, i)) {
        valid_locations.push_back(i);
      }
    }
  }

  bool inline validate(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    return set_valid(data_id, device_id, current_time) == 0;
  }

  inline devicemask_t invalidate_except(dataid_t data_id, devid_t device_id,
                                        timecount_t current_time) {
    assert(data_id < num_data && device_id < num_devices);
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
    assert(src < total_data_movement.size());
    assert(dest < total_data_movement.size());
    total_data_movement[src] += size;
    total_data_movement[dest] += size;
    if (dest == 0) {
      eviction_data_movement[src] += size;
      eviction_data_movement[dest] += size;
    }
  }

  void add_eviction_movement(devid_t device_id, mem_t size) {
    assert(device_id < eviction_data_movement.size());
    eviction_data_movement[device_id] += size;
  }

  const std::vector<mem_t> &get_total_data_movement() const {
    return total_data_movement;
  }

  const mem_t get_total_data_movement(devid_t device_id) const {
    assert(device_id < total_data_movement.size());
    return total_data_movement[device_id];
  }

  const std::vector<mem_t> &get_eviction_data_movement() const {
    return eviction_data_movement;
  }

  const mem_t get_eviction_data_movement(devid_t device_id) const {
    assert(device_id < eviction_data_movement.size());
    return eviction_data_movement[device_id];
  }
};

class DataManager {
protected:
  using EvictionResidencyManager = eviction::ResidencyManager;

  LocationManager mapped_locations;
  LocationManager reserved_locations;
  LocationManager launched_locations;
  MovementManager movement_manager;
  // Residency/eviction tracking policy storage (LRU for now).
  EvictionResidencyManager residency_manager;
  eviction::RuntimeStack eviction_stack;
  MovementCounter movement_counter;
  devid_t n_devices = 0;
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

  bool complete_virtual_move_common(dataid_t data_id, devid_t source, devid_t destination,
                                    const char *move_kind) {
    SPDLOG_DEBUG("Completing virtual {} of data block {} from device {} to device {}", move_kind,
                 data_id, source, destination);

    if (movement_manager.is_moving(data_id, destination)) {
      SPDLOG_DEBUG("Virtual {} of data block {} from device {} to device {} beat the real move",
                   move_kind, data_id, source, destination);
      // Update will happen in the real move
      // Not valid until the real move is completed
      return true;
    }

    // NOTE(wlr): I'm not 100% sure about the source check.
    // Could something that starts at the same time as the move completes be a problem?
    assert(launched_locations.is_valid(data_id, source));
    assert(launched_locations.is_valid(data_id, destination));
    return true;
  }

  void complete_real_move_common(CommunicationManager &comm_manager, dataid_t data_id,
                                 devid_t source, devid_t destination, timecount_t current_time,
                                 const char *move_kind, bool mark_reserved, bool mark_mapped,
                                 mem_t data_size) {
    SPDLOG_DEBUG("Completing real {} of data block {} from device {} to device {}", move_kind,
                 data_id, source, destination);
    assert(movement_manager.is_moving(data_id, destination));

    // Only completed transfers become evictable residency entries.
    residency_manager.read(destination, data_id, data_size);
    launched_locations.set_valid(data_id, destination, current_time);
    if (mark_reserved) {
      reserved_locations.set_valid(data_id, destination, current_time);
    }
    if (mark_mapped) {
      mapped_locations.set_valid(data_id, destination, current_time);
    }
    movement_manager.remove(data_id, destination);
    comm_manager.release_connection(source, destination);
  }

public:
  std::vector<devid_t> valid_location_buffer;

  DataManager() = default;

  DataManager(const Data &data, const Devices &devices)
      : mapped_locations(data.size(), devices.size()),
        reserved_locations(data.size(), devices.size()),
        launched_locations(data.size(), devices.size()), residency_manager(devices),
        movement_counter(devices.size()), n_devices(devices.size()) {
  }

  DataManager(const DataManager &o_)
      : residency_manager(o_.residency_manager), eviction_stack(o_.eviction_stack),
        movement_counter(o_.movement_counter), n_devices(o_.n_devices) {
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
    n_devices = devices.size();
    for (dataid_t i = 0; i < data.size(); i++) {
      auto initial_location = data.get_location(i);
      const auto data_size = data.get_size(i);

      if (initial_location > -1 && (residency_manager.get_mem(initial_location) + data_size) <=
                                       devices.get_max_resources(initial_location).mem) {
        mapped_locations.set_valid(i, initial_location, 0);
        reserved_locations.set_valid(i, initial_location, 0);
        launched_locations.set_valid(i, initial_location, 0);
        device_manager.add_mem<TaskState::MAPPED>(initial_location, data_size, 0);
        device_manager.add_mem<TaskState::RESERVED>(initial_location, data_size, 0);
        device_manager.add_mem<TaskState::LAUNCHED>(initial_location, data_size, 0);
        residency_manager.read(initial_location, i, data_size);
      } else {
        mapped_locations.set_valid(i, 0, 0);
        reserved_locations.set_valid(i, 0, 0);
        launched_locations.set_valid(i, 0, 0);
        device_manager.add_mem<TaskState::MAPPED>(0, data_size, 0);
        device_manager.add_mem<TaskState::RESERVED>(0, data_size, 0);
        device_manager.add_mem<TaskState::LAUNCHED>(0, data_size, 0);
        residency_manager.read(0, i, data_size);
      }
    }
    for (devid_t i = 0; i < devices.size(); i++) {
      SPDLOG_DEBUG("DataManager: Device {} initialized with {}/{} memory", devices.get_name(i),
                   residency_manager.get_mem(i), devices.get_max_resources(i).mem);
    }
    valid_location_buffer.reserve(devices.size());
  }

  void initialize_data_replicate(const Data &data, const Devices &devices,
                                 DeviceManager &device_manager, dataid_t data_id,
                                 devid_t device_id) {
    const auto data_size = data.get_size(data_id);
    if (device_id > -1 &&
        (residency_manager.get_mem(device_id) + data_size) <= devices.get_max_resources(device_id).mem &&
        !mapped_locations.is_valid(data_id, device_id)) {
      mapped_locations.set_valid(data_id, device_id, 0);
      reserved_locations.set_valid(data_id, device_id, 0);
      launched_locations.set_valid(data_id, device_id, 0);
      device_manager.add_mem<TaskState::MAPPED>(device_id, data_size, 0);
      device_manager.add_mem<TaskState::RESERVED>(device_id, data_size, 0);
      device_manager.add_mem<TaskState::LAUNCHED>(device_id, data_size, 0);
      residency_manager.read(device_id, data_id, data_size);
    }
  }

  [[nodiscard]] EvictionResidencyManager &eviction_residency() {
    return residency_manager;
  }

  [[nodiscard]] const EvictionResidencyManager &eviction_residency() const {
    return residency_manager;
  }

  void initialize_eviction_stack(std::size_t n_compute_tasks, std::size_t reserve_hint = 0) {
    eviction_stack.initialize(static_cast<std::size_t>(n_devices), n_compute_tasks, reserve_hint);
  }

  template <class StaticGraphT>
  void build_eviction_ancestor_index(const StaticGraphT &static_graph) {
    eviction_stack.build_compute_task_ancestor_index(static_graph);
  }

  [[nodiscard]] eviction::RuntimeStack &eviction_runtime() {
    return eviction_stack;
  }

  [[nodiscard]] const eviction::RuntimeStack &eviction_runtime() const {
    return eviction_stack;
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

  bool check_valid_reserved(std::span<const dataid_t> &list, devid_t device_id) const {
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
    }
    // Memory change is handled by task request in mapper
  }

  void write_update_mapped(const Data &data, DeviceManager &device_manager,
                           std::span<const dataid_t> list, devid_t device_id,
                           timecount_t current_time) {
    for (auto data_id : list) {
      write_update(data_id, device_id, mapped_locations, current_time);
    }
    // Memory change is handled by task complete
  }

  void read_update_reserved(const Data &data, DeviceManager &device_manager,
                            std::span<const dataid_t> list, devid_t device_id,
                            timecount_t current_time) {
    for (auto data_id : list) {
      read_update(data_id, device_id, reserved_locations, current_time);
    }
    // Memory change is handeled by task request in reserver
  }

  void write_update_reserved(const Data &data, DeviceManager &device_manager,
                             std::span<const dataid_t> list, devid_t device_id,
                             timecount_t current_time) {
    for (auto data_id : list) {
      write_update(data_id, device_id, reserved_locations, current_time);
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
      residency_manager.read(device_id, data_id, size);
      bool changed = read_update(data_id, device_id, launched_locations, current_time);
      if (changed) {
        add_memory(device_manager, device_id, data_id, size, current_time);
      }
    }
  }

  void write_update_launched(const Data &data, DeviceManager &device_manager,
                             std::span<const dataid_t> list, devid_t device_id,
                             timecount_t current_time) {
    for (auto data_id : list) {
      auto changed_flags = write_update(data_id, device_id, launched_locations, current_time);
      const auto size = data.get_size(data_id);
      remove_memory(device_manager, changed_flags, data_id, size, current_time);
    }
  }

  void evict_on_update_launched(const Data &data, DeviceManager &device_manager, dataid_t data_id,
                                devid_t device_id, timecount_t current_time,
                                const eviction::InvalidationInfo &invalidation) {
    auto updated_devices_launched =
        evict_on_update(data_id, device_id, launched_locations, current_time);
    evict_on_update(data_id, device_id, reserved_locations, current_time);

    auto size = data.get_size(data_id);
    const devid_t n_devices = device_manager.n_devices;
    for (devid_t device = 0; device < n_devices; device++) {
      if (updated_devices_launched & (1 << device)) {
        SPDLOG_DEBUG("Evicting data block {} from device {} with size {}", data_id, device, size);
        device_manager.remove_mem<TaskState::RESERVED>(device, size, current_time);
        device_manager.remove_mem<TaskState::LAUNCHED>(device, size, current_time);
        residency_manager.invalidate(device, data_id, true);
      }
    }
    if (invalidation.mapped_cleanup ==
        eviction::InvalidationInfo::MappedCleanupDecision::REMOVE_MAPPED_AND_LOCATION) {
      // No pending local users: remove mapped bytes and mapped validity.
      device_manager.remove_mem<TaskState::MAPPED>(device_id, size, current_time);
      mapped_locations.set_invalid(data_id, device_id, current_time);
    } else if (invalidation.mapped_cleanup ==
               eviction::InvalidationInfo::MappedCleanupDecision::REMOVE_MAPPED_BYTES_ONLY) {
      // Remote-next-writer optimization: drop mapped bytes now while preserving mapped validity.
      device_manager.remove_mem<TaskState::MAPPED>(device_id, size, current_time);
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

    return req;
  }

  MovementStatus start_move(const Topology &topology, CommunicationManager &comm_manager,
                            DeviceManager &device_manager, const Data &data, dataid_t data_id,
                            devid_t source, devid_t destination, timecount_t current_time) {
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

    const auto size = data.get_size(data_id);

    add_memory(device_manager, destination, data_id, size, current_time);

    timecount_t duration = comm_manager.ideal_time_to_transfer(topology, size, source, destination);

    // if (rand() % 100 < 10) {
    //   duration = duration * 146 / 100;
    // }

    // // if (size < 64 * 1024 * 1024) {
    // //   duration = duration * 129 / 100;
    // // }

    if (duration == 0) {
      assert(source != destination);
      SPDLOG_DEBUG("Block moving instantly from {} to {}. Check bandwidth settings.", source,
                   destination);
    }

    movement_manager.set_completion(data_id, destination, current_time + duration);

    comm_manager.reserve_connection(source, destination);

    movement_counter.add_total_movement(source, destination, size);

    return {.is_virtual = false, .duration = duration};
  }

  void complete_move(CommunicationManager &comm_manager, dataid_t data_id, devid_t source,
                     devid_t destination, bool is_virtual, timecount_t current_time,
                     mem_t data_size) {
    if (is_virtual) {
      complete_virtual_move_common(data_id, source, destination, "move");
      return;
    }
    complete_real_move_common(comm_manager, data_id, source, destination, current_time, "move",
                              /*mark_reserved=*/false, /*mark_mapped=*/false, data_size);
  }

  void complete_eviction_move(CommunicationManager &comm_manager, dataid_t data_id, devid_t source,
                              devid_t destination, bool is_virtual, timecount_t current_time,
                              mem_t data_size) {
    if (is_virtual) {
      complete_virtual_move_common(data_id, source, destination, "eviction move");
      return;
    }
    complete_real_move_common(comm_manager, data_id, source, destination, current_time,
                              "eviction move",
                              /*mark_reserved=*/true, /*mark_mapped=*/true, data_size);
  }

  void remove_memory(DeviceManager &device_manager, const devicemask_t changed_flags,
                     dataid_t data_id, mem_t size, timecount_t current_time) {
    const devid_t n_devices = device_manager.n_devices;
    for (devid_t device = 0; device < n_devices; device++) {
      if (changed_flags & (1 << device)) {
        SPDLOG_DEBUG("Removing data block {} from device {} with size {}", data_id, device, size);
        device_manager.remove_mem<TaskState::MAPPED>(device, size, current_time);
        device_manager.remove_mem<TaskState::RESERVED>(device, size, current_time);
        device_manager.remove_mem<TaskState::LAUNCHED>(device, size, current_time);
        residency_manager.invalidate(device, data_id);
      }
    }
  }

  void retire_data(const Data &data, DeviceManager &device_manager, dataid_t data_id,
                   devid_t device_id, timecount_t current_time) {
    auto size = data.get_size(data_id);
    SPDLOG_DEBUG("Retiring data block {} from device {} with size {}", data_id, device_id, size);

    auto mapped_flags = mapped_locations.invalidate_all(data_id, current_time);
    auto reserved_flags = reserved_locations.invalidate_all(data_id, current_time);
    auto launched_flags = launched_locations.invalidate_all(data_id, current_time);
    const devid_t n_devices = device_manager.n_devices;

    for (devid_t device = 0; device < n_devices; device++) {
      const devicemask_t device_mask = (1 << device);
      if (mapped_flags & device_mask) {
        device_manager.remove_mem<TaskState::MAPPED>(device, size, current_time);
      }
      if (reserved_flags & device_mask) {
        device_manager.remove_mem<TaskState::RESERVED>(device, size, current_time);
      }
      if (launched_flags & device_mask) {
        device_manager.remove_mem<TaskState::LAUNCHED>(device, size, current_time);
        residency_manager.invalidate(device, data_id);
      }
    }
  }

  void finalize(timecount_t current_time) {
    mapped_locations.finalize(current_time);
    reserved_locations.finalize(current_time);
    launched_locations.finalize(current_time);
  }

  friend class SchedulerState;
};
