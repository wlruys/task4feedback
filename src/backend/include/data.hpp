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
  std::vector<uint8_t> locations;
  std::vector<ValidEventArray> valid_intervals;
  bool record{false};

public:

  LocationManager() = default;

  LocationManager(dataid_t num_data, devid_t num_devices)
      : num_devices(num_devices), num_data(num_data), locations(num_data, 0){

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

  inline int8_t set_valid(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    const uint8_t mask = (1 << device_id);
    auto old_status = locations[data_id] & mask;

#ifdef SIM_RECORD
    if (record){
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

  inline int8_t set_invalid(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    const uint8_t mask = (1 << device_id);
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

  [[nodiscard]] inline uint8_t get_location_flags(dataid_t data_id) const {
    return locations[data_id];
  }

  [[nodiscard]] inline uint8_t get_location_flags(dataid_t data_id) {
    return locations[data_id];
  }

  void populate_valid_locations(dataid_t data_id, std::vector<devid_t> &valid_locations) const {
    for (uint8_t i = 0; i < num_devices; i++) {
      if (is_valid(data_id, i)) {
        valid_locations.push_back(i);
      }
    }
  }

  bool inline validate(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    return set_valid(data_id, device_id, current_time) == 0;
  }

  inline uint8_t invalidate_except(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    assert(data_id < num_data && device_id < num_devices);
    uint8_t old_status = locations[data_id];
    uint8_t keep_mask = (1 << device_id);
    // Keep only the specified device, invalidate all others
    locations[data_id] &= keep_mask;

    // which bits changed from 1 to 0
    // assumes that data was already valid on device_id
    uint8_t changed_bits = old_status ^ locations[data_id];

    return changed_bits;
  }

  inline uint8_t invalidate_all(dataid_t data_id, timecount_t current_time) {

    uint8_t old_status = locations[data_id];

    // Invalidate all devices
    locations[data_id] = 0;

    // which bits changed from 1 to 0
    uint8_t changed_bits = old_status ^ locations[data_id];

    return changed_bits;
  }

  inline uint8_t invalidate_on(dataid_t data_id, devid_t device_id, timecount_t current_time) {

    uint8_t old_status = locations[data_id];

    locations[data_id] &= ~(1 << device_id); // invalidate the specified device
    uint8_t changed_bits = old_status ^ locations[data_id];

    return changed_bits;
  }

  void finalize(timecount_t current_time) {
    // tie off any open/hanging interval at the end of the simulation
    #ifdef SIM_RECORD
    for (dataid_t i = 0; i < num_data; i++) {
      for (uint8_t j = 0; j < num_devices; j++) {
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

struct MovementStatus {
  bool is_virtual = false;
  timecount_t duration = 0;
};

class LRU_manager {
private:
  mem_t evicted_size = 0;
  uint32_t n_devices_{0};
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
  mutable DataIDList id_buffer;

public:

  LRU_manager() = default;

  // Constructor: initialize for n_devices [0 .. n_devices-1]
  explicit LRU_manager(const Devices &devices)
      : n_devices_(devices.size()), lru_lists_(devices.size()), position_maps_(devices.size()),
        size_maps_(devices.size()), sizes_(devices.size()), max_sizes_(devices.size()) {
    for (auto &size : sizes_) {
      size = 0;
    }
    for (int i = 0; i < devices.size(); i++) {
      max_sizes_[i] = devices.get_max_resources(i).mem;
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
    ZoneScoped;
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
  const std::span<const dataid_t> getLRUids(devid_t device_id, std::size_t mem_size,
                                            std::span<const dataid_t> used_ids) const {
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
};

class DataManager {
protected:
    LocationManager mapped_locations;
    LocationManager reserved_locations;
    LocationManager launched_locations;
    MovementManager movement_manager;
    LRU_manager lru_manager;
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

public:
  std::vector<devid_t> valid_location_buffer;

  DataManager() = default;

  DataManager(const Data &data, const Devices &devices)
      : mapped_locations(data.size(), devices.size()),
        reserved_locations(data.size(), devices.size()),
        launched_locations(data.size(), devices.size()), lru_manager(devices) {
  }

  DataManager(const DataManager &o_) : lru_manager(o_.lru_manager) {
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
      if (initial_location > -1) {
        mapped_locations.set_valid(i, initial_location, 0);
        reserved_locations.set_valid(i, initial_location, 0);
        launched_locations.set_valid(i, initial_location, 0);
        const auto size = data.get_size(i);
        device_manager.add_mem<TaskState::MAPPED>(initial_location, size, 0);
        device_manager.add_mem<TaskState::RESERVED>(initial_location, size, 0);
        device_manager.add_mem<TaskState::LAUNCHED>(initial_location, size, 0);
        lru_manager.read(initial_location, i, size);
      }
    }
    valid_location_buffer.reserve(devices.size());
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
      lru_manager.read(device_id, data_id, size);
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
                                devid_t device_id, timecount_t current_time, bool future_usage,
                                bool write_after_read) {
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
        lru_manager.invalidate(device, data_id, true);
      }
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
  }

  uint8_t get_mapped_location_flags(dataid_t data_id) const {
    return mapped_locations.get_location_flags(data_id);
  }

  uint8_t get_reserved_location_flags(dataid_t data_id) const {
    return reserved_locations.get_location_flags(data_id);
  }

  uint8_t get_launched_location_flags(dataid_t data_id) const {
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

    lru_manager.read(destination, data_id, size);
    add_memory(device_manager, destination, data_id, size, current_time);

    timecount_t duration = comm_manager.ideal_time_to_transfer(topology, size, source, destination);

    if (duration == 0) {
      assert(source != destination);
      SPDLOG_DEBUG("Block moving instantly from {} to {}. Check bandwidth settings.", source,
                   destination);
    }

    movement_manager.set_completion(data_id, destination, current_time + duration);

    comm_manager.reserve_connection(source, destination);

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
        assert(launched_locations.is_valid(data_id, source));
        assert(launched_locations.is_valid(data_id, destination));
      }
      return;
    }

    SPDLOG_DEBUG("Completing real move of data block {} from device {} to device {}", data_id,
                 source, destination);

    assert(movement_manager.is_moving(data_id, destination));
    launched_locations.set_valid(data_id, destination, current_time);
    movement_manager.remove(data_id, destination);

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
    movement_manager.remove(data_id, destination);

    comm_manager.release_connection(source, destination);
  }

  void remove_memory(DeviceManager &device_manager, const uint8_t changed_flags, dataid_t data_id,
                     mem_t size, timecount_t current_time) {
    const devid_t n_devices = device_manager.n_devices;
    for (devid_t device = 0; device < n_devices; device++) {
      if (changed_flags & (1 << device)) {
        SPDLOG_DEBUG("Removing data block {} from device {} with size {}", data_id, device, size);
        device_manager.remove_mem<TaskState::MAPPED>(device, size, current_time);
        device_manager.remove_mem<TaskState::RESERVED>(device, size, current_time);
        device_manager.remove_mem<TaskState::LAUNCHED>(device, size, current_time);
        lru_manager.invalidate(device, data_id);
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
      const uint8_t device_mask = (1 << device);
      if (mapped_flags & device_mask) {
        device_manager.remove_mem<TaskState::MAPPED>(device, size, current_time);
      }
      if (reserved_flags & device_mask) {
        device_manager.remove_mem<TaskState::RESERVED>(device, size, current_time);
      }
      if (launched_flags & device_mask) {
        device_manager.remove_mem<TaskState::LAUNCHED>(device, size, current_time);
        lru_manager.invalidate(device, data_id);
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