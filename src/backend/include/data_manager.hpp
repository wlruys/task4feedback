#pragma once

#include "communication_manager.hpp"
#include "device_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include "tasks.hpp"
#include <algorithm>
#include <functional>
#include <unordered_map>


enum class DataState {
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

public:
  Data() = default;
  Data(std::size_t num_data) : sizes(num_data), initial_location(num_data), data_names(num_data) {
  }

  [[nodiscard]] std::size_t size() const {
    return sizes.size();
  }
  [[nodiscard]] bool empty() const {
    return size() == 0;
  }

  void set_size(dataid_t id, mem_t size) {
    assert(id < sizes.size());
    sizes.at(id) = size;
  }

  void set_location(dataid_t id, devid_t location) {
    assert(id < initial_location.size());

    initial_location.at(id) = location;
  }
  void set_name(dataid_t id, std::string name) {
    assert(id < data_names.size());
    data_names.at(id) = std::move(name);
  }

  void create_block(dataid_t id, mem_t size, devid_t location, std::string name) {
    set_size(id, size);
    set_location(id, location);
    set_name(id, std::move(name));
  }

  [[nodiscard]] mem_t get_size(dataid_t id) const {
    return sizes[id];
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

struct ValidInterval {
  timecount_t start = 0;
  timecount_t stop = 0;
};

struct ValidEventArray{
  timecount_t* starts = NULL;
  timecount_t* stops = NULL;
  std::size_t size = 0;
};


class BlockLocation {
protected:
  dataid_t data_id;
  std::vector<bool> locations;
  std::vector<std::vector<ValidInterval>> valid_intervals;
  std::vector<timecount_t> current_start;

public:
  BlockLocation(dataid_t data_id, std::size_t n_devices) : data_id(data_id), locations(n_devices), valid_intervals(n_devices), current_start(n_devices) {

    #ifdef SIM_TRACK_LOCATION
    for(auto &interval : valid_intervals){
      interval.reserve(50);
    }
    #endif
  }

  bool check_valid_at_time(devid_t device_id, timecount_t query_time) const {
    const auto &intervals = valid_intervals[device_id];

    if(intervals.empty()){
      return false;
    }

    // Use binary search to find the first interval whose start is greater than query_time.
    auto it = std::upper_bound(intervals.begin(), intervals.end(), query_time,
      [](timecount_t t, const ValidInterval &interval) {
        return t < interval.start;
      });
    
    // Check the interval immediately preceding the found one.
    if (it != intervals.begin()) {
      const ValidInterval &candidate = *(it - 1);
      if (candidate.start <= query_time && query_time < candidate.stop) {
        return true;
      }
    }
    
    // If the device is currently valid, check if the open interval covers the query time.
    if (locations[device_id] && query_time >= current_start[device_id]) {
      return true;
    }
    
    return false;
  }

  void set_valid(devid_t device_id, timecount_t current_time) {
    #ifdef SIM_TRACK_LOCATION
    if(!locations[device_id]){
      locations[device_id] = true;
      current_start[device_id] = current_time;
    }
    #else
    locations[device_id] = true;
    #endif
  }

  void set_invalid(devid_t device_id, timecount_t current_time) {
    #ifdef SIM_TRACK_LOCATION
    if(locations[device_id]){
      locations[device_id] = false;
      if(current_start[device_id] != current_time){
        valid_intervals[device_id].emplace_back(current_start[device_id], current_time);
      }
    }
    #else
    locations[device_id] = false;
    #endif
  }

  [[nodiscard]] bool is_valid(devid_t device_id) const {
    return locations.at(device_id);
  }

  [[nodiscard]] bool is_invalid(devid_t device_id) const {
    return !is_valid(device_id);
  }

  // Return the number of valid locations
  [[nodiscard]] std::size_t count_valid() const {
    return static_cast<std::size_t>(std::count(locations.begin(), locations.end(), true));
  }

  // Return indexes of valid locations
  [[nodiscard]] std::vector<devid_t> get_valid_locations() const {
    std::vector<devid_t> valid_locations;
    for (devid_t i = 0; i < locations.size(); i++) {
      if (is_valid(i)) {
        valid_locations.push_back(i);
      }
    }
    return valid_locations;
  }

  void populate_valid_locations(std::vector<devid_t> &valid_locations) const {
    for (devid_t i = 0; i < locations.size(); i++) {
      if (is_valid(i)) {
        valid_locations.push_back(i);
      }
    }
  }

  bool validate(devid_t device_id, timecount_t current_time) {
    if (is_valid(device_id)) {
      return false;
    }
    set_valid(device_id, current_time);
    return true;
  }

  std::vector<devid_t> invalidate_except(devid_t device_id, timecount_t current_time) {
    std::vector<devid_t> updated;
    for (devid_t i = 0; i < locations.size(); i++) {
      if (i != device_id && is_valid(i)) {
        set_invalid(i, current_time);
        updated.push_back(i);
      }
    }
    return updated;
  }

  void finalize(timecount_t current_time){
    //tie off any open/hanging interval at the end of the simulation
    for(devid_t i = 0; i < locations.size(); i++){
      if(is_valid(i)){
        valid_intervals[i].emplace_back(current_start[i], current_time);
        current_start[i] = current_time;
      }
    }
  }

  ValidEventArray get_valid_intervals(devid_t device_id) const {
    assert(device_id < valid_intervals.size());

    const auto &intervals = valid_intervals[device_id];
    ValidEventArray valid_events;

    bool has_open_interval = is_valid(device_id);

    valid_events.size = intervals.size() + has_open_interval;

    if(valid_events.size == 0){
      //Return a single interval from 0 to 0 to indicate no valid intervals
      valid_events.starts = static_cast<timecount_t*>(malloc(1 * sizeof(timecount_t)));
      valid_events.stops = static_cast<timecount_t*>(malloc(1 * sizeof(timecount_t)));
      valid_events.size = 1;
      valid_events.starts[0] = 0;
      valid_events.stops[0] = 0;
      return valid_events;
    }


    valid_events.starts = static_cast<timecount_t*>(malloc((valid_events.size) * sizeof(timecount_t)));
    valid_events.stops = static_cast<timecount_t*>(malloc((valid_events.size) * sizeof(timecount_t)));
    for (std::size_t i = 0; i < intervals.size(); i++) {
      valid_events.starts[i] = intervals[i].start;
      valid_events.stops[i] = intervals[i].stop;
    }
    if (has_open_interval) {
      valid_events.starts[intervals.size()] = current_start[device_id];
      valid_events.stops[intervals.size()] = MAX_TIME;
    }
    return valid_events;
  }


  friend std::ostream &operator<<(std::ostream &os, const BlockLocation &bl) {
    os << "[";
    for (devid_t i = 0; i < bl.locations.size(); i++) {
      os << (bl.is_valid(i) ? "1" : "0");
    }
    os << "]";
    return os;
  }
};

class LocationManager {
protected:
  std::vector<BlockLocation> block_locations;

public:
  LocationManager(std::size_t num_data, std::size_t num_devices) {
    block_locations.reserve(num_data);
    for (dataid_t i = 0; i < num_data; i++) {
      block_locations.emplace_back(i, num_devices);
    }
  }

  void set_valid(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    block_locations.at(data_id).set_valid(device_id, current_time);
  }

  void set_invalid(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    block_locations.at(data_id).set_invalid(device_id, current_time);
  }

  [[nodiscard]] bool is_valid(dataid_t data_id, devid_t device_id) const {
    return block_locations.at(data_id).is_valid(device_id);
  }

  [[nodiscard]] bool is_invalid(dataid_t data_id, devid_t device_id) const {
    return block_locations.at(data_id).is_invalid(device_id);
  }

  [[nodiscard]] std::size_t count_valid(dataid_t data_id) const {
    return block_locations.at(data_id).count_valid();
  }

  [[nodiscard]] std::vector<devid_t> get_valid_locations(dataid_t data_id) const {
    return block_locations.at(data_id).get_valid_locations();
  }

  BlockLocation &at(dataid_t data_id) {
    return block_locations.at(data_id);
  }

  const BlockLocation &at(dataid_t data_id) const {
    return block_locations.at(data_id);
  }

  BlockLocation &operator[](dataid_t data_id) {
    return block_locations.at(data_id);
  }

  const BlockLocation &operator[](dataid_t data_id) const {
    return block_locations.at(data_id);
  }

  ValidEventArray get_valid_intervals(dataid_t data_id, devid_t device_id) const {
    assert(data_id < block_locations.size());
    return block_locations.at(data_id).get_valid_intervals(device_id);
  }

  void finalize(timecount_t current_time){
    for(auto &block : block_locations){
      block.finalize(current_time);
    }
  }
};

struct MovementPair {
  dataid_t data_id = 0;
  devid_t destination = 0;

  MovementPair() = default;

  MovementPair(dataid_t data_id, devid_t destination) : data_id(data_id), destination(destination) {
  }

  bool operator==(const MovementPair &other) const {
    return data_id == other.data_id && destination == other.destination;
  }

  // Hash function for MovementPair
  struct Hash {
    std::size_t operator()(const MovementPair &pair) const {
      // NOTE(wlr): I have no idea what the collision rate of this is
      //            Keep this in mind if something starts failing
      return std::hash<dataid_t>()(pair.data_id) ^ std::hash<devid_t>()(pair.destination);
    }
  };

  bool operator<(const MovementPair &other) const {
    return data_id < other.data_id || (data_id == other.data_id && destination < other.destination);
  }
};

class MovementManager {
protected:
  std::unordered_map<MovementPair, timecount_t, MovementPair::Hash> movement_times;

public:
  MovementManager() = default;

  bool is_moving(dataid_t data_id, devid_t destination) const {
    return movement_times.find({data_id, destination}) != movement_times.end();
  }

  [[nodiscard]] timecount_t get_time(dataid_t data_id, devid_t destination) const {
    auto it = movement_times.find({data_id, destination});
    return it == movement_times.end() ? 0 : it->second;
  }

  void set_completion(dataid_t data_id, devid_t destination, timecount_t time) {
    movement_times[{data_id, destination}] = time;
  }

  void remove(dataid_t data_id, devid_t destination) {
    movement_times.erase({data_id, destination});
  }
};

class DeviceDataCounts {
protected:
  using DataCount = std::unordered_map<dataid_t, std::size_t>;
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

  static bool check_valid(const DataIDList &list, const LocationManager &locations,
                          devid_t device_id) {
    return std::ranges::all_of(
        list, [&](auto data_id) { return !locations.is_invalid(data_id, device_id); });
  }

  static bool read_update(dataid_t data_id, devid_t device_id, LocationManager &locations, timecount_t current_time) {
    return locations.at(data_id).validate(device_id, current_time);
  }

  static auto write_update(dataid_t data_id, devid_t device_id, LocationManager &locations, timecount_t current_time) {
    auto updated_ids = locations[data_id].invalidate_except(device_id, current_time);
    return updated_ids;
  }

public:
  DataManager(Data &data_, DeviceManager &device_manager_,
              CommunicationManager &communication_manager_)
      : data(data_), device_manager(device_manager_), communication_manager(communication_manager_),
        mapped_locations(data.get().size(), device_manager_.size()),
        reserved_locations(data.get().size(), device_manager_.size()),
        launched_locations(data.get().size(), device_manager_.size()),
        counts(device_manager_.size()) {
  }

  DataManager(const DataManager &o_, DeviceManager &device_manager_,
              CommunicationManager &communication_manager_)
      : data(o_.data), device_manager(device_manager_),
        communication_manager(communication_manager_), mapped_locations(o_.mapped_locations),
        reserved_locations(o_.reserved_locations), launched_locations(o_.launched_locations),
        movement_manager(o_.movement_manager), counts(o_.counts) {
  }

  DataManager(const DataManager &o_) = delete;
  DataManager &operator=(const DataManager &o_) = delete;

  void initialize() {
    for (dataid_t i = 0; i < data.get().size(); i++) {
      auto initial_location = data.get().get_location(i);
      mapped_locations.set_valid(i, initial_location, 0);
      reserved_locations.set_valid(i, initial_location, 0);
      launched_locations.set_valid(i, initial_location, 0);
      device_manager.get().add_mem<TaskState::MAPPED>(initial_location, data.get().get_size(i), 0);
      device_manager.get().add_mem<TaskState::RESERVED>(initial_location, data.get().get_size(i), 0);
      device_manager.get().add_mem<TaskState::LAUNCHED>(initial_location, data.get().get_size(i), 0);
    }
  }

  [[nodiscard]] const Data &get_data() const {
    return data;
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

  bool check_valid_reserved(const DataIDList &list, devid_t device_id) const {
    return check_valid(list, reserved_locations, device_id);
  }

  bool check_valid_launched(const DataIDList &list, devid_t device_id) const {
    return check_valid(list, launched_locations, device_id);
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

  void add_memory(dataid_t data_id, devid_t device_id, timecount_t current_time) {
    SPDLOG_DEBUG("Adding data block {} to device {} with size {}", data_id, device_id,
                 data.get().get_size(data_id));
    device_manager.get().add_mem<TaskState::LAUNCHED>(device_id, data.get().get_size(data_id), current_time);
  }

  void read_update_launched(const DataIDList &list, devid_t device_id, timecount_t current_time) {
    for (auto data_id : list) {
      bool changed = read_update(data_id, device_id, launched_locations, current_time);
      if (changed) {
        add_memory(data_id, device_id, current_time);
      }
    }
  }

  auto get_valid_mapped_locations(dataid_t data_id) const {
    return mapped_locations.get_valid_locations(data_id);
  }

  auto get_valid_reserved_locations(dataid_t data_id) const {
    return reserved_locations.get_valid_locations(data_id);
  }

  auto get_valid_launched_locations(dataid_t data_id) const {
    return launched_locations.get_valid_locations(data_id);
  }

  SourceRequest request_source(dataid_t data_id, devid_t destination) const {
    auto valid_locations = launched_locations.get_valid_locations(data_id);
    assert(!valid_locations.empty());

    if (launched_locations.is_valid(data_id, destination)) {
      return {true, destination};
    }

    SPDLOG_DEBUG("Requesting source for data block {} to device {}", data_id, destination);
    SPDLOG_DEBUG("Number of valid locations: {}", valid_locations.size());

    SourceRequest req =
        communication_manager.get().get_best_available_source(destination, valid_locations);

    return req;
  }

  MovementStatus start_move(dataid_t data_id, devid_t source, devid_t destination, timecount_t current_time) {
    assert(launched_locations.is_valid(data_id, source));

    bool is_moving = movement_manager.is_moving(data_id, destination);
    if (is_moving) {
      SPDLOG_DEBUG("Data block {} already moving to device {}", data_id, destination);
      return {true, movement_manager.get_time(data_id, destination)};
    }

    if (launched_locations.is_valid(data_id, destination)) {
      SPDLOG_DEBUG("Data block {} already at device {}", data_id, destination);
      return {true, 0};
    }

    SPDLOG_DEBUG("Starting move of data block {} from device {} to device {}", data_id, source,
                 destination);

    add_memory(data_id, destination, current_time);

    timecount_t duration = communication_manager.get().ideal_time_to_transfer(
        data.get().get_size(data_id), source, destination);

    if (duration == 0) {
      assert(source != destination);
      SPDLOG_DEBUG("Block moving instantly from {} to {}. Check bandwidth settings.", source,
                   destination);
    }

    movement_manager.set_completion(data_id, destination, duration);

    communication_manager.get().reserve_connection(source, destination);

    return {false, duration};
  }

  void complete_move(dataid_t data_id, devid_t source, devid_t destination, bool is_virtual, timecount_t current_time) {

    if (is_virtual) {
      SPDLOG_DEBUG("Completing virtual move of data block {} from device {} to "
                   "device {}",
                   data_id, source, destination);

      if (movement_manager.is_moving(data_id, destination)) {
        SPDLOG_DEBUG("Virtual move of data block {} from device {} to device {} "
                     "beat the real move",
                     data_id, source, destination);
        // Update will happen in the real move
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

  void write_update_launched(const DataIDList &list, devid_t device_id, timecount_t current_time) {
    for (auto data_id : list) {
      auto updated_devices = write_update(data_id, device_id, launched_locations, current_time);
      remove_memory(updated_devices, data_id, current_time);
    }
  }

  void finalize(timecount_t current_time) {
    mapped_locations.finalize(current_time);
    reserved_locations.finalize(current_time);
    launched_locations.finalize(current_time);
  }

  bool check_valid_at_time_mapped(dataid_t data_id, devid_t device_id, timecount_t query_time) const {
    return mapped_locations.at(data_id).check_valid_at_time(device_id, query_time);
  }

  bool check_valid_at_time_reserved(dataid_t data_id, devid_t device_id, timecount_t query_time) const {
    return reserved_locations.at(data_id).check_valid_at_time(device_id, query_time);
  }

  bool check_valid_at_time_launched(dataid_t data_id, devid_t device_id, timecount_t query_time) const {
    return launched_locations.at(data_id).check_valid_at_time(device_id, query_time);
  }

  ValidEventArray get_valid_intervals_mapped(dataid_t data_id, devid_t device_id) const {
    return mapped_locations.get_valid_intervals(data_id, device_id);
  }

  ValidEventArray get_valid_intervals_reserved(dataid_t data_id, devid_t device_id) const {
    return reserved_locations.get_valid_intervals(data_id, device_id);
  } 

  ValidEventArray get_valid_intervals_launched(dataid_t data_id, devid_t device_id) const {
    return launched_locations.get_valid_intervals(data_id, device_id);
  }

  friend class SchedulerState;
};