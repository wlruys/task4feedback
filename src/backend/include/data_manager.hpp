#pragma once

#include "communication_manager.hpp"
#include "device_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <algorithm>
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
  Data(std::size_t num_data)
      : sizes(num_data), initial_location(num_data), data_names(num_data) {}

  [[nodiscard]] std::size_t size() const { return sizes.size(); }
  [[nodiscard]] bool empty() const { return size() == 0; }

  void set_size(dataid_t id, mem_t size) { sizes[id] = size; }
  void set_location(dataid_t id, devid_t location) {
    initial_location[id] = location;
  }
  void set_name(dataid_t id, std::string name) {
    data_names[id] = std::move(name);
  }

  void create_block(dataid_t id, mem_t size, devid_t location,
                    std::string name) {
    set_size(id, size);
    set_location(id, location);
    set_name(id, std::move(name));
  }

  [[nodiscard]] mem_t get_size(dataid_t id) const { return sizes[id]; }
  [[nodiscard]] devid_t get_location(dataid_t id) const {
    return initial_location[id];
  }
  [[nodiscard]] const std::string &get_name(dataid_t id) const {
    return data_names[id];
  }

  friend class DataManager;
};

class BlockLocation {
protected:
  std::vector<bool> locations;

public:
  BlockLocation(std::size_t n_devices) : locations(n_devices) {}

  void set_valid(devid_t device_id) { locations[device_id] = true; }

  void set_invalid(devid_t device_id) { locations[device_id] = false; }

  [[nodiscard]] bool is_valid(devid_t device_id) const {
    return locations[device_id];
  }

  [[nodiscard]] bool is_invalid(devid_t device_id) const {
    return !is_valid(device_id);
  }

  // Return the number of valid locations
  [[nodiscard]] std::size_t count_valid() const {
    return static_cast<std::size_t>(
        std::count(locations.begin(), locations.end(), true));
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

  bool validate(devid_t device_id) {
    if (is_valid(device_id)) {
      return false;
    }
    set_valid(device_id);
    return true;
  }

  std::vector<devid_t> invalidate_except(devid_t device_id) {
    std::vector<devid_t> updated;
    for (devid_t i = 0; i < locations.size(); i++) {
      if (i != device_id && is_valid(i)) {
        set_invalid(i);
        updated.push_back(i);
      }
    }
    return updated;
  }
};

class LocationManager {
protected:
  std::vector<BlockLocation> block_locations;

public:
  LocationManager(std::size_t num_data, std::size_t num_devices)
      : block_locations(num_data, BlockLocation(num_devices)) {}

  void set_valid(dataid_t data_id, devid_t device_id) {
    block_locations[data_id].set_valid(device_id);
  }

  void set_invalid(dataid_t data_id, devid_t device_id) {
    block_locations[data_id].set_invalid(device_id);
  }

  [[nodiscard]] bool is_valid(dataid_t data_id, devid_t device_id) const {
    return block_locations[data_id].is_valid(device_id);
  }

  [[nodiscard]] bool is_invalid(dataid_t data_id, devid_t device_id) const {
    return block_locations[data_id].is_invalid(device_id);
  }

  [[nodiscard]] std::size_t count_valid(dataid_t data_id) const {
    return block_locations[data_id].count_valid();
  }

  [[nodiscard]] std::vector<devid_t>
  get_valid_locations(dataid_t data_id) const {
    return block_locations[data_id].get_valid_locations();
  }

  BlockLocation &operator[](dataid_t data_id) {
    return block_locations[data_id];
  }
};

struct MovementPair {
  dataid_t data_id = 0;
  devid_t destination = 0;

  MovementPair() = default;

  MovementPair(dataid_t data_id, devid_t destination)
      : data_id(data_id), destination(destination) {}

  bool operator==(const MovementPair &other) const {
    return data_id == other.data_id && destination == other.destination;
  }

  // Hash function for MovementPair
  struct Hash {
    std::size_t operator()(const MovementPair &pair) const {
      // NOTE(wlr): I have no idea what the collision rate of this is
      //            Keep this in mind if something starts failing
      return std::hash<dataid_t>()(pair.data_id) ^
             std::hash<devid_t>()(pair.destination);
    }
  };

  bool operator<(const MovementPair &other) const {
    return data_id < other.data_id ||
           (data_id == other.data_id && destination < other.destination);
  }
};

class MovementManager {
protected:
  std::unordered_map<MovementPair, timecount_t, MovementPair::Hash>
      movement_times;

public:
  MovementManager() = default;

  bool is_moving(dataid_t data_id, devid_t destination) const {
    return movement_times.find({data_id, destination}) != movement_times.end();
  }

  [[nodiscard]] timecount_t get_time(dataid_t data_id,
                                     devid_t destination) const {
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
      counts[data_id] = 1;
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
  DataCounts(std::size_t n_devices) : device_counts(n_devices) {}

  bool increment_mapped(dataid_t data_id, devid_t device_id) {
    return device_counts[device_id].increment_mapped(data_id);
  }

  bool increment_reserved(dataid_t data_id, devid_t device_id) {
    return device_counts[device_id].increment_reserved(data_id);
  }

  bool increment_launched(dataid_t data_id, devid_t device_id) {
    return device_counts[device_id].increment_launched(data_id);
  }

  bool increment_moving(dataid_t data_id, devid_t device_id) {
    return device_counts[device_id].increment_moving(data_id);
  }

  bool decrement_mapped(dataid_t data_id, devid_t device_id) {
    return device_counts[device_id].decrement_mapped(data_id);
  }

  bool decrement_reserved(dataid_t data_id, devid_t device_id) {
    return device_counts[device_id].decrement_reserved(data_id);
  }

  bool decrement_launched(dataid_t data_id, devid_t device_id) {
    return device_counts[device_id].decrement_launched(data_id);
  }

  bool decrement_moving(dataid_t data_id, devid_t device_id) {
    return device_counts[device_id].decrement_moving(data_id);
  }

  [[nodiscard]] std::size_t count_mapped(dataid_t data_id,
                                         devid_t device_id) const {
    return device_counts[device_id].count_mapped(data_id);
  }

  [[nodiscard]] std::size_t count_reserved(dataid_t data_id,
                                           devid_t device_id) const {
    return device_counts[device_id].count_reserved(data_id);
  }

  [[nodiscard]] std::size_t count_launched(dataid_t data_id,
                                           devid_t device_id) const {
    return device_counts[device_id].count_launched(data_id);
  }

  [[nodiscard]] std::size_t count_moving(dataid_t data_id,
                                         devid_t device_id) const {
    return device_counts[device_id].count_moving(data_id);
  }
};

class DataManager {
protected:
  Data &data;
  DeviceManager &device_manager;
  CommunicationManager &communication_manager;
  LocationManager mapped_locations;
  LocationManager reserved_locations;
  LocationManager launched_locations;
  MovementManager movement_manager;
  DataCounts counts;

  static bool check_valid(const DataIDList &list,
                          const LocationManager &locations, devid_t device_id) {
    return std::ranges::all_of(list, [&](auto data_id) {
      return !locations.is_invalid(data_id, device_id);
    });
  }

  static bool read_update(dataid_t data_id, devid_t device_id,
                          LocationManager &locations) {
    return locations[data_id].validate(device_id);
  }

  static auto write_update(dataid_t data_id, devid_t device_id,
                           LocationManager &locations) {
    auto updated_ids = locations[data_id].invalidate_except(device_id);
    return updated_ids;
  }

public:
  DataManager(Data &data_, DeviceManager &device_manager_,
              CommunicationManager &communication_manager_)
      : data(data_), device_manager(device_manager_),
        communication_manager(communication_manager_),
        mapped_locations(data.size(), device_manager_.size()),
        reserved_locations(data.size(), device_manager_.size()),
        launched_locations(data.size(), device_manager_.size()),
        counts(device_manager_.size()) {}

  void initialize() {
    for (dataid_t i = 0; i < data.size(); i++) {
      auto initial_location = data.get_location(i);
      mapped_locations.set_valid(i, initial_location);
      reserved_locations.set_valid(i, initial_location);
      launched_locations.set_valid(i, initial_location);
      device_manager.add_mem<TaskState::MAPPED>(initial_location,
                                                data.get_size(i));
      device_manager.add_mem<TaskState::RESERVED>(initial_location,
                                                  data.get_size(i));
      device_manager.add_mem<TaskState::LAUNCHED>(initial_location,
                                                  data.get_size(i));
    }
  }

  [[nodiscard]] const Data &get_data() const { return data; }

  [[nodiscard]] const LocationManager &get_mapped_locations() const {
    return mapped_locations;
  }

  [[nodiscard]] const LocationManager &get_reserved_locations() const {
    return reserved_locations;
  }

  [[nodiscard]] const LocationManager &get_launched_locations() const {
    return launched_locations;
  }

  [[nodiscard]] const DataCounts &get_counts() const { return counts; }

  bool check_valid_mapped(const DataIDList &list, devid_t device_id) const {
    return check_valid(list, mapped_locations, device_id);
  }

  bool check_valid_reserved(const DataIDList &list, devid_t device_id) const {
    return check_valid(list, reserved_locations, device_id);
  }

  bool check_valid_launched(const DataIDList &list, devid_t device_id) const {
    return check_valid(list, launched_locations, device_id);
  }

  [[nodiscard]] mem_t local_size(const DataIDList &list,
                                 const LocationManager &locations,
                                 devid_t device_id) const {
    mem_t local_size = 0;
    for (auto data_id : list) {
      if (locations.is_valid(data_id, device_id)) {
        local_size += data.get_size(data_id);
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

  [[nodiscard]] mem_t non_local_size(const DataIDList &list,
                                     const LocationManager &locations,
                                     devid_t device_id) const {
    mem_t non_local_size = 0;
    for (auto data_id : list) {
      if (locations.is_invalid(data_id, device_id)) {
        non_local_size += data.get_size(data_id);
      }
    }
    return non_local_size;
  }

  mem_t non_local_size_mapped(const DataIDList &list, devid_t device_id) const {
    return non_local_size(list, mapped_locations, device_id);
  }

  mem_t non_local_size_reserved(const DataIDList &list,
                                devid_t device_id) const {
    return non_local_size(list, reserved_locations, device_id);
  }

  mem_t non_local_size_launched(const DataIDList &list,
                                devid_t device_id) const {
    return non_local_size(list, launched_locations, device_id);
  }

  mem_t shared_size(DataIDList &list1, DataIDList &list2) {
    mem_t shared_size = 0;
    for (auto data_id : list1) {
      if (std::find(list2.begin(), list2.end(), data_id) != list2.end()) {
        shared_size += data.get_size(data_id);
      }
    }
    return shared_size;
  }

  void read_update_mapped(const DataIDList &list, devid_t device_id) {
    for (auto data_id : list) {
      read_update(data_id, device_id, mapped_locations);
    }
    // Memory change is handled by task request in mapper
  }

  void write_update_mapped(const DataIDList &list, devid_t device_id) {
    for (auto data_id : list) {
      write_update(data_id, device_id, mapped_locations);
    }
    // Memory change is handled by task complete
  }

  void read_update_reserved(const DataIDList &list, devid_t device_id) {
    for (auto data_id : list) {
      read_update(data_id, device_id, reserved_locations);
    }
    // Memory change is handeled by task request in reserver
  }

  void write_update_reserved(const DataIDList &list, devid_t device_id) {
    for (auto data_id : list) {
      write_update(data_id, device_id, reserved_locations);
    }
    // Memory change is handled by task complete
  }

  void add_memory(dataid_t data_id, devid_t device_id) {
    device_manager.add_mem<TaskState::LAUNCHED>(device_id,
                                                data.get_size(data_id));
  }

  void read_update_launched(const DataIDList &list, devid_t device_id) {
    for (auto data_id : list) {
      bool changed = read_update(data_id, device_id, launched_locations);
      if (changed) {
        add_memory(data_id, device_id);
      }
    }
  }

  SourceRequest request_source(dataid_t data_id, devid_t destination) {
    auto valid_locations = launched_locations.get_valid_locations(data_id);
    std::cout << "Valid locations: " << valid_locations.size() << std::endl;
    for (auto loc : valid_locations) {
      std::cout << loc << " ";
    }
    std::cout << std::endl;
    SourceRequest req = communication_manager.get_best_available_source(
        destination, valid_locations);

    if (!req.found) {
      return req;
    }

    communication_manager.increase_active_links(req.source, destination);

    return req;
  }

  timecount_t start_move(dataid_t data_id, devid_t source,
                         devid_t destination) {
    assert(launched_locations.is_valid(data_id, source));

    bool is_moving = movement_manager.is_moving(data_id, destination);
    if (is_moving) {
      return movement_manager.get_time(data_id, destination);
    }

    if (launched_locations.is_valid(data_id, destination)) {
      return 0;
    }

    add_memory(data_id, destination);

    timecount_t completion_time = communication_manager.ideal_time_to_transfer(
        data.get_size(data_id), source, destination);

    movement_manager.set_completion(data_id, destination, completion_time);

    return completion_time;
  }

  void complete_move(dataid_t data_id, devid_t source, devid_t destination,
                     bool existed) {
    if (!existed) {
      // NOTE(wlr): I'm not 100% sure about the source check
      // Could something that starts at the same time as the move completes be a
      // problem?
      assert(launched_locations.is_valid(data_id, source));
      assert(launched_locations.is_valid(data_id, destination));
      return;
    }

    assert(movement_manager.is_moving(data_id, destination));
    launched_locations.set_valid(data_id, destination);
    movement_manager.remove(data_id, destination);

    communication_manager.decrease_active_links(source, destination);
  }

  void remove_memory(const DeviceIDList &device_list, devid_t device_id) {
    for (auto device : device_list) {
      auto size = data.get_size(device);
      device_manager.remove_mem<TaskState::MAPPED>(device_id, size);
      device_manager.remove_mem<TaskState::RESERVED>(device_id, size);
      device_manager.remove_mem<TaskState::LAUNCHED>(device_id, size);
    }
  }

  void write_update_launched(const DataIDList &list, devid_t device_id) {
    for (auto data_id : list) {
      auto updated_devices =
          write_update(data_id, device_id, launched_locations);
      remove_memory(updated_devices, device_id);
    }
  }
};