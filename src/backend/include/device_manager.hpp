#pragma once
#include "devices.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <cassert>
#include <functional>
#include <unordered_map>

class DeviceManager;

template <typename T> struct ResourceEventArray {
  std::vector<timecount_t> times;
  std::vector<T> resources;
  std::size_t size;
};

template <typename T> struct ResourceEvent {
  timecount_t time = 0;
  T resource = 0;
};

template <typename T> class ResourceTracker {
  std::vector<ResourceEvent<T>> events;

public:
  ResourceTracker() {
#ifdef SIM_TRACK_RESOURCES
    events.reserve(2000);
#endif // SIM_TRACK_RESOURCES
  }

  void add_set(timecount_t time, T resource) {
#ifdef SIM_TRACK_RESOURCES
    events.emplace_back(time, resource);
#endif // SIM_TRACK_RESOURCES
  }

  void add_change(timecount_t time, T resource) {
#ifdef SIM_TRACK_RESOURCES
    events.emplace_back(time, resource);
#endif // SIM_TRACK_RESOURCES
  }

  [[nodiscard]] T get_resource(timecount_t time) const {
    // Assume events are sorted access with binary search

#ifndef SIM_TRACK_RESOURCES
    return 0;
#endif // SIM_TRACK_RESOURCES

    if (events.empty()) {
      return 0;
    }

    auto it = std::lower_bound(events.begin(), events.end(), time,
                               [](const ResourceEvent<T> &e, timecount_t t) { return e.time < t; });
    if (it == events.end()) {
      return events.back().resource;
    }
    return it->resource;
  }

  void reset() {
    events.clear();
  }

  ResourceEventArray<T> get_events() const {
    // Returns a copy of the events
    ResourceEventArray<T> result;

    result.size = events.size();

    if (result.size == 0) {
      // If no events, return a single event with time 0 and resource 0
      result.times.resize(1);
      result.resources.resize(1);
      result.times[0] = 0;
      result.resources[0] = 0;
      return result;
    }

    result.times.resize(events.size());
    result.resources.resize(events.size());

    for (int i = 0; i < events.size(); i++) {
      result.times[i] = events[i].time;
      result.resources[i] = events[i].resource;
    }

    return result;
  }
};

class DeviceResources {
protected:
  void resize(std::size_t n) {
    vcu.resize(n, 0);
    mem.resize(n, 0);
    vcu_tracker.resize(n);
    mem_tracker.resize(n);
  }

public:
  std::vector<vcu_t> vcu;
  std::vector<mem_t> mem;

  std::vector<ResourceTracker<vcu_t>> vcu_tracker;
  std::vector<ResourceTracker<mem_t>> mem_tracker;

  DeviceResources(){};

  DeviceResources(std::size_t n) : vcu(n, 0), mem(n, 0), vcu_tracker(n), mem_tracker(n) {
  }

  DeviceResources(const DeviceResources &other) {
    vcu = other.vcu;
    mem = other.mem;
    vcu_tracker = other.vcu_tracker;
    mem_tracker = other.mem_tracker;
  }

  DeviceResources &operator=(const DeviceResources &other) = default;

  void set_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    vcu[id] = vcu_;
    vcu_tracker[id].add_set(current_time, vcu_);
  }
  void set_mem(devid_t id, mem_t m, timecount_t current_time) {
    mem[id] = m;
    mem_tracker[id].add_set(current_time, m);
  }

  [[nodiscard]] vcu_t get_vcu(devid_t id) const {
    assert(id < vcu.size());
    return vcu.at(id);
  }
  [[nodiscard]] mem_t get_mem(devid_t id) const {
    assert(id < mem.size());
    return mem.at(id);
  }

  vcu_t add_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    assert(id < vcu.size());
    vcu.at(id) += vcu_;
    vcu_tracker[id].add_change(current_time, vcu.at(id));
    return vcu.at(id);
  }
  mem_t add_mem(devid_t id, mem_t m, timecount_t current_time) {
    assert(id < mem.size());
    mem.at(id) += m;
    mem_tracker[id].add_change(current_time, mem.at(id));
    return mem.at(id);
  }

  vcu_t remove_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    assert(id < vcu.size());
    assert(vcu.at(id) >= vcu_);
    vcu.at(id) -= vcu_;
    vcu_tracker[id].add_change(current_time, vcu.at(id));
    return vcu.at(id);
  }
  mem_t remove_mem(devid_t id, mem_t m, timecount_t current_time) {
    assert(id < mem.size());
    assert(mem.at(id) >= m);
    mem.at(id) -= m;
    mem_tracker[id].add_change(current_time, mem.at(id));
    return mem.at(id);
  }

  Resources add_resources(devid_t id, const Resources &r, timecount_t current_time) {
    add_vcu(id, r.vcu, current_time);
    add_mem(id, r.mem, current_time);
    return {vcu.at(id), mem.at(id)};
  }

  Resources remove_resources(devid_t id, const Resources &r, timecount_t current_time) {
    remove_vcu(id, r.vcu, current_time);
    remove_mem(id, r.mem, current_time);
    return {vcu.at(id), mem.at(id)};
  }

  [[nodiscard]] vcu_t overflow_vcu(devid_t id, vcu_t query, vcu_t max) const {
    const vcu_t request = vcu.at(id) + query;
    if (request <= max) {
      return 0;
    }
    return request - max;
  }

  [[nodiscard]] mem_t overflow_mem(devid_t id, mem_t query, mem_t max) const {
    const mem_t request = mem.at(id) + query;
    if (request <= max) {
      return 0;
    }
    return request - max;
  }

  [[nodiscard]] bool fit_vcu(devid_t id, vcu_t query, vcu_t max) const {
    return vcu[id] + query <= max;
  }
  [[nodiscard]] bool fit_mem(devid_t id, mem_t query, mem_t max) const {
    return mem[id] + query <= max;
  }

  [[nodiscard]] bool fit_resources(devid_t id, Resources &r, Resources &max) const {
    return fit_vcu(id, r.vcu, max.vcu) && fit_mem(id, r.mem, max.mem);
  }

  Resources overflow_resources(devid_t id, Resources &r, Resources &max) const {
    vcu_t vcu_overflow = overflow_vcu(id, r.vcu, max.vcu);
    mem_t mem_overflow = overflow_mem(id, r.mem, max.mem);
    return {vcu_overflow, mem_overflow};
  }

  vcu_t get_vcu_at_time(devid_t id, timecount_t time) const {
    return vcu_tracker[id].get_resource(time);
  }

  mem_t get_mem_at_time(devid_t id, timecount_t time) const {
    return mem_tracker[id].get_resource(time);
  }

  ResourceEventArray<vcu_t> get_vcu_events(devid_t id) const {
    return vcu_tracker[id].get_events();
  }

  ResourceEventArray<mem_t> get_mem_events(devid_t id) const {
    return mem_tracker[id].get_events();
  }

  friend class DeviceManager;
};

class Devices {

protected:
  std::vector<Device> devices;
  std::array<DeviceIDList, num_device_types> type_map;
  std::vector<std::string> device_names;

  std::unordered_map<std::string, devid_t> device_name_map;
  std::unordered_map<devid_t, devid_t> global_to_local;

  void resize(std::size_t n_devices) {
    devices.resize(n_devices);
    device_names.resize(n_devices);
  }

  [[nodiscard]] Device &get_device(devid_t id) {
    return devices[id];
  }

public:
  Devices() = default;
  Devices(std::size_t n_devices) {
    resize(n_devices);
  }

  [[nodiscard]] const Device &get_device(devid_t id) const {
    return devices.at(id);
  }

  [[nodiscard]] std::string &get_name(devid_t id) {
    return device_names[id];
  }
  [[nodiscard]] const std::string &get_name(devid_t id) const {
    return device_names.at(id);
  }

  [[nodiscard]] std::size_t size() const {
    return devices.size();
  }

  [[nodiscard]] const DeviceIDList &get_devices(DeviceType type) const {
    return type_map.at(static_cast<std::size_t>(type));
  }

  [[nodiscard]] const Resources &get_max_resources(devid_t id) const {
    return devices.at(id).max_resources;
  }

  [[nodiscard]] DeviceType get_type(devid_t id) const {
    return devices.at(id).arch;
  }

  [[nodiscard]] devid_t get_device_id(std::string name) const {
    return device_name_map.at(name);
  }

  [[nodiscard]] devid_t get_local_id(devid_t global_id) const {
    return global_to_local.at(global_id);
  }

  [[nodiscard]] devid_t get_global_id(DeviceType arch, devid_t local_id) const {
    return type_map.at(static_cast<std::size_t>(arch)).at(local_id);
  }

  void create_device(devid_t id, std::string name, DeviceType arch, vcu_t vcu, mem_t mem) {
    if (id >= devices.size()) {
      resize(id + 1);
    }

    assert(id < devices.size());
    devices.at(id) = Device(id, arch, vcu, mem);
    type_map.at(static_cast<std::size_t>(arch)).push_back(id);

    device_name_map[name] = id;
    devid_t local_id = type_map.at(static_cast<std::size_t>(arch)).size() - 1;
    global_to_local[id] = local_id;

    device_names.at(id) = std::move(name);
  }

  devid_t append_device(std::string name, DeviceType arch, vcu_t vcu, mem_t mem) {
    devid_t id = devices.size();
    create_device(id, std::move(name), arch, vcu, mem);
    return id;
  }

  friend class DeviceManager;
};

class DeviceManager {
protected:
  void resize(std::size_t n_devices) {
    mapped.resize(n_devices);
    reserved.resize(n_devices);
    launched.resize(n_devices);
  }

  [[nodiscard]] Devices &get_devices() {
    return devices.get();
  }

public:
  std::reference_wrapper<Devices> devices;

  DeviceResources mapped;
  DeviceResources reserved;
  DeviceResources launched;

  DeviceManager(Devices &devices_)
      : devices(devices_), mapped(devices_.size()), reserved(devices_.size()),
        launched(devices_.size()){};

  DeviceManager(const DeviceManager &other) = default;

  DeviceManager &operator=(const DeviceManager &other) = default;

  void initialize() {
  }

  [[nodiscard]] std::size_t size() const {
    return devices.get().size();
  }

  [[nodiscard]] const Devices &get_devices() const {
    return devices;
  }

  template <TaskState State> [[nodiscard]] const DeviceResources &get_resources() const {
    if constexpr (State == TaskState::MAPPED) {
      return mapped;
    } else if constexpr (State == TaskState::RESERVED) {
      return reserved;
    } else if constexpr (State == TaskState::LAUNCHED) {
      return launched;
    } else {
      static_assert(State == TaskState::COMPLETED, "Invalid task state in get_resources()");
    }
  }

  template <TaskState State> DeviceResources &get_resources() {
    if constexpr (State == TaskState::MAPPED) {
      return mapped;
    } else if constexpr (State == TaskState::RESERVED) {
      return reserved;
    } else if constexpr (State == TaskState::LAUNCHED) {
      return launched;
    } else {
      static_assert(State == TaskState::COMPLETED, "Invalid task state in get_resources()");
    }
  }

  template <TaskState State> [[nodiscard]] Resources get_resources(devid_t id) const {
    auto &resources = get_resources<State>();
    return {resources.get_vcu(id), resources.get_mem(id)};
  }

  template <TaskState State> [[nodiscard]] mem_t get_mem(devid_t id) const {
    auto &resources = get_resources<State>();
    return resources.get_mem(id);
  }

  template <TaskState State> [[nodiscard]] vcu_t get_vcu(devid_t id) const {
    auto &resources = get_resources<State>();
    return resources.get_vcu(id);
  }

  template <TaskState State> mem_t add_mem(devid_t id, mem_t mem_, timecount_t current_time) {
    auto &resources = get_resources<State>();
    return resources.add_mem(id, mem_, current_time);
  }

  template <TaskState State> mem_t remove_mem(devid_t id, mem_t mem_, timecount_t current_time) {
    auto &resources = get_resources<State>();
    return resources.remove_mem(id, mem_, current_time);
  }

  template <TaskState State> void remove_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    auto &resources = get_resources<State>();
    resources.remove_vcu(id, vcu_, current_time);
  }

  template <TaskState State> void add_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    auto &resources = get_resources<State>();
    resources.add_vcu(id, vcu_, current_time);
  }

  template <TaskState State> [[nodiscard]] bool can_fit_mem(devid_t id, mem_t mem_) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get().get_max_resources(id);
    return state_resources.fit_mem(id, mem_, device_max_resources.mem);
  }

  template <TaskState State> [[nodiscard]] bool can_fit_vcu(devid_t id, vcu_t vcu_) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get().get_max_resources(id);
    return state_resources.fit_vcu(id, vcu_, device_max_resources.vcu);
  }

  template <TaskState State> [[nodiscard]] mem_t overflow_mem(devid_t id, mem_t mem_) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get().get_max_resources(id);
    return state_resources.overflow_mem(id, mem_, device_max_resources.mem);
  }

  template <TaskState State> [[nodiscard]] vcu_t overflow_vcu(devid_t id, vcu_t vcu_) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get().get_max_resources(id);
    return state_resources.overflow_vcu(id, vcu_, device_max_resources.vcu);
  }

  template <TaskState State>
  void add_resources(devid_t id, const Resources &r, timecount_t current_time) {
    auto &state_resources = get_resources<State>();
    state_resources.add_resources(id, r, current_time);
  }

  template <TaskState State>
  void remove_resources(devid_t id, const Resources &r, timecount_t current_time) {
    auto &state_resources = get_resources<State>();
    state_resources.remove_resources(id, r, current_time);
  }

  template <TaskState State>
  [[nodiscard]] Resources overflow_resources(devid_t id, const Resources &r) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get().get_max_resources(id);
    return state_resources.overflow_resources(id, r, device_max_resources);
  }

  template <TaskState State>
  [[nodiscard]] vcu_t get_vcu_at_time(devid_t id, timecount_t time) const {
    auto &state_resources = get_resources<State>();
    return state_resources.get_vcu_at_time(id, time);
  }

  template <TaskState State>
  [[nodiscard]] mem_t get_mem_at_time(devid_t id, timecount_t time) const {
    auto &state_resources = get_resources<State>();
    return state_resources.get_mem_at_time(id, time);
  }

  template <TaskState State> ResourceEventArray<vcu_t> get_vcu_events(devid_t id) const {
    auto &state_resources = get_resources<State>();
    return state_resources.get_vcu_events(id);
  }

  template <TaskState State> ResourceEventArray<mem_t> get_mem_events(devid_t id) const {
    auto &state_resources = get_resources<State>();
    return state_resources.get_mem_events(id);
  }

  friend class SchedulerState;
};