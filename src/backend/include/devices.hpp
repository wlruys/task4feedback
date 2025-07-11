#pragma once
#include "macros.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <ankerl/unordered_dense.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <type_traits>
#include <unordered_map>

#define MAX_MEM std::numeric_limits<mem_t>::max()
#define HOST_ID 0

class Device {
public:
  Resources max_resources;
  devid_t id = 0;
  copy_t max_copy = 0;
  DeviceType arch = DeviceType::CPU;

  Device() = default;
  Device(devid_t id, DeviceType arch, copy_t max_copy, vcu_t vcu, mem_t mem)
      : id(id), max_resources(vcu, mem), max_copy(max_copy), arch(arch) {
  }

  [[nodiscard]] mem_t get_mem() const {
    return max_resources.mem;
  }
  [[nodiscard]] vcu_t get_vcu() const {
    return max_resources.vcu;
  }

  [[nodiscard]] copy_t get_max_copy() const {
    return max_copy;
  }
};

class DeviceManager;

template <typename T> struct ResourceEventArray {
  std::vector<timecount_t> times;
  std::vector<T> resources;

  void add_set(timecount_t time, T resource) {
    times.push_back(time);
    resources.push_back(resource);
  }

  void add_change(timecount_t time, T resource) {
    times.push_back(time);
    resources.push_back(resource);
  }

  [[nodiscard]] std::size_t size() const {
    return times.size();
  }

  [[nodiscard]] bool empty() const {
    return times.empty();
  }

  void clear() {
    times.clear();
    resources.clear();
  }

  [[nodiscard]] timecount_t get_time(std::size_t index) const {
    assert(index < times.size());
    return times[index];
  }

  [[nodiscard]] T get_resource(std::size_t index) const {
    assert(index < resources.size());
    return resources[index];
  }

  [[nodiscard]] T get_resource_at_time(timecount_t time) const {
    // Assume events are sorted access with binary search
    if (empty()) {
      return 0;
    }

    auto it = std::lower_bound(times.begin(), times.end(), time);
    if (it == times.end()) {
      return resources.back();
    }
    return resources[it - times.begin()];
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
  std::vector<mem_t> mem_max;

  std::vector<ResourceEventArray<vcu_t>> vcu_tracker;
  std::vector<ResourceEventArray<mem_t>> mem_tracker;
  bool record{false};

  DeviceResources() {};

  DeviceResources(devid_t n)
      : vcu(n, 0), mem(n, 0), mem_max(n, MAX_MEM), vcu_tracker(n), mem_tracker(n) {
  }

  void start_record() {
    record = true;
  }

  void stop_record(){
    record = false;
    for (auto &tracker : vcu_tracker) {
      tracker.clear();
    }
    for (auto &tracker : mem_tracker) {
      tracker.clear();
    }
  }

  DeviceResources(const DeviceResources &other) {
    vcu = other.vcu;
    mem = other.mem;
    vcu_tracker = other.vcu_tracker;
    mem_tracker = other.mem_tracker;
    mem_max = other.mem_max;
    assert(vcu.size() == mem.size());
  }

  DeviceResources &operator=(const DeviceResources &other) = default;

  void set_max_mem(devid_t id, mem_t m) {
    assert(id < mem_max.size());
    mem_max[id] = m;
  }

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
    return vcu[id];
  }
  [[nodiscard]] mem_t get_mem(devid_t id) const {
    assert(id < mem.size());
    return mem[id];
  }

  vcu_t add_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    assert(id < vcu.size());
    auto &v = vcu[id];
    v += vcu_;
    vcu_tracker[id].add_change(current_time, v);
    return v;
  }
  mem_t add_mem(devid_t id, mem_t m, timecount_t current_time) {
    assert(id < mem.size());
    auto &v = mem[id];
    v += m;
    mem_tracker[id].add_change(current_time, v);
    return v;
  }

  vcu_t remove_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    assert(id < vcu.size());
    assert(vcu[id] >= vcu_);
    auto &v = vcu[id];
    v -= vcu_;
    vcu_tracker[id].add_change(current_time, v);
    return v;
  }
  mem_t remove_mem(devid_t id, mem_t m, timecount_t current_time) {
    assert(id < mem.size());
    assert(mem[id] >= m);
    auto &v = mem[id];
    v -= m;
    mem_tracker[id].add_change(current_time, v);
    return v;
  }

  Resources add_resources(devid_t id, const Resources &r, timecount_t current_time) {
    add_vcu(id, r.vcu, current_time);
    add_mem(id, r.mem, current_time);
    return {vcu[id], mem[id]};
  }

  Resources remove_resources(devid_t id, const Resources &r, timecount_t current_time) {
    remove_vcu(id, r.vcu, current_time);
    remove_mem(id, r.mem, current_time);
    return {vcu[id], mem[id]};
  }

  [[nodiscard]] vcu_t overflow_vcu(devid_t id, vcu_t query) const {
    const vcu_t request = vcu[id] + query;
    if (request <= MAX_VCUS) {
      return 0;
    }
    return request - MAX_VCUS;
  }

  [[nodiscard]] mem_t overflow_mem(devid_t id, mem_t query) const {
    const mem_t request = mem[id] + query;
    const auto max = mem_max[id];
    if (request <= max) {
      return 0;
    }
    return request - max;
  }

  [[nodiscard]] bool fit_vcu(devid_t id, vcu_t query) const {
    return vcu[id] + query <= MAX_VCUS;
  }
  [[nodiscard]] bool fit_mem(devid_t id, mem_t query) const {
    return mem[id] + query <= mem_max[id];
  }

  [[nodiscard]] bool fit_resources(devid_t id, Resources &r) const {
    return fit_vcu(id, r.vcu) && fit_mem(id, r.mem);
  }

  Resources overflow_resources(devid_t id, Resources &r) const {
    vcu_t vcu_overflow = overflow_vcu(id, r.vcu);
    mem_t mem_overflow = overflow_mem(id, r.mem);
    return {vcu_overflow, mem_overflow};
  }

  vcu_t get_vcu_at_time(devid_t id, timecount_t time) const {
    return vcu_tracker[id].get_resource(time);
  }

  mem_t get_mem_at_time(devid_t id, timecount_t time) const {
    return mem_tracker[id].get_resource(time);
  }

  friend class DeviceManager;
};

class Devices {

protected:
  std::vector<Device> devices;
  std::array<DeviceIDList, num_device_types> type_map;
  std::vector<std::string> device_names;
  ankerl::unordered_dense::map<std::string, devid_t> device_name_map;
  ankerl::unordered_dense::map<devid_t, devid_t> global_to_local;

  void resize(devid_t n_devices) {
    devices.resize(n_devices);
    device_names.resize(n_devices);
  }

  [[nodiscard]] Device &get_device(devid_t id) {
    return devices[id];
  }

public:
  Devices() = default;
  Devices(devid_t n_devices) {
    resize(n_devices);
  }

  [[nodiscard]] const Device &get_device(devid_t id) const {
    return devices[id];
  }

  [[nodiscard]] std::string &get_name(devid_t id) {
    return device_names[id];
  }
  [[nodiscard]] const std::string &get_name(devid_t id) const {
    return device_names[id];
  }

  [[nodiscard]] std::size_t size() const {
    return devices.size();
  }

  [[nodiscard]] const DeviceIDList &get_devices(DeviceType type) const {
    return type_map[static_cast<std::size_t>(type)];
  }

  [[nodiscard]] const Resources &get_max_resources(devid_t id) const {
    return devices[id].max_resources;
  }

  [[nodiscard]] DeviceType get_type(devid_t id) const {
    return devices[id].arch;
  }

  [[nodiscard]] devid_t get_device_id(std::string name) const {
    return device_name_map.at(name);
  }

  [[nodiscard]] devid_t get_local_id(devid_t global_id) const {
    return global_to_local.at(global_id);
  }

  [[nodiscard]] devid_t get_global_id(DeviceType arch, devid_t local_id) const {
    const auto idx = __builtin_ctz(static_cast<uint8_t>(arch));
    assert(idx < type_map.size() && "Invalid device type index");
    assert(local_id < type_map[idx].size() && "Local ID out of bounds for device type");
    return type_map[idx][local_id];
  }

  void create_device(devid_t id, std::string name, DeviceType arch, copy_t max_copy, mem_t mem) {
    if (id >= devices.size()) {
      resize(id + 1);
    }

    assert(id < devices.size());
    devices[id] = Device(id, arch, max_copy, MAX_VCUS, mem);
    const auto idx = __builtin_ctz(static_cast<uint8_t>(arch));
    assert(idx < type_map.size() && "Invalid device type index");
    type_map[idx].push_back(id);

    device_name_map[name] = id;
    devid_t local_id = type_map[idx].size() - 1;
    global_to_local[id] = local_id;

    device_names[id] = std::move(name);
  }

  devid_t append_device(std::string name, DeviceType arch, copy_t max_copy, mem_t mem) {
    devid_t id = devices.size();
    create_device(id, std::move(name), arch, max_copy, mem);
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

public:
  DeviceResources mapped;
  DeviceResources reserved;
  DeviceResources launched;
  std::size_t n_devices{0};
  bool initialized = false;

  DeviceManager() = default;

  DeviceManager(const Devices &devices_)
      : mapped(devices_.size()), reserved(devices_.size()), launched(devices_.size()),
        n_devices{devices_.size()}{};

  DeviceManager(const DeviceManager &other) = default;

  DeviceManager &operator=(const DeviceManager &other) = default;

  void initialize(const Devices &devices_) {
    if (initialized) {
      SPDLOG_WARN("DeviceManager already initialized. Skipping re-initialization.");
      return;
    }
    initialized = true;
    for (devid_t id = 0; id < devices_.size(); ++id) {
      mapped.set_max_mem(id, devices_.get_max_resources(id).mem);
      reserved.set_max_mem(id, devices_.get_max_resources(id).mem);
      launched.set_max_mem(id, devices_.get_max_resources(id).mem);
    }
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
    return state_resources.fit_mem(id, mem_);
  }

  template <TaskState State> [[nodiscard]] bool can_fit_vcu(devid_t id, vcu_t vcu_) const {
    auto &state_resources = get_resources<State>();
    return state_resources.fit_vcu(id, vcu_);
  }

  template <TaskState State> [[nodiscard]] mem_t overflow_mem(devid_t id, mem_t mem_) const {
    auto &state_resources = get_resources<State>();
    return state_resources.overflow_mem(id, mem_);
  }

  template <TaskState State> [[nodiscard]] vcu_t overflow_vcu(devid_t id, vcu_t vcu_) const {
    auto &state_resources = get_resources<State>();
    return state_resources.overflow_vcu(id, vcu_);
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
    return state_resources.overflow_resources(id, r);
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