#pragma once
#include "devices.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <cassert>

class DeviceManager;

class DeviceResources {
protected:
  void resize(std::size_t n) {
    vcu.resize(n, 0);
    mem.resize(n, 0);
  }

public:
  std::vector<vcu_t> vcu;
  std::vector<mem_t> mem;

  DeviceResources() = default;

  DeviceResources(std::size_t n) { resize(n); }

  void set_vcu(devid_t id, vcu_t vcu_) { vcu[id] = vcu_; }
  void set_mem(devid_t id, mem_t m) { mem[id] = m; }

  [[nodiscard]] vcu_t get_vcu(devid_t id) const { return vcu[id]; }
  [[nodiscard]] mem_t get_mem(devid_t id) const { return mem[id]; }

  vcu_t add_vcu(devid_t id, vcu_t vcu_) {
    vcu[id] += vcu_;
    return vcu[id];
  }
  mem_t add_mem(devid_t id, mem_t m) {
    mem[id] += m;
    return mem[id];
  }

  vcu_t remove_vcu(devid_t id, vcu_t vcu_) {
    vcu[id] -= vcu_;
    return vcu[id];
  }
  mem_t remove_mem(devid_t id, mem_t m) {
    mem[id] -= m;
    return mem[id];
  }

  Resources add_resources(devid_t id, const Resources &r) {
    add_vcu(id, r.vcu);
    add_mem(id, r.mem);
    return {vcu[id], mem[id]};
  }

  Resources remove_resources(devid_t id, const Resources &r) {
    remove_vcu(id, r.vcu);
    remove_mem(id, r.mem);
    return {vcu[id], mem[id]};
  }

  [[nodiscard]] vcu_t overflow_vcu(devid_t id, vcu_t query, vcu_t max) const {
    const vcu_t request = vcu[id] + query;
    if (request <= max) {
      return 0;
    }
    return request - max;
  }

  [[nodiscard]] mem_t overflow_mem(devid_t id, mem_t query, mem_t max) const {
    const mem_t request = mem[id] + query;
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

  [[nodiscard]] bool fit_resources(devid_t id, Resources &r,
                                   Resources &max) const {
    return fit_vcu(id, r.vcu, max.vcu) && fit_mem(id, r.mem, max.mem);
  }

  Resources overflow_resources(devid_t id, Resources &r, Resources &max) const {
    vcu_t vcu_overflow = overflow_vcu(id, r.vcu, max.vcu);
    mem_t mem_overflow = overflow_mem(id, r.mem, max.mem);
    return {vcu_overflow, mem_overflow};
  }

  friend class DeviceManager;
};

class Devices {

protected:
  std::vector<Device> devices;
  std::array<DeviceIDList, num_device_types> type_map;
  std::vector<std::string> device_names;

  void resize(std::size_t n_devices) {
    devices.resize(n_devices);
    device_names.resize(n_devices);
  }

  [[nodiscard]] Device &get_device(devid_t id) { return devices[id]; }

public:
  Devices() = default;
  Devices(std::size_t n_devices) { resize(n_devices); }

  [[nodiscard]] const Device &get_device(devid_t id) const {
    return devices[id];
  }

  [[nodiscard]] std::string &get_name(devid_t id) { return device_names[id]; }
  [[nodiscard]] const std::string &get_name(devid_t id) const {
    return device_names[id];
  }

  [[nodiscard]] std::size_t size() const { return devices.size(); }

  [[nodiscard]] const DeviceIDList &get_devices(DeviceType type) const {
    return type_map[static_cast<std::size_t>(type)];
  }

  [[nodiscard]] const Resources &get_max_resources(devid_t id) const {
    return devices[id].max_resources;
  }

  [[nodiscard]] DeviceType get_type(devid_t id) const {
    return devices[id].arch;
  }

  void create_device(devid_t id, std::string name, DeviceType arch, vcu_t vcu,
                     mem_t mem) {
    assert(id < devices.size());
    devices[id] = Device(id, arch, vcu, mem);
    type_map[static_cast<std::size_t>(arch)].push_back(id);
    device_names[id] = std::move(name);
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

  Devices &get_devices() { return devices; }

public:
  Devices &devices;

  DeviceResources mapped;
  DeviceResources reserved;
  DeviceResources launched;

  DeviceManager(Devices &devices_) : devices(devices_) {
    std::size_t n_devices = devices.size();
    resize(n_devices);
  };

  void initialize() {}

  [[nodiscard]] std::size_t size() const { return devices.size(); }

  [[nodiscard]] const Devices &get_devices() const { return devices; }

  template <TaskState State> const DeviceResources &get_resources() const {
    if constexpr (State == TaskState::MAPPED) {
      return mapped;
    } else if constexpr (State == TaskState::RESERVED) {
      return reserved;
    } else if constexpr (State == TaskState::LAUNCHED) {
      return launched;
    } else {
      static_assert(State == TaskState::COMPLETED,
                    "Invalid task state in get_resources()");
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
      static_assert(State == TaskState::COMPLETED,
                    "Invalid task state in get_resources()");
    }
  }

  template <TaskState State>
  [[nodiscard]] Resources get_resources(devid_t id) const {
    auto &resources = get_resources<State>();
    return {resources.get_vcu(id), resources.get_mem(id)};
  }

  template <TaskState State> mem_t get_mem(devid_t id) const {
    auto &resources = get_resources<State>();
    return resources.get_mem(id);
  }

  template <TaskState State> vcu_t get_vcu(devid_t id) const {
    auto &resources = get_resources<State>();
    return resources.get_vcu(id);
  }

  template <TaskState State> mem_t add_mem(devid_t id, mem_t mem_) {
    auto &resources = get_resources<State>();
    return resources.add_mem(id, mem_);
  }

  template <TaskState State> mem_t remove_mem(devid_t id, mem_t mem_) {
    auto &resources = get_resources<State>();
    return resources.remove_mem(id, mem_);
  }

  template <TaskState State> void remove_vcu(devid_t id, vcu_t vcu_) {
    auto &resources = get_resources<State>();
    resources.remove_vcu(id, vcu_);
  }

  template <TaskState State> void add_vcu(devid_t id, vcu_t vcu_) {
    auto &resources = get_resources<State>();
    resources.add_vcu(id, vcu_);
  }

  template <TaskState State> bool can_fit_mem(devid_t id, mem_t mem_) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get_max_resources(id);
    return state_resources.fit_mem(id, mem_, device_max_resources.mem);
  }

  template <TaskState State> bool can_fit_vcu(devid_t id, vcu_t vcu_) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get_max_resources(id);
    return state_resources.fit_vcu(id, vcu_, device_max_resources.vcu);
  }

  template <TaskState State> mem_t overflow_mem(devid_t id, mem_t mem_) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get_max_resources(id);
    return state_resources.overflow_mem(id, mem_, device_max_resources.mem);
  }

  template <TaskState State> vcu_t overflow_vcu(devid_t id, vcu_t vcu_) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get_max_resources(id);
    return state_resources.overflow_vcu(id, vcu_, device_max_resources.vcu);
  }

  template <TaskState State>
  void add_resources(devid_t id, const Resources &r) {
    auto &state_resources = get_resources<State>();
    state_resources.add_resources(id, r);
  }

  template <TaskState State>
  void remove_resources(devid_t id, const Resources &r) {
    auto &state_resources = get_resources<State>();
    state_resources.remove_resources(id, r);
  }

  template <TaskState State>
  Resources overflow_resources(devid_t id, const Resources &r) const {
    auto &state_resources = get_resources<State>();
    const auto &device_max_resources = devices.get_max_resources(id);
    return state_resources.overflow_resources(id, r, device_max_resources);
  }
};