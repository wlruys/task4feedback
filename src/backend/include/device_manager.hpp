#pragma once
#include "devices.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <cassert>

class DeviceManager;

class DeviceResources {
protected:
  void resize(std::size_t n) {
    vcu.resize(n);
    mem.resize(n);
    time.resize(n);
  }

public:
  std::vector<vcu_t> vcu;
  std::vector<mem_t> mem;
  std::vector<timecount_t> time;

  DeviceResources() = default;

  DeviceResources(std::size_t n) { resize(n); }

  void set_vcus(devid_t id, vcu_t vcu_) { vcu[id] = vcu_; }
  void set_mem(devid_t id, mem_t m) { mem[id] = m; }
  void set_time(devid_t id, timecount_t t) { time[id] = t; }

  [[nodiscard]] vcu_t get_vcus(devid_t id) const { return vcu[id]; }
  [[nodiscard]] mem_t get_mem(devid_t id) const { return mem[id]; }
  [[nodiscard]] timecount_t get_time(devid_t id) const { return time[id]; }

  void add_vcus(devid_t id, vcu_t vcu_) { vcu[id] += vcu_; }
  void add_mem(devid_t id, mem_t m) { mem[id] += m; }
  void add_time(devid_t id, timecount_t t) { time[id] += t; }

  void sub_vcus(devid_t id, vcu_t vcu_) { vcu[id] -= vcu_; }
  void sub_mem(devid_t id, mem_t m) { mem[id] -= m; }
  void sub_time(devid_t id, timecount_t t) { time[id] -= t; }

  void add_resources(devid_t id, Resources &r) {
    add_vcus(id, r.vcu);
    add_mem(id, r.mem);
  }

  void sub_resources(devid_t id, Resources &r) {
    sub_vcus(id, r.vcu);
    sub_mem(id, r.mem);
  }

  [[nodiscard]] vcu_t overflow_vcus(devid_t id, vcu_t query, vcu_t max) const {
    vcu_t overflow = vcu[id] + query - max;
    return overflow;
  }

  [[nodiscard]] mem_t overflow_mem(devid_t id, mem_t query, mem_t max) const {
    mem_t overflow = mem[id] + query - max;
    return overflow;
  }

  [[nodiscard]] bool fit_vcus(devid_t id, vcu_t query, vcu_t max) const {
    return vcu[id] + query <= max;
  }
  [[nodiscard]] bool fit_mem(devid_t id, mem_t query, mem_t max) const {
    return mem[id] + query <= max;
  }

  [[nodiscard]] bool fit_resources(devid_t id, Resources &r,
                                   Resources &max) const {
    return fit_vcus(id, r.vcu, max.vcu) && fit_mem(id, r.mem, max.mem);
  }

  Resources overflow_resources(devid_t id, Resources &r, Resources &max) const {
    vcu_t vcu_overflow = overflow_vcus(id, r.vcu, max.vcu);
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
  [[nodiscard]] std::string &get_name(devid_t id) { return device_names[id]; }

public:
  Devices() = default;
  Devices(std::size_t n_devices) { resize(n_devices); }

  [[nodiscard]] const Device &get_device(devid_t id) const {
    return devices[id];
  }
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

  [[nodiscard]] const Devices &get_devices() const { return devices; }

  template <TaskState State>
  [[nodiscard]] const DeviceResources &get_resources() const;
  template <TaskState State> DeviceResources &get_resources();
  template <TaskState State>
  [[nodiscard]] Resources get_resources(devid_t id) const;

  template <TaskState State> mem_t get_mem(devid_t id);
  template <TaskState State> mem_t add_mem(devid_t id, mem_t mem_);
  template <TaskState State> mem_t sub_mem(devid_t id, mem_t mem_);
  template <TaskState State>
  [[nodiscard]] bool can_fit_mem(devid_t id, mem_t mem_) const;
  template <TaskState State>
  [[nodiscard]] mem_t overflow_mem(devid_t id, mem_t mem_) const;

  template <TaskState State> vcu_t get_vcu(devid_t id);
  template <TaskState State> void sub_vcu(devid_t id, vcu_t vcu_);
  template <TaskState State> void add_vcu(devid_t id, vcu_t vcu_);
  template <TaskState State>
  [[nodiscard]] bool can_fit_vcu(devid_t id, vcu_t vcu_) const;
  template <TaskState>
  [[nodiscard]] vcu_t overflow_vcu(devid_t id, vcu_t vcu_) const;

  template <TaskState> void add_resources(devid_t id, const Resources &r);
  template <TaskState> void sub_resources(devid_t id, const Resources &r);
  template <TaskState> bool can_fit_resources(devid_t id, Resources &r) const;
  template <TaskState>
  [[nodiscard]] Resources overflow_resources(devid_t id,
                                             const Resources &r) const;
};