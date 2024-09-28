#pragma once
#include "devices.hpp"

class DeviceResources {
public:
  std::vector<vcu_t> vcu;
  std::vector<mem_t> mem;
  std::vector<timecount_t> time;

  DeviceResources() = default;

  DeviceResources(std::size_t n) {
    vcu.resize(n, 0);
    mem.resize(n, 0);
    time.resize(n, 0);
  }

  void set_vcus(devid_t id, vcu_t vcu) { vcu[id] = vcu; }
  void set_mem(devid_t id, mem_t m) { mem[id] = m; }
  void set_time(devid_t id, timecount_t t) { time[id] = t; }

  vcu_t get_vcus(devid_t id) { return vcu[id]; }
  mem_t get_mem(devid_t id) { return mem[id]; }
  timecount_t get_time(devid_t id) { return time[id]; }

  void add_vcus(devid_t id, vcu_t vcu) { vcu[id] += vcu; }
  void add_mem(devid_t id, mem_t m) { mem[id] += m; }
  void add_time(devid_t id, timecount_t t) { time[id] += t; }

  void sub_vcus(devid_t id, vcu_t vcu) { vcu[id] -= vcu; }
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

  bool fit_vcus(devid_t id, vcu_t query, vcu_t max) {
    return vcu[id] + query <= max;
  }
  bool fit_mem(devid_t id, mem_t query, mem_t max) {
    return mem[id] + query <= max;
  }

  bool fit_resources(devid_t id, Resources &r, Resources &max) {
    return fit_vcus(id, r.vcu, max.vcu) && fit_mem(id, r.mem, max.mem);
  }
};

class DeviceManager {
public:
  std::vector<Device> devices;
  std::array<std::vector<devid_t>, num_device_types> type_map;
  std::vector<std::string> device_names;

  DeviceResources mapped_resources;
  DeviceResources reserved_resources;
  DeviceResources launched_resources;

  DeviceManager() = default;

  void add_device(devid_t id, std::string name, DeviceType arch, vcu_t vcu,
                  mem_t mem) {
    devices[id] = Device{id, arch, vcu, mem};
    device_names.push_back(name);
    type_map[static_cast<size_t>(arch)].push_back(id);
  }

  void initialize_resources(std::size_t n) {
    mapped_resources = DeviceResources(n);
    reserved_resources = DeviceResources(n);
    launched_resources = DeviceResources(n);
  }

  Device &get_device(devid_t id) { return devices[id]; }
  auto &get_name(devid_t id) { return device_names[id]; }
};