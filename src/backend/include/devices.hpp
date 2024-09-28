#pragma once
#include "resources.hpp"
#include <iostream>
#include <type_traits>

enum class DeviceType { NONE = -1, CPU = 0, GPU = 1 };
constexpr std::size_t num_device_types = 2;

inline auto to_string(const DeviceType &arch) {
  switch (arch) {
  case DeviceType::NONE:
    return "NONE";
    break;
  case DeviceType::CPU:
    return "CPU";
    break;
  case DeviceType::GPU:
    return "GPU";
    break;
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const DeviceType &arch) {
  os << to_string(arch);
  return os;
}

struct Resources {
  vcu_t vcu = 0;
  mem_t mem = 0;
  timecount_t time = 0;

  Resources() = default;
  Resources(vcu_t vcu, mem_t mem) : vcu(vcu), mem(mem) {}
  Resources(vcu_t vcu, mem_t mem, timecount_t time)
      : vcu(vcu), mem(mem), time(time) {}
};

class Device : public Resources {
public:
  devid_t id;
  DeviceType arch = DeviceType::NONE;

  Device() = default;
  Device(devid_t id, DeviceType arch, vcu_t vcu, mem_t mem)
      : Resources(vcu, mem), id(id), arch(arch) {}
};