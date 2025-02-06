#pragma once
#include "resources.hpp"
#include <iostream>
#include <type_traits>

enum class DeviceType {
  NONE = -1,
  CPU = 0,
  GPU = 1
};
constexpr std::size_t num_device_types = 2;

inline auto to_string(const DeviceType& arch) {
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

inline std::ostream& operator<<(std::ostream& os, const DeviceType& arch) {
  os << to_string(arch);
  return os;
}

class Device {
public:
  devid_t id = 0;
  DeviceType arch = DeviceType::NONE;
  Resources max_resources;

  Device() = default;
  Device(devid_t id, DeviceType arch, vcu_t vcu, mem_t mem)
      : id(id), arch(arch), max_resources(vcu, mem) {
  }
};