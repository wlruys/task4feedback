#pragma once
#include "resources.hpp"
#include <iostream>
#include <type_traits>

#define HOST_ID 0

enum class DeviceType : int8_t {
  NONE = -1,
  CPU = 0,
  GPU = 1
};
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

class Device {
public:
  Resources max_resources;
  devid_t id = 0;
  copy_t max_copy = 0;
  DeviceType arch = DeviceType::NONE;

  Device() = default;
  Device(devid_t id, DeviceType arch, copy_t max_copy, vcu_t vcu, mem_t mem)
      : id(id), arch(arch), max_resources(vcu, mem), max_copy(max_copy) {
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