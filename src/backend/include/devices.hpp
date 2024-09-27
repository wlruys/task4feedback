#pragma once
#include "macros.hpp"
#include <iostream>
#include <type_traits>

enum class DeviceType { NONE = -1, CPU = 0, GPU = 1 };
constexpr std::size_t num_device_types = 2;

inline std::ostream &operator<<(std::ostream &os, const DeviceType &arch) {
  switch (arch) {
  case DeviceType::NONE:
    os << "NONE";
    break;
  case DeviceType::CPU:
    os << "CPU";
    break;
  case DeviceType::GPU:
    os << "GPU";
    break;
  default:
    os << "UNKNOWN";
  }
  return os;
}
