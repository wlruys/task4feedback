#include "include/tasks.hpp"

std::vector<DeviceType> ComputeTask::get_supported_architectures() const {
  std::vector<DeviceType> supported_architectures;
  for (std::size_t i = 0; i < num_device_types; i++) {
    if (variants[i].arch != DeviceType::NONE) {
      supported_architectures.push_back(static_cast<DeviceType>(i));
    }
  }
  return supported_architectures;
}