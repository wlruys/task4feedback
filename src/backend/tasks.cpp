#include "tasks.hpp"
#include "devices.hpp"
#include <limits>
#include <type_traits>

devicemask_t StaticTaskInfo::get_supported_devices_mask(taskid_t compute_task_id) const {
  using UMask = std::make_unsigned_t<devicemask_t>;
  // assumes exactly two DeviceType values:
  // CPU maps to device bit 0, and GPU maps to all remaining device bits.
  // If additional DeviceType values are added, this mapping must be revisited.
  constexpr UMask cpu_device_bit = UMask{1};
  constexpr UMask gpu_device_bits = std::numeric_limits<UMask>::max() & ~cpu_device_bit;

  const auto arch_mask = get_supported_architecture_mask(compute_task_id);
  SPDLOG_DEBUG("Getting supported devices mask for task {} with arch mask: {}", compute_task_id,
               arch_mask);

  UMask mask = 0;
  if ((arch_mask & static_cast<uint8_t>(DeviceType::CPU)) != 0) {
    mask |= cpu_device_bit;
  }
  if ((arch_mask & static_cast<uint8_t>(DeviceType::GPU)) != 0) {
    mask |= gpu_device_bits;
  }

  const auto device_mask = static_cast<devicemask_t>(mask);
  SPDLOG_DEBUG("Supported devices mask for task {}: {}", compute_task_id,
               static_cast<int>(device_mask));
  assert(device_mask != 0 && "No supported devices found for the task");
  return device_mask;
}
