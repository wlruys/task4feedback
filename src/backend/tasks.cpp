uint8_t StaticTaskInfo::get_supported_devices_mask(taskid_t compute_task_id,
                                                   const Devices &devices) const {
  uint8_t mask = 0;
  const devid_t n_devices = devices.size();

  auto arch_mask = get_supported_architecture_mask(compute_task_id);
  SPDLOG_DEBUG("Getting supported devices mask for task {} with arch mask: {}", compute_task_id,
               arch_mask);
  for (devid_t i = 0; i < n_devices; ++i) {
    const auto arch = devices.get_type(i);
    uint8_t arch_type = static_cast<uint8_t>(arch);
    SPDLOG_DEBUG("Checking device {} with arch type {}", i, arch_type);
    if ((arch_mask & arch_type) != 0) {
      SPDLOG_DEBUG("Device {} is supported for task {}", i, compute_task_id);
      mask |= (1 << i);
    }
  }
  SPDLOG_DEBUG("Supported devices mask for task {}: {}", compute_task_id, static_cast<int>(mask));
  assert(mask != 0 && "No supported devices found for the task");
  return mask;
}