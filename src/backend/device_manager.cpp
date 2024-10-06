#include "device_manager.hpp"

template <TaskState State>
const DeviceResources &DeviceManager::get_resources() const {
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

template <TaskState State> DeviceResources &DeviceManager::get_resources() {
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
Resources DeviceManager::get_resources(devid_t id) const {
  auto &resources = get_resources<State>();
  return {resources.get_vcus(id), resources.get_mem(id)};
}

template <TaskState State> mem_t DeviceManager::get_mem(devid_t id) {
  auto &resources = get_resources<State>();
  return resources.get_mem(id);
}

template <TaskState State> vcu_t DeviceManager::get_vcu(devid_t id) {
  auto &resources = get_resources<State>();
  return resources.get_vcu(id);
}

template <TaskState State>
mem_t DeviceManager::add_mem(devid_t id, mem_t mem_) {
  auto &resources = get_resources<State>();
  return resources.add_mem(id, mem_);
}

template <TaskState State>
mem_t DeviceManager::sub_mem(devid_t id, mem_t mem_) {
  auto &resources = get_resources<State>();
  return resources.sub_mem(id, mem_);
}

template <TaskState State> void DeviceManager::sub_vcu(devid_t id, vcu_t vcu_) {
  auto &resources = get_resources<State>();
  resources.sub_vcu(id, vcu_);
}

template <TaskState State> void DeviceManager::add_vcu(devid_t id, vcu_t vcu_) {
  auto &resources = get_resources<State>();
  resources.add_vcu(id, vcu_);
}

template <TaskState State>
bool DeviceManager::can_fit_mem(devid_t id, mem_t mem_) const {
  auto &state_resources = get_resources<State>();
  const auto &device_max_resources = devices.get_max_resources(id);
  return state_resources.fit_mem(id, mem_, device_max_resources.mem);
}

template <TaskState State>
bool DeviceManager::can_fit_vcu(devid_t id, vcu_t vcu_) const {
  auto &state_resources = get_resources<State>();
  const auto &device_max_resources = devices.get_max_resources(id);
  return state_resources.fit_vcu(id, vcu_, device_max_resources.vcu);
}

template <TaskState State>
mem_t DeviceManager::overflow_mem(devid_t id, mem_t mem_) const {
  auto &state_resources = get_resources<State>();
  const auto &device_max_resources = devices.get_max_resources(id);
  return state_resources.overflow_mem(id, mem_, device_max_resources.mem);
}

template <TaskState State>
vcu_t DeviceManager::overflow_vcu(devid_t id, vcu_t vcu_) const {
  auto &state_resources = get_resources<State>();
  const auto &device_max_resources = devices.get_max_resources(id);
  return state_resources.overflow_vcu(id, vcu_, device_max_resources.vcu);
}

template <TaskState State>
void DeviceManager::add_resources(devid_t id, const Resources &r) {
  auto &state_resources = get_resources<State>();
  state_resources.add_resources(id, r);
}

template <TaskState State>
void DeviceManager::sub_resources(devid_t id, const Resources &r) {
  auto &state_resources = get_resources<State>();
  state_resources.sub_resources(id, r);
}

template <TaskState State>
Resources DeviceManager::overflow_resources(devid_t id,
                                            const Resources &r) const {
  auto &state_resources = get_resources<State>();
  const auto &device_max_resources = devices.get_max_resources(id);
  return state_resources.overflow_resources(id, r, device_max_resources);
}
