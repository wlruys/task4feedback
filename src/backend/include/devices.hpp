#pragma once
#include "macros.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <type_traits>
#include <unordered_map>

#define MAX_MEM std::numeric_limits<mem_t>::max()
#define HOST_ID 0

class Device {
public:
  Resources max_resources;
  devid_t id = 0;
  copy_t h2d_max_copy = 0;
  copy_t d2d_max_copy = 0;
  DeviceType arch = DeviceType::CPU;

  Device() = default;
  Device(devid_t id, DeviceType arch, copy_t h2d_max_copy, copy_t d2d_max_copy, vcu_t vcu,
         mem_t mem)
      : id(id), max_resources(vcu, mem), h2d_max_copy(h2d_max_copy), d2d_max_copy(d2d_max_copy),
        arch(arch) {
  }

  [[nodiscard]] mem_t get_mem() const {
    return max_resources.mem;
  }
  [[nodiscard]] vcu_t get_vcu() const {
    return max_resources.vcu;
  }

  [[nodiscard]] copy_t get_h2d_max_copy() const {
    return h2d_max_copy;
  }
  [[nodiscard]] copy_t get_d2d_max_copy() const {
    return d2d_max_copy;
  }
};

class DeviceManager;

template <typename T> struct ResourceEventArray {
  std::vector<timecount_t> times;
  std::vector<T> resources;

  void add_set(timecount_t time, T resource) {
    times.push_back(time);
    resources.push_back(resource);
  }

  void add_change(timecount_t time, T resource) {
    times.push_back(time);
    resources.push_back(resource);
  }

  [[nodiscard]] std::size_t size() const {
    return times.size();
  }

  [[nodiscard]] bool empty() const {
    return times.empty();
  }

  void clear() {
    times.clear();
    resources.clear();
  }

  [[nodiscard]] timecount_t get_time(std::size_t index) const {
    T4F_INVARIANT(index < times.size());
    return times[index];
  }

  [[nodiscard]] T get_resource(std::size_t index) const {
    T4F_INVARIANT(index < resources.size());
    return resources[index];
  }

  [[nodiscard]] T get_resource_at_time(timecount_t time) const {
    // Assume events are sorted access with binary search
    if (empty()) {
      return 0;
    }

    auto it = std::lower_bound(times.begin(), times.end(), time);
    if (it == times.end()) {
      return resources.back();
    }
    return resources[it - times.begin()];
  }
};

class DeviceResources {
protected:
  SoABuffer buf_;
  std::size_t n_{0};

  // __restrict__ tells the compiler none of these alias each other,
  // enabling reordering/CSE across same-type fields (vcu/vcu_peak and mem/mem_peak/mem_max
  // are all int64_t* so aliasing analysis would otherwise be conservative).
  void seat_pointers_impl(char *base,
                          std::size_t off_vcu, std::size_t off_mem,
                          std::size_t off_vcu_peak, std::size_t off_mem_peak,
                          std::size_t off_mem_max) noexcept {
    vcu      = soa_ptr_at<vcu_t, soa_hot_alignment_v<vcu_t>>(base, off_vcu);
    mem      = soa_ptr_at<mem_t, soa_hot_alignment_v<mem_t>>(base, off_mem);
    vcu_peak = soa_ptr_at<vcu_t, soa_hot_alignment_v<vcu_t>>(base, off_vcu_peak);
    mem_peak = soa_ptr_at<mem_t, soa_hot_alignment_v<mem_t>>(base, off_mem_peak);
    mem_max  = soa_ptr_at<mem_t, soa_hot_alignment_v<mem_t>>(base, off_mem_max);
  }

  void seat_pointers(std::size_t n) {
    SoALayout layout;
    layout.begin();
    auto off_vcu      = layout.add_hot_field<vcu_t>(n);
    auto off_mem      = layout.add_hot_field<mem_t>(n);
    auto off_vcu_peak = layout.add_hot_field<vcu_t>(n);
    auto off_mem_peak = layout.add_hot_field<mem_t>(n);
    auto off_mem_max  = layout.add_hot_field<mem_t>(n);
    seat_pointers_impl(buf_.base(), off_vcu, off_mem, off_vcu_peak, off_mem_peak, off_mem_max);
  }

  void resize(std::size_t n) {
    n_ = n;
    SoALayout layout;
    layout.begin();
    auto off_vcu      = layout.add_hot_field<vcu_t>(n);
    auto off_mem      = layout.add_hot_field<mem_t>(n);
    auto off_vcu_peak = layout.add_hot_field<vcu_t>(n);
    auto off_mem_peak = layout.add_hot_field<mem_t>(n);
    auto off_mem_max  = layout.add_hot_field<mem_t>(n);
    buf_ = SoABuffer::allocate(layout.total()); // zero-fills vcu, mem, vcu_peak, mem_peak
    seat_pointers_impl(buf_.base(), off_vcu, off_mem, off_vcu_peak, off_mem_peak, off_mem_max);
    std::fill(mem_max, mem_max + n, MAX_MEM);
    vcu_tracker.resize(n);
    mem_tracker.resize(n);
  }

  void reset_moved_from() noexcept {
    vcu = nullptr;
    mem = nullptr;
    vcu_peak = nullptr;
    mem_peak = nullptr;
    mem_max = nullptr;
    n_ = 0;
    record = false;
  }

public:
  vcu_t * __restrict__ vcu{nullptr};
  mem_t * __restrict__ mem{nullptr};
  vcu_t * __restrict__ vcu_peak{nullptr};
  mem_t * __restrict__ mem_peak{nullptr};
  mem_t * __restrict__ mem_max{nullptr};

  std::vector<ResourceEventArray<vcu_t>> vcu_tracker;
  std::vector<ResourceEventArray<mem_t>> mem_tracker;
  bool record{false};

  DeviceResources() = default;

  DeviceResources(devid_t n) {
    resize(static_cast<std::size_t>(n));
  }

  // Deep copy: single memcpy + pointer fixup.
  DeviceResources(const DeviceResources &other)
      : vcu_tracker(other.vcu_tracker), mem_tracker(other.mem_tracker),
        record(other.record), n_(other.n_) {
    if (n_ > 0) {
      buf_ = other.buf_.deep_copy();
      seat_pointers(n_);
    }
  }

  DeviceResources &operator=(const DeviceResources &other) {
    if (this == &other) return *this;
    DeviceResources tmp(other);
    std::swap(buf_, tmp.buf_);
    std::swap(vcu, tmp.vcu);
    std::swap(mem, tmp.mem);
    std::swap(vcu_peak, tmp.vcu_peak);
    std::swap(mem_peak, tmp.mem_peak);
    std::swap(mem_max, tmp.mem_max);
    std::swap(vcu_tracker, tmp.vcu_tracker);
    std::swap(mem_tracker, tmp.mem_tracker);
    std::swap(record, tmp.record);
    std::swap(n_, tmp.n_);
    return *this;
  }

  DeviceResources(DeviceResources &&other) noexcept
      : buf_(std::move(other.buf_)), n_(other.n_), vcu(other.vcu), mem(other.mem),
        vcu_peak(other.vcu_peak), mem_peak(other.mem_peak), mem_max(other.mem_max),
        vcu_tracker(std::move(other.vcu_tracker)), mem_tracker(std::move(other.mem_tracker)),
        record(other.record) {
    other.reset_moved_from();
  }

  DeviceResources &operator=(DeviceResources &&other) noexcept {
    if (this != &other) {
      buf_ = std::move(other.buf_);
      n_ = other.n_;
      vcu = other.vcu;
      mem = other.mem;
      vcu_peak = other.vcu_peak;
      mem_peak = other.mem_peak;
      mem_max = other.mem_max;
      vcu_tracker = std::move(other.vcu_tracker);
      mem_tracker = std::move(other.mem_tracker);
      record = other.record;
      other.reset_moved_from();
    }
    return *this;
  }

  void start_record() {
    record = true;
  }

  void stop_record() {
    record = false;
    for (auto &tracker : vcu_tracker) {
      tracker.clear();
    }
    for (auto &tracker : mem_tracker) {
      tracker.clear();
    }
  }

  void set_max_mem(devid_t id, mem_t m) {
    T4F_INVARIANT(id < n_);
    mem_max[id] = m;
  }

  void set_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    vcu[id] = vcu_;
    vcu_peak[id] = std::max(vcu_peak[id], vcu_);
    if (record) {
      vcu_tracker[id].add_set(current_time, vcu_);
    }
  }
  void set_mem(devid_t id, mem_t m, timecount_t current_time) {
    mem[id] = m;
    mem_peak[id] = std::max(mem_peak[id], m);
    if (record) {
      mem_tracker[id].add_set(current_time, m);
    }
  }

  [[nodiscard]] vcu_t get_vcu(devid_t id) const {
    T4F_INVARIANT(id < n_);
    return vcu[id];
  }
  [[nodiscard]] mem_t get_mem(devid_t id) const {
    T4F_INVARIANT(id < n_);
    return mem[id];
  }

  vcu_t add_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    T4F_INVARIANT(id < n_);
    auto &v = vcu[id];
    v += vcu_;
    vcu_peak[id] = std::max(vcu_peak[id], v);
    if (record) {
      vcu_tracker[id].add_change(current_time, v);
    }
    return v;
  }
  mem_t add_mem(devid_t id, mem_t m, timecount_t current_time) {
    T4F_INVARIANT(id < n_);
    auto &v = mem[id];
    v += m;
    mem_peak[id] = std::max(mem_peak[id], v);
    if (record) {
      mem_tracker[id].add_change(current_time, v);
    }
    return v;
  }

  vcu_t remove_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    T4F_INVARIANT(id < n_);
    T4F_INVARIANT(vcu[id] >= vcu_);
    auto &v = vcu[id];
    v -= vcu_;
    if (record) {
      vcu_tracker[id].add_change(current_time, v);
    }
    return v;
  }
  mem_t remove_mem(devid_t id, mem_t m, timecount_t current_time) {
    T4F_INVARIANT(id < n_);
    T4F_INVARIANT(mem[id] >= m);
    auto &v = mem[id];
    v -= m;
    if (record) {
      mem_tracker[id].add_change(current_time, v);
    }
    return v;
  }

  Resources add_resources(devid_t id, const Resources &r, timecount_t current_time) {
    add_vcu(id, r.vcu, current_time);
    add_mem(id, r.mem, current_time);
    return {vcu[id], mem[id]};
  }

  Resources remove_resources(devid_t id, const Resources &r, timecount_t current_time) {
    remove_vcu(id, r.vcu, current_time);
    remove_mem(id, r.mem, current_time);
    return {vcu[id], mem[id]};
  }

  [[nodiscard]] vcu_t overflow_vcu(devid_t id, vcu_t query) const {
    const vcu_t request = vcu[id] + query;
    if (request <= MAX_VCUS) {
      return 0;
    }
    return request - MAX_VCUS;
  }

  [[nodiscard]] mem_t overflow_mem(devid_t id, mem_t query) const {
    const mem_t request = mem[id] + query;
    const auto max = mem_max[id];
    if (request <= max) {
      return 0;
    }
    return request - max;
  }

  [[nodiscard]] bool fit_vcu(devid_t id, vcu_t query) const {
    return vcu[id] + query <= MAX_VCUS;
  }
  [[nodiscard]] bool fit_mem(devid_t id, mem_t query) const {
    return mem[id] + query <= mem_max[id];
  }

  [[nodiscard]] mem_t get_mem_peak(devid_t id) const {
    T4F_INVARIANT(id < n_);
    return mem_peak[id];
  }

  [[nodiscard]] bool fit_resources(devid_t id, Resources &r) const {
    return fit_vcu(id, r.vcu) && fit_mem(id, r.mem);
  }

  Resources overflow_resources(devid_t id, Resources &r) const {
    vcu_t vcu_overflow = overflow_vcu(id, r.vcu);
    mem_t mem_overflow = overflow_mem(id, r.mem);
    return {vcu_overflow, mem_overflow};
  }

  vcu_t get_vcu_at_time(devid_t id, timecount_t time) const {
    return vcu_tracker[id].get_resource(time);
  }

  mem_t get_mem_at_time(devid_t id, timecount_t time) const {
    return mem_tracker[id].get_resource(time);
  }

  friend class DeviceManager;
};

class Devices {

protected:
  std::vector<Device> devices;
  std::array<DeviceIDList, num_device_types> type_map;
  std::vector<std::string> device_names;
  ankerl::unordered_dense::map<std::string, devid_t> device_name_map;
  ankerl::unordered_dense::map<devid_t, devid_t> global_to_local;

  void resize(devid_t n_devices) {
    devices.resize(n_devices);
    device_names.resize(n_devices);
  }

  [[nodiscard]] Device &get_device(devid_t id) {
    return devices[id];
  }

public:
  Devices() = default;
  Devices(devid_t n_devices) {
    resize(n_devices);
  }

  [[nodiscard]] const Device &get_device(devid_t id) const {
    return devices[id];
  }

  [[nodiscard]] std::string &get_name(devid_t id) {
    return device_names[id];
  }
  [[nodiscard]] const std::string &get_name(devid_t id) const {
    return device_names[id];
  }

  [[nodiscard]] std::size_t size() const {
    return devices.size();
  }

  [[nodiscard]] const DeviceIDList &get_devices(DeviceType type) const {
    return type_map[static_cast<std::size_t>(type)];
  }

  [[nodiscard]] const Resources &get_max_resources(devid_t id) const {
    return devices[id].max_resources;
  }

  [[nodiscard]] DeviceType get_type(devid_t id) const {
    return devices[id].arch;
  }

  [[nodiscard]] devid_t get_device_id(std::string name) const {
    return device_name_map.at(name);
  }

  [[nodiscard]] devid_t get_local_id(devid_t global_id) const {
    return global_to_local.at(global_id);
  }

  [[nodiscard]] devid_t get_global_id(DeviceType arch, devid_t local_id) const {
    const auto idx = __builtin_ctz(static_cast<uint8_t>(arch));
    T4F_INVARIANT(idx < type_map.size() && "Invalid device type index");
    T4F_INVARIANT(local_id < type_map[idx].size() && "Local ID out of bounds for device type");
    return type_map[idx][local_id];
  }

  void create_device(devid_t id, std::string name, DeviceType arch, copy_t h2d_max_copy,
                     copy_t d2d_max_copy, mem_t mem) {
    if (id >= devices.size()) {
      resize(id + 1);
    }

    T4F_INVARIANT(id < devices.size());
    devices[id] = Device(id, arch, h2d_max_copy, d2d_max_copy, MAX_VCUS, mem);
    const auto idx = __builtin_ctz(static_cast<uint8_t>(arch));
    T4F_INVARIANT(idx < type_map.size() && "Invalid device type index");
    type_map[idx].push_back(id);

    device_name_map[name] = id;
    devid_t local_id = type_map[idx].size() - 1;
    global_to_local[id] = local_id;

    device_names[id] = std::move(name);
  }

  devid_t append_device(std::string name, DeviceType arch, copy_t h2d_max_copy, copy_t d2d_max_copy,
                        mem_t mem) {
    devid_t id = devices.size();
    create_device(id, std::move(name), arch, h2d_max_copy, d2d_max_copy, mem);
    return id;
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

public:
  DeviceResources mapped;
  DeviceResources reserved;
  DeviceResources launched;
  std::size_t n_devices{0};
  bool initialized = false;

  DeviceManager() = default;

  DeviceManager(const Devices &devices_)
      : mapped(devices_.size()), reserved(devices_.size()), launched(devices_.size()),
        n_devices{devices_.size()} {};

  DeviceManager(const DeviceManager &other) = default;

  DeviceManager &operator=(const DeviceManager &other) = default;

  void initialize(const Devices &devices_) {
    if (initialized) {
      SPDLOG_WARN("DeviceManager already initialized. Skipping re-initialization.");
      return;
    }
    initialized = true;
    for (devid_t id = 0; id < devices_.size(); ++id) {
      mapped.set_max_mem(id, devices_.get_max_resources(id).mem);
      reserved.set_max_mem(id, devices_.get_max_resources(id).mem);
      launched.set_max_mem(id, devices_.get_max_resources(id).mem);
    }
  }

  void start_record() {
    mapped.start_record();
    reserved.start_record();
    launched.start_record();
  }

  void stop_record() {
    mapped.stop_record();
    reserved.stop_record();
    launched.stop_record();
  }

  template <TaskState State> [[nodiscard]] const DeviceResources &get_resources() const {
    if constexpr (State == TaskState::MAPPED) {
      return mapped;
    } else if constexpr (State == TaskState::RESERVED) {
      return reserved;
    } else if constexpr (State == TaskState::LAUNCHED) {
      return launched;
    } else {
      static_assert(State == TaskState::COMPLETED, "Invalid task state in get_resources()");
    }
  }

  template <TaskState State> DeviceResources &get_resources() {
    if constexpr (State == TaskState::MAPPED) {
      return mapped;
    } else if constexpr (State == TaskState::RESERVED) {
      return reserved;
    } else if constexpr (State == TaskState::LAUNCHED) {
      return launched;
    } else {
      static_assert(State == TaskState::COMPLETED, "Invalid task state in get_resources()");
    }
  }

  template <TaskState State> [[nodiscard]] Resources get_resources(devid_t id) const {
    auto &resources = get_resources<State>();
    return {resources.get_vcu(id), resources.get_mem(id)};
  }

  template <TaskState State> [[nodiscard]] mem_t get_mem(devid_t id) const {
    auto &resources = get_resources<State>();
    return resources.get_mem(id);
  }

  template <TaskState State> [[nodiscard]] vcu_t get_vcu(devid_t id) const {
    auto &resources = get_resources<State>();
    return resources.get_vcu(id);
  }

  template <TaskState State> mem_t add_mem(devid_t id, mem_t mem_, timecount_t current_time) {
    auto &resources = get_resources<State>();
    return resources.add_mem(id, mem_, current_time);
  }

  template <TaskState State> mem_t remove_mem(devid_t id, mem_t mem_, timecount_t current_time) {
    auto &resources = get_resources<State>();
    return resources.remove_mem(id, mem_, current_time);
  }

  template <TaskState State> void remove_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    auto &resources = get_resources<State>();
    resources.remove_vcu(id, vcu_, current_time);
  }

  template <TaskState State> void add_vcu(devid_t id, vcu_t vcu_, timecount_t current_time) {
    auto &resources = get_resources<State>();
    resources.add_vcu(id, vcu_, current_time);
  }

  template <TaskState State> [[nodiscard]] bool can_fit_mem(devid_t id, mem_t mem_) const {
    auto &state_resources = get_resources<State>();
    return state_resources.fit_mem(id, mem_);
  }

  template <TaskState State> [[nodiscard]] bool can_fit_vcu(devid_t id, vcu_t vcu_) const {
    auto &state_resources = get_resources<State>();
    return state_resources.fit_vcu(id, vcu_);
  }

  template <TaskState State> [[nodiscard]] mem_t overflow_mem(devid_t id, mem_t mem_) const {
    auto &state_resources = get_resources<State>();
    return state_resources.overflow_mem(id, mem_);
  }

  template <TaskState State> [[nodiscard]] vcu_t overflow_vcu(devid_t id, vcu_t vcu_) const {
    auto &state_resources = get_resources<State>();
    return state_resources.overflow_vcu(id, vcu_);
  }

  template <TaskState State>
  void add_resources(devid_t id, const Resources &r, timecount_t current_time) {
    auto &state_resources = get_resources<State>();
    state_resources.add_resources(id, r, current_time);
  }

  template <TaskState State>
  void remove_resources(devid_t id, const Resources &r, timecount_t current_time) {
    auto &state_resources = get_resources<State>();
    state_resources.remove_resources(id, r, current_time);
  }

  template <TaskState State>
  [[nodiscard]] Resources overflow_resources(devid_t id, const Resources &r) const {
    auto &state_resources = get_resources<State>();
    return state_resources.overflow_resources(id, r);
  }

  template <TaskState State>
  [[nodiscard]] vcu_t get_vcu_at_time(devid_t id, timecount_t time) const {
    auto &state_resources = get_resources<State>();
    return state_resources.get_vcu_at_time(id, time);
  }

  template <TaskState State>
  [[nodiscard]] mem_t get_mem_at_time(devid_t id, timecount_t time) const {
    auto &state_resources = get_resources<State>();
    return state_resources.get_mem_at_time(id, time);
  }

  template <TaskState State> ResourceEventArray<vcu_t> get_vcu_events(devid_t id) const {
    auto &state_resources = get_resources<State>();
    return state_resources.get_vcu_events(id);
  }

  template <TaskState State> ResourceEventArray<mem_t> get_mem_events(devid_t id) const {
    auto &state_resources = get_resources<State>();
    return state_resources.get_mem_events(id);
  }

  friend class SchedulerState;
};
