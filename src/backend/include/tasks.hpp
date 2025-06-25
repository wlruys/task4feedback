#pragma once
#include "resources.hpp"
#include "settings.hpp"

#include "devices.hpp"
#include "spdlog/spdlog.h"
#include <ankerl/unordered_dense.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <set>
#include <span>
#include <string>
#include <tracy/Tracy.hpp>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define TASK_BUFFER_SIZE 20
#define EXPECTED_EVICTION_TASKS 1000

template <typename T> inline std::vector<T> as_vector(ankerl::unordered_dense::set<T> &set) {
  std::vector<T> vec;
  vec.reserve(set.size());
  for (const auto &item : set) {
    vec.push_back(item);
  }
  return vec;
}

template <typename K, typename V>
inline std::vector<T> as_vector(ankerl::unordered_dense::map<K, V> &map) {
  std::vector<T> vec;
  vec.reserve(map.size());
  for (const auto &item : map) {
    vec.push_back(item.second);
  }
  return vec;
}

enum class TaskType : int8_t {
  COMPUTE = 0,
  DATA = 1,
  EVICTION = 2,
};
constexpr std::size_t num_task_types = 3;

static constexpr uint8_t flags_for_state[5] = {
    /*SPAWNED  */ 1u,
    /*MAPPED   */ 2u,
    /*RESERVED */ 4u,
    /*LAUNCHED */ 8u,
    /*COMPLETED*/ 16u};

enum class TaskState : int8_t {
  SPAWNED = 0,
  MAPPED = 1,
  RESERVED = 2,
  LAUNCHED = 3,
  COMPLETED = 4
};
constexpr std::size_t num_task_states = 5;

inline std::string to_string(const TaskState &state) {
  switch (state) {
  case TaskState::SPAWNED:
    return "SPAWNED";
    break;
  case TaskState::MAPPED:
    return "MAPPED";
    break;
  case TaskState::RESERVED:
    return "RESERVED";
    break;
  case TaskState::LAUNCHED:
    return "LAUNCHED";
    break;
  case TaskState::COMPLETED:
    return "COMPLETED";
    break;
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const TaskState &state) {
  os << to_string(state);
  return os;
}

enum class TaskStatus : int8_t {
  NONE = -1,
  MAPPABLE = 0,
  RESERVABLE = 1,
  LAUNCHABLE = 2,
};
constexpr std::size_t num_task_statuses = 3;

inline std::string to_string(const TaskStatus &status) {
  switch (status) {
  case TaskStatus::MAPPABLE:
    return "MAPPABLE";
    break;
  case TaskStatus::RESERVABLE:
    return "RESERVABLE";
    break;
  case TaskStatus::LAUNCHABLE:
    return "LAUNCHABLE";
    break;
  case TaskStatus::NONE:
  default:
    return "UNKNOWN";
  }
}

inline std::ostream &operator<<(std::ostream &os, const TaskStatus &state) {
  os << to_string(state);
  return os;
}

class Variant {
public:
  Resources resources;
  timecount_t time = 0;
  DeviceType arch = DeviceType::NONE;

  Variant() = default;
  Variant(DeviceType arch_, vcu_t vcu_, mem_t mem_, timecount_t time_)
      : resources(vcu_, mem_), time(time_), arch(arch_) {
  }

  [[nodiscard]] DeviceType get_arch() const {
    return arch;
  }

  [[nodiscard]] vcu_t get_vcus() const {
    return resources.vcu;
  }
  [[nodiscard]] mem_t get_mem() const {
    return resources.mem;
  }

  [[nodiscard]] const Resources &get_resources() const {
    return resources;
  }

  [[nodiscard]] timecount_t get_mean_duration() const {
    return time;
  }
};

using VariantList = std::array<Variant, num_device_types>;

struct alignas(8) ComputeTaskStaticInfo {
  int32_t tag{};
  int32_t type{};
}

struct ComputeTaskVariantInfo {
  int8_t mask = 0; // bitmask for supported architectures
  std::array<Variant, num_device_types> variants{};
};

struct alignas(32) ComputeTaskDepInfo {
  int32_t s_dependencies;
  int32_t e_dependencies;
  int32_t s_dependents;
  int32_t e_dependents;
  int32_t s_data_dependencies;
  int32_t e_data_dependencies;
  int32_t s_data_dependents;
  int32_t e_data_dependents;
}

struct alignas(32) ComputeTaskDataInfo {
  int32_t s_read{};
  int32_t e_read{};
  int32_t s_write{};
  int32_t e_write{};
  int32_t s_retire{};
  int32_t e_retire{};
  int32_t s_unique{};
  int32_t e_unique{};
}

struct alignas(32) DataTaskStaticInfo {
  int32_t s_dependencies{};
  int32_t e_dependencies{};
  int32_t s_dependents{};
  int32_t e_dependents{};
  int32_t data_id{};
  int32_t compute_task{};
  int64_t pad{}; // padding to align to 32 bytes
};

struct alignas(32) ComputeTaskRuntimeInfo {
  int8_t state{};
  int8_t flags{};
  int16_t unmapped{};
  int16_t unreserved{};
  int16_t incomplete{};
  int32_t mapped_device{-1};
  int32_t reserve_priority{};
  int32_t launch_priority{};
  int32_t depth{};
};

struct alignas(16) DataTaskRuntimeInfo {
  int8_t state{};
  int8_t flags{};
  int16_t incomplete{};
  int32_t source_device{};
  int32_t mapped_device{-1};
  int32_t launch_priority{};
};

struct alignas(16) EvictionTaskRuntimeInfo {
  int8_t state{};
  int8_t flags{};
  int16_t pad{}; // padding to align to 32 bytes
  int32_t data_id{};
  int32_t evicting_on{};
  int32_t backup_to{};
}

struct alignas(32) TaskTimeRecord {
  timecount_t mapped_time{};
  timecount_t reserved_time{};
  timecount_t launched_time{};
  timecount_t completed_time{};
};

class StaticTaskInfo {

protected:
  std::vector<ComputeTaskDepInfo> compute_task_dep_info;
  std::vector<ComputeTaskDataInfo> compute_task_data_info;
  std::vector<ComputeTaskVariantInfo> compute_task_variant_info;
  std::vector<DataTaskStaticInfo> data_task_static_info;
  std::vector<ComputeTaskStaticInfo> compute_task_static_info;

  std::vector<taskid_t> compute_task_dependencies;
  std::vector<taskid_t> compute_task_dependents;
  std::vector<taskid_t> compute_task_data_dependencies;
  std::vector<taskid_t> compute_task_data_dependents;
  std::vector<dataid_t> compute_task_read;
  std::vector<dataid_t> compute_task_write;
  std::vector<dataid_t> compute_task_retire;
  std::vector<dataid_t> compute_task_recent_writers;
  std::vector<dataid_t> compute_task_unique;
  std::vector<taskid_t> compute_task_most_recent_writers;
  std::vector<taskid_t> data_task_dependencies;
  std::vector<taskid_t> data_task_dependents;

  std::vector<std::string> compute_task_names;
  std::vector<std::string> data_task_names;

public:
  StaticTaskInfo() = default;

  StaticTaskInfo(int32_t num_compute_tasks, int32_t num_data_tasks) {
    compute_task_dep_info.resize(num_compute_tasks);
    compute_task_data_info.resize(num_compute_tasks);
    compute_task_variant_info.resize(num_compute_tasks);
    compute_task_recent_writers.resize(num_compute_tasks);
    data_task_static_info.resize(num_data_tasks);
    compute_task_static_info.resize(num_compute_tasks);

    compute_task_names.resize(num_compute_tasks);
    data_task_names.resize(num_data_tasks);
  }

  // Creation and Initialization

  void add_compute_task(taskid_t id, const std::string &name, const ComputeTaskDepInfo &dep_info,
                        const ComputeTaskDataInfo &data_info,
                        const ComputeTaskStaticInfo &compute_info) {
    compute_task_dep_info[id] = dep_info;
    compute_task_data_info[id] = data_info;
    compute_task_static_info[id] = compute_info;
    compute_task_names[id] = name;
  }

  void add_data_task(taskid_t id, const std::string &name, const DataTaskStaticInfo &static_info) {
    data_task_static_info[id] = static_info;
    data_task_names[id] = name;
  }

  void add_compute_dependencies(taskid_t id, const std::vector<taskid_t> &dependencies) {
    auto &info = compute_task_dep_info[id];
    // copy dependencies to corresponding location
    auto s_idx = info.s_dependencies;
    auto e_idx = info.e_dependencies;
    std::copy(compute_task_dependencies.begin() + s_idx, compute_task_dependencies.begin() + e_idx,
              dependencies.begin());
  }

  void add_compute_dependents(taskid_t id, const std::vector<taskid_t> &dependents) {
    auto &info = compute_task_dep_info[id];
    // copy dependents to corresponding location
    auto s_idx = info.s_dependents;
    auto e_idx = info.e_dependents;
    std::copy(compute_task_dependents.begin() + s_idx, compute_task_dependents.begin() + e_idx,
              dependents.begin());
  }

  void add_compute_task_data_dependencies(taskid_t id, const std::vector<taskid_t> &dependencies) {
    auto &info = compute_task_dep_info[id];
    // copy dependencies to corresponding location
    auto s_idx = info.s_data_dependencies;
    auto e_idx = info.e_data_dependencies;
    std::copy(compute_task_data_dependencies.begin() + s_idx,
              compute_task_data_dependencies.begin() + e_idx, dependencies.begin());
  }

  void add_data_task_dependencies(taskid_t id, const std::vector<taskid_t> &dependencies) {
    auto &info = data_task_static_info[id];
    // copy dependencies to corresponding location
    auto s_idx = info.s_dependencies;
    auto e_idx = info.e_dependencies;
    std::copy(data_task_dependencies.begin() + s_idx, data_task_dependencies.begin() + e_idx,
              dependencies.begin());
  }

  void add_data_task_dependents(taskid_t id, const std::vector<taskid_t> &dependents) {
    auto &info = data_task_static_info[id];
    // copy dependents to corresponding location
    auto s_idx = info.s_dependents;
    auto e_idx = info.e_dependents;
    std::copy(data_task_dependents.begin() + s_idx, data_task_dependents.begin() + e_idx,
              dependents.begin());
  }

  void add_read(taskid_t id, const std::vector<dataid_t> &read) {
    auto &info = compute_task_data_info[id];
    // copy read data to corresponding location
    auto s_idx = info.s_read;
    auto e_idx = info.e_read;
    std::copy(compute_task_read.begin() + s_idx, compute_task_read.begin() + e_idx, read.begin());
  }

  void add_most_recent_writers(taskid_t id, const std::vector<taskid_t> &writers) {
    auto &info = compute_task_data_info[id];
    // copy most recent writers to corresponding location
    auto s_idx = info.s_read; // assuming s_read is used for recent writers
    auto e_idx = info.e_read; // assuming e_read is used for recent writers
    std::copy(compute_task_recent_writers.begin() + s_idx,
              compute_task_recent_writers.begin() + e_idx, writers.begin());
  }

  void add_write(taskid_t id, const std::vector<dataid_t> &write) {
    auto &info = compute_task_data_info[id];
    // copy write data to corresponding location
    auto s_idx = info.s_write;
    auto e_idx = info.e_write;
    std::copy(compute_task_write.begin() + s_idx, compute_task_write.begin() + e_idx,
              write.begin());
  }

  void add_retire(taskid_t id, const std::vector<dataid_t> &retire) {
    auto &info = compute_task_data_info[id];
    // copy retire data to corresponding location
    auto s_idx = info.s_retire;
    auto e_idx = info.e_retire;
    std::copy(compute_task_retire.begin() + s_idx, compute_task_retire.begin() + e_idx,
              retire.begin());
  }

  void add_unique(taskid_t id, const std::vector<dataid_t> &unique) {
    auto &info = compute_task_data_info[id];
    // copy unique data to corresponding location
    auto s_idx = info.s_unique;
    auto e_idx = info.e_unique;
    std::copy(compute_task_unique.begin() + s_idx, compute_task_unique.begin() + e_idx,
              unique.begin());
  }

  void add_compute_variant(taskid_t id, DeviceType arch, mem_t mem, vcu_t vcu, timecount_t time) {
    auto &info = compute_task_variant_info[id];
    int8_t arch_type = static_cast<int8_t>(arch);
    info.mask |= (1 << arch_type);
    info.variants[arch_type] = Variant(arch, vcu, mem, time);
  }

  // Getters

  [[nodiscard]] int32_t get_num_compute_tasks() const {
    return static_cast<int32_t>(compute_task_dep_info.size());
  }

  [[nodiscard]] int32_t get_num_data_tasks() const {
    return static_cast<int32_t>(data_task_static_info.size());
  }

  [[nodiscard]] int32_t get_num_tasks() const {
    return get_num_compute_tasks() + get_num_data_tasks();
  }

  [[nodiscard]] bool empty() const {
    return (compute_task_dep_info.empty() && data_task_static_info.empty());
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_dependencies(taskid_t id) const {
    auto &info = compute_task_dep_info[id];
    return {compute_task_dependencies.data() + info.s_dependencies,
            compute_task_dependencies.data() + info.e_dependencies};
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_dependents(taskid_t id) const {
    auto &info = compute_task_dep_info[id];
    return {compute_task_dependents.data() + info.s_dependents,
            compute_task_dependents.data() + info.e_dependents};
  }

  [[nodiscard]] std::span<const taskid_t> get_data_task_dependencies(taskid_t id) const {
    auto &info = data_task_static_info[id];
    return {data_task_dependencies.data() + info.s_dependencies,
            data_task_dependencies.data() + info.e_dependencies};
  }

  [[nodiscard]] std::span<const taskid_t> get_data_task_dependents(taskid_t id) const {
    auto &info = data_task_static_info[id];
    return {data_task_dependents.data() + info.s_dependents,
            data_task_dependents.data() + info.e_dependents};
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_data_dependencies(taskid_t id) const {
    auto &info = compute_task_dep_info[id];
    return {compute_task_data_dependencies.data() + info.s_data_dependencies,
            compute_task_data_dependencies.data() + info.e_data_dependencies};
  }

  [[nodiscard]] std::span<const taskid_t> get_compute_task_data_dependents(taskid_t id) const {
    auto &info = compute_task_dep_info[id];
    return {compute_task_data_dependents.data() + info.s_data_dependents,
            compute_task_data_dependents.data() + info.e_data_dependents};
  }

  [[nodiscard]] std::span<const dataid_t> get_read(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_read.data() + info.s_read, compute_task_read.data() + info.e_read};
  }

  [[nodiscard]] std::span<const dataid_t> get_write(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_write.data() + info.s_write, compute_task_write.data() + info.e_write};
  }

  [[nodiscard]] std::span<const dataid_t> get_retire(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_retire.data() + info.s_retire, compute_task_retire.data() + info.e_retire};
  }

  [[nodiscard]] std::span<const dataid_t> get_unique(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_unique.data() + info.s_unique, compute_task_unique.data() + info.e_unique};
  }

  [[nodiscard]] std::span<const taskid_t> get_most_recent_writers(taskid_t id) const {
    auto &info = compute_task_data_info[id];
    return {compute_task_recent_writers.data() + info.s_read,
            compute_task_recent_writers.data() + info.e_read};
  }

  [[nodiscard]] const VariantList &get_variants(taskid_t id) const {
    return compute_task_variant_info[id].variants;
  }

  [[nodiscard]] const Variant &get_variant(taskid_t id, DeviceType arch) const {
    return variants(id)[static_cast<std::size_t>(arch)];
  }

  [[nodiscard]] const Resources &get_compute_task_resources(taskid_t id, Devicetype arch) const {
    auto &info = compute_task_variant_info[id];
    int8_t arch_type = static_cast<int8_t>(arch);
    // assert that mask flag is set for the given architecture
    assert((info.mask & arch_type) != 0 && "Architecture not supported for this compute task");
    const auto &variant = info.variants[arch_type];
    return variant.get_resources();
  }

  [[nodiscard]] const timecount_t get_mean_duration(taskid_t id, DeviceType arch) const {
    auto &info = compute_task_variant_info[id];
    int8_t arch_type = static_cast<int8_t>(arch);
    // assert that mask flag is set for the given architecture
    assert((info.mask & arch_type) != 0 && "Architecture not supported for this compute task");
    const auto &variant = info.variants[arch_type];
    return variant.get_mean_duration();
  }

  [[nodiscard]] const int8_t get_compute_task_variant_mask(taskid_t id) const {
    return compute_task_variant_info[id].mask;
  }

  // TODO(wlr): Deprecate this to avoid allocation. Loop over mask direcly where this is used.
  [[nodiscard]] std::vector<Devicetype>
  get_compute_task_supported_architectures(taskid_t id) const {
    std::vector<Devicetype> supported_architectures;
    auto &info = compute_task_variant_info[id];
    for (int i = 0; i < num_device_types; ++i) {
      if ((info.mask & (1 << i)) != 0) {
        supported_architectures.push_back(static_cast<Devicetype>(i));
      }
    }
    return supported_architectures;
  }

  [[nodiscard]] bool is_compute_task_architecture_supported(taskid_t id, DeviceType arch) const {
    auto &info = compute_task_variant_info[id];
    int8_t arch_type = static_cast<int8_t>(arch);
    // assert that mask flag is set for the given architecture
    return (info.mask & arch_type) != 0;
  }

  [[nodiscard]] const std::string &get_compute_task_name(taskid_t id) const {
    return compute_task_names[id];
  }

  [[nodiscard]] const std::string &get_data_task_name(taskid_t id) const {
    return data_task_names[id];
  }

  [[nodiscard]] const taskid_t get_data_id(taskid_t id) const {
    return data_task_static_info[id].data_id;
  }

  [[nodiscard]] const taskid_t get_compute_task(taskid_t id) const {
    return data_task_static_info[id].compute_task;
  }
};

class RuntimeTaskInfo {
protected:
  std::vector<ComputeTaskRuntimeInfo> compute_task_runtime_info;
  std::vector<DataTaskRuntimeInfo> data_task_runtime_info;
  std::vector<EvictionTaskRuntimeInfo> eviction_task_runtime_info;

  std::vector<taskid_t> task_buffer;

  std::vector<TaskTimeRecord> compute_task_time_records;
  std::vector<TaskTimeRecord> data_task_time_records;
  std::vector<TaskTimeRecord> eviction_task_time_records;

  std::vector<taskid_t> eviction_task_names;

public:
  RuntimeTaskInfo() = default;

  RuntimeTaskInfo(int32_t num_compute_tasks, int32_t num_data_tasks)
      : compute_task_runtime_info(num_compute_tasks), data_task_runtime_info(num_data_tasks), ,
        compute_task_time_records(num_compute_tasks), data_task_time_records(num_data_tasks), {
    task_buffer.reserve(TASK_BUFFER_SIZE);
  }

  RuntimeTaskInfo(TaskStaticInfo &static_info) {
    int32_t num_compute_tasks = static_cast<int32_t>(static_info.get_num_compute_tasks());
    int32_t num_data_tasks = static_cast<int32_t>(static_info.get_num_data_tasks());
    compute_task_runtime_info.resize(num_compute_tasks);
    data_task_runtime_info.resize(num_data_tasks);
    compute_task_time_records.resize(num_compute_tasks);
    data_task_time_records.resize(num_data_tasks);

    for (int32_t i = 0; i < num_compute_tasks; ++i) {
      initialize_compute_runtime(i, static_info);
    }

    for (int32_t i = 0; i < num_data_tasks; ++i) {
      initialize_data_runtime(i, static_info);
    }
  }

  // Creation and Initialization

  void initialize_compute_runtime(int32_t compute_task_id, const TaskStaticInfo &static_info) {
    set_compute_task_state(compute_task_id, TaskState::SPAWNED);

    auto &dep_info = static_info.get_compute_task_dep_info(compute_task_id);
    auto n_dependencies = dep_info.e_dependencies - dep_info.s_dependencies;
    auto n_data_dependencies = dep_info.e_data_dependencies - dep_info.s_data_dependencies;
    set_compute_task_unmapped(compute_task_id, n_dependencies);
    set_compute_task_unreserved(compute_task_id, n_dependencies);
    set_compute_task_incomplete(compute_task_id, n_dependencies + n_data_dependencies);
  }

  void initialize_data_runtime(int32_t data_task_id, const TaskStaticInfo &static_info) {
    set_data_task_state(data_task_id, TaskState::SPAWNED);

    auto &static_info = static_info.get_data_task_static_info(data_task_id);
    auto n_dependencies = static_info.e_dependencies - static_info.s_dependencies;
    set_data_task_incomplete(data_task_id, n_dependencies);
  }

  int32_t add_eviction_task(const std::string &name, int32_t data_id, int32_t evicting_on,
                            int32_t backup_to) {
    taskid_t id = static_cast<taskid_t>(eviction_task_runtime_info.size());
    eviction_task_runtime_info.emplace_back();
    eviction_task_time_records.emplace_back();
    eviction_task_names.push_back(name);

    set_eviction_task_state(id, TaskState::SPAWNED);
    set_eviction_task_evicting_on(id, evicting_on);
    set_eviction_task_backup_to(id, backup_to);
    set_eviction_task_data_id(id, data_id);
    return id;
  }

  // Getters

  [[nodiscard]] const std::string get_eviction_task_name(taskid_t id) const {
    return eviction_task_names[id];
  }

  [[nodiscard]] const int16_t get_compute_task_unmapped(taskid_t id) const {
    return compute_task_runtime_info[id].unmapped;
  }

  [[nodiscard]] const int16_t get_compute_task_unreserved(taskid_t id) const {
    return compute_task_runtime_info[id].unreserved;
  }

  [[nodiscard]] const int16_t get_compute_task_incomplete(taskid_t id) const {
    return compute_task_runtime_info[id].incomplete;
  }

  [[nodiscard]] const int32_t get_compute_task_mapped_device(taskid_t id) const {
    return compute_task_runtime_info[id].mapped_device;
  }

  [[nodiscard]] const int32_t get_compute_task_reserve_priority(taskid_t id) const {
    return compute_task_runtime_info[id].reserve_priority;
  }

  [[nodiscard]] const int32_t get_compute_task_launch_priority(taskid_t id) const {
    return compute_task_runtime_info[id].launch_priority;
  }

  [[nodiscard]] const int8_t get_compute_task_state(taskid_t id) const {
    return compute_task_runtime_info[id].state;
  }

  [[nodiscard]] const int8_t get_compute_task_flags(taskid_t id) const {
    return compute_task_runtime_info[id].flags;
  }

  [[nodiscard]] const int32_t get_compute_task_depth(taskid_t id) const {
    return compute_task_runtime_info[id].depth;
  }

  [[nodiscard]] const int8_t get_data_task_state(taskid_t id) const {
    return data_task_runtime_info[id].state;
  }

  [[nodiscard]] const int8_t get_data_task_flags(taskid_t id) const {
    return data_task_runtime_info[id].flags;
  }

  [[nodiscard]] const int32_t get_data_task_source_device(taskid_t id) const {
    return data_task_runtime_info[id].source_device;
  }

  [[nodiscard]] const int32_t get_data_task_mapped_device(taskid_t id) const {
    return data_task_runtime_info[id].mapped_device;
  }

  [[nodiscard]] const int32_t get_data_task_launch_priority(taskid_t id) const {
    return data_task_runtime_info[id].launch_priority;
  }

  [[nodiscard]] const int8_t get_eviction_task_state(taskid_t id) const {
    return eviction_task_runtime_info[id].state;
  }

  [[nodiscard]] const int32_t get_eviction_task_evicting_on(taskid_t id) const {
    return eviction_task_runtime_info[id].evicting_on;
  }

  [[nodiscard]] const int32_t get_eviction_task_backup_to(taskid_t id) const {
    return eviction_task_runtime_info[id].backup_to;
  }

  [[nodiscard]] const timecount_t get_compute_task_mapped_time(taskid_t id) const {
    return compute_task_time_records[id].mapped_time;
  }

  [[nodiscard]] const timecount_t get_compute_task_reserved_time(taskid_t id) const {
    return compute_task_time_records[id].reserved_time;
  }

  [[nodiscard]] const timecount_t get_compute_task_launched_time(taskid_t id) const {
    return compute_task_time_records[id].launched_time;
  }

  [[nodiscard]] const timecount_t get_compute_task_completed_time(taskid_t id) const {
    return compute_task_time_records[id].completed_time;
  }

  [[nodiscard]] const timecount_t get_data_task_reserved_time(taskid_t id) const {
    return data_task_time_records[id].reserved_time;
  }

  [[nodiscard]] const timecount_t get_data_task_launched_time(taskid_t id) const {
    return data_task_time_records[id].launched_time;
  }

  [[nodiscard]] const timecount_t get_data_task_completed_time(taskid_t id) const {
    return data_task_time_records[id].completed_time;
  }

  [[nodiscard]] const timecount_t get_eviction_task_reserved_time(taskid_t id) const {
    return eviction_task_time_records[id].reserved_time;
  }

  [[nodiscard]] const timecount_t get_eviction_task_launched_time(taskid_t id) const {
    return eviction_task_time_records[id].launched_time;
  }

  [[nodiscard]] const timecount_t get_eviction_task_completed_time(taskid_t id) const {
    return eviction_task_time_records[id].completed_time;
  }

  [[nodiscard]] TaskState get_compute_task_state_at_time(taskid_t id, timecount_t query) const {
    if (query < compute_task_time_records[id].mapped_time) {
      return TaskState::SPAWNED;
    } else if (query < compute_task_time_records[id].reserved_time) {
      return TaskState::MAPPED;
    } else if (query < compute_task_time_records[id].launched_time) {
      return TaskState::RESERVED;
    } else if (query < compute_task_time_records[id].completed_time) {
      return TaskState::LAUNCHED;
    } else {
      return TaskState::COMPLETED;
    }
  }

  [[nodiscard]] TaskState get_data_task_state_at_time(taskid_t id, timecount_t query) const {
    if (query < data_task_time_records[id].reserved_time) {
      return TaskState::SPAWNED;
    } else if (query < data_task_time_records[id].launched_time) {
      return TaskState::MAPPED;
    } else if (query < data_task_time_records[id].completed_time) {
      return TaskState::RESERVED;
    } else {
      return TaskState::COMPLETED;
    }
  }

  [[nodiscard]] TaskState get_eviction_task_state_at_time(taskid_t id, timecount_t query) const {
    if (query < eviction_task_time_records[id].reserved_time) {
      return TaskState::SPAWNED;
    } else if (query < eviction_task_time_records[id].launched_time) {
      return TaskState::MAPPED;
    } else if (query < eviction_task_time_records[id].completed_time) {
      return TaskState::RESERVED;
    } else {
      return TaskState::COMPLETED;
    }
  }

  [[nodiscard]] const bool is_data_task_virtual(taskid_t id) const {
    // Virtual tasks have first bit of flags set to 1
    return (data_task_runtime_info[id].flags & 0x01) != 0;
  }

  [[nodiscard]] const bool is_eviction_task_virtual(taskid_t id) const {
    // Virtual tasks have first bit of flags set to 1
    return (eviction_task_runtime_info[id].flags & 0x01) != 0;
  }

  // TODO(wlr): Change status to flag for bit-wise comparison

  bool is_compute_mappable(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.unmapped == 0 && info.state >= static_cast<int8_t>(TaskState::SPAWNED);
  }

  bool is_compute_reservable(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.unreserved == 0 && info.state >= static_cast<int8_t>(TaskState::MAPPED);
  }

  bool is_compute_launchable(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.state >= static_cast<int8_t>(TaskState::RESERVED);
  }

  bool is_compute_completed(taskid_t id) const {
    auto &info = compute_task_runtime_info[id];
    return info.state >= static_cast<int8_t>(TaskState::COMPLETED);
  }

  bool is_data_launchable(taskid_t id) const {
    auto &info = data_task_runtime_info[id];
    return info.state >= static_cast<int8_t>(TaskState::RESERVED);
  }

  bool is_data_completed(taskid_t id) const {
    auto &info = data_task_runtime_info[id];
    return info.state >= static_cast<int8_t>(TaskState::COMPLETED);
  }

  bool is_eviction_launchable(taskid_t id) const {
    auto &info = eviction_task_runtime_info[id];
    return info.state >= static_cast<int8_t>(TaskState::RESERVED);
  }

  bool is_eviction_completed(taskid_t id) const {
    auto &info = eviction_task_runtime_info[id];
    return info.state >= static_cast<int8_t>(TaskState::COMPLETED);
  }

  TaskStatus get_compute_task_status(taskid_t id) const {
    if (is_compute_mappable(id)) {
      return TaskStatus::MAPPABLE;
    } else if (is_compute_reservable(id)) {
      return TaskStatus::RESERVABLE;
    } else if (is_compute_launchable(id)) {
      return TaskStatus::LAUNCHABLE;
    } else {
      return TaskStatus::NONE;
    }
  }

  TaskStatus get_data_task_status(taskid_t id) const {
    if (is_data_launchable(id)) {
      return TaskStatus::LAUNCHABLE;
    } else if (is_data_completed(id)) {
      return TaskStatus::NONE;
    } else {
      return TaskStatus::NONE;
    }
  }

  TaskStatus get_eviction_task_status(taskid_t id) const {
    if (is_eviction_launchable(id)) {
      return TaskStatus::LAUNCHABLE;
    } else if (is_eviction_completed(id)) {
      return TaskStatus::NONE;
    } else {
      return TaskStatus::NONE;
    }
  }

  // Non const grab fields

  [[nodiscard]] ComputeTaskRuntimeInfo &get_compute_task_runtime_info(taskid_t id) {
    return compute_task_runtime_info[id];
  }
  [[nodiscard]] DataTaskRuntimeInfo &get_data_task_runtime_info(taskid_t id) {
    return data_task_runtime_info[id];
  }
  [[nodiscard]] EvictionTaskRuntimeInfo &get_eviction_task_runtime_info(taskid_t id) {
    return eviction_task_runtime_info[id];
  }

  [[nodiscard]] TaskTimeRecord &get_compute_task_time_record(taskid_t id) {
    return compute_task_time_records[id];
  }
  [[nodiscard]] TaskTimeRecord &get_data_task_time_record(taskid_t id) {
    return data_task_time_records[id];
  }
  [[nodiscard]] TaskTimeRecord &get_eviction_task_time_record(taskid_t id) {
    return eviction_task_time_records[id];
  }

  // Setters

  void set_compute_task_state(taskid_t id, TaskState state) {
    compute_task_runtime_info[id].state = static_cast<int8_t>(state);
  }

  void set_data_task_state(taskid_t id, TaskState state) {
    data_task_runtime_info[id].state = static_cast<int8_t>(state);
  }

  void set_eviction_task_state(taskid_t id, TaskState state) {
    eviction_task_runtime_info[id].state = static_cast<int8_t>(state);
  }

  void set_compute_task_unmapped(taskid_t id, int16_t unmapped) {
    compute_task_runtime_info[id].unmapped = unmapped;
  }
  void set_compute_task_unreserved(taskid_t id, int16_t unreserved) {
    compute_task_runtime_info[id].unreserved = unreserved;
  }
  void set_compute_task_incomplete(taskid_t id, int16_t incomplete) {
    compute_task_runtime_info[id].incomplete = incomplete;
  }
  void set_compute_task_mapped_device(taskid_t id, int32_t mapped_device) {
    compute_task_runtime_info[id].mapped_device = mapped_device;
  }
  void set_compute_task_reserve_priority(taskid_t id, int32_t reserve_priority) {
    compute_task_runtime_info[id].reserve_priority = reserve_priority;
  }
  void set_compute_task_launch_priority(taskid_t id, int32_t launch_priority) {
    compute_task_runtime_info[id].launch_priority = launch_priority;
  }

  void set_compute_task_depth(taskid_t id, int32_t depth) {
    compute_task_runtime_info[id].depth = depth;
  }

  void set_compute_task_state(taskid_t id, int8_t state) {
    compute_task_runtime_info[id].state = state;
  }
  void set_compute_task_flags(taskid_t id, int8_t flags) {
    compute_task_runtime_info[id].flags = flags;
  }
  void set_data_task_state(taskid_t id, int8_t state) {
    data_task_runtime_info[id].state = state;
  }
  void set_data_task_virtual(taskid_t id, bool virtual_task) {
    data_task_runtime_info[id].flags = virtual_task ? (data_task_runtime_info[id].flags | 0x01)
                                                    : (data_task_runtime_info[id].flags & ~0x01);
  }
  void set_data_task_source_device(taskid_t id, int32_t source_device) {
    data_task_runtime_info[id].source_device = source_device;
  }
  void set_data_task_mapped_device(taskid_t id, int32_t mapped_device) {
    data_task_runtime_info[id].mapped_device = mapped_device;
  }
  void set_data_task_launch_priority(taskid_t id, int32_t launch_priority) {
    data_task_runtime_info[id].launch_priority = launch_priority;
  }
  void set_eviction_task_state(taskid_t id, int8_t state) {
    eviction_task_runtime_info[id].state = state;
  }
  void set_eviction_task_virtual(taskid_t id, bool virtual_task) {
    eviction_task_runtime_info[id].flags = virtual_task
                                               ? (eviction_task_runtime_info[id].flags | 0x01)
                                               : (eviction_task_runtime_info[id].flags & ~0x01);
  }
  void set_eviction_task_evicting_on(taskid_t id, int32_t evicting_on) {
    eviction_task_runtime_info[id].evicting_on = evicting_on;
  }
  void set_eviction_task_backup_to(taskid_t id, int32_t backup_to) {
    eviction_task_runtime_info[id].backup_to = backup_to;
  }

  void set_eviction_task_data_id(taskid_t id, int32_t data_id) {
    eviction_task_runtime_info[id].data_id = data_id;
  }

  void record_mapped(taskid_t id, timecount_t mapped_time) {
    task_time_records[id].mapped_time = mapped_time;
  }
  void record_reserved(taskid_t id, timecount_t reserved_time) {
    task_time_records[id].reserved_time = reserved_time;
  }
  void record_launched(taskid_t id, timecount_t launched_time) {
    task_time_records[id].launched_time = launched_time;
  }
  void record_completed(taskid_t id, timecount_t completed_time) {
    task_time_records[id].completed_time = completed_time;
  }

  // Task State modifiers

  bool decrement_compute_task_unmapped(taskid_t id) {
    auto &info = compute_task_runtime_info[id];
    info.unmapped--;
    return (info.unmapped == 0) && (info.state >= static_cast<int8_t>(TaskState::SPAWNED));
  }

  bool decrement_compute_task_unreserved(taskid_t id) {
    auto &info = compute_task_runtime_info[id];
    info.unreserved--;
    return (info.unreserved == 0) && (info.state >= static_cast<int8_t>(TaskState::MAPPED));
  }

  bool decrement_compute_task_incomplete(taskid_t id) {
    auto &info = compute_task_runtime_info[id];
    info.incomplete--;
    return (info.incomplete == 0) && (info.state >= static_cast<int8_t>(TaskState::RESERVED));
  }

  bool decrement_data_task_incomplete(taskid_t id) {
    auto &info = data_task_runtime_info[id];
    info.incomplete--;
    return (info.incomplete == 0) && (info.state >= static_cast<int8_t>(TaskState::RESERVED));
  }

  bool decrement_eviction_task_incomplete(taskid_t id) {
    auto &info = eviction_task_runtime_info[id];
    info.incomplete--;
    return (info.incomplete == 0) && (info.state >= static_cast<int8_t>(TaskState::RESERVED));
  }

  const std::vector<taskid_t> &compute_notify_mapped(taskid_t compute_task_id,
                                                     devid_t mapped_device, timecount_t time) {
    auto &my_info = compute_task_runtime_info[compute_task_id];
    auto &my_time_record = compute_task_time_records[compute_task_id];
    my_info.mapped_device = mapped_device;
    my_info.state = static_cast<int8_t>(TaskState::MAPPED);
    my_time_record.mapped_time = time;

    task_buffer.clear();

    auto my_dependents = get_compute_task_dependents(compute_task_id);

    for (const auto &dependent_id : my_dependents) {
      if (decrement_compute_task_unmapped(dependent_id)) {
        task_buffer.push_back(dependent_id);
      }
    }

    return task_buffer;
  }

  const std::vector<taskid_t> &compute_notify_reserved(taskid_t compute_task_id,
                                                       devid_t mapped_device, timecount_t time) {
    auto &my_info = compute_task_runtime_info[compute_task_id];
    auto &my_time_record = compute_task_time_records[compute_task_id];
    my_info.mapped_device = mapped_device;
    my_info.state = static_cast<int8_t>(TaskState::RESERVED);
    my_time_record.reserved_time = time;

    task_buffer.clear();

    auto my_dependents = get_compute_task_dependents(compute_task_id);

    for (const auto &dependent_id : my_dependents) {
      if (decrement_compute_task_unreserved(dependent_id)) {
        task_buffer.push_back(dependent_id);
      }
    }

    return task_buffer;
  }

  void compute_notify_launched(taskid_t compute_task_id, timecount_t time) {
    auto &my_info = compute_task_runtime_info[compute_task_id];
    auto &my_time_record = compute_task_time_records[compute_task_id];
    my_info.state = static_cast<int8_t>(TaskState::LAUNCHED);
    my_time_record.launched_time = time;
  }

  const std::vector<taskid_t> &compute_notify_completed(taskid_t compute_task_id,
                                                        timecount_t time) {
    auto &my_info = compute_task_runtime_info[compute_task_id];
    auto &my_time_record = compute_task_time_records[compute_task_id];
    my_info.state = static_cast<int8_t>(TaskState::COMPLETED);
    my_time_record.completed_time = time;

    task_buffer.clear();

    auto my_dependents = get_compute_task_dependents(compute_task_id);

    for (const auto &dependent_id : my_dependents) {
      if (decrement_compute_task_incomplete(dependent_id)) {
        task_buffer.push_back(dependent_id);
      }
    }

    return task_buffer;
  }

  const std::vector<taskid_t> &compute_notify_data_completed(taskid_t compute_task_id,
                                                             timecount_t time) {
    auto &my_info = compute_task_runtime_info[compute_task_id];

    // state and time assumed to be updated by prior call to notify_completed

    task_buffer.clear();
    auto my_data_dependents = get_compute_task_data_dependents(compute_task_id);

    for (const auto &dependent_id : my_data_dependents) {
      if (decrement_data_task_incomplete(dependent_id)) {
        task_buffer.push_back(dependent_id);
      }
    }

    return task_buffer;
  }

  const std::vector<taskid_t> &data_notify_completed(taskid_t data, timecount_t time) {
    auto &my_info = data_task_runtime_info[data];
    auto &my_time_record = data_task_time_records[data];

    my_info.state = static_cast<int8_t>(TaskState::COMPLETED);
    my_time_record.completed_time = time;

    task_buffer.clear();
    auto my_dependents = get_data_task_dependents(data);

    for (const auto &dependent_id : my_dependents) {
      if (decrement_data_task_incomplete(dependent_id)) {
        task_buffer.push_back(dependent_id);
      }
    }

    return task_buffer;
  }
};

class Task {
public:
  taskid_t id{};
  std::string name;

  ankerl::unordered_dense::set<taskid_t> dependencies;
  ankerl::unordered_dense::set<taskid_t> dependents;

  ankerl::unordered_dense::set<taskid_t> data_dependencies;
  ankerl::unordered_dense::set<taskid_t> data_dependents;

  ankerl::unordered_dense::set<taskid_t> temp_dependencies;
  ankerl::unordered_dense::set<taskid_t> temp_dependents;

  ankerl::unordered_dense::set<dataid_t> read;
  ankerl::unordered_dense::set<dataid_t> write;
  ankerl::unordered_dense::set<dataid_t> retire;

  std::vector<dataid_t> unique;
  std::vector<dataid_t> v_read;
  std::vector<taskid_t> most_recent_writers;

  std::vector<uint8_t> arch_mask;
  std::vector<vcu_t> vcu;
  std::vector<mem_t> mem;
  std::vector<timecount_t> time;

  int32_t depth{};
  int32_t type{-1};
  int32_t tag{-1};
};

class DataTask {
public:
  taskid_t id{};
  std::string name;
  taskid_t compute_task{-1}; // ID of the compute task that produces this data
  dataid_t data_id{-1};      // Unique ID for the data
  ankerl::unordered_dense::set<taskid_t> dependencies;
  ankerl::unordered_dense::set<taskid_t> dependents;

  ankerl::unordered_dense::set<taskid_t> temp_dependencies;
  ankerl::unordered_dense::set<taskid_t> temp_dependents;
};

class Graph {
public:
  std::vector<Task> tasks;
  std::vector<DataTask> data_tasks;
  std::vector<taskid_t> sorted;
  std::vector<taskid_t> initial_tasks;

  ankerl::unordered_dense::map<dataid_t, taskid_t>
      writers; // Maps data IDs to their most recent writer task ID

  taskid_t add_task(const std::string &name) {
    taskid_t id = static_cast<taskid_t>(tasks.size());
    tasks.emplace_back();
    tasks.back().id = id;
    tasks.back().name = name;
    return id;
  }

  taskid_t add_data_task(const std::string &name, taskid_t compute_task, dataid_t data_id) {
    taskid_t id = static_cast<taskid_t>(data_tasks.size());
    data_tasks.emplace_back();
    data_tasks.back().id = id;
    data_tasks.back().name = name;
    data_tasks.back().compute_task = compute_task;
    data_tasks.back().data_id = data_id;
    return id;
  }

  void add_read_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    auto &task = tasks[task_id];
    for (const auto &data_id : data_ids) {
      task.read.insert(data_id);
      task.v_read.push_back(data_id);         // Store read data in a vector for ordered access
      task.most_recent_writers.push_back(-1); // Initialize with -1 for no writer
    }
  }

  void add_write_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    auto &task = tasks[task_id];
    for (const auto &data_id : data_ids) {
      task.write.insert(data_id);
    }
  }

  void add_retire_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    auto &task = tasks[task_id];
    for (const auto &data_id : data_ids) {
      task.retire.insert(data_id);
    }
  }

  void set_tag(taskid_t task_id, int32_t tag) {
    tasks[task_id].tag = tag;
  }

  void set_type(taskid_t task_id, int32_t type) {
    tasks[task_id].type = type;
  }

  void set_variant(taskid_t task_id, DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time) {
    auto &task = tasks[task_id];
    size_t arch_index = static_cast<size_t>(arch);
    if (arch_index >= task.arch_mask.size()) {
      task.arch_mask.resize(arch_index + 1, 0);
      task.vcu.resize(arch_index + 1);
      task.mem.resize(arch_index + 1);
      task.time.resize(arch_index + 1);
    }
    task.arch_mask[arch_index] = 1; // Set the architecture mask
    task.vcu[arch_index] = vcu;
    task.mem[arch_index] = mem;
    task.time[arch_index] = time;
  }

  void add_dependency(taskid_t task_id, taskid_t dependency_id) {
    tasks[task_id].dependencies.insert(dependency_id);
  }

  void add_dependencies(taskid_t task_id, const std::vector<taskid_t> &dependency_ids) {
    auto &task = tasks[task_id];
    for (const auto &dependency_id : dependency_ids) {
      task.dependencies.insert(dependency_id);
    }
  }

  void populate_dependencies_from_dataflow() {
    ankerl::unordered_dense::map<dataid_t, taskid_t> last_writer;
    for (auto &task : tasks) {
      for (const auto &data_id : task.read) {
        auto it = last_writer.find(data_id);
        if (it != last_writer.end()) {
          add_dependency(task.id, it->second);
        }
      }
      for (const auto &data_id : task.write) {
        last_writer[data_id] = task.id;
      }
    }
  }

  void populate_dependents() {
    for (auto &task : tasks) {
      for (const auto &dependency_id : task.dependencies) {
        tasks[dependency_id].dependents.insert(task.id);
      }
    }
  }

  void populate_data_dependents() {
    // Data tasks depend on compute tasks
    // Iterate data tasks
    // Update data dependents of the compute tasks based on data dependencies
    for (auto &data_task : data_tasks) {
      for (const auto &dependency_id : data_task.dependencies) {
        tasks[dependency_id].data_dependents.insert(data_task.id);
      }
    }

    // Compute tasks depend on data tasks
    // Iterate compute tasks
    // Update dependents of the data tasks based on data dependencies
    for (auto &task : tasks) {
      for (const auto &data_task_id : task.data_dependencies) {
        data_tasks[data_task_id].dependents.insert(task.id);
      }
    }
  }

  void populate_unique_data() {
    for (auto &task : tasks) {
      std::set<dataid_t> unique_data_set;
      for (const auto &data_id : task.read) {
        unique_data_set.insert(data_id);
      }
      for (const auto &data_id : task.write) {
        unique_data_set.insert(data_id);
      }
      task.unique.assign(unique_data_set.begin(), unique_data_set.end());
    }
  }

  void populate_initial_tasks() {
    for (const auto &task : tasks) {
      if (task.dependencies.empty()) {
        initial_tasks.push_back(task.id);
      }
    }
  }

  void bfs() const {
    sorted.clear();
    sorted.resize(tasks.size());

    std::queue<taskid_t> queue;

    for (auto task_id : initial_tasks) {
      queue.push(task_id);
    }

    for (auto &task : tasks) {
      task.temp_dependencies.clear();
      task.temp_dependents.clear();
      task.temp_dependencies.insert(task.dependencies.begin(), task.dependencies.end());
      task.temp_dependents.insert(task.dependents.begin(), task.dependents.end());
    }

    while (!queue.empty()) {
      taskid_t current = queue.front();
      queue.pop();
      sorted.push_back(current);

      for (const auto &dependent : tasks[current].temp_dependents) {
        tasks[dependent].temp_dependencies.erase(current);
        if (tasks[dependent].temp_dependencies.empty()) {
          queue.push(dependent);
        }
      }
    }
  }

  void dfs() {
    sorted.clear();
    sorted.resize(tasks.size());

    std::stack<taskid_t> stack;

    for (auto &task : initial_tasks) {
      stack.push(task);
    }

    for (auto &task : tasks) {
      task.temp_dependencies.clear();
      task.temp_dependents.clear();
      task.temp_dependencies.insert(task.dependencies.begin(), task.dependencies.end());
      task.temp_dependents.insert(task.dependents.begin(), task.dependents.end());
    }

    while (!stack.empty()) {
      taskid_t current = stack.top();
      stack.pop();
      sorted.push_back(current);

      for (const auto &dependent : tasks[current].temp_dependents) {
        tasks[dependent].temp_dependencies.erase(current);
        if (tasks[dependent].temp_dependencies.empty()) {
          stack.push(dependent);
        }
      }
    }
  }

  void populate_depth() {
    for (auto &task : tasks) {
      task.depth = 0; // Reset depth
    }

    for (const auto &task_id : sorted) {
      auto &task = tasks[task_id];
      for (const auto &dependency_id : task.dependencies) {
        auto &dependency_task = tasks[dependency_id];
        if (dependency_task.depth < task.depth + 1) {
          dependency_task.depth = task.depth + 1;
        }
      }
    }
  }

  void create_data_task(taskid_t task_id, dataid_t data_id, bool has_writer,
                        taskid_t writer_id = -1) {

    auto &task = tasks[task_id];
    auto data_name = task.name + "_data_" + std::to_string(data_id);
    auto data_task_id = add_data_task(data_name, task_id, data_id);
    auto &data_task = data_tasks[data_task_id];

    if (has_writer) {
      data_task.dependencies.insert(writer_id);
      auto &writer_task = tasks[writer_id];
      writer_task.data_dependents.insert(data_task_id);
    }

    data_task.dependents.insert(task_id);
    task.data_dependencies.insert(data_task_id);
  }

  void populate_data_dependencies(bool ensure_dependencies = false, bool create_data_tasks = true) {
    writers.clear();

    // Iterate in a valid topological order
    for (auto task_id : sorted) {

      auto &task = tasks[task_id];

      // Create data tasks for all reads from current task
      if (create_data_tasks) {
        for (int32_t i = 0; i < task.v_read.size(); ++i) {
          dataid_t data_id = task.v_read[i];
          auto it = writers.find(data_id);
          taskid_t writer_id = -1;
          bool has_writer = it != writers.end();
          if (has_writer) {
            task.most_recent_writers[i] = it->second;
          }
          create_data_task(task_id, data_id, has_writer, writer_id);
        }
      }

      if (ensure_dependencies) {
        // writers are compute tasks
        // Ensure that the compute task depends on all writers of the data it reads
        for (const auto &data_id : task.read) {
          auto it = writers.find(data_id);
          if (it != writers.end()) {
            task.dependencies.insert(it->second);
            tasks[it->second].dependents.insert(task_id);
          }
        }

        // Ensure that the compute task depends on all writers of the data it writes
        for (const auto &data_id : task.write) {
          auto it = writers.find(data_id);
          if (it != writers.end()) {
            task.dependencies.insert(it->second);
            tasks[it->second].dependents.insert(task_id);
          }
        }

        // Ensure that the compute task depends on all writers of the data it retires
        for (const auto &data_id : task.retire) {
          auto it = writers.find(data_id);
          if (it != writers.end()) {
            task.dependencies.insert(it->second);
            tasks[it->second].dependents.insert(task_id);
          }
        }
      }

      // Update writers map with current task's writes
      for (const auto &data_id : task.write) {
        writers[data_id] = task_id;
      }
    }
  }

  void finalize(bool ensure_dependencies = false, bool create_data_tasks = true) {
    populate_dependencies_from_dataflow();
    populate_unique_data();
    populate_dependents();
    populate_depth();
    populate_initial_tasks();
    bfs();
    populate_data_dependencies(ensure_dependencies, create_data_tasks);
    populate_data_dependents();
  }

  TaskStaticInfo create_static_graph() const {
    TaskStaticInfo static_info(tasks.size(), data_tasks.size());

    taskid_t compute_dependency_offset = 0;
    taskid_t compute_dependent_offset = 0;
    taskid_t compute_data_dependency_offset = 0;
    taskid_t compute_data_dependent_offset = 0;

    dataid_t read_offset = 0;
    dataid_t write_offset = 0;
    dataid_t retire_offset = 0;
    dataid_t unique_offset = 0;

    taskid_t data_dependency_offset = 0;
    taskid_t data_dependent_offset = 0;

    for (const auto &task : tasks) {
      auto compute_dep_info = ComputeTaskDepInfo();
      auto compute_data_info = ComputeTaskDataInfo();
      auto compute_task_info = ComputeTaskStaticInfo();

      compute_dep_info.s_dependencies = compute_dependency_offset;
      compute_dep_info.e_dependencies = compute_dependency_offset + task.dependencies.size();
      compute_dependency_offset += task.dependencies.size();

      compute_dep_info.s_dependents = compute_dependent_offset;
      compute_dep_info.e_dependents = compute_dependent_offset + task.dependents.size();
      compute_dependent_offset += task.dependents.size();

      compute_dep_info.s_data_dependencies = compute_data_dependency_offset;
      compute_dep_info.e_data_dependencies =
          compute_data_dependency_offset + task.data_dependencies.size();
      compute_data_dependency_offset += task.data_dependencies.size();

      compute_dep_info.s_data_dependents = compute_data_dependent_offset;
      compute_dep_info.e_data_dependents =
          compute_data_dependent_offset + task.data_dependents.size();
      compute_data_dependent_offset += task.data_dependents.size();

      compute_data_info.s_read = read_offset;
      compute_data_info.e_read = read_offset + task.read.size();
      read_offset += task.read.size();

      compute_data_info.s_write = write_offset;
      compute_data_info.e_write = write_offset + task.write.size();
      write_offset += task.write.size();

      compute_data_info.s_retire = retire_offset;
      compute_data_info.e_retire = retire_offset + task.retire.size();
      retire_offset += task.retire.size();

      compute_data_info.s_unique = unique_offset;
      compute_data_info.e_unique = unique_offset + task.unique.size();
      unique_offset += task.unique.size();

      compute_task_info.tag = task.tag;
      compute_task_info.type = task.type;

      static_info.add_compute_task(task.id, task.name, compute_variant_info, compute_dep_info,
                                   compute_data_info, compute_task_info);

      static_info.add_compute_dependencies(task.id, as_vector(task.dependencies));
      static_info.add_compute_dependents(task.id, as_vector(task.dependents));
      static_info.add_compute_data_dependencies(task.id, as_vector(task.data_dependencies));
      static_info.add_compute_data_dependents(task.id, as_vector(task.data_dependents));
      static_info.add_compute_read(task.id, task.v_read);
      static_info.add_most_recent_writers(task.id, task.most_recent_writers);
      static_info.static_info.add_compute_write(task.id, as_vector(task.write));
      static_info.add_compute_retire(task.id, as_vector(task.retire));
      static_info.add_compute_unique(task.id, as_vector(task.unique));

      for (int i = 0; i < task.arch_mask.size(); ++i) {
        if (task.arch_mask[i]) {
          static_info.add_compute_variant(task.id, static_cast<DeviceType>(i), task.vcu[i],
                                          task.mem[i], task.time[i]);
        }
      }
    }

    for (const auto &data_task : data_tasks) {
      auto data_task_info = DataTaskStaticInfo();

      data_task_info.s_dependencies = data_dependency_offset;
      data_task_info.e_dependencies = data_dependency_offset + data_task.dependencies.size();
      data_dependency_offset += data_task.dependencies.size();

      data_task_info.s_dependents = data_dependent_offset;
      data_task_info.e_dependents = data_dependent_offset + data_task.dependents.size();
      data_dependent_offset += data_task.dependents.size();

      data_task_info.data_id = data_task.data_id;
      data_task_info.compute_task = data_task.compute_task;

      static_info.add_data_task(data_task.id, data_task.name, data_task_info);

      static_info.add_data_dependencies(data_task.id, as_vector(data_task.dependencies));
      static_info.add_data_dependents(data_task.id, as_vector(data_task.dependents));
    }

    return static_info;
  }
};

// class Task {
// protected:
//   TaskIDList dependencies;
//   TaskIDList dependents;

// public:
//   taskid_t id = 0;
//   int32_t depth = 0;

//   Task() = default;
//   Task(taskid_t id) : id(id) {
//   }

//   void set_depth(int32_t depth_) {
//     this->depth = depth_;
//   }

//   [[nodiscard]] taskid_t get_id() const {
//     return id;
//   }
//   [[nodiscard]] int64_t get_depth() const {
//     return depth;
//   }

//   void set_dependencies(TaskIDList _dependencies) {
//     this->dependencies = std::move(_dependencies);
//   }

//   void set_dependents(TaskIDList _dependents) {
//     this->dependents = std::move(_dependents);
//   }

//   [[nodiscard]] const TaskIDList &get_dependencies() const {
//     return dependencies;
//   }
//   [[nodiscard]] const TaskIDList &get_dependents() const {
//     return dependents;
//   }

//   void add_dependency(taskid_t dependency) {
//     dependencies.push_back(dependency);
//   }

//   void add_dependent(taskid_t dependent) {
//     dependents.push_back(dependent);
//   }
// };

// class ComputeTask : public Task {
// protected:
//   TaskIDList data_dependencies;
//   TaskIDList data_dependents;

//   DataIDList read;
//   DataIDList write;
//   DataIDList unique;
//   DataIDList retire;
//   TaskIDList most_recent_writers;

//   VariantList variants;

//   int type = -1;
//   int tag = -1;

// public:
//   static constexpr TaskType task_type = TaskType::COMPUTE;

//   ComputeTask() = default;
//   ComputeTask(taskid_t id) {
//     this->id = id;
//   }

//   bool is_supported_architecture(DeviceType arch) const;

//   void add_variant(DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time) {
//     variants.at(static_cast<std::size_t>(arch)) = Variant(arch, vcu, mem, time);
//   }

//   void add_recent_writer(taskid_t writer_id, dataid_t data_id) {
//     if (most_recent_writers.size() <= this->read.size()) {
//       most_recent_writers.resize(this->read.size(), -1);
//     }

//     // Find the index of the data_id in the read list
//     auto it = std::find(this->read.begin(), this->read.end(), data_id);
//     if (it != this->read.end()) {
//       std::size_t index = std::distance(this->read.begin(), it);
//       most_recent_writers[index] = writer_id;
//     } else {
//       SPDLOG_DEBUG("ComputeTask::add_writer : Data ID {} not found in read list, writer {}",
//                    data_id, writer_id);
//     }
//   }

//   TaskIDList &get_most_recent_writers() {
//     return most_recent_writers;
//   }

//   [[nodiscard]] const TaskIDList &get_most_recent_writers() const {
//     return most_recent_writers;
//   }

//   taskid_t get_recent_writer_by_dataid(dataid_t data_id) const {
//     auto it = std::find(this->read.begin(), this->read.end(), data_id);
//     if (it != this->read.end()) {
//       std::size_t index = std::distance(this->read.begin(), it);
//       return most_recent_writers[index];
//     }
//     return -1;
//   }

//   taskid_t get_recent_writer(std::size_t index) const {
//     assert(index < most_recent_writers.size());
//     return most_recent_writers[index];
//   }

//   void add_variant(DeviceType arch, Variant variant) {
//     variants.at(static_cast<std::size_t>(arch)) = variant;
//   }

//   Variant &get_variant(DeviceType arch) {
//     return variants.at(static_cast<std::size_t>(arch));
//   }

//   [[nodiscard]] const Variant &get_variant(DeviceType arch) const {
//     return variants.at(static_cast<std::size_t>(arch));
//   }

//   [[nodiscard]] const VariantList &get_variants() const {
//     return variants;
//   }
//   [[nodiscard]] std::vector<Variant> get_variant_vector() const;

//   void set_read(DataIDList _read) {
//     this->read = std::move(_read);
//     this->most_recent_writers.resize(read.size(), -1);
//   }
//   void set_write(DataIDList _write) {
//     this->write = std::move(_write);
//   }
//   void set_retire(DataIDList _retire) {
//     this->retire = std::move(_retire);
//   }

//   void set_type(int type_) {
//     this->type = type_;
//   }

//   void set_tag(int tag_) {
//     this->tag = tag_;
//   }

//   [[nodiscard]] int get_type() const {
//     return type;
//   }

//   [[nodiscard]] int get_tag() const {
//     return tag;
//   }

//   [[nodiscard]] std::vector<DeviceType> get_supported_architectures() const;

//   [[nodiscard]] const DataIDList &get_read() const {
//     return read;
//   }
//   [[nodiscard]] const DataIDList &get_write() const {
//     return write;
//   }

//   [[nodiscard]] const DataIDList &get_retire() const {
//     return retire;
//   }

//   void add_data_dependency(taskid_t dependency) {
//     data_dependencies.push_back(dependency);
//   }

//   void add_data_dependent(taskid_t dependent) {
//     data_dependents.push_back(dependent);
//   }

//   [[nodiscard]] const TaskIDList &get_data_dependencies() const {
//     return data_dependencies;
//   }

//   [[nodiscard]] const TaskIDList &get_data_dependents() const {
//     return data_dependents;
//   }

//   void find_unique_data() {
//     std::unordered_set<dataid_t> unique_set;
//     for (auto data_id : read) {
//       unique_set.insert(data_id);
//     }
//     for (auto data_id : write) {
//       unique_set.insert(data_id);
//     }
//     unique.assign(unique_set.begin(), unique_set.end());
//   }

//   [[nodiscard]] const DataIDList &get_unique() const {
//     return unique;
//   }
// };

// class DataTask : public Task {
// private:
//   dataid_t data_id;
//   taskid_t compute_task;

// public:
//   static constexpr TaskType task_type = TaskType::DATA;

//   DataTask() = default;
//   DataTask(taskid_t id_) {
//     this->id = id_;
//   }

//   void set_data_id(dataid_t data_id_) {
//     this->data_id = data_id_;
//   }
//   void set_compute_task(taskid_t compute_task_) {
//     this->compute_task = compute_task_;
//   }

//   [[nodiscard]] taskid_t get_compute_task() const {
//     return compute_task;
//   }

//   [[nodiscard]] dataid_t get_data_id() const {
//     return data_id;
//   }
// };

// class EvictionTask : public DataTask {
// protected:
//   TaskIDList data_dependencies;
//   TaskIDList data_dependents;
//   devid_t device_id = HOST_ID;         // the backup device
//   devid_t invalidate_device = HOST_ID; // the device to invalidate

// public:
//   static constexpr TaskType task_type = TaskType::EVICTION;

//   EvictionTask() = default;
//   EvictionTask(taskid_t id_) {
//     this->id = id_;
//   }

//   [[nodiscard]] const TaskIDList &get_data_dependencies() const {
//     return data_dependencies;
//   }
//   [[nodiscard]] const TaskIDList &get_data_dependents() const {
//     return data_dependents;
//   }
//   void add_data_dependency(taskid_t dependency) {
//     data_dependencies.push_back(dependency);
//   }
//   void add_data_dependent(taskid_t dependent) {
//     data_dependents.push_back(dependent);
//   }

//   [[nodiscard]] devid_t get_device_id() const {
//     return device_id;
//   }

//   void set_invalidate_device(devid_t invalidate_device_) {
//     this->invalidate_device = invalidate_device_;
//   }

//   [[nodiscard]] devid_t get_invalidate_device() const {
//     return invalidate_device;
//   }

//   void set_device_id(devid_t device_id_) {
//     this->device_id = device_id_;
//   }

//   [[nodiscard]] bool is_eviction() const {
//     return true;
//   }
// };

// class MinimalTask {
// public:
//   taskid_t id;
//   std::unordered_map<taskid_t, taskid_t> dependencies;
//   std::vector<taskid_t> dependents;

//   MinimalTask() = default;
//   MinimalTask(taskid_t id) : id(id) {
//   }

//   MinimalTask(const MinimalTask &other) = default;

//   MinimalTask(MinimalTask &&other) noexcept
//       : id(std::exchange(other.id, 0)), dependencies(std::move(other.dependencies)),
//         dependents(std::move(other.dependents)) {
//   }

//   MinimalTask(const Task &task) : id(task.id) {
//     const auto &task_dependencies = task.get_dependencies();
//     const auto &task_dependents = task.get_dependents();

//     for (auto dep : task_dependencies) {
//       dependencies[dep] = dep;
//     }
//     dependents.assign(task_dependents.begin(), task_dependents.end());
//   }

//   // Copy assignment operator
//   MinimalTask &operator=(const MinimalTask &other) {
//     if (this != &other) {
//       id = other.id;
//       dependencies = other.dependencies;
//       dependents = other.dependents;
//     }
//     return *this;
//   }

//   // Move assignment operator
//   MinimalTask &operator=(MinimalTask &&other) noexcept {
//     if (this != &other) {
//       id = std::exchange(other.id, 0);
//       dependencies = std::move(other.dependencies);
//       dependents = std::move(other.dependents);
//     }
//     return *this;
//   }
// };

// struct TaskTypeBundle {
//   TaskType type;
//   taskid_t id;
// };

// using ComputeTaskList = std::vector<ComputeTask>;
// using DataTaskList = std::vector<DataTask>;
// using EvictionTaskList = std::vector<EvictionTask>;
// using TaskList = std::vector<Task>;
// using MixedTaskIDList = std::vector<TaskTypeBundle>;

// class GraphManager;

// class EvictionTasks {
// protected:
//   EvictionTaskList tasks;
//   std::vector<std::string> task_names;
//   taskid_t n_noneviction_tasks = 0;
//   taskid_t n_eviction_tasks = 0;

// public:
//   EvictionTasks() = default;
//   EvictionTasks(const EvictionTasks &other) = default;

//   EvictionTasks(taskid_t n_tasks) : n_noneviction_tasks(n_tasks) {
//     const int EXPECTED_EVICTION_TASKS = 1000;
//     tasks.reserve(n_tasks + EXPECTED_EVICTION_TASKS);
//     task_names.reserve(n_tasks + EXPECTED_EVICTION_TASKS);
//   }

//   taskid_t get_local_eviction_task_id(taskid_t id) const {
//     assert(id < n_noneviction_tasks + n_eviction_tasks);
//     assert(id >= n_noneviction_tasks);
//     return id - n_noneviction_tasks;
//   }

//   taskid_t add_eviction_task(taskid_t compute_task, dataid_t data_id, devid_t backup_device,
//                              devid_t invalidate_device) {
//     taskid_t id = n_noneviction_tasks + n_eviction_tasks++;
//     tasks.emplace_back(id);
//     task_names.emplace_back("eviction[" + std::to_string(id) + "]");
//     tasks.back().set_data_id(data_id);
//     tasks.back().set_compute_task(compute_task);
//     tasks.back().set_device_id(backup_device);
//     tasks.back().set_invalidate_device(invalidate_device);
//     return id;
//   }

//   bool is_eviction(taskid_t id) const {
//     assert(id < n_noneviction_tasks + n_eviction_tasks);
//     return id >= n_noneviction_tasks;
//   }

//   [[nodiscard]] EvictionTask &get_eviction_task(taskid_t id) {
//     return tasks.at(get_local_eviction_task_id(id));
//   }

//   [[nodiscard]] const EvictionTask &get_eviction_task(taskid_t id) const {
//     return tasks.at(get_local_eviction_task_id(id));
//   }

//   [[nodiscard]] const std::string &get_name(taskid_t id) const {
//     assert(id < n_noneviction_tasks + n_eviction_tasks);
//     return task_names.at(get_local_eviction_task_id(id));
//   }

//   [[nodiscard]] std::size_t get_n_eviction_tasks() const {
//     return n_eviction_tasks;
//   }
// };

// class Tasks {
// protected:
//   const taskid_t num_compute_tasks;
//   taskid_t current_task_id = 0;
//   ComputeTaskList compute_tasks;
//   DataTaskList data_tasks;
//   std::vector<std::string> task_names;
//   mutable bool initialized = false;

//   ComputeTaskList &get_compute_tasks() {
//     return compute_tasks;
//   }
//   DataTaskList &get_data_tasks() {
//     return data_tasks;
//   }

//   ComputeTask &get_compute_task(taskid_t id) {
//     return compute_tasks[id];
//   }
//   DataTask &get_data_task(taskid_t id) {
//     return data_tasks[id - num_compute_tasks];
//   }
//   Task &get_task(taskid_t id);

//   void create_data_task(ComputeTask &task, bool has_writer, taskid_t writer_id, dataid_t
//   data_id);

// public:
//   Tasks(taskid_t num_compute_tasks);

//   [[nodiscard]] bool is_initialized() const {
//     return initialized;
//   }

//   void set_initalized() const {
//     assert(!initialized);
//     initialized = true;
//   }

//   StatsBundle<timecount_t> get_duration_statistics(std::vector<DeviceType> &device_types)
//   const;

//   [[nodiscard]] std::size_t size() const;
//   [[nodiscard]] std::size_t compute_size() const;
//   [[nodiscard]] std::size_t data_size() const;
//   [[nodiscard]] bool empty() const;
//   [[nodiscard]] bool is_compute(taskid_t id) const;
//   [[nodiscard]] bool is_data(taskid_t id) const;
//   [[nodiscard]] bool is_eviction(taskid_t id) const;

//   void add_compute_task(ComputeTask task);
//   void add_data_task(DataTask task);

//   void create_compute_task(taskid_t tid, std::string name, TaskIDList dependencies);
//   void add_variant(taskid_t id, DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time);
//   void set_read(taskid_t id, DataIDList read);
//   void set_write(taskid_t id, DataIDList write);
//   void set_retire(taskid_t id, DataIDList retire);
//   void set_type(taskid_t id, int type);
//   void set_tag(taskid_t id, int tag);

//   [[nodiscard]] int get_type(taskid_t id) const;
//   [[nodiscard]] int get_tag(taskid_t id) const;

//   [[nodiscard]] const ComputeTaskList &get_compute_tasks() const {
//     return compute_tasks;
//   }
//   [[nodiscard]] const DataTaskList &get_data_tasks() const {
//     return data_tasks;
//   }

//   [[nodiscard]] const ComputeTask &get_compute_task(taskid_t id) const {
//     assert(id < num_compute_tasks);
//     return compute_tasks.at(id);
//   }
//   [[nodiscard]] const DataTask &get_data_task(taskid_t id) const {
//     assert(id >= num_compute_tasks);
//     return data_tasks.at(id - num_compute_tasks);
//   }

//   [[nodiscard]] const TaskIDList &get_dependencies(taskid_t id) const;
//   [[nodiscard]] const TaskIDList &get_dependents(taskid_t id) const;
//   [[nodiscard]] const VariantList &get_variants(taskid_t id) const;
//   [[nodiscard]] const Variant &get_variant(taskid_t id, DeviceType arch) const;
//   [[nodiscard]] const DataIDList &get_read(taskid_t id) const;
//   [[nodiscard]] const DataIDList &get_write(taskid_t id) const;
//   [[nodiscard]] const DataIDList &get_retire(taskid_t id) const;

//   [[nodiscard]] const Resources &get_task_resources(taskid_t id) const;
//   [[nodiscard]] const Resources &get_task_resources(taskid_t id, DeviceType arch) const;

//   [[nodiscard]] const TaskIDList &get_data_dependencies(taskid_t id) const;
//   [[nodiscard]] const TaskIDList &get_data_dependents(taskid_t id) const;

//   [[nodiscard]] std::size_t get_depth(taskid_t id) const;
//   [[nodiscard]] dataid_t get_data_id(taskid_t id) const;

//   [[nodiscard]] std::string const &get_name(taskid_t id) const {
//     return task_names.at(id);
//   }

//   [[nodiscard]] std::vector<DeviceType> get_supported_architectures(taskid_t id) const;

//   [[nodiscard]] std::vector<Variant> get_variant_vector(taskid_t id) const {
//     return get_compute_task(id).get_variant_vector();
//   }

//   [[nodiscard]] const Task &get_task(taskid_t id) const;

//   friend class GraphManager;
// };