#pragma once
#include "devices.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <span>
#include <sys/types.h>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

using op_t = uint32_t;
using f_t = float_t;
using namespace nb::literals;

#if NB_VERSION_MAJOR > 1
// shapes are signed starting with NB 2, nb::any now stands for any type, not any shape
constexpr auto any_size = -1;
#else
constexpr auto any_size = nb::any;
#endif

using TorchArr = nb::ndarray<nb::pytorch, nb::device::cpu, float>;

template <typename T>
using TorchArr2D = nb::ndarray<T, nb::shape<any_size, any_size>, nb::c_contig, nb::pytorch>;
using TorchFloatArr2D = TorchArr2D<float>;
using TorchIntArr2D = TorchArr2D<int>;
using TorchInt64Arr2D = TorchArr2D<int64_t>;
using TorchUInt32Arr2D = TorchArr2D<uint32_t>;
using TorchUInt64Arr2D = TorchArr2D<uint64_t>;

template <typename T>
using TorchArr1D = nb::ndarray<T, nb::shape<any_size>, nb::c_contig, nb::pytorch>;
using TorchFloatArr1D = TorchArr1D<float>;
using TorchIntArr1D = TorchArr1D<int>;
using TorchInt64Arr1D = TorchArr1D<int64_t>;
using TorchUInt32Arr1D = TorchArr1D<uint32_t>;
using TorchUInt64Arr1D = TorchArr1D<uint64_t>;

taskid_t max(taskid_t a, taskid_t b) {
  return a > b ? a : b;
}

taskid_t min(taskid_t a, taskid_t b) {
  return a < b ? a : b;
}

enum class NodeType {
  ANY = -1,
  TASK = 0,
  DATA_BLOCK = 1,
  DEVICE = 2
};

enum class EdgeType {
  ANY = -1,
  TASK_TASK = 0,
  TASK_DATA = 1,
  TASK_DEVICE = 2,
  DATA_DEVICE = 3
};

struct GraphSpec {
public:
  taskid_t max_in_degree = 0;
  taskid_t max_out_degree = 0;
  dataid_t max_data_in_degree = 0;
  dataid_t max_data_out_degree = 0;
  dataid_t max_data_usage = 0;

  taskid_t max_candidates = 3;

  taskid_t max_edges_tasks_tasks = 0;
  taskid_t max_edges_tasks_data = 0;
  taskid_t max_edges_tasks_devices = 0;
  taskid_t max_edges_data_devices = 0;

  taskid_t max_tasks = 0;
  taskid_t max_data = 0;
  taskid_t max_devices = 0;

  void compute_max_degree(const SchedulerState &state) {
    const auto &tasks = state.get_task_manager().get_tasks();
    for (const auto &task : tasks.get_compute_tasks()) {
      max_in_degree = max(max_in_degree, task.get_dependencies().size());
      max_out_degree = max(max_out_degree, task.get_dependents().size());
      max_data_in_degree = max(max_data_in_degree, task.get_data_dependencies().size());
      max_data_out_degree = max(max_data_out_degree, task.get_data_dependents().size());
      max_data_usage = max(max_data_usage, task.get_unique().size());
    }
  }
};

using TaskSet = std::set<taskid_t>;

class GraphExtractor {
protected:
  std::reference_wrapper<const SchedulerState> state;
  std::vector<op_t> source_list;
  std::vector<op_t> target_list;
  std::set<taskid_t> visited;
  std::unordered_set<dataid_t> data_visited;
  std::unordered_set<taskid_t> local_visited;
  std::unordered_map<taskid_t, taskid_t> task_index_map;
  std::unordered_map<dataid_t, dataid_t> data_index_map;
  std::unordered_map<devid_t, devid_t> device_index_map;

  GraphSpec spec;

public:
  GraphExtractor(const SchedulerState &state) : state(state) {
    source_list.reserve(400);
    target_list.reserve(400);
    local_visited.reserve(400);
    task_index_map.reserve(400);
    data_index_map.reserve(400);
    device_index_map.reserve(400);
    data_visited.reserve(400);
  }

  GraphSpec &get_spec() {
    return spec;
  }

  void set_spec(const GraphSpec &spec) {
    this->spec = spec;
  }

  [[nodiscard]] TaskIDList get_active_tasks() const {
    const auto &s = this->state.get();
    return s.counts.get_active_task_list();
  }

  void _get_k_hop_task_dependents(TaskSet &visited, taskid_t task_id, int k) {

    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    local_visited.clear();

    std::queue<taskid_t> q;
    q.push(task_id);
    visited.insert(task_id);

    int current_hop = 0;

    while (!q.empty() && current_hop < k) {
      std::size_t level_size = q.size();

      for (std::size_t i = 0; i < level_size; ++i) {
        taskid_t current_task_id = q.front();
        q.pop();

        const auto &task = tasks.get_compute_task(current_task_id);
        for (const auto &dep_id : task.get_dependents()) {
          if (local_visited.insert(dep_id).second) {
            q.push(dep_id);
            visited.insert(dep_id);
          }
        }
      }

      current_hop++;
    }
  }

  [[nodiscard]] TaskIDList get_k_hop_task_dependents(TaskSet &visited, TaskIDList &initial_tasks,
                                                     int k) {

    for (const auto &task_id : initial_tasks) {
      _get_k_hop_task_dependents(visited, task_id, k);
    }

    // Only keep the first max_tasks tasks
    TaskIDList result;
    result.resize(min(spec.max_tasks, visited.size()));
    std::copy_n(visited.begin(), result.size(), result.begin());

    return result;
  }

  void _get_k_hop_task_dependencies(TaskSet &visited, taskid_t task_id, int k) {

    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    local_visited.clear();

    std::queue<taskid_t> q;
    q.push(task_id);
    visited.insert(task_id);

    int current_hop = 0;

    while (!q.empty() && current_hop < k) {
      std::size_t level_size = q.size();

      for (std::size_t i = 0; i < level_size; ++i) {
        taskid_t current_task_id = q.front();
        q.pop();

        const auto &task = tasks.get_compute_task(current_task_id);
        for (const auto &dep_id : task.get_dependencies()) {
          if (local_visited.insert(dep_id).second) {
            q.push(dep_id);
            visited.insert(dep_id);
          }
        }
      }

      current_hop++;
    }
  }

  [[nodiscard]] TaskIDList get_k_hop_task_dependencies(TaskSet &visited, TaskIDList &initial_tasks,
                                                       int k) {

    for (const auto &task_id : initial_tasks) {
      _get_k_hop_task_dependencies(visited, task_id, k);
    }

    // Only keep the first max_tasks tasks
    TaskIDList result;
    result.resize(min(spec.max_tasks, visited.size()));
    std::copy_n(visited.begin(), result.size(), result.begin());

    return result;
  }

  void i_get_k_hop_task_bidirectional(TaskSet &visited, taskid_t task_id, int k) {

    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    local_visited.clear();

    std::queue<taskid_t> q;
    q.push(task_id);
    visited.insert(task_id);

    int current_hop = 0;

    while (!q.empty() && current_hop < k) {
      std::size_t level_size = q.size();

      for (std::size_t i = 0; i < level_size; ++i) {
        taskid_t current_task_id = q.front();
        q.pop();

        const auto &task = tasks.get_compute_task(current_task_id);
        for (const auto &dep_id : task.get_dependencies()) {
          if (local_visited.insert(dep_id).second) {
            q.push(dep_id);
            visited.insert(dep_id);
          }
        }

        for (const auto &dep_id : task.get_dependents()) {
          if (local_visited.insert(dep_id).second) {
            q.push(dep_id);
            visited.insert(dep_id);
          }
        }
      }

      current_hop++;
    }
  }

  [[nodiscard]] TaskIDList get_k_hop_task_bidirectional(TaskSet &visited, TaskIDList &initial_tasks,
                                                        int k) {

    for (const auto &task_id : initial_tasks) {
      i_get_k_hop_task_bidirectional(visited, task_id, k);
    }

    // Only keep the first max_tasks tasks
    TaskIDList result;
    result.resize(min(spec.max_tasks, visited.size()));
    std::copy_n(visited.begin(), result.size(), result.begin());

    return result;
  }

  size_t get_task_task_edges(TaskIDList &sources, TorchUInt64Arr2D &output) {

    // Check first dimension is 2
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    // Clear data structures before use
    task_index_map.clear();
    task_index_map.reserve(sources.size());

    for (std::size_t i = 0; i < sources.size(); i++) {
      task_index_map[sources[i]] = i;
    }

    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    std::size_t edge_count = 0;

    for (std::size_t source_idx = 0; source_idx < sources.size(); source_idx++) {
      const auto source_id = sources[source_idx];
      const auto &source_task = tasks.get_compute_task(source_id);
      const auto &dependencies = source_task.get_dependencies();

      // Process each dependency
      for (const auto &dep_id : dependencies) {
        auto it = task_index_map.find(dep_id);
        if (it != task_index_map.end()) {
          output(0, edge_count) = source_idx;
          output(1, edge_count) = it->second;
          edge_count++;
        }
      }
    }
    return edge_count;
  }

  [[nodiscard]] DataIDList get_unique_data(TaskIDList &task_ids) {
    data_visited.clear();
    data_visited.reserve(400);

    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    for (const auto &task_id : task_ids) {
      const auto &task = tasks.get_compute_task(task_id);
      for (auto data_id : task.get_unique()) {
        data_visited.insert(data_id);
      }
    }

    DataIDList result;
    auto max_size = min(spec.max_data, data_visited.size());
    result.reserve(max_size);
    std::copy(data_visited.begin(), data_visited.end(), std::back_inserter(result));

    return result;
  }

  size_t get_task_data_edges(TaskIDList &task_ids, DataIDList &data_ids, TorchUInt64Arr2D &output) {
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    data_index_map.clear();
    data_index_map.reserve(data_ids.size());

    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    std::size_t edge_count = 0;
    for (std::size_t i = 0; i < data_ids.size(); i++) {
      data_index_map[data_ids[i]] = i;
    }

    for (std::size_t i = 0; i < task_ids.size(); i++) {
      const auto &task_id = task_ids[i];
      const auto &task = tasks.get_compute_task(task_id);
      for (auto data_id : task.get_unique()) {
        auto it = data_index_map.find(data_id);
        if (it != data_index_map.end()) {
          output(0, edge_count) = i;
          output(1, edge_count) = it->second;
          edge_count++;
        }
      }
    }
    return edge_count;
  }

  size_t get_data_device_edges(DataIDList &data_ids, TorchUInt64Arr2D &output) {

    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    const auto &device_manager = state.get().get_device_manager();
    const auto &devices = device_manager.get_devices();
    const auto &data_manager = state.get().get_data_manager();

    std::size_t edge_count = 0;
    for (std::size_t i = 0; i < data_ids.size(); i++) {
      for (std::size_t j = 0; j < devices.size(); j++) {
        if (data_manager.check_valid_mapped(data_ids[i], j)) {
          output(0, i) = i;
          output(1, i) = j;
          edge_count++;
        }
      }
    }
    return edge_count;
  }
};

f_t guarded_divide(double a, double b) {
  if (b == 0) {
    return static_cast<f_t>(a);
  }
  return static_cast<f_t>(a / b);
}

void one_hot(int index, std::span<f_t> output) {
  for (std::size_t i = 0; i < output.size(); i++) {
    output[i] = static_cast<f_t>(i == index);
  }
}

// CRTP base for features
template <typename Derived> struct Feature {
  [[nodiscard]] size_t getFeatureDim() const {
    return static_cast<const Derived *>(this)->getFeatureDimImpl();
  }

  void extractFeature(uint32_t object_id, TorchArr &output) const {
    std::span<float> sp(output.data(), output.size());
    static_cast<const Derived *>(this)->extractFeatureImpl(object_id, sp);
  }

  template <typename Span> void extractFeature(uint32_t object_id, Span output) const {
    static_cast<const Derived *>(this)->extractFeatureImpl(object_id, output);
  }
};

template <typename Derived> struct EdgeFeature {
  [[nodiscard]] size_t getFeatureDim() const {
    return static_cast<const Derived *>(this)->getFeatureDimImpl();
  }

  void extractFeature(uint32_t source_id, uint32_t target_id, TorchArr &output) const {
    std::span<float> sp(output.data(), output.size());
    static_cast<const Derived *>(this)->extractFeatureImpl(source_id, target_id, sp);
  }

  template <typename Span>
  void extractFeature(uint32_t source_id, uint32_t target_id, Span output) const {
    static_cast<const Derived *>(this)->extractFeatureImpl(source_id, target_id, output);
  }
};

// Compile-time FeatureExtractor using variadic templates
template <typename... Features> class FeatureExtractor {
  std::tuple<Features...> features;

  // Helper to compute total feature dimension at compile-time
  template <size_t... Is> size_t computeFeatureDim(std::index_sequence<Is...>) const {
    return (std::get<Is>(features).getFeatureDim() + ...);
  }

public:
  FeatureExtractor(Features... feats) : features(std::move(feats)...) {
  }

  size_t getFeatureDim() const {
    return computeFeatureDim(std::make_index_sequence<sizeof...(Features)>{});
  }

  void getFeatures(int object_id, TorchArr &output) const {
    std::span<float> sp(output.data(), output.size());
    getFeatures(object_id, sp);
  }

  template <typename Span> void getFeatures(int object_id, Span output) const {
    size_t offset = 0;
    std::apply(
        [&](const auto &...feats) {
          (..., (feats.extractFeature(object_id, output.subspan(offset, feats.getFeatureDim())),
                 offset += feats.getFeatureDim()));
        },
        features);
  }
};

template <typename... Features> class EdgeFeatureExtractor {
  std::tuple<Features...> features;

  // Helper to compute total feature dimension at compile-time
  template <size_t... Is> size_t computeFeatureDim(std::index_sequence<Is...>) const {
    return (std::get<Is>(features).getFeatureDim() + ...);
  }

public:
  EdgeFeatureExtractor(Features... feats) : features(std::move(feats)...) {
  }

  size_t getFeatureDim() const {
    return computeFeatureDim(std::make_index_sequence<sizeof...(Features)>{});
  }

  void getFeatures(int source_id, int target_id, TorchArr &output) const {
    std::span<float> sp(output.data(), output.size());
    getFeatures(source_id, target_id, sp);
  }

  template <typename Span> void getFeatures(int source_id, int target_id, Span output) const {
    size_t offset = 0;
    std::apply(
        [&](const auto &...feats) {
          (..., (feats.extractFeature(source_id, target_id,
                                      output.subspan(offset, feats.getFeatureDim())),
                 offset += feats.getFeatureDim()));
        },
        features);
  }
};

template <typename Derived> struct StateFeature : Feature<Derived> {
  const SchedulerState &state;
  const NodeType node_type;

  StateFeature(const SchedulerState &state, const NodeType node_type)
      : state(state), node_type(node_type) {
  }
};

template <typename Derived> struct StateEdgeFeature : EdgeFeature<Derived> {
  const SchedulerState &state;
  const EdgeType edge_type;

  StateEdgeFeature(const SchedulerState &state, const EdgeType edge_type)
      : state(state), edge_type(edge_type) {
  }
};

template <typename Derived> struct IntFeature : Feature<Derived> {
  const size_t v;
  const NodeType node_type;

  IntFeature(size_t v, NodeType node_type) : v(v), node_type(node_type) {
  }
};

template <typename Derived> struct IntEdgeFeature : EdgeFeature<Derived> {
  const size_t v;
  const EdgeType edge_type;

  IntEdgeFeature(size_t v, EdgeType edge_type) : v(v), edge_type(edge_type) {
  }
};

struct InDegreeTaskFeature : public StateFeature<InDegreeTaskFeature> {
  InDegreeTaskFeature(const SchedulerState &state)
      : StateFeature<InDegreeTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  static f_t get_in_degree(const ComputeTask &task) {
    return static_cast<f_t>(task.get_dependencies().size());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    output[0] = log(get_in_degree(task));
  }
};

struct OutDegreeTaskFeature : public StateFeature<OutDegreeTaskFeature> {
  OutDegreeTaskFeature(const SchedulerState &state)
      : StateFeature<OutDegreeTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  static f_t get_out_degree(const ComputeTask &task) {
    return static_cast<f_t>(task.get_dependents().size());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    output[0] = log(get_out_degree(task));
  }
};

struct DurationTaskFeature : public StateFeature<DurationTaskFeature> {
  DurationTaskFeature(const SchedulerState &state)
      : StateFeature<DurationTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  static f_t get_duration(const ComputeTask &task, DeviceType arch) {
    return static_cast<f_t>(task.get_variant(arch).get_observed_time());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    output[0] = log(get_duration(task, DeviceType::CPU));
    output[1] = log(get_duration(task, DeviceType::GPU));
  }
};

struct MemoryTaskFeature : public StateFeature<MemoryTaskFeature> {
  MemoryTaskFeature(const SchedulerState &state)
      : StateFeature<MemoryTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  static f_t get_task_memcost(const ComputeTask &task, DeviceType arch) {
    return static_cast<f_t>(task.get_variant(arch).get_mem());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    output[0] = log(get_task_memcost(task, DeviceType::CPU));
    output[1] = log(get_task_memcost(task, DeviceType::GPU));
  }
};

struct EmptyTaskFeature : public IntFeature<EmptyTaskFeature> {
  size_t dimension;
  EmptyTaskFeature(size_t dimension)
      : IntFeature<EmptyTaskFeature>(dimension, NodeType::TASK), dimension(dimension) {
  }

  size_t getFeatureDimImpl() const {
    return dimension;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
  }
};

struct OneHotMappedDeviceTaskFeature : public StateFeature<OneHotMappedDeviceTaskFeature> {
  OneHotMappedDeviceTaskFeature(const SchedulerState &state)
      : StateFeature<OneHotMappedDeviceTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task_manager = state.get_task_manager();
    one_hot(task_manager.state.get_mapping(task_id), output);
  }
};

struct DataMappedLocations : public StateFeature<DataMappedLocations> {
  DataMappedLocations(const SchedulerState &state)
      : StateFeature<DataMappedLocations>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_device_manager().get_devices();
    for (std::size_t i = 0; i < devices.size(); i++) {
      output[i] = static_cast<f_t>(data_manager.check_valid_mapped(data_id, i));
    }
  }
};

struct DataReservedLocations : public StateFeature<DataReservedLocations> {
  DataReservedLocations(const SchedulerState &state)
      : StateFeature<DataReservedLocations>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_device_manager().get_devices();
    for (std::size_t i = 0; i < devices.size(); i++) {
      output[i] = static_cast<f_t>(data_manager.check_valid_reserved(data_id, i));
    }
  }
};

struct DataLaunchedLocations : public StateFeature<DataLaunchedLocations> {
  DataLaunchedLocations(const SchedulerState &state)
      : StateFeature<DataLaunchedLocations>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_device_manager().get_devices();
    for (std::size_t i = 0; i < devices.size(); i++) {
      output[i] = static_cast<f_t>(data_manager.check_valid_launched(data_id, i));
    }
  }
};

struct DataSizeFeature : public StateFeature<DataSizeFeature> {
  DataSizeFeature(const SchedulerState &state)
      : StateFeature<DataSizeFeature>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data = state.get_data_manager().get_data();
    output[0] = log(static_cast<f_t>(data.get_size(data_id)));
  }
};

struct DeviceMemoryFeature : public StateFeature<DeviceMemoryFeature> {
  DeviceMemoryFeature(const SchedulerState &state)
      : StateFeature<DeviceMemoryFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 3;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    const auto &device_manager = state.get_device_manager();
    auto mapped_mem = static_cast<double>(device_manager.get_mem<TaskState::MAPPED>(device_id));
    auto reserved_mem = static_cast<double>(device_manager.get_mem<TaskState::RESERVED>(device_id));
    auto launched_mem = static_cast<double>(device_manager.get_mem<TaskState::LAUNCHED>(device_id));
    output[0] = log(mapped_mem);
    output[1] = log(reserved_mem);
    output[2] = log(launched_mem);
  }
};

struct DeviceTimeFeature : public StateFeature<DeviceTimeFeature> {
  DeviceTimeFeature(const SchedulerState &state)
      : StateFeature<DeviceTimeFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 3;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    const auto &s = this->state;
    const auto &device_manager = s.get_device_manager();
    auto mapped_time = static_cast<double>(s.costs.get_mapped_time(device_id));
    auto reserved_time = static_cast<double>(s.costs.get_reserved_time(device_id));
    auto launched_time = static_cast<double>(s.costs.get_launched_time(device_id));
    output[0] = log(mapped_time);
    output[1] = log(reserved_time);
    output[2] = log(launched_time);
  }
};

struct DeviceIDFeature : public StateFeature<DeviceIDFeature> {
  DeviceIDFeature(const SchedulerState &state)
      : StateFeature<DeviceIDFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    one_hot(device_id, output);
  }
};

struct DeviceArchitectureFeature : public StateFeature<DeviceArchitectureFeature> {
  DeviceArchitectureFeature(const SchedulerState &state)
      : StateFeature<DeviceArchitectureFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    const auto &devices = state.get_device_manager().get_devices();
    output[0] = static_cast<f_t>(devices.get_type(device_id) == DeviceType::CPU);
    output[1] = static_cast<f_t>(devices.get_type(device_id) == DeviceType::GPU);
  }
};

struct TaskTaskSharedDataFeature : public StateEdgeFeature<TaskTaskSharedDataFeature> {
  TaskTaskSharedDataFeature(const SchedulerState &state)
      : StateEdgeFeature<TaskTaskSharedDataFeature>(state, EdgeType::TASK_TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    const auto &tasks = state.get_task_manager().get_tasks();
    const auto &source_task = tasks.get_compute_task(source_id);
    const auto &target_task = tasks.get_compute_task(target_id);
    const auto &data_manager = state.get_data_manager();
    const auto &data = data_manager.get_data();

    double total_memory_cost_source = 0;
    for (auto data_id : source_task.get_unique()) {
      total_memory_cost_source += data.get_size(data_id);
    }

    double total_memory_cost_target = 0;
    for (auto data_id : target_task.get_unique()) {
      total_memory_cost_target += data.get_size(data_id);
    }

    auto shared_memory_cost = static_cast<double>(
        data_manager.shared_size(source_task.get_unique(), target_task.get_unique()));

    output[0] = guarded_divide(shared_memory_cost, total_memory_cost_source);
  }
};

struct TaskDataRelativeSizeFeature : public StateEdgeFeature<TaskDataRelativeSizeFeature> {
  TaskDataRelativeSizeFeature(const SchedulerState &state)
      : StateEdgeFeature<TaskDataRelativeSizeFeature>(state, EdgeType::TASK_DATA) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    const auto &tasks = state.get_task_manager().get_tasks();
    const auto &source_task = tasks.get_compute_task(source_id);
    const auto &data_manager = state.get_data_manager();
    const auto &data = data_manager.get_data();

    double total_memory_cost_source = 0;
    for (auto data_id : source_task.get_unique()) {
      total_memory_cost_source += data.get_size(data_id);
    }

    double data_size = static_cast<double>(data.get_size(target_id));
    output[0] = guarded_divide(data_size, total_memory_cost_source);
  }
};

struct TaskDataUsageFeature : public StateEdgeFeature<TaskDataUsageFeature> {
  TaskDataUsageFeature(const SchedulerState &state)
      : StateEdgeFeature<TaskDataUsageFeature>(state, EdgeType::TASK_DATA) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    const auto &tasks = state.get_task_manager().get_tasks();
    const auto &source_task = tasks.get_compute_task(source_id);

    bool is_read_access = std::find(source_task.get_read().begin(), source_task.get_read().end(),
                                    target_id) != source_task.get_read().end();
    bool is_write_access = std::find(source_task.get_write().begin(), source_task.get_write().end(),
                                     target_id) != source_task.get_write().end();

    output[0] = static_cast<f_t>(is_read_access);
    output[1] = static_cast<f_t>(is_write_access);
  }
};