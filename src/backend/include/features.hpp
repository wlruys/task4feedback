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

using TorchArr = nb::ndarray<nb::pytorch, float, nb::device::cpu>;

enum class NodeType {
  TASK = 0,
  DATA_BLOCK = 1,
  DEVICE = 2
};

enum class EdgeType {
  TASK_TASK = 0,
  TASK_DATA = 1,
  TASK_DEVICE = 2,
  DATA_DEVICE = 3
};

struct GraphSpec {
public:
  const SchedulerState &state;
  taskid_t max_in_degree = 0;
  taskit_t max_out_degree = 0;
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

  GraphSpec(const SchedulerState &state) : state(state) {
  }

  void compute_max_degree() {
    const auto &tasks = state.get_task_manager().get_tasks();
    for (const auto &task : tasks.get_compute_tasks()) {
      max_in_degree = std::max(max_in_degree, task.get_dependencies().size());
      max_out_degree = std::max(max_out_degree, task.get_dependents().size());
      max_data_in_degree = std::max(max_data_in_degree, task.get_data_dependencies().size());
      max_data_out_degree = std::max(max_data_out_degree, task.get_data_dependents().size());
      max_data_usage = std::max(dataid_t max_data_usage = 0;, task.get_unique().size());
    }
  }
};

using TaskSet = std::set<taskid_t>;

class GraphExtractor {
protected:
  const SchedulerState &state;
  std::vector<op_t> source_list;
  std::vector<op_t> target_list;
  std::set<taskid_t> visited;
  std::unordered_set<taskid_t> local_visited;
  std::unordered_map<taskid_t, taskid_t> task_index_map;
  std::unordered_map<dataid_t, dataid_t> data_index_map;
  std::unordered_map<devid_t, devid_t> device_index_map;

  GraphSpec spec;

public:
  GraphExtractor(const SchedulerState &state) : state(state) {
    source_list.reserve(400);
    target_list.reserve(400);
    visited.reserve(400);
    local_visited.reserve(400);
    task_index_map.reserve(400);
    data_index_map.reserve(400);
    device_index_map.reserve(400);
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

  [[nodiscard]] void _get_k_hop_task_dependents(TaskSet &visited, taskid_t task_id, int k) {

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
    result.resize(std::min(spec.max_tasks, visited.size()));
    std::copy_n(visited.begin(), result.size(), result.begin());

    return result;
  }

  [[nodiscard]] void _get_k_hop_task_dependencies(TaskSet &visited, taskid_t task_id, int k) {

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
      _get_k_hop_task_dependencies(task_id, k);
    }

    // Only keep the first max_tasks tasks
    TaskIDList result;
    result.resize(std::min(spec.max_tasks, visited.size()));
    std::copy_n(visited.begin(), result.size(), result.begin());

    return result;
  }

  void _get_k_hop_task_bidirectional(TaskSet &visited, taskid_t task_id, int k) {

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
      _get_k_hop_task_bidirectional(task_id, k);
    }

    // Only keep the first max_tasks tasks
    TaskIDList result;
    result.resize(std::min(spec.max_tasks, visited.size()));
    std::copy_n(visited.begin(), result.size(), result.begin());

    return result;
  }

  void get_task_task_edges(TaskIDList &sources, TaskIDList &targets, TorchArr &output) {
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

  void getFeatures(int task_id, TorchArr &output) const {
    std::span<float> sp(output.data(), output.size());
    getFeatures(task_id, sp);
  }

  template <typename Span> void getFeatures(int task_id, Span output) const {
    size_t offset = 0;
    std::apply(
        [&](const auto &...feats) {
          (..., (feats.extractFeature(task_id, output.subspan(offset, feats.getFeatureDim())),
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

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span &&output) const {
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

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span &&output) const {
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

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span &&output) const {
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

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span &&output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    output[0] = log(get_task_memcost(task, DeviceType::CPU));
    output[1] = log(get_task_memcost(task, DeviceType::GPU));
  }
};

struct EmptyTaskFeature : public StateFeature<EmptyTaskFeature> {
  size_t dimension;
  EmptyTaskFeature(const SchedulerState &state, size_t dimension)
      : StateFeature<EmptyTaskFeature>(state, NodeType::TASK), dimension(dimension) {
  }

  size_t getFeatureDimImpl() const {
    return dimension;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span &&output) const {
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

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span &&output) const {
    const auto &task_manager = state.get_task_manager();
    one_hot(task_manager.state.get_mapping(task_id), output);
  }
};