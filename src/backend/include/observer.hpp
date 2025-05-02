#pragma once
#include "devices.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include "simulator.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory>
#include <span>
#include <sys/types.h>
#include <unordered_map>

using op_t = int32_t;
using f_t = float_t;

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

template <typename Derived> struct Feature {
  const SchedulerState &state;
  const NodeType node_type;

  Feature(const SchedulerState &state, NodeType node_type) : state(state), node_type(node_type) {
  }

  [[nodiscard]] size_t getFeatureDim() const {
    return static_cast<const Derived *>(this)->getFeatureDimImpl();
  }

  template <typename ID, typename Span> void extractFeature(ID object_id, Span output) const {
    static_cast<const Derived *>(this)->extractFeatureImpl(object_id, output);
  }
};

template <typename Derived> struct SimpleFeature {

  [[nodiscard]] size_t getFeatureDim() const {
    return static_cast<const Derived *>(this)->getFeatureDimImpl();
  }

  template <typename ID, typename Span> void extractFeature(ID object_id, Span output) const {
    static_cast<const Derived *>(this)->extractFeatureImpl(object_id, output);
  }
};

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

class RuntimeExtractorInterface {
public:
  virtual ~RuntimeExtractorInterface() = default;
  virtual size_t getFeatureDim() const = 0;
  template <typename ID, typename Span>
  virtual void RuntimeExtractorInterface(ID object_id, Span output) const = 0;
};

template <typename Extractor> class ExtractorWrapper : public RuntimeExtractorInterface {
  Extractor extractor;

public:
  ExtractorWrapper(Extractor ext) : extractor(std::move(ext)) {
  }

  size_t getFeatureDim() const override {
    return extractor.getFeatureDim();
  }

  void getFeatures(int task_id, std::span<float> output) const override {
    extractor.getFeatures(task_id, output);
  }
};

class NewObserver {

private:
  std::unique_ptr<RuntimeExtractorInterface> task_extractor;
  std::unique_ptr<RuntimeExtractorInterface> data_extractor;

public:
  template <typename Extractor> void setTaskExtractor(Extractor extractor) {
    task_extractor = std::make_unique<ExtractorWrapper<Extractor>>(std::move(extractor));
  }

  template <typename Extractor> void setDataExtractor(Extractor extractor) {
    data_extractor = std::make_unique<ExtractorWrapper<Extractor>>(std::move(extractor));
  }

  size_t getTaskFeatureDim() const {
    return task_extractor->getFeatureDim();
  }

  size_t getDataFeatureDim() const {
    return data_extractor->getFeatureDim();
  }

  void getTaskFeatures(taskid_t task_id, std::span<float> output) const {
    task_extractor->getFeatures(task_id, output);
  }

  void getDataFeatures(dataid_t data_id, std::span<float> output) const {
    data_extractor->getFeatures(data_id, output);
  }

}

struct InDegreeTaskFeature : public Feature<InDegreeTaskFeature> {
  InDegreeTaskFeature(const SchedulerState &state)
      : Feature<InDegreeTaskFeature>(state, NodeType::TASK) {
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

struct OutDegreeTaskFeature : public Feature<OutDegreeTaskFeature> {
  OutDegreeTaskFeature(const SchedulerState &state)
      : Feature<OutDegreeTaskFeature>(state, NodeType::TASK) {
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

struct DurationTaskFeature : public Feature<DurationTaskFeature> {
  DurationTaskFeature(const SchedulerState &state)
      : Feature<DurationTaskFeature>(state, NodeType::TASK) {
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

struct MemoryTaskFeature : public Feature<MemoryTaskFeature> {
  MemoryTaskFeature(const SchedulerState &state)
      : Feature<MemoryTaskFeature>(state, NodeType::TASK) {
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

struct EmptyTaskFeature : public Feature<EmptyTaskFeature> {
  size_t dimension;
  EmptyTaskFeature(const SchedulerState &state, size_t dimension)
      : Feature<EmptyTaskFeature>(state, NodeType::TASK), dimension(dimension) {
  }

  size_t getFeatureDimImpl() const {
    return dimension;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span &&output) const {
  }
};

struct OneHotMappedDeviceTaskFeature : public Feature<OneHotMappedDeviceTaskFeature> {
  OneHotMappedDeviceTaskFeature(const SchedulerState &state)
      : Feature<OneHotMappedDeviceTaskFeature>(state, NodeType::TASK) {
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

struct NormalizationInfo {
  depcount_t max_in_degree;
  double average_in_degree;
  double stddev_in_degree;

  depcount_t max_out_degre;
  double average_out_degree;
  double stddev_out_degree;

  timecount_t serial_execution_time;
  std::array<double, num_device_types> average_duration;
  std::array<double, num_device_types> stddev_duration;

  std::array<double, num_device_types> average_task_memcost;
  std::array<double, num_device_types> stddev_task_memcost;

  double average_data_size;
  double stddev_data_size;
};

struct FeatureDimSpec {
  std::size_t max_candidates = 5;
  std::size_t max_devices = 8;

  std::size_t max_edges_tasks_tasks = 20;
  std::size_t max_edges_tasks_data = 20;

  std::size_t max_tasks = max_candidates * max_devices;
  std::size_t max_data = max_tasks * max_edges_tasks_data;

  std::size_t task_feature_dim = 8 + max_devices;
  std::size_t data_feature_dim = 1 + max_devices;
  std::size_t device_feature_dim = 8 + max_devices;

  std::size_t task_data_feature_dim = 3;
  std::size_t task_device_feature_dim = 2;
  std::size_t task_task_feature_dim = 1;

  FeatureDimSpec() = default;

  FeatureDimSpec(std::size_t max_candidates, std::size_t max_devices,
                 std::size_t max_edges_tasks_tasks, std::size_t max_edges_tasks_data)
      : max_candidates(max_candidates), max_devices(max_devices),
        max_edges_tasks_tasks(max_edges_tasks_tasks), max_edges_tasks_data(max_edges_tasks_data),
        max_tasks(max_candidates * max_devices), max_data(max_tasks * max_edges_tasks_data) {
  }
};

template <typename... TaskFeatures> class NewObserverBase {

public:
  std::reference_wrapper<const SchedulerState> state;

  FeatureExtractor<... TaskFeatures> task_feature_extractor;

  NewObserverBase(const Simulator &simulator) : state(simulator.get_state()) {
  }
};

struct Features {
  f_t *features = nullptr;
  std::size_t feature_dim = 0;
  std::size_t feature_len = 0;
};

struct TaskFeatures : public Features {};
struct DataFeatures : public Features {};
struct DeviceFeatures : public Features {};
struct TaskDataEdges : public Features {
  op_t *data2id;
  size_t data2id_len;
  op_t *edges;
};

struct TaskDeviceEdges : public Features {
  op_t *device2id;
  size_t device2id_len;
  op_t *edges;
};

struct DataDeviceEdges : public Features {
  op_t *data2id;
  size_t data2id_len;
  op_t *device2id;
  size_t device2id_len;
  op_t *edges;
};

struct TaskTaskEdges : public Features {
  op_t *edges;
};

class Observer {
private:
  void reserve() {
    source_list.reserve(1000);
    target_list.reserve(1000);
  }

public:
  std::reference_wrapper<const SchedulerState> state;
  NormalizationInfo graph_info;

  std::vector<op_t> source_list;
  std::vector<op_t> target_list;

  Observer(const Simulator &simulator) : state(simulator.get_state()) {
    reserve();
  }

  Observer(const SchedulerState &state) : state(state) {
    reserve();
  }

  void read_state(const Simulator &simulator) {
    state = simulator.get_state();
  }

  static double get_in_degree(const ComputeTask &task) {
    return static_cast<double>(task.get_dependencies().size());
  }

  static double get_out_degree(const ComputeTask &task) {
    return static_cast<double>(task.get_dependents().size());
  }

  static double get_duration(const ComputeTask &task, DeviceType arch) {
    return static_cast<double>(task.get_variant(arch).get_observed_time());
  }

  static double get_task_memcost(const ComputeTask &task, DeviceType arch) {
    return static_cast<double>(task.get_variant(arch).get_mem());
  }

  static double get_task_data_memcost(const ComputeTask &task, const Data &data) {
    double memcost = 0;
    for (const auto &data_id : task.get_unique()) {
      memcost += static_cast<double>(data.get_size(data_id));
    }
    return memcost;
  }

  int get_n_tasks() const {
    return state.get().get_task_manager().get_tasks().size();
  }

  void preprocess_global(const Tasks &tasks, const Data &data) {

    // NOTE(wlr): This is AN AWFUL estimator of stddev
    //            I just want something running for now
    //            TODO: Implement this better

    double in_degree_sum = 0;
    double in_degree_sum2 = 0;
    depcount_t max_in_degree = 0;

    double out_degree_sum = 0;
    double out_degree_sum2 = 0;
    depcount_t max_out_degree = 0;

    std::array<double, num_device_types> duration_sum = {0, 0};
    std::array<double, num_device_types> duration_sum2 = {0, 0};
    timecount_t duration_worst_sum = 0;

    std::array<double, num_device_types> task_memcost_sum = {0, 0};
    std::array<double, num_device_types> task_memcost_sum2 = {0, 0};

    double data_size_sum = 0;
    double data_size_sum2 = 0;

    for (const auto &task : tasks.get_compute_tasks()) {
      in_degree_sum += get_in_degree(task);
      in_degree_sum2 += get_in_degree(task) * get_in_degree(task);
      max_in_degree = std::max(max_in_degree, (int32_t)task.get_dependencies().size());

      out_degree_sum += get_out_degree(task);
      out_degree_sum2 += get_out_degree(task) * get_out_degree(task);
      max_out_degree = std::max(max_out_degree, (int32_t)task.get_dependents().size());

      double task_data_memcost = get_task_data_memcost(task, data);

      timecount_t duration_worst = 0;
      for (std::size_t i = 0; i < num_device_types; i++) {
        const auto d = get_duration(task, static_cast<DeviceType>(i));
        duration_sum[i] += d;
        duration_sum2[i] += d * d;

        const auto m = get_task_memcost(task, static_cast<DeviceType>(i)) + task_data_memcost;
        task_memcost_sum[i] += m;
        task_memcost_sum2[i] += m * m;

        if (static_cast<timecount_t>(d) > duration_worst) {
          duration_worst = static_cast<timecount_t>(d);
        }
      }

      duration_worst_sum += duration_worst;
    }

    for (auto size : data.get_sizes()) {
      data_size_sum += static_cast<double>(size);
      data_size_sum2 += static_cast<double>(size) * static_cast<double>(size);
    }

    const auto num_tasks = static_cast<double>(tasks.compute_size());
    const auto num_data = static_cast<double>(data.size());

    graph_info.max_in_degree = max_in_degree;
    graph_info.average_in_degree = in_degree_sum / num_tasks;
    graph_info.stddev_in_degree = std::sqrt(
        in_degree_sum2 / num_tasks - graph_info.average_in_degree * graph_info.average_in_degree);

    graph_info.max_out_degre = max_out_degree;
    graph_info.average_out_degree = out_degree_sum / num_tasks;
    graph_info.stddev_out_degree =
        std::sqrt(out_degree_sum2 / num_tasks -
                  graph_info.average_out_degree * graph_info.average_out_degree);

    graph_info.serial_execution_time = duration_worst_sum;
    for (std::size_t i = 0; i < num_device_types; i++) {
      graph_info.average_duration[i] = duration_sum[i] / num_tasks;
      graph_info.stddev_duration[i] =
          std::sqrt(duration_sum2[i] / num_tasks -
                    graph_info.average_duration[i] * graph_info.average_duration[i]);

      graph_info.average_task_memcost[i] = task_memcost_sum[i] / num_tasks;
      graph_info.stddev_task_memcost[i] =
          std::sqrt(task_memcost_sum2[i] / num_tasks -
                    graph_info.average_task_memcost[i] * graph_info.average_task_memcost[i]);
    }

    graph_info.average_data_size = data_size_sum / num_data;
    graph_info.stddev_data_size = std::sqrt(
        data_size_sum2 / num_data - graph_info.average_data_size * graph_info.average_data_size);

    SPDLOG_DEBUG("Graph properties:");
    SPDLOG_DEBUG("Max in-degree: {}", graph_info.max_in_degree);
    SPDLOG_DEBUG("Average in-degree: {}", graph_info.average_in_degree);
    SPDLOG_DEBUG("Stddev in-degree: {}", graph_info.stddev_in_degree);

    SPDLOG_DEBUG("Max out-degree: {}", graph_info.max_out_degre);
    SPDLOG_DEBUG("Average out-degree: {}", graph_info.average_out_degree);
    SPDLOG_DEBUG("Stddev out-degree: {}", graph_info.stddev_out_degree);

    SPDLOG_DEBUG("Serial execution time: {}", graph_info.serial_execution_time);
    for (std::size_t i = 0; i < num_device_types; i++) {
      SPDLOG_DEBUG("Average duration on device type {}: {}", to_string(static_cast<DeviceType>(i)),
                   graph_info.average_duration[i]);
      SPDLOG_DEBUG("Stddev duration on device type {}: {}", to_string(static_cast<DeviceType>(i)),
                   graph_info.stddev_duration[i]);

      SPDLOG_DEBUG("Average task memcost on device type {}: {}",
                   to_string(static_cast<DeviceType>(i)), graph_info.average_task_memcost[i]);
      SPDLOG_DEBUG("Stddev task memcost on device type {}: {}",
                   to_string(static_cast<DeviceType>(i)), graph_info.stddev_task_memcost[i]);
    }

    SPDLOG_DEBUG("Average data size: {}", graph_info.average_data_size);
    SPDLOG_DEBUG("Stddev data size: {}", graph_info.stddev_data_size);
  }

  void global_features() {
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    const auto &data = s.get_data_manager().get_data();
    preprocess_global(tasks, data);
  }

  void get_device_mask_int8(taskid_t task_id, int8_t *valid_devices, size_t max_devices) const {
    const auto &s = this->state.get();
    assert(max_devices > s.get_device_manager().get_devices().size());
    s.fill_supported_devices(task_id, std::span<int8_t>(valid_devices, max_devices));
  }

  [[nodiscard]] TaskIDList get_active_tasks() const {
    const auto &s = this->state.get();
    return s.counts.get_active_task_list();
  }

  [[nodiscard]] TaskIDList get_k_hop_dependents(taskid_t *initial_tasks, size_t n, int k) const {
    // NOTE(wlr): Sorry this is messy, I just wanted to get something running

    if (k <= 0) {
      return {};
    }

    std::span initial(initial_tasks, n);

    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    TaskIDList result;
    result.reserve(initial.size());

    std::unordered_set<taskid_t> visited;
    std::queue<taskid_t> q;

    for (const auto &task_id : initial) {
      if (visited.insert(task_id).second) {
        q.push(task_id);
      }
    }

    int current_hop = 0;

    while (!q.empty() && current_hop < k) {
      std::size_t level_size = q.size();

      for (std::size_t i = 0; i < level_size; ++i) {
        taskid_t current_task_id = q.front();
        q.pop();

        const auto &task = tasks.get_compute_task(current_task_id);
        for (const auto &dep_id : task.get_dependents()) {
          if (visited.insert(dep_id).second) {
            q.push(dep_id);
            result.push_back(dep_id);
          }
        }
      }

      current_hop++;
    }

    return result;
  }

  [[nodiscard]] TaskIDList get_k_hop_dependencies(taskid_t *initial_tasks, size_t n, int k) const {
    // NOTE(wlr): Sorry this is messy, I just wanted to get something running

    if (k <= 0) {
      return {};
    }

    std::span initial(initial_tasks, n);

    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    TaskIDList result;
    result.reserve(initial.size());

    std::unordered_set<taskid_t> visited;
    std::queue<taskid_t> q;

    for (const auto &task_id : initial) {
      if (visited.insert(task_id).second) {
        q.push(task_id);
      }
    }

    int current_hop = 0;

    while (!q.empty() && current_hop < k) {
      std::size_t level_size = q.size();

      for (std::size_t i = 0; i < level_size; ++i) {
        taskid_t current_task_id = q.front();
        q.pop();

        const auto &task = tasks.get_compute_task(current_task_id);
        for (const auto &dep_id : task.get_dependencies()) {
          if (visited.insert(dep_id).second) {
            q.push(dep_id);
            result.push_back(dep_id);
          }
        }
      }

      current_hop++;
    }

    return result;
  }

  void get_task_features(taskid_t task_id, std::span<f_t> features) const {
    const auto &devices = this->state.get().get_device_manager().get_devices();
    const std::size_t FEATURE_LENGTH = 8 + devices.size();
    assert(features.size() == FEATURE_LENGTH);

    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    const auto &task = tasks.get_compute_task(task_id);

    features[0] = guarded_divide(get_in_degree(task), graph_info.average_in_degree);
    features[1] = guarded_divide(get_out_degree(task), graph_info.average_out_degree);

    features[2] = static_cast<f_t>(s.is_mapped(task_id));
    features[3] = static_cast<f_t>(s.is_reserved(task_id));
    features[4] = static_cast<f_t>(s.is_launched(task_id));
    features[5] = static_cast<f_t>(s.is_completed(task_id));
    if (task_manager.state.get_state(task_id) != TaskState::SPAWNED) {
      // One-hot encode the device
      for (std::size_t i = 0; i < devices.size(); i++) {
        features[6 + i] = static_cast<f_t>(task_manager.state.get_mapping(task_id) == i);
      }
    } else {
      for (std::size_t i = 0; i < devices.size(); i++) {
        features[6 + i] = 0;
      }
    }
    features[6 + devices.size()] = 0; // static_cast<f_t>(task.get_depth());
    features[7 + devices.size()] = 0; // is mapping candidate (set in Python layer)

    // // Test print
    // std::cout << "Task features: ";
    // for (std::size_t i = 0; i < features.size(); i++) {
    //   std::cout << features[i] << " ";
    // }
    // std::cout << std::endl;
  }

  void get_device_features(devid_t device_id, std::span<f_t> features) const {
    const auto &s = this->state.get();
    const auto &device_manager = s.get_device_manager();
    const auto &devices = s.get_device_manager().get_devices();

    const std::size_t FEATURE_LENGTH = 8 + devices.size();
    assert(features.size() == FEATURE_LENGTH);

    const auto &device = devices.get_device(device_id);

    features[0] = static_cast<f_t>(device.arch == DeviceType::CPU);
    features[1] = static_cast<f_t>(device.arch == DeviceType::GPU);

    double total_mapped_mem = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_mapped_mem += static_cast<double>(device_manager.get_mem<TaskState::MAPPED>(i));
    }

    double total_reserved_mem = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_reserved_mem += static_cast<double>(device_manager.get_mem<TaskState::RESERVED>(i));
    }

    double total_launched_mem = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_launched_mem += static_cast<double>(device_manager.get_mem<TaskState::LAUNCHED>(i));
    }

    auto mapped_mem = static_cast<double>(device_manager.get_mem<TaskState::MAPPED>(device_id));
    auto reserved_mem = static_cast<double>(device_manager.get_mem<TaskState::RESERVED>(device_id));
    auto launched_mem = static_cast<double>(device_manager.get_mem<TaskState::LAUNCHED>(device_id));

    features[2] = guarded_divide(mapped_mem, total_mapped_mem);
    features[3] = guarded_divide(reserved_mem, total_reserved_mem);
    features[4] = guarded_divide(launched_mem, total_launched_mem);

    double total_mapped_time = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_mapped_time += static_cast<double>(s.costs.get_mapped_time(i));
    }
    double total_reserved_time = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_reserved_time += static_cast<double>(s.costs.get_reserved_time(i));
    }

    double total_launched_time = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_launched_time += static_cast<double>(s.costs.get_launched_time(i));
    }

    auto mapped_time = static_cast<double>(s.costs.get_mapped_time(device_id));
    auto reserved_time = static_cast<double>(s.costs.get_reserved_time(device_id));
    auto launched_time = static_cast<double>(s.costs.get_launched_time(device_id));

    features[5] = guarded_divide(mapped_time, total_mapped_time);
    features[6] = guarded_divide(reserved_time, total_reserved_time);
    features[7] = guarded_divide(launched_time, total_launched_time);

    // One-hot encode the device id
    for (std::size_t i = 0; i < devices.size(); i++) {
      features[8 + i] = static_cast<f_t>(device_id == i);
    }
  }

  void get_data_features(dataid_t data_id, std::span<f_t> features) const {
    const auto &s = this->state.get();

    const auto &data_manager = s.get_data_manager();
    const auto &devices = s.get_device_manager().get_devices();
    const auto &data = s.get_data_manager().get_data();

    const auto &data_size = static_cast<double>(data.get_size(data_id));

    const std::size_t FEATURE_LENGTH = 1 + devices.size();
    assert(features.size() == FEATURE_LENGTH);

    features[0] = guarded_divide(data_size, graph_info.average_data_size);
    for (std::size_t i = 0; i < devices.size(); i++) {
      features[i + 1] = static_cast<f_t>(data_manager.get_mapped_locations().is_valid(data_id, i));
    }
  }

  void get_task_device_features(taskid_t task_id, devid_t device_id, std::span<f_t> features) {
    constexpr std::size_t FEATURE_LENGTH = 2;
    assert(features.size() == FEATURE_LENGTH);

    const auto &s = this->state.get();
    const auto &task = s.get_task_manager().get_tasks().get_compute_task(task_id);
    const auto &device = s.get_device_manager().get_devices().get_device(device_id);

    DeviceType arch = device.arch;

    features[0] = guarded_divide(get_duration(task, arch),
                                 graph_info.average_duration.at(static_cast<std::size_t>(arch)));

    features[1] =
        guarded_divide(get_task_memcost(task, arch),
                       graph_info.average_task_memcost.at(static_cast<std::size_t>(arch)));
  }

  void get_task_data_features(taskid_t task_id, dataid_t data_id, std::span<f_t> features) const {
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    const auto &data = s.get_data_manager().get_data();

    const auto &task = tasks.get_compute_task(task_id);
    const auto data_size = static_cast<double>(data.get_size(data_id));

    const std::size_t FEATURE_LENGTH = 3;
    assert(features.size() == FEATURE_LENGTH);

    double task_data_memcost = get_task_data_memcost(task, data);
    bool is_read =
        std::find(task.get_read().begin(), task.get_read().end(), data_id) != task.get_read().end();
    bool is_write = std::find(task.get_write().begin(), task.get_write().end(), data_id) !=
                    task.get_write().end();

    features[0] = guarded_divide(data_size, task_data_memcost);
    features[1] = static_cast<f_t>(is_read);
    features[2] = static_cast<f_t>(is_write);
  }

  void get_task_task_features(taskid_t task_id, taskid_t dep_id, std::span<f_t> features) const {
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    const auto &data_manager = s.get_data_manager();

    const auto &task = tasks.get_compute_task(task_id);
    const auto &dep = tasks.get_compute_task(dep_id);

    const std::size_t FEATURE_LENGTH = 1;
    assert(features.size() == FEATURE_LENGTH);

    double total_mem_cost = 0;
    for (auto data_id : task.get_unique()) {
      total_mem_cost += static_cast<double>(data_manager.get_data().get_size(data_id));
    }

    auto shared_mem =
        static_cast<double>(data_manager.shared_size(task.get_unique(), dep.get_unique()));

    features[0] = guarded_divide(shared_mem, total_mem_cost);
  }

  void get_data_device_features(dataid_t data_id, devid_t device_id,
                                std::span<f_t> features) const {
    const std::size_t FEATURE_LENGTH = 1;
    assert(features.size() == FEATURE_LENGTH);
    features[0] = 1.0;
  }

  std::size_t get_number_of_unique_data(taskid_t *ids, std::size_t n) {
    std::set<dataid_t> unique_set;
    for (std::size_t i = 0; i < n; i++) {
      const auto &task = state.get().get_task_manager().get_tasks().get_compute_task(ids[i]);
      for (auto data_id : task.get_unique()) {
        unique_set.insert(data_id);
      }
    }
    return unique_set.size();
  }

  auto get_unique_datamap(std::span<taskid_t> task_ids) {
    std::map<dataid_t, std::size_t> unique_map;

    for (auto task_id : task_ids) {
      const auto &task = state.get().get_task_manager().get_tasks().get_compute_task(task_id);
      for (auto data_id : task.get_unique()) {
        if (unique_map.find(data_id) == unique_map.end()) {
          unique_map[data_id] = unique_map.size();
        }
      }
    }
    return unique_map;
  }

  TaskTaskEdges get_task_task_edges(taskid_t *source_pointer, std::size_t n,
                                    taskid_t *target_pointer, std::size_t m) {
    std::span source(source_pointer, n);
    std::span targets(target_pointer, m);
    TaskTaskEdges edges;

    std::unordered_map<taskid_t, taskid_t> target_index;
    for (std::size_t i = 0; i < targets.size(); i++) {
      target_index[targets[i]] = i;
    }

    source_list.clear();
    target_list.clear();

    taskid_t source_idx = 0;
    for (const auto &source_id : source) {
      const auto &source_task =
          state.get().get_task_manager().get_tasks().get_compute_task(source_id);
      for (const auto &dep_id : source_task.get_dependencies()) {
        if (target_index.find(dep_id) != target_index.end()) {
          source_list.push_back(source_idx);
          target_list.push_back(target_index[dep_id]);
        }
      }
      source_idx++;
    }

    edges.feature_dim = 1;
    edges.feature_len = source_list.size();

    edges.edges = static_cast<op_t *>(malloc(2 * edges.feature_len * sizeof(op_t)));

    // fill COO edges row major (2 x N)
    for (std::size_t i = 0; i < edges.feature_len; i++) {
      edges.edges[i] = source_list[i];
      edges.edges[i + edges.feature_len] = target_list[i];
    }

    edges.features =
        static_cast<f_t *>(malloc(edges.feature_len * edges.feature_dim * sizeof(f_t)));

    std::span<f_t> feature_span(edges.features, edges.feature_len * edges.feature_dim);
    for (std::size_t i = 0; i < edges.feature_len; i++) {
      get_task_task_features(source[source_list[i]], targets[target_list[i]],
                             feature_span.subspan(i * edges.feature_dim, edges.feature_dim));
    }

    return edges;
  }

  TaskDataEdges get_task_data_edges(taskid_t *task_ids_pointer, std::size_t n) {
    std::span task_ids(task_ids_pointer, n);
    TaskDataEdges edges;

    std::unordered_map<dataid_t, int32_t> data_map;

    source_list.clear();
    target_list.clear();

    for (std::size_t i = 0; i < task_ids.size(); i++) {
      const auto &task_id = task_ids[i];
      const auto &task = state.get().get_task_manager().get_tasks().get_compute_task(task_id);
      for (auto data_id : task.get_unique()) {

        if (data_map.find(data_id) == data_map.end()) {
          data_map[data_id] = data_map.size();
        }

        source_list.push_back(i);
        target_list.push_back(data_map[data_id]);
      }
    }

    edges.feature_dim = 3;
    edges.feature_len = source_list.size();
    edges.data2id_len = data_map.size();

    edges.edges = static_cast<op_t *>(malloc(2 * edges.feature_len * sizeof(op_t)));

    // fill COO edges row major (2 x N)
    for (std::size_t i = 0; i < edges.feature_len; i++) {
      edges.edges[i] = source_list[i];
      edges.edges[i + edges.feature_len] = target_list[i];
    }

    edges.data2id = static_cast<op_t *>(malloc(edges.data2id_len * sizeof(op_t)));

    for (const auto &data_id : data_map) {
      edges.data2id[data_id.second] = data_id.first;
    }

    edges.features =
        static_cast<f_t *>(malloc(edges.feature_len * edges.feature_dim * sizeof(f_t)));

    std::span<f_t> feature_span(edges.features, edges.feature_len * edges.feature_dim);

    for (std::size_t i = 0; i < edges.feature_len; i++) {
      get_task_data_features(task_ids[source_list[i]], edges.data2id[target_list[i]],
                             feature_span.subspan(i * edges.feature_dim, edges.feature_dim));
    }

    return edges;
  }

  TaskDeviceEdges get_task_device_edges(taskid_t *task_ids_pointer, std::size_t n) {
    std::span task_ids(task_ids_pointer, n);
    TaskDeviceEdges edges;

    std::map<devid_t, int32_t> device_map;

    source_list.clear();
    target_list.clear();

    for (std::size_t i = 0; i < task_ids.size(); i++) {
      const auto &task_id = task_ids[i];
      const auto &task = state.get().get_task_manager().get_tasks().get_compute_task(task_id);
      const auto &supported_architectures = task.get_supported_architectures();

      for (auto arch : supported_architectures) {
        const auto &device_ids = state.get().get_device_manager().get_devices().get_devices(arch);
        for (auto device_id : device_ids) {

          if (device_map.find(device_id) == device_map.end()) {
            device_map[device_id] = device_map.size();
          }

          source_list.push_back(i);
          target_list.push_back(device_map[device_id]);
        }
      }
    }

    edges.device2id_len = device_map.size();
    edges.feature_dim = 2;
    edges.feature_len = source_list.size();

    edges.edges = static_cast<op_t *>(malloc(2 * edges.feature_len * sizeof(op_t)));

    // fill COO edges row major (2 x N)
    for (std::size_t i = 0; i < edges.feature_len; i++) {
      edges.edges[i] = source_list[i];
      edges.edges[i + edges.feature_len] = target_list[i];
    }

    edges.device2id = static_cast<op_t *>(malloc(edges.device2id_len * sizeof(op_t)));

    for (const auto &device_id : device_map) {
      edges.device2id[device_id.second] = device_id.first;
    }

    edges.features =
        static_cast<f_t *>(malloc(edges.feature_len * edges.feature_dim * sizeof(f_t)));

    std::span<f_t> feature_span(edges.features, edges.feature_len * edges.feature_dim);

    for (std::size_t i = 0; i < edges.feature_len; i++) {
      get_task_device_features(task_ids[source_list[i]], edges.device2id[target_list[i]],
                               feature_span.subspan(i * edges.feature_dim, edges.feature_dim));
    }

    return edges;
  }

  DataDeviceEdges get_data_device_edges(taskid_t *task_ids_pointer, std::size_t n) {
    std::span task_ids(task_ids_pointer, n);
    DataDeviceEdges edges;

    source_list.clear();
    target_list.clear();

    auto data_map = get_unique_datamap(task_ids);

    edges.data2id_len = data_map.size();
    edges.data2id = static_cast<op_t *>(malloc(edges.data2id_len * sizeof(op_t)));

    for (const auto &data_id : data_map) {
      edges.data2id[data_id.second] = data_id.first;
    }

    std::map<devid_t, int32_t> device_map;

    for (const auto &data_id : data_map) {
      const auto &valid_sources =
          state.get().get_data_manager().get_valid_mapped_locations(data_id.first);
      for (const auto &device_id : valid_sources) {

        if (device_map.find(device_id) == device_map.end()) {
          device_map[device_id] = device_map.size();
        }

        source_list.push_back(data_id.second);
        target_list.push_back(device_map[device_id]);
      }
    }

    edges.device2id_len = device_map.size();
    edges.device2id = static_cast<op_t *>(malloc(edges.device2id_len * sizeof(op_t)));
    for (const auto &device_id : device_map) {
      edges.device2id[device_id.second] = device_id.first;
    }

    edges.feature_dim = 1;
    edges.feature_len = source_list.size();

    edges.edges = static_cast<op_t *>(malloc(2 * edges.feature_len * sizeof(op_t)));

    // fill COO edges row major (2 x N)

    for (std::size_t i = 0; i < edges.feature_len; i++) {
      edges.edges[i] = source_list[i];
      edges.edges[i + edges.feature_len] = target_list[i];
    }

    edges.features =
        static_cast<f_t *>(malloc(edges.feature_len * edges.feature_dim * sizeof(f_t)));
    std::span<f_t> feature_span(edges.features, edges.feature_len * edges.feature_dim);
    for (std::size_t i = 0; i < edges.feature_len; i++) {
      get_data_device_features(edges.data2id[source_list[i]], edges.device2id[target_list[i]],
                               feature_span.subspan(i * edges.feature_dim, edges.feature_dim));
    }

    return edges;
  }

  TaskFeatures get_task_features(taskid_t *task_ids_pointer, std::size_t n) {
    std::span task_ids(task_ids_pointer, n);
    TaskFeatures features;

    const auto &devices = state.get().get_device_manager().get_devices();
    features.feature_dim = 8 + devices.size();
    features.feature_len = task_ids.size();

    features.features =
        static_cast<f_t *>(malloc(features.feature_len * features.feature_dim * sizeof(f_t)));
    std::span<f_t> feature_span(features.features, features.feature_len * features.feature_dim);

    for (std::size_t i = 0; i < task_ids.size(); i++) {
      auto start_i = i * features.feature_dim;
      get_task_features(task_ids[i], feature_span.subspan(start_i, features.feature_dim));
    }

    return features;
  }

  DeviceFeatures get_device_features(const devid_t *device_ids_pointer, std::size_t n) {
    std::span device_ids(device_ids_pointer, n);
    DeviceFeatures features;
    const auto &devices = state.get().get_device_manager().get_devices();
    features.feature_dim = 8 + devices.size();
    features.feature_len = device_ids.size();

    features.features =
        static_cast<f_t *>(malloc(features.feature_len * features.feature_dim * sizeof(f_t)));
    std::span<f_t> feature_span(features.features, features.feature_len * features.feature_dim);

    for (std::size_t i = 0; i < device_ids.size(); i++) {
      get_device_features(device_ids[i],
                          feature_span.subspan(i * features.feature_dim, features.feature_dim));
    }

    return features;
  }

  DataFeatures get_data_features(dataid_t *data_ids_pointer, std::size_t n) {
    std::span data_ids(data_ids_pointer, n);
    DataFeatures features;
    features.feature_dim = 1 + state.get().get_device_manager().size();
    features.feature_len = data_ids.size();

    features.features =
        static_cast<f_t *>(malloc(features.feature_len * features.feature_dim * sizeof(f_t)));

    std::span<f_t> feature_span(features.features, features.feature_len * features.feature_dim);

    for (std::size_t i = 0; i < data_ids.size(); i++) {
      get_data_features(data_ids[i],
                        feature_span.subspan(i * features.feature_dim, features.feature_dim));
    }

    return features;
  }
};