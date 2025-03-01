// nanobind_module.cpp
#include "devices.hpp"
#include "features.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include "simulator.hpp"
#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/trampoline.h>
#include <span>
#include <sys/types.h>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// Runtime Polymorphism Interface

struct IFeature {
  virtual ~IFeature() = default;
  virtual size_t get_feature_dim() const = 0;
  virtual void extract_feature(uint32_t object_id, std::span<float> output) const = 0;
};

template <typename Derived> struct FeatureAdapter : IFeature {
  Derived feature;
  FeatureAdapter(Derived feat) : feature(std::move(feat)) {
  }

  size_t get_feature_dim() const override {
    return feature.getFeatureDim();
  }

  void extract_feature(uint32_t object_id, std::span<float> output) const override {
    feature.extractFeature(object_id, output);
  }
};

struct IEdgeFeature {
  virtual ~IEdgeFeature() = default;
  virtual size_t get_feature_dim() const = 0;
  virtual void extract_feature(uint32_t source_id, uint32_t target_id,
                               std::span<float> output) const = 0;
};

template <typename Derived> struct EdgeFeatureAdapter : IEdgeFeature {
  Derived feature;
  EdgeFeatureAdapter(Derived feat) : feature(std::move(feat)) {
  }

  size_t get_feature_dim() const override {
    return feature.getFeatureDim();
  }

  void extract_feature(uint32_t source_id, uint32_t target_id,
                       std::span<float> output) const override {
    feature.extractFeature(source_id, target_id, output);
  }
};
struct RuntimeFeatureExtractor {
  std::vector<std::shared_ptr<IFeature>> features;

  void addFeature(std::shared_ptr<IFeature> feature) {
    features.push_back(std::move(feature));
  }

  size_t getFeatureDim() const {
    size_t total = 0;
    for (const auto &f : features)
      total += f->get_feature_dim();
    return total;
  }

  void getFeatures(uint32_t object_id, std::span<float> arr) const {
    float *data = arr.data();
    size_t offset = 0;
    for (const auto &f : features) {
      size_t dim = f->get_feature_dim();
      std::span<float> sp(data + offset, dim);
      f->extract_feature(object_id, sp);
      offset += dim;
    }
  }
};

struct RuntimeEdgeFeatureExtractor {
  std::vector<std::shared_ptr<IEdgeFeature>> features;

  void addFeature(std::shared_ptr<IEdgeFeature> feature) {
    features.push_back(std::move(feature));
  }

  size_t getFeatureDim() const {
    size_t total = 0;
    for (const auto &f : features)
      total += f->get_feature_dim();
    return total;
  }

  void getFeatures(uint32_t source_id, uint32_t target_id, std::span<float> arr) const {
    float *data = arr.data();
    size_t offset = 0;
    for (const auto &f : features) {
      size_t dim = f->get_feature_dim();
      std::span<float> sp(data + offset, dim);
      f->extract_feature(source_id, target_id, sp);
      offset += dim;
    }
  }
};

template <typename E>
void get_features_batch(const E &extractor, const TorchInt64Arr1D &object_ids,
                        TorchFloatArr2D &tensor) {
  float *data = tensor.data();
  auto v = object_ids.view();
  int num_cols = extractor.getFeatureDim();
  for (int i = 0; i < object_ids.size(); ++i) {
    std::span<float> row(data + i * num_cols, num_cols);
    extractor.getFeatures(static_cast<uint32_t>(v(i)), row);
  }
}

template <typename E>
void get_edge_features_batch(const E &extractor, TorchInt64Arr2D &edges, TorchFloatArr2D &tensor) {
  float *data = tensor.data();
  int num_cols = extractor.getFeatureDim();
  auto v = edges.view();
  for (int i = 0; i < edges.shape(1); ++i) {
    std::span<float> row(data + i * num_cols, num_cols);
    extractor.getFeatures(static_cast<uint32_t>(v(0, i)), static_cast<uint32_t>(v(1, i)), row);
  }
}

template <typename FEType> void bind_int_feature(nb::module_ &m, const char *class_name) {
  nb::class_<FEType>(m, class_name)
      .def(nb::init<SchedulerState &, size_t>())
      .def_prop_ro("feature_dim", &FEType::getFeatureDim)
      .def("extract_feature",
           [](const FEType &self, uint32_t task_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.extractFeature(task_id, sp);
           })
      .def_static("create", [](SchedulerState &state, size_t n) -> std::shared_ptr<IFeature> {
        return std::make_shared<FeatureAdapter<FEType>>(FEType(state, n));
      });
}

template <typename FEType> void bind_int_edge_feature(nb::module_ &m, const char *class_name) {
  nb::class_<FEType>(m, class_name)
      .def(nb::init<SchedulerState &, size_t>())
      .def_prop_ro("feature_dim", &FEType::getFeatureDim)
      .def("extract_feature",
           [](const FEType &self, uint32_t source_id, uint32_t target_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.extractFeature(source_id, target_id, sp);
           })
      .def_static("create", [](SchedulerState &state, size_t n) -> std::shared_ptr<IEdgeFeature> {
        return std::make_shared<EdgeFeatureAdapter<FEType>>(FEType(state, n));
      });
}

template <typename FEType> void bind_state_feature(nb::module_ &m, const char *class_name) {
  nb::class_<FEType>(m, class_name)
      .def(nb::init<SchedulerState &>())
      .def_prop_ro("feature_dim", &FEType::getFeatureDim)
      .def("extract_feature",
           [](const FEType &self, uint32_t task_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.extractFeature(task_id, sp);
           })
      .def_static("create", [](SchedulerState &n) -> std::shared_ptr<IFeature> {
        return std::make_shared<FeatureAdapter<FEType>>(FEType(n));
      });
}

template <typename FEType> void bind_state_edge_feature(nb::module_ &m, const char *class_name) {
  nb::class_<FEType>(m, class_name)
      .def(nb::init<SchedulerState &>())
      .def_prop_ro("feature_dim", &FEType::getFeatureDim)
      .def("extract_feature",
           [](const FEType &self, uint32_t source_id, uint32_t target_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.extractFeature(source_id, target_id, sp);
           })
      .def_static("create", [](SchedulerState &n) -> std::shared_ptr<IEdgeFeature> {
        return std::make_shared<EdgeFeatureAdapter<FEType>>(FEType(n));
      });
}

template <typename... Features>
void bind_feature_extractor(nb::module_ &m, const char *class_name) {
  using FEType = FeatureExtractor<Features...>;
  nb::class_<FEType>(m, class_name)
      .def(nb::init<Features...>())
      .def_prop_ro("feature_dim", &FEType::getFeatureDim)
      .def("get_features",
           [](const FEType &self, uint32_t task_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.getFeatures(task_id, sp);
           })
      .def("get_features_batch", &get_features_batch<FEType>);
}

template <typename... Features>
void bind_edge_feature_extractor(nb::module_ &m, const char *class_name) {
  using FEType = EdgeFeatureExtractor<Features...>;
  nb::class_<FEType>(m, class_name)
      .def(nb::init<Features...>())
      .def_prop_ro("feature_dim", &FEType::getFeatureDim)
      .def("get_features",
           [](const FEType &self, uint32_t source_id, uint32_t target_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.getFeatures(source_id, target_id, sp);
           })
      .def("get_features_batch", &get_edge_features_batch<FEType>);
}

// ----- Nanobind Module -----
void init_observer_ext(nb::module_ &m) {
  nb::bind_vector<std::vector<std::shared_ptr<IFeature>>>(m, "IFeatureVector");
  nb::bind_vector<std::vector<std::shared_ptr<IEdgeFeature>>>(m, "IEdgeFeatureVector");

  // Task Features
  bind_int_feature<EmptyTaskFeature>(m, "EmptyTaskFeature");
  bind_state_feature<InDegreeTaskFeature>(m, "InDegreeTaskFeature");
  bind_state_feature<OutDegreeTaskFeature>(m, "OutDegreeTaskFeature");
  bind_state_feature<DurationTaskFeature>(m, "DurationTaskFeature");
  bind_state_feature<MemoryTaskFeature>(m, "MemoryTaskFeature");
  bind_state_feature<OneHotMappedDeviceTaskFeature>(m, "OneHotMappedDeviceTaskFeature");
  bind_state_feature<TaskStateFeature>(m, "TaskStateFeature");

  // Data Features
  bind_int_feature<EmptyDataFeature>(m, "EmptyDataFeature");
  bind_state_feature<DataMappedLocations>(m, "DataMappedLocationsFeature");
  bind_state_feature<DataReservedLocations>(m, "DataReservedLocationsFeature");
  bind_state_feature<DataLaunchedLocations>(m, "DataLaunchedLocationsFeature");
  bind_state_feature<DataSizeFeature>(m, "DataSizeFeature");

  // Device Features
  bind_int_feature<EmptyDeviceFeature>(m, "EmptyDeviceFeature");
  bind_state_feature<DeviceMemoryFeature>(m, "DeviceMemoryFeature");
  bind_state_feature<DeviceTimeFeature>(m, "DeviceTimeFeature");
  bind_state_feature<DeviceIDFeature>(m, "DeviceIDFeature");
  bind_state_feature<DeviceArchitectureFeature>(m, "DeviceArchitectureFeature");

  // Task Task Features
  bind_int_edge_feature<EmptyTaskTaskFeature>(m, "EmptyTaskTaskFeature");
  bind_state_edge_feature<TaskTaskSharedDataFeature>(m, "TaskTaskSharedDataFeature");

  // Task Data Features
  bind_int_edge_feature<EmptyTaskDataFeature>(m, "EmptyTaskDataFeature");
  bind_state_edge_feature<TaskDataRelativeSizeFeature>(m, "TaskDataRelativeSizeFeature");
  bind_state_edge_feature<TaskDataUsageFeature>(m, "TaskDataUsageFeature");

  // Task Device Features

  // Data Device Features

  // PrecompiledFeatureExtractors
  bind_feature_extractor<InDegreeTaskFeature, OutDegreeTaskFeature, OneHotMappedDeviceTaskFeature>(
      m, "TaskFeatureExtractor");
  bind_feature_extractor<DataSizeFeature, DataMappedLocations>(m, "DataFeatureExtractor");
  bind_feature_extractor<DeviceArchitectureFeature, DeviceIDFeature, DeviceMemoryFeature,
                         DeviceTimeFeature>(m, "DeviceFeatureExtractor");
  bind_edge_feature_extractor<TaskTaskSharedDataFeature>(m, "TaskTaskFeatureExtractor");
  bind_edge_feature_extractor<TaskDataRelativeSizeFeature, TaskDataUsageFeature>(
      m, "TaskDataFeatureExtractor");

  nb::class_<IFeature>(m, "IFeature")
      .def_prop_ro("feature_dim", &IFeature::get_feature_dim)
      .def("extract_feature", &IFeature::extract_feature);

  nb::class_<IEdgeFeature>(m, "IEdgeFeature")
      .def_prop_ro("feature_dim", &IEdgeFeature::get_feature_dim)
      .def("extract_feature", &IEdgeFeature::extract_feature);

  nb::class_<RuntimeFeatureExtractor>(m, "RuntimeFeatureExtractor")
      .def(nb::init<>())
      .def("add_feature", &RuntimeFeatureExtractor::addFeature)
      .def_prop_ro("feature_dim", &RuntimeFeatureExtractor::getFeatureDim)
      .def("get_features",
           [](const RuntimeFeatureExtractor &self, uint32_t task_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.getFeatures(task_id, sp);
           })
      .def("get_features_batch", &get_features_batch<RuntimeFeatureExtractor>);

  nb::class_<RuntimeEdgeFeatureExtractor>(m, "RuntimeEdgeFeatureExtractor")
      .def(nb::init<>())
      .def("add_feature", &RuntimeEdgeFeatureExtractor::addFeature)
      .def_prop_ro("feature_dim", &RuntimeEdgeFeatureExtractor::getFeatureDim)
      .def("get_features",
           [](const RuntimeEdgeFeatureExtractor &self, uint32_t source_id, uint32_t target_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.getFeatures(source_id, target_id, sp);
           })
      .def("get_features_batch", &get_edge_features_batch<RuntimeEdgeFeatureExtractor>);

  nb::class_<GraphSpec>(m, "GraphSpec")
      .def(nb::init<>())
      .def_rw("max_in_degree", &GraphSpec::max_in_degree)
      .def_rw("max_out_degree", &GraphSpec::max_out_degree)
      .def_rw("max_data_usage", &GraphSpec::max_data_usage)
      .def_rw("max_candidates", &GraphSpec::max_candidates)
      .def_rw("max_edges_tasks_tasks", &GraphSpec::max_edges_tasks_tasks)
      .def_rw("max_edges_tasks_data", &GraphSpec::max_edges_tasks_data)
      .def_rw("max_edges_tasks_devices", &GraphSpec::max_edges_tasks_devices)
      .def_rw("max_edges_data_devices", &GraphSpec::max_edges_data_devices)
      .def_rw("max_tasks", &GraphSpec::max_tasks)
      .def_rw("max_data", &GraphSpec::max_data)
      .def_rw("max_devices", &GraphSpec::max_devices)
      .def("compute_max_degree",
           nb::overload_cast<const SchedulerState &>(&GraphSpec::compute_max_degree))
      .def("compute_max_tasks", &GraphSpec::compute_max_tasks)
      .def("compute_max_data", &GraphSpec::compute_max_data)
      .def("finalize", &GraphSpec::finalize)
      .def("__str__", [](const GraphSpec &self) {
        return "GraphSpec {\n"
               "  Nodes:\n"
               "    max_tasks: " +
               std::to_string(self.max_tasks) +
               "\n"
               "    max_data: " +
               std::to_string(self.max_data) +
               "\n"
               "    max_devices: " +
               std::to_string(self.max_devices) +
               "\n"
               "  Degrees:\n"
               "    max_in_degree: " +
               std::to_string(self.max_in_degree) +
               "\n"
               "    max_out_degree: " +
               std::to_string(self.max_out_degree) +
               "\n"
               "    max_data_usage: " +
               std::to_string(self.max_data_usage) +
               "\n"
               "    max_candidates: " +
               std::to_string(self.max_candidates) +
               "\n"
               "  Edges:\n"
               "    max_edges_tasks_tasks: " +
               std::to_string(self.max_edges_tasks_tasks) +
               "\n"
               "    max_edges_tasks_data: " +
               std::to_string(self.max_edges_tasks_data) +
               "\n"
               "    max_edges_tasks_devices: " +
               std::to_string(self.max_edges_tasks_devices) +
               "\n"
               "    max_edges_data_devices: " +
               std::to_string(self.max_edges_data_devices) +
               "\n"
               "}";
      });

  nb::class_<GraphExtractor>(m, "GraphExtractor")
      .def(nb::init<SchedulerState &>())
      .def("get_k_hop_dependencies", &GraphExtractor::get_k_hop_dependencies)
      .def("get_k_hop_dependents", &GraphExtractor::get_k_hop_dependents)
      .def("get_k_hop_bidirectional", &GraphExtractor::get_k_hop_bidirectional)
      .def("get_active_tasks", &GraphExtractor::get_active_tasks)
      .def("get_task_task_edges", &GraphExtractor::get_task_task_edges)
      .def("get_task_task_edges_reverse", &GraphExtractor::get_task_task_edges_reverse)
      .def("get_task_data_edges", &GraphExtractor::get_task_data_edges)
      .def("get_data_device_edges", &GraphExtractor::get_data_device_edges)
      .def("get_unique_data", &GraphExtractor::get_unique_data);
}
