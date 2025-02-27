// nanobind_module.cpp
#include "devices.hpp"
#include "features.hpp"
#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/shared_ptr.h>
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

struct PyIFeature : IFeature {
  NB_TRAMPOLINE(IFeature, 2);

  size_t get_feature_dim() const override {
    NB_OVERRIDE_PURE(get_feature_dim);
  }

  void extract_feature(uint32_t object_id, std::span<float> output) const override {
    // nb::ndarray<nb::pytorch, float, nb::device::cpu> arr(output.data(), output.size());
    NB_OVERRIDE_PURE(extract_feature, object_id, output);
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

template <typename E>
void get_features_batch(const E &extractor, const std::vector<uint32_t> &object_ids,
                        nb::ndarray<nb::pytorch, float, nb::device::cpu> tensor) {
  float *data = tensor.data();
  auto num_cols = extractor.getFeatureDim();
  for (size_t i = 0; i < object_ids.size(); ++i) {
    std::span<float> row(data + i * num_cols, num_cols);
    extractor.getFeatures(object_ids[i], row);
  }
}

template <typename FEType> void bind_state_feature(nb::module_ &m, const char *class_name) {
  nb::class_<FEType>(m, class_name)
      .def(nb::init<size_t>())
      .def("get_feature_dim", &DeviceLocationFeature::getFeatureDim)
      .def("extract_feature",
           [](const DeviceLocationFeature &self, uint32_t task_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.extractFeature(task_id, sp);
           })
      .def_static("make_shared", [](size_t n) -> std::shared_ptr<IFeature> {
        return std::make_shared<FeatureAdapter<FEType>>(FEType(n));
      });
}

template <typename... Features>
void bind_feature_extractor(nb::module_ &m, const char *class_name) {
  using FEType = FeatureExtractor<Features...>;
  nb::class_<FEType>(m, class_name)
      .def(nb::init<Features...>())
      .def("get_feature_dim", &FEType::getFeatureDim)
      .def("get_features",
           [](const FEType &self, uint32_t task_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.getFeatures(task_id, sp);
           })
      .def("get_features_batch", &get_features_batch<FEType>);
}

// ----- Nanobind Module -----
void init_observer_ext(nb::module_ &m) {
  nb::bind_vector<std::vector<std::shared_ptr<IFeature>>>(m, "IFeatureVector");

  bind_state_feature<DeviceLocationFeature>(m, "DeviceLocationFeature");

  bind_feature_extractor<DeviceLocationFeature, DeviceLocationFeature>(m, "FeatureExtractor");

  nb::class_<IFeature, PyIFeature>(m, "IFeature")
      .def(nb::init<>())
      .def("get_feature_dim", &IFeature::get_feature_dim)
      .def("extract_feature", &IFeature::extract_feature);

  nb::class_<RuntimeFeatureExtractor>(m, "RuntimeFeatureExtractor")
      .def(nb::init<>())
      .def("add_feature", &RuntimeFeatureExtractor::addFeature)
      .def("get_feature_dim", &RuntimeFeatureExtractor::getFeatureDim)
      .def("get_features",
           [](const RuntimeFeatureExtractor &self, uint32_t task_id,
              nb::ndarray<nb::pytorch, float, nb::device::cpu> arr) {
             float *data = arr.data();
             std::span<float> sp(data, self.getFeatureDim());
             self.getFeatures(task_id, sp);
           })
      .def("get_features_batch", &get_features_batch<RuntimeFeatureExtractor>);
}
