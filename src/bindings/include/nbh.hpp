#pragma once
#include "settings.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

#if NB_VERSION_MAJOR > 1
// shapes are signed starting with NB 2, nb::any now stands for any type, not any shape
constexpr auto any_size = -1;
#else
constexpr auto any_size = nb::any;
#endif

using TorchArr = nb::ndarray<nb::pytorch, nb::device::cpu, float>;

template <typename T>
using TorchArr2D = nb::ndarray<T, nb::shape<any_size, any_size>, nb::c_contig>;
using TorchFloatArr2D = TorchArr2D<float>;
using TorchIntArr2D = TorchArr2D<int>;
using TorchInt64Arr2D = TorchArr2D<int64_t>;
using TorchUInt32Arr2D = TorchArr2D<uint32_t>;
using TorchUInt64Arr2D = TorchArr2D<uint64_t>;

template <typename T> using TorchArr1D = nb::ndarray<T, nb::shape<any_size>, nb::c_contig>;
using TorchFloatArr1D = TorchArr1D<float>;
using TorchIntArr1D = TorchArr1D<int>;
using TorchInt64Arr1D = TorchArr1D<int64_t>;
using TorchUInt32Arr1D = TorchArr1D<uint32_t>;
using TorchUInt64Arr1D = TorchArr1D<uint64_t>;