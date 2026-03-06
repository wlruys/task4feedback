#pragma once
#include "macros.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <list>
#include <limits>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>

using priority_t = int32_t;
using taskid_t = int32_t;
using dataid_t = int32_t;
using devid_t = int32_t;
using depcount_t = int32_t;

#ifndef T4F_MAX_DEVICES
#define T4F_MAX_DEVICES 16
#endif

constexpr std::size_t kMaxDevices = static_cast<std::size_t>(T4F_MAX_DEVICES);
static_assert(kMaxDevices >= 5 && kMaxDevices <= 64,
              "T4F_MAX_DEVICES must be in [5, 64]");

// Number of bits is chosen from {8,16,32,64} and must cover max devices.
using devicemask_t = std::conditional_t<
    (kMaxDevices <= 8), uint8_t,
    std::conditional_t<(kMaxDevices <= 16), uint16_t,
                       std::conditional_t<(kMaxDevices <= 32), uint32_t, uint64_t>>>;
using devicemask_unsigned_t = std::make_unsigned_t<devicemask_t>;
constexpr std::size_t kDeviceMaskBits = std::numeric_limits<devicemask_unsigned_t>::digits;
static_assert(kMaxDevices <= kDeviceMaskBits, "Mask width must cover T4F_MAX_DEVICES");

constexpr devicemask_unsigned_t kAllDeviceBits =
    (kMaxDevices == kDeviceMaskBits)
        ? std::numeric_limits<devicemask_unsigned_t>::max()
        : ((devicemask_unsigned_t{1} << kMaxDevices) - devicemask_unsigned_t{1});

[[nodiscard]] constexpr devicemask_t device_bit(devid_t device_id) {
  return static_cast<devicemask_t>(
      devicemask_unsigned_t{1} << static_cast<devicemask_unsigned_t>(device_id));
}

[[nodiscard]] constexpr devicemask_unsigned_t mask_for_n_devices(std::size_t n_devices) {
  if (n_devices == 0) return devicemask_unsigned_t{0};
  if (n_devices >= kDeviceMaskBits) return std::numeric_limits<devicemask_unsigned_t>::max();
  return (devicemask_unsigned_t{1} << n_devices) - devicemask_unsigned_t{1};
}

// using priority_t = int32_t;
// using taskid_t = int32_t;
// using dataid_t = int32_t;
// using devid_t = int32_t;
// using depcount_t = int32_t;

using vcu_t = int64_t;
using mem_t = int64_t;
using timecount_t = int64_t;
using copy_t = int32_t;

using TaskIDList = std::vector<taskid_t>;
using TaskIDLinkedList = std::list<taskid_t>;

using DataIDList = std::vector<dataid_t>;
using TaskIDLinkedList = std::list<taskid_t>;

using DeviceIDList = std::vector<devid_t>;
using DeviceIDLinkedList = std::list<devid_t>;

using PriorityList = std::vector<priority_t>;

using TaskDeviceList = std::vector<std::tuple<taskid_t, devid_t>>;

class SchedulerState;

template <typename T> void print(std::vector<T> vec) {
  for (auto &elem : vec) {
    std::cout << elem << " ";
  }
  std::cout << std::endl;
}

template <typename K, typename T> void print(std::unordered_map<K, T> map) {
  for (auto &elem : map) {
    std::cout << elem.first << " ";
  }
  std::cout << std::endl;
}

template <typename T> void print(std::list<T> list) {
  for (auto &elem : list) {
    std::cout << elem << " ";
  }
  std::cout << std::endl;
}

template <typename G> void labeled_print(const std::string &name, const G &g) {
  std::cout << name << " ";
  print(g);
}

template <typename T> struct StatsBundle {
  T min = 0;
  T max = 0;
  double mean = 0;
  T median = 0;
  double stddev = 0;

  StatsBundle() = default;

  StatsBundle(std::vector<T> &v) {
    if (v.empty()) {
      return;
    }

    // std::sort(v.begin(), v.end());

    min = v.front();
    max = v.back();
    mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    median = v.size() % 2 == 0 ? (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2.0 : v[v.size() / 2];

    double sum = 0;
    for (const auto &val : v) {
      sum += (val - mean) * (val - mean);
    }
    // stddev = std::sqrt(sum / v.size());
  }
};
