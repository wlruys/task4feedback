#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <list>
#include <numeric>
#include <unordered_map>
#include <vector>

using priority_t = int32_t;
using taskid_t = int32_t;
using dataid_t = int32_t;
using devid_t = int32_t;
using depcount_t = int32_t;

// using priority_t = int32_t;
// using taskid_t = int32_t;
// using dataid_t = int32_t;
// using devid_t = int32_t;
// using depcount_t = int32_t;

using TaskIDList = std::vector<taskid_t>;
using TaskIDLinkedList = std::list<taskid_t>;

using DataIDList = std::vector<dataid_t>;
using TaskIDLinkedList = std::list<taskid_t>;

using DeviceIDList = std::vector<devid_t>;
using DeviceIDLinkedList = std::list<devid_t>;

using PriorityList = std::vector<priority_t>;

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
