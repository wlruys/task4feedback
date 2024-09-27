#pragma once
#include <array>
#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>

using taskid_t = unsigned long long;
using dataid_t = unsigned long long;
using devid_t = unsigned int;

using depcount_t = unsigned int;

using TaskIDList = std::vector<taskid_t>;
using TaskIDLinkedList = std::list<taskid_t>;

using DataIDList = std::vector<dataid_t>;
using TaskIDLinkedList = std::list<taskid_t>;

using DeviceIDList = std::vector<devid_t>;
using DeviceIDLinkedList = std::list<devid_t>;

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