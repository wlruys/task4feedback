#pragma once
#include "device_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include "tasks.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class EvictionManager;

class DataPool {
private:
  std::unordered_set<dataid_t> _pool;

public:
  bool add(dataid_t id) {
    _pool.insert(id);
    return true;
  }

  bool remove(dataid_t id) {
    _pool.erase(id);
    return true;
  }

  bool contains(dataid_t id) {
    return _pool.find(id) != _pool.end();
  }

  size_t size() {
    return _pool.size();
  }
  void clear() {
    _pool.clear();
  }
};

class EvictionPool {
protected:
  mem_t evictable_size = 0;

public:
  EvictionPool() = default;

  bool add(dataid_t id, mem_t size);
  bool remove(dataid_t id, mem_t size);
  dataid_t peek();
  dataid_t pop();

  bool contains(dataid_t id) const;
  size_t size() const;

  [[nodiscard]] mem_t get_evictable_size() const {
    return evictable_size;
  }
};

class DataNodeList {
private:
  struct DataNode {
    // Ownership of a node is held by the previous node
    dataid_t id = -1;
    mem_t size = 0;
    std::unique_ptr<DataNode> next;
    DataNode *prev = nullptr;

    explicit DataNode(dataid_t id_ = -1) : id(id_) {
    }
    explicit DataNode(dataid_t id_, mem_t size_) : id(id_), size(size_) {
    }
  };

  // The list owns the head node, which is a sentinel node (does not contain data)
  // The tail node is a sentinel node (does not contain data)
  std::unique_ptr<DataNode> head;
  DataNode *tail;
  size_t size_ = 0;
  std::unordered_map<dataid_t, DataNode *> data_map;
  mem_t evictable_size = 0;

  friend void swap(DataNodeList &a, DataNodeList &b) noexcept {
    using std::swap;
    swap(a.head, b.head);
    swap(a.tail, b.tail);
    swap(a.size_, b.size_);
    swap(a.data_map, b.data_map);
  }

public:
  DataNodeList() {
    head = std::make_unique<DataNode>();
    head->next = std::make_unique<DataNode>();
    tail = head->next.get();
    tail->prev = head.get();
  }

  DataNodeList(const DataNodeList &other) : head(std::make_unique<DataNode>()), size_(other.size_) {
    // Deep copy is needed for simulator copies to pass eviction state

    DataNode *cur_new = head.get();
    for (DataNode *cur_old = other.head->next.get(); cur_old != nullptr;
         cur_old = cur_old->next.get()) {
      auto node = std::make_unique<DataNode>(cur_old->id, cur_old->size);
      node->prev = cur_new;
      cur_new->next = std::move(node);
      cur_new = cur_new->next.get();
    }
    tail = cur_new;

    // rebuild map (tail sentinel id == -1, which is not in the map)
    for (DataNode *n = head->next.get(); n != nullptr; n = n->next.get())
      if (n->id != -1)
        data_map[n->id] = n;
  }

  DataNodeList &operator=(DataNodeList other) {
    swap(*this, other);
    return *this;
  }

  DataNodeList(DataNodeList &&) noexcept = default;
  DataNodeList &operator=(DataNodeList &&) noexcept = default;

  bool append(dataid_t id, mem_t size = 0) {
    if (data_map.count(id))
      return false;

    auto new_node = std::make_unique<DataNode>(id, size);
    DataNode *prev_last = tail->prev;

    // splice in front of tail
    new_node->prev = prev_last;
    new_node->next = std::move(prev_last->next); // takes old tail
    prev_last->next = std::move(new_node);
    prev_last->next->next->prev = prev_last->next.get();

    data_map[id] = prev_last->next.get();
    ++size_;
    evictable_size += size;
    return true;
  }

  bool remove(dataid_t id) {
    auto it = data_map.find(id);
    if (it == data_map.end())
      return false;

    DataNode *node = it->second;
    node->prev->next = std::move(node->next);
    if (node->prev->next)
      node->prev->next->prev = node->prev;

    assert(id == node->id);
    evictable_size -= node->size;

    data_map.erase(it);
    --size_;

    return true;
  }

  dataid_t pop() {
    if (size_ == 0)
      return -1;
    DataNode *node = head->next.get();
    dataid_t id = node->id;
    evictable_size -= node->size;

    head->next = std::move(node->next);
    head->next->prev = head.get();

    data_map.erase(id);
    --size_;
    return id;
  }

  dataid_t peek() const {
    return (size_ > 0 ? head->next->id : -1);
  }

  bool contains(dataid_t id) const {
    return data_map.count(id) != 0;
  }

  size_t size() const {
    return size_;
  }

  mem_t get_evictable_size() const {
    return evictable_size;
  }
};

class LRUEvictionPool : public EvictionPool {
protected:
  DataNodeList data_list;

public:
  LRUEvictionPool() = default;

  bool add(dataid_t id, mem_t size) {
    return data_list.append(id, size);
  }

  bool remove(dataid_t id) {
    return data_list.remove(id);
  }

  dataid_t peek() {
    return data_list.peek();
  }

  dataid_t pop() {
    return data_list.pop();
  }

  bool contains(dataid_t id) const {
    return data_list.contains(id);
  }

  size_t size() const {
    return data_list.size();
  }

  mem_t get_evictable_size() const {
    return data_list.get_evictable_size();
  }
};

class EvictionManager {
public:
  std::vector<LRUEvictionPool> eviction_pools;
  std::unordered_map<dataid_t, taskid_t> active_eviction_tasks;

  EvictionManager() = default;

  EvictionManager(const int n_devices) {
    eviction_pools.resize(n_devices);
  }

  bool add(dataid_t data_id, devid_t id, mem_t size) {
    assert(id < eviction_pools.size());
    return eviction_pools.at(id).add(data_id, size);
  }

  bool remove(dataid_t data_id, devid_t id) {
    assert(id < eviction_pools.size());
    return eviction_pools.at(id).remove(data_id);
  }

  dataid_t peek(devid_t id) {
    assert(id < eviction_pools.size());
    return eviction_pools.at(id).peek();
  }

  dataid_t pop(devid_t id) {
    assert(id < eviction_pools.size());
    return eviction_pools.at(id).pop();
  }

  bool contains(dataid_t data_id, devid_t id) {
    assert(id < eviction_pools.size());
    return eviction_pools.at(id).contains(data_id);
  }

  size_t size(devid_t id) const {
    assert(id < eviction_pools.size());
    return eviction_pools.at(id).size();
  }

  mem_t get_evictable_memory(devid_t id) const {
    assert(id < eviction_pools.size());
    return eviction_pools.at(id).get_evictable_size();
  }

  bool has_evictable_memory(devid_t id) const {
    assert(id < eviction_pools.size());
    std::cout << "Evictable size: " << eviction_pools.at(id).get_evictable_size() << std::endl;
    std::cout << "Number of elements in eviction pool: " << eviction_pools.at(id).size()
              << std::endl;
    return eviction_pools.at(id).get_evictable_size() > 0;
  }

  bool add_active_eviction_task(taskid_t eviction_task_id, dataid_t data_id) {
    if (active_eviction_tasks.find(data_id) != active_eviction_tasks.end()) {
      return false;
    }
    active_eviction_tasks[data_id] = eviction_task_id;
    return true;
  }
};