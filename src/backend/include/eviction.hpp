#pragma once

#include "communication_manager.hpp"
#include "device_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include "tasks.hpp"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cstddef>
#include <functional>
#include <ranges>
#include <string>
#include <unordered_map>

class LRUCache {
public:
  LRUCache(std::size_t num_devices)
      : num_devices_(num_devices), nodes_(num_devices), maps_(num_devices), lru_ids_(num_devices) {
  }

  // deep-copy per-device lists and rebuild iterator maps
  LRUCache(const LRUCache &other)
      : num_devices_(other.num_devices_), nodes_(other.num_devices_), maps_(other.num_devices_),
        lru_ids_(other.lru_ids_) {
    for (devid_t d = 0; d < num_devices_; ++d) {
      nodes_[d] = other.nodes_[d];
      for (auto it = nodes_[d].begin(); it != nodes_[d].end(); ++it) {
        maps_[d][it->id] = it;
      }
    }
  }

  // Touch: increment in_use, move to MRU (front) on given device
  void touch(devid_t device_id, dataid_t id, mem_t size, bool in_use = true) {
    assert(device_id < num_devices_);
    auto &lst = nodes_[device_id];
    auto &mp = maps_[device_id];

    auto it = mp.find(id);
    if (it == mp.end()) {
      // new entry
      lst.push_front(Node{.id = id, .in_use = in_use ? 1 : 0, .size = size});
      mp[id] = lst.begin();
    } else {
      // existing entry: bump in_use, move to front
      auto nodeIt = it->second;
      nodeIt->in_use++;
      lst.splice(lst.begin(), lst, nodeIt);
    }
  }

  // Untouch: decrement in_use (must remain ≥0) on given device
  void untouch(devid_t device_id, dataid_t id) {
    assert(device_id < num_devices_);
    auto &mp = maps_[device_id];

    auto it = mp.find(id);
    assert(it != mp.end() && "untouch on non-existent id");
    auto nodeIt = it->second;
    assert(nodeIt->in_use > 0 && "in_use would go negative");
    nodeIt->in_use--;
  }

  // Return all ids with in_use == 0 in true LRU order (oldest first)
  const DataIDList &getLRU(devid_t device_id) {
    assert(device_id < num_devices_);
    auto &lst = nodes_[devid_t(device_id)];
    auto &out = lru_ids_[device_id];

    out.clear();
    for (auto it = lst.rbegin(); it != lst.rend(); ++it) {
      if (it->in_use == 0) {
        out.push_back(it->id);
      }
    }
    return out;
  }

  // Evict a single id on given device
  void evict(devid_t device_id, dataid_t id) {
    assert(device_id < num_devices_);
    auto &lst = nodes_[device_id];
    auto &mp = maps_[device_id];

    auto it = mp.find(id);
    if (it == mp.end())
      return;
    lst.erase(it->second);
    mp.erase(it);
  }

  // Evict multiple ids on given device
  void evict(devid_t device_id, const DataIDList &ids) {
    for (auto id : ids) {
      evict(devid_t(device_id), id);
    }
  }

private:
  struct Node {
    dataid_t id;
    int in_use;
    mem_t size;
  };

  devid_t num_devices_;
  std::vector<std::list<Node>> nodes_;
  std::vector<ankerl::unordered_dense::map<dataid_t, typename std::list<Node>::iterator>> maps_;
  std::vector<DataIDList> lru_ids_;
  mem_t evictable_size_ = 0;
};
