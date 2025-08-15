#pragma once
#include "devices.hpp"
#include "macros.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <span>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

struct CommunicationStats {
  timecount_t latency = 0;
  mem_t bandwidth = 0;
};

struct CommunicationRequest {
  taskid_t data_task_id = 0;
  devid_t source = 0;
  devid_t destination = 0;
  mem_t size = 0;

  bool operator==(const CommunicationRequest &other) const {
    return data_task_id == other.data_task_id && source == other.source &&
           destination == other.destination && size == other.size;
  }

  struct Hash {
    std::size_t operator()(const CommunicationRequest &req) const {
      // NOTE(wlr): I have no idea what the collision rate of this is
      //            Keep this in mind if something starts failing
      return std::hash<taskid_t>()(req.data_task_id) ^ std::hash<devid_t>()(req.source) ^
             std::hash<devid_t>()(req.destination) ^ std::hash<mem_t>()(req.size);
    }
  };

  bool operator<(const CommunicationRequest &other) const {
    return data_task_id < other.data_task_id ||
           (data_task_id == other.data_task_id &&
            (source < other.source || (source == other.source &&
                                       (destination < other.destination ||
                                        (destination == other.destination && size < other.size)))));
  }
};

class Topology {

public:
  std::vector<timecount_t> latency;
  std::vector<mem_t> bandwidths;
  std::vector<copy_t> links;
  devid_t num_devices = 0;

  Topology(devid_t num_devices)
      : latency(num_devices * num_devices), bandwidths(num_devices * num_devices),
        links(num_devices * num_devices), num_devices(num_devices) {

    for (devid_t i = 0; i < num_devices; ++i) {
      for (devid_t j = 0; j < num_devices; ++j) {
        if (i == j) {
          latency[i * num_devices + j] = 0;
          bandwidths[i * num_devices + j] = MAX_MEM;
          links[i * num_devices + j] =
              std::numeric_limits<copy_t>::max(); // Self-links are always available
        } else {
          bandwidths[i * num_devices + j] = 0;
          links[i * num_devices + j] = 0; // No links by default
        }
      }
    }
  }

  void set_bandwidth(devid_t src, devid_t dst, mem_t bandwidth) {
    bandwidths[src * num_devices + dst] = bandwidth;
  }

  void set_max_connections(devid_t src, devid_t dst, uint8_t max_links) {
    links[src * num_devices + dst] = max_links;
  }

  void set_latency(devid_t src, devid_t dst, timecount_t latency_) {
    latency[src * num_devices + dst] = latency_;
  }

  [[nodiscard]] timecount_t get_latency(devid_t src, devid_t dst) const {
    return latency[src * num_devices + dst];
  }

  [[nodiscard]] mem_t get_bandwidth(devid_t src, devid_t dst) const {
    return bandwidths[src * num_devices + dst];
  }

  [[nodiscard]] bool is_connected(devid_t src, devid_t dst) const {
    return links[src * num_devices + dst] > 0;
  }

  [[nodiscard]] copy_t get_max_connections(devid_t src, devid_t dst) const {
    return links[src * num_devices + dst];
  }
};

struct SourceRequest {
  bool found = false;
  devid_t source = 0;
};

struct DeviceUsage {
  copy_t incoming = 0;
  copy_t outgoing = 0;
  copy_t max = 0;
};

struct LinkUsage {
  copy_t active = 0;
  copy_t max = 0;
};

class CommunicationManager {
  devid_t num_devices = 0;
  std::vector<DeviceUsage> device_usage;
  std::vector<LinkUsage> link_usage;
  std::vector<double> bandwidth_reciprocals;

  void precompute_reciprocals(const Topology &topology) {
    bandwidth_reciprocals.resize(num_devices * num_devices);
    const auto &bandwidth = topology.bandwidths;
    for (size_t i = 0; i < bandwidth.size(); ++i) {
      bandwidth_reciprocals[i] = 1.0 / static_cast<double>(bandwidth[i]);
    }
  }

  void precompute_max_copies(const Devices &devices) {
    device_usage.resize(num_devices);
    for (devid_t i = 0; i < num_devices; ++i) {
      const auto &device = devices.get_device(i);
      device_usage[i].max = device.get_max_copy();
      SPDLOG_DEBUG("Precomputed max copies for device {}: {}", i, device_usage[i].max);
    }
  }

  void precompute_link_max_copies(const Topology &topology) {
    link_usage.resize(num_devices * num_devices);
    for (devid_t src = 0; src < num_devices; ++src) {
      for (devid_t dst = 0; dst < num_devices; ++dst) {
        link_usage[src * num_devices + dst].max = topology.get_max_connections(src, dst);
        SPDLOG_DEBUG("Precomputed max connections from device {} to device {}: {}", src, dst,
                     link_usage[src * num_devices + dst].max);
      }
    }
  }

public:
  CommunicationManager() = default;

  CommunicationManager(const Topology &topology_, const Devices &devices_)
      : num_devices(devices_.size()), device_usage(num_devices),
        link_usage(num_devices * num_devices) {
    precompute_reciprocals(topology_);
    precompute_max_copies(devices_);
    precompute_link_max_copies(topology_);
  }

  CommunicationManager(const CommunicationManager &c) = default;

  CommunicationManager &operator=(const CommunicationManager &c) = default;

  inline void increase_incoming(devid_t device_id) {
    device_usage[device_id].incoming += 1;
  }
  inline void decrease_incoming(devid_t device_id) {
    assert(device_usage[device_id].incoming >= 1);
    device_usage[device_id].incoming -= 1;
  }

  inline void increase_outgoing(devid_t device_id) {
    device_usage[device_id].outgoing += 1;
  }
  inline void decrease_outgoing(devid_t device_id) {
    assert(device_usage[device_id].outgoing >= 1);
    device_usage[device_id].outgoing -= 1;
  }

  inline void increase_active_links(devid_t src, devid_t dst) {
    link_usage[src * num_devices + dst].active += 1;
  }

  inline void decrease_active_links(devid_t src, devid_t dst) {
    assert(link_usage[src * num_devices + dst].active >= 1);
    link_usage[src * num_devices + dst].active -= 1;
  }

  inline void reserve_connection(devid_t src, devid_t dst) {
    increase_incoming(dst);
    increase_outgoing(src);
    increase_active_links(src, dst);
  }

  inline void release_connection(devid_t src, devid_t dst) {
    decrease_incoming(dst);
    decrease_outgoing(src);
    decrease_active_links(src, dst);
  }

  [[nodiscard]] inline copy_t get_active(devid_t src, devid_t dst) const {
    return link_usage[src * num_devices + dst].active;
  }

  [[nodiscard]] inline copy_t get_incoming(devid_t device_id) const {
    return device_usage[device_id].incoming;
  }

  [[nodiscard]] inline copy_t get_outgoing(devid_t device_id) const {
    return device_usage[device_id].outgoing;
  }

  [[nodiscard]] inline copy_t get_total_usage(devid_t device_id) const {
    return device_usage[device_id].incoming + device_usage[device_id].outgoing;
  }

  [[nodiscard]] inline bool is_device_available(devid_t device_id) const {
    const auto used = get_total_usage(device_id);
    const auto available = device_usage[device_id].max;
    return used < available;
  }

  [[nodiscard]] inline bool is_link_available(devid_t src, devid_t dst) const {
    const auto used = get_active(src, dst);
    const auto available = link_usage[src * num_devices + dst].max;
    return used < available;
  }

  [[nodiscard]] inline bool check_connection(devid_t src, devid_t dst) const {
    // No copy if same device
    if (src == dst)
      return true;

    // check link availability
    if (!is_link_available(src, dst)) {
      return false;
    }

    // check device availability
    return is_device_available(src) && is_device_available(dst);
  }

  [[nodiscard]] mem_t get_bandwidth(const Topology &topology, devid_t src, devid_t dst) const {
    return topology.get_bandwidth(src, dst);
  }

  [[nodiscard]] mem_t get_available_bandwidth(const Topology &topology, devid_t src,
                                              devid_t dst) const {
    return get_bandwidth(topology, src, dst);
  }

  [[nodiscard]] inline timecount_t ideal_time_to_transfer(const Topology &topology, mem_t size,
                                                          devid_t src, devid_t dst) const {

    if (src == dst || size == 0) {
      return 0;
    }

    const auto bw_r = bandwidth_reciprocals[src * num_devices + dst];
    const auto latency = static_cast<timecount_t>(topology.get_latency(src, dst));
    const auto s = static_cast<double>(size);
    auto time = latency + static_cast<timecount_t>(s * bw_r);

    SPDLOG_DEBUG("Calculating ideal time to transfer {} bytes from device {} "
                 "to device {} with bandwidth {} and latency {}: {}",
                 size, src, dst, bw_r, latency, time);
    return time;
  }

  [[nodiscard]] inline SourceRequest
  get_best_available_source(const Topology &topology, devid_t dst,
                            const uint8_t possible_source_flags) const {

    const uint8_t destination_mask = (1 << dst);

    // Early return for local data
    if (possible_source_flags & destination_mask) {
      SPDLOG_DEBUG("Data is local, returning {} as best source", dst);
      return {true, dst};
    }

    devid_t best_source = 0;
    mem_t best_bandwidth = 0;
    bool found = false;

    const auto size = topology.num_devices;
    for (devid_t src = 0; src < size; ++src) {
      const uint8_t src_mask = (1 << src);

      const bool is_valid = (possible_source_flags & src_mask) && is_link_available(src, dst) &&
                            is_device_available(src) && is_device_available(dst);

      SPDLOG_DEBUG("Checking source {} for destination {}: is_valid = {}", src, dst, is_valid);
      SPDLOG_DEBUG("HAS_DATA = {}", possible_source_flags & src_mask);
      SPDLOG_DEBUG("LINK_AVAILABLE = {}", is_link_available(src, dst));
      SPDLOG_DEBUG("SRC_AVAILABLE = {}", is_device_available(src));
      SPDLOG_DEBUG("DST_AVAILABLE = {}", is_device_available(dst));

      const auto bandwidth = topology.get_bandwidth(src, dst);

      const bool is_better = is_valid && (bandwidth > best_bandwidth);
      best_bandwidth = is_better ? bandwidth : best_bandwidth;
      best_source = is_better ? src : best_source;
      found = found || is_better;
    }

    return {found, best_source};
  }

  [[nodiscard]] inline SourceRequest get_best_source(const Topology &topology, devid_t dst,
                                                     const uint8_t possible_source_flags) const {

    const uint8_t destination_mask = (1 << dst);
    if (possible_source_flags & destination_mask) {
      return {true, dst}; // Local data is always available
    }

    devid_t best_source = 0;
    mem_t best_bandwidth = 0;
    bool found = false;

    for (devid_t src = 0; src < topology.num_devices; ++src) {
      const uint8_t src_mask = (1 << src);
      bool is_valid = (possible_source_flags & src_mask);

      auto bandwidth = get_available_bandwidth(topology, src, dst);
      const bool is_better = is_valid && (bandwidth > best_bandwidth);
      best_bandwidth = is_better ? bandwidth : best_bandwidth;
      best_source = is_better ? src : best_source;
      found = found || is_better;
    }
    return {found, best_source};
  }

  friend class SchedulerState;
};
