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
  devid_t num_devices = 0;
  std::vector<timecount_t> latency;
  std::vector<mem_t> bandwidths;
  std::vector<copy_t> links;

  Topology(devid_t num_devices)
      : latency(num_devices * num_devices), bandwidths(num_devices * num_devices),
        links(num_devices * num_devices), num_devices(num_devices) {
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

// constant array of device types to max copies
// TODO(wlr): Should be a property of the device thats configurable at runtime
// but I'm lazy rn and don't want to change the interface
constexpr std::array<copy_t, num_device_types> max_incoming_copies = {4, 1};
constexpr std::array<copy_t, num_device_types> max_outgoing_copies = {4, 1};
constexpr std::array<copy_t, num_device_types> max_total_copies = {4, 2};

struct SourceRequest {
  bool found = false;
  devid_t source = 0;
};

class CommunicationManager {
  devid_t num_devices = 0;
  std::vector<copy_t> incoming;
  std::vector<copy_t> outgoing;
  std::vector<copy_t> active_links;
  std::vector<copy_t> max_total_copies;
  std::vector<double> bandwidth_reciprocals;

  void precompute_reciprocals(const Topology &topology) {
    bandwidth_reciprocals.resize(num_devices * num_devices);
    const auto &bandwidth = topology.bandwidths;
    for (size_t i = 0; i < bandwidth.size(); ++i) {
      bandwidth_reciprocals[i] = 1.0 / static_cast<double>(bandwidth[i]);
    }
  }

  void precompute_max_copies(const Devices &devices) {
    max_total_copies.resize(num_devices);
    for (devid_t i = 0; i < num_devices; ++i) {
      const auto &device = devices.get_device(i);
      max_total_copies[i] = device.get_max_copy();
    }
  }

public:
  CommunicationManager(const Topology &topology_, const Devices &devices_)
      : num_devices(devices_.size()), incoming(devices_.size(), 0), outgoing(devices_.size(), 0),
        active_links(devices_.size() * devices_.size(), 0) {
    precompute_reciprocals(topology_);
    precompute_max_copies(devices_);
  }

  CommunicationManager(const CommunicationManager &c) = default;

  CommunicationManager &operator=(const CommunicationManager &c) = default;

  inline void increase_incoming(devid_t device_id) {
    incoming[device_id] += 1;
  }
  inline void decrease_incoming(devid_t device_id) {
    assert(incoming[device_id] >= 1);
    incoming[device_id] -= 1;
  }

  inline void increase_outgoing(devid_t device_id) {
    outgoing[device_id] += 1;
  }
  inline void decrease_outgoing(devid_t device_id) {
    assert(outgoing[device_id] >= 1);
    outgoing[device_id] -= 1;
  }

  inline void increase_active_links(devid_t src, devid_t dst) {
    active_links[src * num_devices + dst] += 1;
  }

  inline void decrease_active_links(devid_t src, devid_t dst) {
    assert(active_links[src * num_devices + dst] >= 1);
    active_links[src * num_devices + dst] -= 1;
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
    return active_links[src * num_devices + dst];
  }

  [[nodiscard]] inline copy_t get_incoming(devid_t device_id) const {
    return incoming[device_id];
  }

  [[nodiscard]] inline copy_t get_outgoing(devid_t device_id) const {
    return outgoing[device_id];
  }

  [[nodiscard]] inline copy_t get_total_usage(devid_t device_id) const {
    return incoming[device_id] + outgoing[device_id];
  }

  [[nodiscard]] inline bool is_device_available(devid_t device_id) const {
    const auto used = get_total_usage(device_id);
    const auto available = max_total_copies[device_id];
    return used <= available;
  }

  [[nodiscard]] inline bool is_link_available(const Topology &topology, devid_t src,
                                              devid_t dst) const {
    const auto used = get_active(src, dst);
    const auto available = topology.get_max_connections(src, dst);
    return used <= available;
  }

  [[nodiscard]] inline bool check_connection(const Topology &topology, devid_t src,
                                             devid_t dst) const {
    // Early return for same device
    if (src == dst)
      return true;

    if (!is_link_available(topology, src, dst)) {
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
                 "to device {} with bandwidth {} and latency {}",
                 size, src, dst, bw_r, latency);
    return time;
  }

  /// TODO(wlr) test repacked data
  // Better: Structure of Arrays → Array of Structures for better cache locality
  // struct LinkInfo {
  //   copy_t used_links;
  //   copy_t max_links;
  //   mem_t bandwidth;
  // };
  // std::vector<LinkInfo> link_data; // Size: num_devices * num_devices

  // TODO(wlr): Test something like this that is "branchless" but computes for all devices
  //  [[nodiscard]] SourceRequest
  //  get_best_available_source(const Topology &topology, devid_t dst,
  //                            std::span<const int8_t> possible_source_flags) const {
  //    // Early return for local data
  //    if (possible_source_flags[dst] != 0) {
  //      return {true, dst};
  //    }

  //   const auto *bandwidth_data = topology.bandwidths.data();
  //   const auto *active_links_data = active_links.data();
  //   const auto *max_links_data = topology.links.data();

  //   devid_t best_source = 0;
  //   mem_t best_bandwidth = 0;
  //   bool found = false;

  //   const auto size = static_cast<devid_t>(possible_source_flags.size());
  //   for (devid_t src = 0; src < size; ++src) {
  //     const std::size_t index = src * num_devices + dst;

  //     // Check all conditions without branching
  //     const bool has_source = possible_source_flags[src] != 0;
  //     const bool links_available = active_links_data[index] <= max_links_data[index];
  //     const bool src_available = (incoming[src] + outgoing[src]) <= max_total_copies[src];
  //     const bool dst_available = (incoming[dst] + outgoing[dst]) <= max_total_copies[dst];

  //     // All conditions must be true for this source to be valid
  //     const bool is_valid = has_source & links_available & src_available & dst_available;

  //     // Get bandwidth (safe to access even if invalid since we mask it out)
  //     const auto bandwidth = bandwidth_data[index];

  //     // Only consider this bandwidth if source is valid AND better than current best
  //     const bool is_better = is_valid & (bandwidth > best_bandwidth);

  //     // Update best values using conditional expressions
  //     best_bandwidth = is_better ? bandwidth : best_bandwidth;
  //     best_source = is_better ? src : best_source;
  //     found = found | is_better;
  //   }

  //   return {found, best_source};
  // }

  [[nodiscard]] SourceRequest get_best_available_source(const Topology &topology, devid_t dst,
                                                        const uint8_t possible_source_flags) const {

    const uint8_t destination_mask = (1 << dst);

    // Early return for local data
    if (possible_source_flags & destination_mask) {
      return {true, dst};
    }

    const auto *bandwidth_data = topology.bandwidths.data();
    const auto *active_links_data = active_links.data();
    const auto *max_links_data = topology.links.data();

    devid_t best_source = 0;
    mem_t best_bandwidth = 0;
    bool found = false;

    const auto n_devices = topology.num_devices;
    for (devid_t src = 0; src < n_devices; ++src) {
      const uint8_t src_mask = (1 << src);
      if (possible_source_flags & src_mask) {
        continue;
      }

      const std::size_t index = src * num_devices + dst;
      const auto used_links = active_links_data[index];
      const auto max_links = max_links_data[index];

      if (used_links > max_links) {
        continue;
      }

      // Inline device availability check
      const auto src_usage = incoming[src] + outgoing[src];
      const auto dst_usage = incoming[dst] + outgoing[dst];
      const auto src_limit = max_total_copies[src];
      const auto dst_limit = max_total_copies[dst];

      if (src_usage > src_limit || dst_usage > dst_limit) {
        continue;
      }

      const auto bandwidth = bandwidth_data[index];

      const bool is_better = bandwidth > best_bandwidth;
      best_bandwidth = is_better ? bandwidth : best_bandwidth;
      best_source = is_better ? src : best_source;
      found = found || is_better;
    }

    return {found, best_source};
  }

  [[nodiscard]] SourceRequest get_best_source(const Topology &topology, devid_t dst,
                                              uint8_t possible_source_flags) const {
    // Return the source with the highest bandwidth
    // If no source is available, return found=false

    bool found = false;
    devid_t best_source = 0;
    mem_t best_bandwidth = 0;

    for (devid_t src = 0; src < topology.num_devices; ++src) {
      if (possible_source_flags & (1 << src)) {
        continue; // Skip invalid sources
      }

      if (src == dst) {
        return {true, src};
      }

      auto bandwidth = get_available_bandwidth(topology, src, dst);
      if (bandwidth > best_bandwidth) {
        best_bandwidth = bandwidth;
        best_source = src;
        found = true;
      }
    }
    return {found, best_source};
  }

  friend class SchedulerState;
};
