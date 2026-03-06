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
    T4F_INVARIANT(num_devices >= 0 && static_cast<std::size_t>(num_devices) <= kMaxDevices);

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
  copy_t h2d_incoming = 0;
  copy_t h2d_outgoing = 0;
  copy_t d2d_outgoing = 0;
  copy_t d2d_incoming = 0;
  copy_t h2d_max = 0;
  copy_t d2d_max = 0;
};

struct LinkUsage {
  copy_t active = 0;
  copy_t max = 0;
};

class CommunicationManager {
  devid_t num_devices = 0;
  std::vector<DeviceUsage> device_usage;
  std::vector<LinkUsage> link_usage; // (src, dst) links
  std::vector<double> bandwidth_reciprocals;
  std::vector<uint8_t> is_host;
  std::vector<std::vector<devid_t>> preferred_sources_by_destination;
  std::vector<uint8_t> source_rank_by_destination_source;

  void precompute_reciprocals(const Topology &topology) {
    bandwidth_reciprocals.resize(num_devices * num_devices);
    const auto &bandwidth = topology.bandwidths;
    const std::size_t n = bandwidth.size();
    const auto * __restrict__ bw = bandwidth.data();
    auto * __restrict__ recip = bandwidth_reciprocals.data();
    for (std::size_t i = 0; i < n; ++i) {
      recip[i] = 1.0 / static_cast<double>(bw[i]);
    }
  }

  void precompute_max_copies(const Devices &devices) {
    device_usage.resize(num_devices);
    is_host.resize(num_devices, 0);
    for (devid_t i = 0; i < num_devices; ++i) {
      const auto &device = devices.get_device(i);
      device_usage[i].h2d_max = device.get_h2d_max_copy();
      device_usage[i].d2d_max = device.get_d2d_max_copy();
      SPDLOG_DEBUG("Precomputed max copies for device {}: h2d={}, d2d={}", i,
                   device_usage[i].h2d_max, device_usage[i].d2d_max);
      is_host[i] = (device.arch == DeviceType::CPU) ? 1 : 0;
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

  void precompute_source_order(const Topology &topology) {
    source_rank_by_destination_source.assign(num_devices * num_devices,
                                             std::numeric_limits<uint8_t>::max());
    preferred_sources_by_destination.resize(num_devices);

    for (devid_t dst = 0; dst < num_devices; ++dst) {
      auto &order = preferred_sources_by_destination[dst];
      order.clear();
      order.reserve(num_devices);
      for (devid_t src = 0; src < num_devices; ++src) {
        order.push_back(src);
      }
      std::stable_sort(order.begin(), order.end(), [&](devid_t lhs, devid_t rhs) {
        return topology.get_bandwidth(lhs, dst) > topology.get_bandwidth(rhs, dst);
      });

      for (std::size_t rank = 0; rank < order.size(); ++rank) {
        source_rank_by_destination_source[dst * num_devices + order[rank]] =
            static_cast<uint8_t>(rank);
      }
    }
  }

  [[nodiscard]] inline uint8_t get_source_rank(devid_t dst, devid_t src) const {
    return source_rank_by_destination_source[dst * num_devices + src];
  }

public:
  CommunicationManager() = default;

  CommunicationManager(const Topology &topology_, const Devices &devices_)
      : num_devices(devices_.size()), device_usage(num_devices),
        link_usage(num_devices * num_devices) {
    T4F_INVARIANT(num_devices >= 0 && static_cast<std::size_t>(num_devices) <= kMaxDevices);
    precompute_reciprocals(topology_);
    precompute_max_copies(devices_);
    precompute_link_max_copies(topology_);
    precompute_source_order(topology_);
  }

  CommunicationManager(const CommunicationManager &c) = default;

  CommunicationManager &operator=(const CommunicationManager &c) = default;

  inline bool is_h2d(devid_t src, devid_t dst) const {
    return (is_host[src] != 0 && is_host[dst] == 0) || (is_host[src] == 0 && is_host[dst] != 0);
  }

  inline bool is_d2d(devid_t src, devid_t dst) const {
    return is_host[src] == 0 && is_host[dst] == 0;
  }

  inline bool is_h2h(devid_t src, devid_t dst) const {
    return is_host[src] && is_host[dst];
  }

  inline void reserve_copy_engine(devid_t dst, devid_t src) {
    if (is_h2d(src, dst)) {
      device_usage[src].h2d_outgoing += 1;
      device_usage[dst].h2d_incoming += 1;
    } else if (is_d2d(src, dst)) {
      device_usage[src].d2d_outgoing += 1;
      device_usage[dst].d2d_incoming += 1;
    }
  }

  inline void release_copy_engine(devid_t dst, devid_t src) {
    if (is_h2d(src, dst)) {
      device_usage[src].h2d_outgoing -= 1;
      device_usage[dst].h2d_incoming -= 1;
    } else if (is_d2d(src, dst)) {
      device_usage[src].d2d_outgoing -= 1;
      device_usage[dst].d2d_incoming -= 1;
    }
  }

  inline void increase_active_links(devid_t src, devid_t dst) {
    link_usage[src * num_devices + dst].active += 1;
  }

  inline void decrease_active_links(devid_t src, devid_t dst) {
    T4F_INVARIANT(link_usage[src * num_devices + dst].active >= 1);
    link_usage[src * num_devices + dst].active -= 1;
  }

  inline void reserve_connection(devid_t src, devid_t dst) {
    reserve_copy_engine(dst, src);
    increase_active_links(src, dst);
  }

  inline void release_connection(devid_t src, devid_t dst) {
    release_copy_engine(dst, src);
    decrease_active_links(src, dst);
  }

  [[nodiscard]] inline copy_t get_active(devid_t src, devid_t dst) const {
    return link_usage[src * num_devices + dst].active;
  }

  [[nodiscard]] inline bool is_device_available(devid_t src, devid_t dst) const {
    if (is_h2d(src, dst)) {
      const auto used_h2d_outgoing = device_usage[src].h2d_outgoing;
      const auto used_h2d_incoming = device_usage[dst].h2d_incoming;
      const auto available_h2d_outgoing = device_usage[src].h2d_max;
      const auto available_h2d_incoming = device_usage[dst].h2d_max;
      return used_h2d_outgoing < available_h2d_outgoing &&
             used_h2d_incoming < available_h2d_incoming;
    } else if (is_d2d(src, dst)) {
      const auto used_d2d_outgoing = device_usage[src].d2d_outgoing;
      const auto used_d2d_incoming = device_usage[dst].d2d_incoming;
      const auto available_d2d_outgoing = device_usage[src].d2d_max;
      const auto available_d2d_incoming = device_usage[dst].d2d_max;
      return used_d2d_outgoing < available_d2d_outgoing &&
             used_d2d_incoming < available_d2d_incoming;
    }
    if (is_h2h(src, dst)) {
      return true;
    }
    return false;
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
    return is_device_available(src, dst);
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
                            const devicemask_t possible_source_flags) const {
    MONUnusedParameter(topology);

    const devicemask_t destination_mask = device_bit(dst);

    // Early return for local data
    if (possible_source_flags & destination_mask) {
      SPDLOG_DEBUG("Data is local, returning {} as best source", dst);
      return {true, dst};
    }

    devicemask_t candidates = possible_source_flags & ~destination_mask;
    if (candidates == 0) {
      return {false, 0};
    }

    bool found = false;
    devid_t best_source = 0;
    auto best_rank = std::numeric_limits<uint8_t>::max();

    while (candidates) {
      const auto candidate_bits = static_cast<devicemask_unsigned_t>(candidates);
      const auto src = static_cast<devid_t>(std::countr_zero(candidate_bits));
      candidates = static_cast<devicemask_t>(candidate_bits & (candidate_bits - 1));

      if (src >= num_devices) {
        continue;
      }
      if (!is_link_available(src, dst) || !is_device_available(src, dst)) {
        continue;
      }

      const auto rank = get_source_rank(dst, src);
      if (!found || rank < best_rank) {
        found = true;
        best_source = src;
        best_rank = rank;
        if (best_rank == 0) {
          break;
        }
      }
    }

    return {found, best_source};
  }

  [[nodiscard]] inline SourceRequest
  get_best_source(const Topology &topology, devid_t dst,
                  const devicemask_t possible_source_flags) const {
    MONUnusedParameter(topology);

    const devicemask_t destination_mask = device_bit(dst);
    if (possible_source_flags & destination_mask) {
      return {true, dst}; // Local data is always available
    }

    devicemask_t candidates = possible_source_flags & ~destination_mask;
    if (candidates == 0) {
      return {false, 0};
    }

    bool found = false;
    devid_t best_source = 0;
    auto best_rank = std::numeric_limits<uint8_t>::max();

    while (candidates) {
      const auto candidate_bits = static_cast<devicemask_unsigned_t>(candidates);
      const auto src = static_cast<devid_t>(std::countr_zero(candidate_bits));
      candidates = static_cast<devicemask_t>(candidate_bits & (candidate_bits - 1));

      if (src >= num_devices) {
        continue;
      }

      const auto rank = get_source_rank(dst, src);
      if (!found || rank < best_rank) {
        found = true;
        best_source = src;
        best_rank = rank;
        if (best_rank == 0) {
          break;
        }
      }
    }

    return {found, best_source};
  }

  friend class SchedulerState;
};
