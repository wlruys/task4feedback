#pragma once

#include "device_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

class Topology {
  std::vector<mem_t> bandwidths;
  std::vector<uint8_t> links;
  std::size_t num_devices;

public:
  Topology(std::size_t num_devices)
      : bandwidths(num_devices * num_devices), links(num_devices * num_devices),
        num_devices(num_devices) {}

  void set_bandwidth(devid_t src, devid_t dst, mem_t bandwidth) {
    bandwidths[src * num_devices + dst] = bandwidth;
  }

  [[nodiscard]] mem_t get_bandwidth(devid_t src, devid_t dst) const {
    return bandwidths[src * num_devices + dst];
  }

  [[nodiscard]] bool is_connected(devid_t src, devid_t dst) const {
    return links[src * num_devices + dst] > 0;
  }

  [[nodiscard]] uint8_t max_connections(devid_t src, devid_t dst) const {
    return links[src * num_devices + dst];
  }
};

// constant array of device types to max copies
constexpr std::array<copy_t, num_device_types> max_incoming_copies = {4, 1};
constexpr std::array<copy_t, num_device_types> max_outgoing_copies = {4, 1};
constexpr std::array<copy_t, num_device_types> max_total_copies = {4, 2};

struct SourceRequest {
  bool found;
  devid_t source;
};

class CommunicationManager {

  std::vector<copy_t> incoming;
  std::vector<copy_t> outgoing;
  std::vector<copy_t> active_links;
  Topology &topology;
  Devices &devices;

  std::size_t get_device_type_idx(devid_t device_id) const {
    return static_cast<std::size_t>(devices.get_type(device_id));
  }

public:
  CommunicationManager(Topology &topology, Devices &devices)
      : topology(topology), devices(devices) {
    incoming.resize(devices.size(), 0);
    outgoing.resize(devices.size(), 0);
    active_links.resize(devices.size() * devices.size(), 0);
  }

  void increase_incoming(devid_t device_id) { incoming[device_id] += 1; }
  void decrease_incoming(devid_t device_id) { incoming[device_id] -= 1; }

  void increase_outgoing(devid_t device_id) { outgoing[device_id] += 1; }
  void decrease_outgoing(devid_t device_id) { outgoing[device_id] -= 1; }

  void increase_active_links(devid_t src, devid_t dst) {
    active_links[src * devices.size() + dst] += 1;
  }

  void decrease_active(devid_t src, devid_t dst) {
    active_links[src * devices.size() + dst] -= 1;
  }

  void reserve_connection(devid_t src, devid_t dst) {
    increase_incoming(dst);
    increase_outgoing(src);
    increase_active_links(src, dst);
  }

  void release_connection(devid_t src, devid_t dst) {
    decrease_incoming(dst);
    decrease_outgoing(src);
    decrease_active(src, dst);
  }

  [[nodiscard]] copy_t get_active(devid_t src, devid_t dst) const {
    return active_links[src * devices.size() + dst];
  }

  [[nodiscard]] copy_t get_incoming(devid_t device_id) const {
    return incoming[device_id];
  }

  [[nodiscard]] copy_t get_outgoing(devid_t device_id) const {
    return outgoing[device_id];
  }

  [[nodiscard]] copy_t get_total_usage(devid_t device_id) const {
    return incoming[device_id] + outgoing[device_id];
  }

  [[nodiscard]] bool is_device_available(devid_t device_id) const {
    auto used = get_total_usage(device_id);
    auto available = max_total_copies[get_device_type_idx(device_id)];
    return used <= available;
  }

  [[nodiscard]] bool is_link_available(devid_t src, devid_t dst) const {
    auto used = get_active(src, dst);
    auto available = topology.max_connections(src, dst);
    return used <= available;
  }

  [[nodiscard]] bool check_connection(devid_t src, devid_t dst) const {
    return is_device_available(src) && is_device_available(dst) &&
           is_link_available(src, dst);
  }

  [[nodiscard]] mem_t get_bandwidth(devid_t src, devid_t dst) const {
    return topology.get_bandwidth(src, dst);
  }

  [[nodiscard]] mem_t get_available_bandwidth(devid_t src, devid_t dst) const {
    return get_bandwidth(src, dst);
  }

  [[nodiscard]] mem_t time_to_transfer(mem_t size, devid_t src,
                                       devid_t dst) const {
    return size / get_available_bandwidth(src, dst);
  }

  [[nodiscard]] mem_t ideal_time_to_transfer(mem_t size, devid_t src,
                                             devid_t dst) const {
    return size / get_bandwidth(src, dst);
  }

  SourceRequest get_best_source(devid_t dst,
                                DeviceIDList &possible_sources) const {
    // Return the source with the highest bandwidth
    // Device and connection must be available

    bool found = false;
    devid_t best_source = 0;
    mem_t best_bandwidth = 0;
    for (auto src : possible_sources) {
      if (check_connection(src, dst)) {
        auto bandwidth = get_available_bandwidth(src, dst);
        if (bandwidth > best_bandwidth) {
          best_bandwidth = bandwidth;
          best_source = src;
          found = true;
        }
      }
    }
    return {found, best_source};
  }
};