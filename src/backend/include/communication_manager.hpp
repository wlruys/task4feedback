#pragma once
#include "device_manager.hpp"
#include "devices.hpp"
#include "macros.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

struct CommunicationStats {
  timecount_t latency;
  mem_t bandwidth;
};

struct CommunicationRequest {
  taskid_t data_task_id;
  devid_t source;
  devid_t destination;
  mem_t size;

  bool operator==(const CommunicationRequest &other) const {
    return data_task_id == other.data_task_id && source == other.source &&
           destination == other.destination && size == other.size;
  }

  struct Hash {
    std::size_t operator()(const CommunicationRequest &req) const {
      // NOTE(wlr): I have no idea what the collision rate of this is
      //            Keep this in mind if something starts failing
      return std::hash<taskid_t>()(req.data_task_id) ^
             std::hash<devid_t>()(req.source) ^
             std::hash<devid_t>()(req.destination) ^
             std::hash<mem_t>()(req.size);
    }
  };

  bool operator<(const CommunicationRequest &other) const {
    return data_task_id < other.data_task_id ||
           (data_task_id == other.data_task_id &&
            (source < other.source ||
             (source == other.source &&
              (destination < other.destination ||
               (destination == other.destination && size < other.size)))));
  }
};

class Topology {
  std::vector<timecount_t> latency;
  std::vector<mem_t> bandwidths;
  std::vector<uint8_t> links;
  std::size_t num_devices;

public:
  Topology(std::size_t num_devices)
      : latency(num_devices * num_devices),
        bandwidths(num_devices * num_devices), links(num_devices * num_devices),
        num_devices(num_devices) {}

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

  [[nodiscard]] uint8_t get_max_connections(devid_t src, devid_t dst) const {
    return links[src * num_devices + dst];
  }
};

class CommunicationNoise {
protected:
  std::unordered_map<CommunicationRequest, CommunicationStats,
                     CommunicationRequest::Hash>
      record;
  unsigned int seed = 0;
  mutable std::mt19937 gen;
  Topology &topology;

  struct request_high_precision {
    uint64_t data_task_id;
    uint64_t source;
    uint64_t destination;
    uint64_t size;

    request_high_precision() = default;

    request_high_precision(const CommunicationRequest &req) {
      data_task_id = req.data_task_id;
      source = req.source;
      destination = req.destination;
      size = req.size;
    }
  };

  struct stats_high_precision {
    uint64_t latency;
    uint64_t bandwidth;

    stats_high_precision() = default;

    stats_high_precision(const CommunicationStats &stats) {
      latency = stats.latency;
      bandwidth = stats.bandwidth;
    }
  };

  [[nodiscard]] virtual CommunicationStats
  sample_stats(const CommunicationRequest &req) const {
    MONUnusedParameter(req);
    return {0, 0};
  };

  static uint64_t calculate_checksum(
      const std::unordered_map<CommunicationRequest, CommunicationStats,
                               CommunicationRequest::Hash> &data) {
    const uint64_t prime = 0x100000001B3ull; // FNV prime
    uint64_t hash = 0xcbf29ce484222325ull;   // FNV offset basis

    // Store all data in a vector, sort by request
    std::vector<std::pair<CommunicationRequest, CommunicationStats>> sdata(
        data.begin(), data.end());

    std::sort(sdata.begin(), sdata.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    for (const auto &[req, stats] : sdata) {
      hash ^= static_cast<uint64_t>(req.source);
      hash *= prime;
      hash ^= static_cast<uint64_t>(req.destination);
      hash *= prime;
      hash ^= static_cast<uint64_t>(req.data_task_id);
      hash *= prime;
      hash ^= static_cast<uint64_t>(req.size);
      hash *= prime;

      // Hash CommunicationStats
      const char *stats_data = reinterpret_cast<const char *>(&stats);
      for (size_t i = 0; i < sizeof(CommunicationStats); ++i) {
        hash ^= static_cast<uint64_t>(stats_data[i]);
        hash *= prime;
      }
    }

    return hash;
  }

public:
  static constexpr uint32_t FILE_VERSION = 1;
  static constexpr size_t BUFFER_SIZE = 8192;

  CommunicationNoise(Topology &topology_, unsigned int seed_ = 0)
      : seed(seed_), gen(seed_), topology(topology_) {}

  [[nodiscard]] CommunicationStats get(const CommunicationRequest &req) {
    auto it = record.find(req);
    if (it != record.end()) {
      return it->second;
    }
    set(req, sample_stats(req));
    return record.at(req);
  }

  void set(const CommunicationRequest &req, const CommunicationStats &stats) {
    record[req] = stats;
  }

  void set(std::unordered_map<CommunicationRequest, CommunicationStats,
                              CommunicationRequest::Hash>
               record_) {
    record = std::move(record_);
  }

  [[nodiscard]] CommunicationStats operator()(const CommunicationRequest &req) {
    return get(req);
  }

  void operator()(const CommunicationRequest &req,
                  const CommunicationStats &stats) {
    set(req, stats);
  }

  void operator()(const CommunicationRequest &req, timecount_t latency,
                  mem_t bandwidth) {
    set(req, {latency, bandwidth});
  }

  void dump_to_binary(const std::string &filename) const {

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    // Set up buffering
    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    // Write header
    file.write("COMM", 4);
    file.write(reinterpret_cast<const char *>(&FILE_VERSION),
               sizeof(FILE_VERSION));

    // Write data size
    uint64_t data_size = record.size();
    file.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));

    // Write data
    for (const auto &[lreq, lstats] : record) {
      request_high_precision req(lreq);
      stats_high_precision stats(lstats);
      file.write(reinterpret_cast<const char *>(&req),
                 sizeof(request_high_precision));
      file.write(reinterpret_cast<const char *>(&stats),
                 sizeof(stats_high_precision));
    }

    // Write checksum
    uint64_t checksum = calculate_checksum(record);
    file.write(reinterpret_cast<const char *>(&checksum), sizeof(checksum));

    if (file.fail()) {
      throw std::runtime_error("Error writing to file: " + filename);
    }
  }

  void load_from_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    // Set up buffering
    std::array<char, BUFFER_SIZE> buffer;
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    // Read and verify header
    std::array<char, 4> header;
    file.read(header.data(), header.size());
    if (std::string(header.data(), header.size()) != "COMM") {
      throw std::runtime_error("Invalid file format");
    }

    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != FILE_VERSION) {
      throw std::runtime_error("Unsupported file version");
    }

    // Read data size
    uint64_t data_size;
    file.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));

    // Read data
    record.clear();
    for (uint64_t i = 0; i < data_size; ++i) {
      request_high_precision hreq;
      stats_high_precision hstats;
      file.read(reinterpret_cast<char *>(&hreq),
                sizeof(request_high_precision));
      file.read(reinterpret_cast<char *>(&hstats),
                sizeof(stats_high_precision));
      CommunicationRequest req = {hreq.data_task_id, hreq.source,
                                  hreq.destination, hreq.size};
      CommunicationStats stats = {hstats.latency, hstats.bandwidth};
      record[req] = stats;
    }

    // Read and verify checksum
    uint64_t stored_checksum;
    file.read(reinterpret_cast<char *>(&stored_checksum),
              sizeof(stored_checksum));
    uint64_t calculated_checksum = calculate_checksum(record);

    if (stored_checksum != calculated_checksum) {
      throw std::runtime_error("Checksum mismatch - file may be corrupted");
    }

    if (file.fail()) {
      throw std::runtime_error("Error reading from file: " + filename);
    }
  }

  void clear() { record.clear(); }
};

class UniformCommunicationNoise : public CommunicationNoise {
protected:
  CommunicationStats
  sample_stats(const CommunicationRequest &req) const override {
    auto mean_bw = topology.get_bandwidth(req.source, req.destination);
    auto mean_latency = topology.get_latency(req.source, req.destination);

    std::uniform_int_distribution<timecount_t> latency_dist(
        mean_latency - mean_latency / 2, mean_latency + mean_latency / 2);

    std::uniform_int_distribution<mem_t> bandwidth_dist(mean_bw - mean_bw / 2,
                                                        mean_bw + mean_bw / 2);

    return {latency_dist(gen), bandwidth_dist(gen)};
  }

public:
  UniformCommunicationNoise(Topology &topology_, unsigned int seed_ = 0)
      : CommunicationNoise(topology_, seed_) {}
};

// constant array of device types to max copies
// TODO(wlr): Should be a property of the device thats configurable at runtime
// but I'm lazy rn and don't want to change the interface
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

  [[nodiscard]] std::size_t get_device_type_idx(devid_t device_id) const {
    return static_cast<std::size_t>(devices.get_type(device_id));
  }

public:
  CommunicationManager(Topology &topology, Devices &devices)
      : topology(topology), devices(devices) {
    incoming.resize(devices.size(), 0);
    outgoing.resize(devices.size(), 0);
    active_links.resize(devices.size() * devices.size(), 0);
  }
  void initialize() {}

  void increase_incoming(devid_t device_id) { incoming[device_id] += 1; }
  void decrease_incoming(devid_t device_id) { incoming[device_id] -= 1; }

  void increase_outgoing(devid_t device_id) { outgoing[device_id] += 1; }
  void decrease_outgoing(devid_t device_id) { outgoing[device_id] -= 1; }

  void increase_active_links(devid_t src, devid_t dst) {
    active_links[src * devices.size() + dst] += 1;
  }

  void decrease_active_links(devid_t src, devid_t dst) {
    active_links[src * devices.size() + dst] -= 1;
  }

  void reserve_connection(devid_t src, devid_t dst) {
    increase_incoming(dst);
    increase_outgoing(src);
    increase_active_links(src, dst);
    std::cout << "+Active links: " << static_cast<int>(get_active(src, dst))
              << std::endl;
  }

  void release_connection(devid_t src, devid_t dst) {
    decrease_incoming(dst);
    decrease_outgoing(src);
    decrease_active_links(src, dst);
    std::cout << "-Active links: " << static_cast<int>(get_active(src, dst))
              << std::endl;
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
    auto available = topology.get_max_connections(src, dst);
    return used <= available;
  }

  [[nodiscard]] bool check_connection(devid_t src, devid_t dst) const {

    std::cout << "Checking connection between " << src << " and " << dst
              << std::endl;
    std::cout << "Device available: " << is_device_available(src) << " "
              << is_device_available(dst) << std::endl;
    std::cout << "Link available: " << is_link_available(src, dst) << std::endl;
    std::cout << "Active links: " << static_cast<int>(get_active(src, dst))
              << std::endl;
    std::cout << "Max links: "
              << static_cast<int>(topology.get_max_connections(src, dst))
              << std::endl;

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
    const auto bw = static_cast<double>(get_bandwidth(src, dst));
    const auto s = static_cast<double>(size);
    return static_cast<mem_t>(s / bw);
  }

  [[nodiscard]] mem_t ideal_time_to_transfer(mem_t size, devid_t src,
                                             devid_t dst) const {
    const auto bw = static_cast<double>(get_bandwidth(src, dst));
    const auto s = static_cast<double>(size);
    return static_cast<mem_t>(s / bw);
  }

  SourceRequest
  get_best_available_source(devid_t dst, DeviceIDList &possible_sources) const {
    // Return the source with the highest bandwidth
    // If no source is available, return found=false

    bool found = false;
    devid_t best_source = 0;
    mem_t best_bandwidth = 0;
    for (auto src : possible_sources) {
      if (check_connection(src, dst)) {
        auto bandwidth = get_available_bandwidth(src, dst);
        if (bandwidth >= best_bandwidth) {
          best_bandwidth = bandwidth;
          best_source = src;
          found = true;
        }
      }
    }
    return {found, best_source};
  }
};