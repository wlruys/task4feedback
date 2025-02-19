#include <sys/types.h>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "include/communication_manager.hpp"

class UniformCommNoiseFixture {
protected:
  const unsigned int seed = 42;
  const std::size_t num_devices = 4;
  const mem_t bandwidth = 1000;
  const timecount_t latency = 100;
  const u_int8_t max_connections = 1;
  Topology topology;
  UniformCommunicationNoise noise;

public:
  UniformCommNoiseFixture() : topology(num_devices), noise(topology, seed) {
    for (devid_t i = 0; i < num_devices; ++i) {
      for (devid_t j = 0; j < num_devices; ++j) {
        topology.set_bandwidth(i, j, bandwidth);
        topology.set_latency(i, j, latency);
        topology.set_max_connections(i, j, max_connections);
      }
    }
  }
};

TEST_CASE_FIXTURE(UniformCommNoiseFixture,
                  "UniformCommunicationNoise generate and get") {

  const int LATENCY_100 = 100;
  const int LATENCY_200 = 200;
  const int LATENCY_300 = 300;
  const int LATENCY_400 = 400;

  std::vector<CommunicationRequest> requests = {
      {0, 1, 0, LATENCY_100},
      {1, 2, 1, LATENCY_200},
      {2, 3, 2, LATENCY_300},
      {3, 0, 3, LATENCY_400},
  };

  std::vector<CommunicationStats> statlist;

  for (std::size_t i = 0; i < requests.size(); ++i) {
    CommunicationStats stats = noise.get(requests[i]);
    CHECK(stats.latency >= 0);
    CHECK(stats.bandwidth >= 0);
    statlist.push_back(stats);
  }

  for (std::size_t i = 0; i < requests.size(); ++i) {
    CommunicationStats stats = noise.get(requests[i]);
    CHECK(stats.latency == statlist[i].latency);
    CHECK(stats.bandwidth == statlist[i].bandwidth);
  }
}

TEST_CASE_FIXTURE(UniformCommNoiseFixture,
                  "UniformCommunicationNoise: dump and load") {
  const std::string filename = "test_comm_noise.bin";
  const int LATENCY_100 = 100;
  const int LATENCY_200 = 200;
  const int LATENCY_300 = 300;
  const int LATENCY_400 = 400;

  std::vector<CommunicationRequest> requests = {
      {0, 1, 0, LATENCY_100},
      {1, 2, 1, LATENCY_200},
      {2, 3, 2, LATENCY_300},
      {3, 0, 3, LATENCY_400},
  };

  std::vector<CommunicationStats> statlist;

  for (auto &req : requests) {
    CommunicationStats stats = noise.get(req);
    CHECK(stats.latency >= 0);
    CHECK(stats.bandwidth >= 0);
    statlist.push_back(stats);
  }

  noise.dump_to_binary(filename);

  UniformCommunicationNoise new_noise(topology, seed);
  new_noise.load_from_binary(filename);

  for (std::size_t i = 0; i < requests.size(); ++i) {
    CommunicationStats stats = new_noise.get(requests[i]);
    CHECK(stats.latency >= 0);
    CHECK(stats.bandwidth >= 0);
    CHECK(stats.latency == statlist[i].latency);
    CHECK(stats.bandwidth == statlist[i].bandwidth);
  }
}