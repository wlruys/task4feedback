#pragma once

#include "data.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#ifdef ENABLE_KAHYPAR
#include <libkahypar.h>
#endif

struct KaHyParHypergraph {
  int32_t num_vertices = 0;
  std::vector<int32_t> eptr;
  std::vector<int32_t> eind;
  std::vector<int32_t> vwgts;
  std::vector<int32_t> hewgts;
  std::vector<dataid_t> edge_data_ids;
  std::vector<uint32_t> edge_generations;
  std::vector<taskid_t> vertex_task_ids;
  std::vector<std::size_t> vertex_input_positions;

  void clear() {
    num_vertices = 0;
    eptr.clear();
    eind.clear();
    vwgts.clear();
    hewgts.clear();
    edge_data_ids.clear();
    edge_generations.clear();
    vertex_task_ids.clear();
    vertex_input_positions.clear();
  }

  [[nodiscard]] int32_t num_hyperedges() const {
    if (eptr.empty()) {
      return 0;
    }
    return static_cast<int32_t>(eptr.size()) - 1;
  }

  [[nodiscard]] bool empty() const {
    return num_vertices == 0 || num_hyperedges() == 0;
  }
};

class KaHyPar_wrapper {
public:
  [[nodiscard]] static bool available() {
#ifdef ENABLE_KAHYPAR
    return true;
#else
    return false;
#endif
  }

  [[nodiscard]] static int32_t clamp_weight(uint64_t value) {
    constexpr auto max_weight =
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    return static_cast<int32_t>(std::clamp<uint64_t>(value, 1, max_weight));
  }

#ifdef ENABLE_KAHYPAR
  bool call_kahypar_partition(const KaHyParHypergraph &hypergraph, int32_t nparts,
                              std::vector<int32_t> &part, uint64_t seed = 0,
                              double imbalance = 0.03) const {
    if (hypergraph.num_vertices <= 0 || hypergraph.num_hyperedges() <= 0 || nparts <= 0) {
      return false;
    }
    if (hypergraph.vwgts.size() != static_cast<std::size_t>(hypergraph.num_vertices) ||
        hypergraph.eptr.size() != static_cast<std::size_t>(hypergraph.num_hyperedges()) + 1 ||
        hypergraph.hewgts.size() != static_cast<std::size_t>(hypergraph.num_hyperedges())) {
      return false;
    }

    kahypar_context_t *context = kahypar_context_new();
    kahypar_configure_context_from_file(context, KAHYPAR_CONFIG_PATH);
    kahypar_set_seed(context, static_cast<int>(seed));

    std::vector<size_t> hyperedge_indices(hypergraph.eptr.begin(), hypergraph.eptr.end());
    std::vector<kahypar_hyperedge_id_t> hyperedges(hypergraph.eind.begin(), hypergraph.eind.end());
    std::vector<kahypar_hypernode_weight_t> vertex_weights(hypergraph.vwgts.begin(),
                                                           hypergraph.vwgts.end());
    std::vector<kahypar_hyperedge_weight_t> edge_weights(hypergraph.hewgts.begin(),
                                                         hypergraph.hewgts.end());
    std::vector<kahypar_partition_id_t> partition(
        static_cast<std::size_t>(hypergraph.num_vertices), -1);
    kahypar_hyperedge_weight_t objective = 0;

    kahypar_partition(static_cast<kahypar_hypernode_id_t>(hypergraph.num_vertices),
                      static_cast<kahypar_hyperedge_id_t>(hypergraph.num_hyperedges()),
                      imbalance, static_cast<kahypar_partition_id_t>(nparts),
                      vertex_weights.data(), edge_weights.data(), hyperedge_indices.data(),
                      hyperedges.data(), &objective, context, partition.data());

    kahypar_context_free(context);

    part.resize(partition.size());
    std::transform(partition.begin(), partition.end(), part.begin(),
                   [](kahypar_partition_id_t block) { return static_cast<int32_t>(block); });
    return true;
  }
#else
  static bool call_kahypar_partition(const KaHyParHypergraph & /*unused*/, int32_t /*unused*/,
                                     std::vector<int32_t> & /*unused*/,
                                     uint64_t /*unused*/ = 0,
                                     double /*unused*/ = 0.03) {
    std::cerr << "[KaHyPar] error: support was disabled at compile time\n";
    return false;
  }
#endif
};
