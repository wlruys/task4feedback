#pragma once

#include "settings.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#ifdef ENABLE_METIS
#include <metis.h>
#endif

struct MetisGraph {
  int32_t num_vertices = 0;
  std::vector<int32_t> xadj;
  std::vector<int32_t> adjncy;
  std::vector<int32_t> vwgts;
  std::vector<int32_t> adjwgt;
  std::vector<taskid_t> vertex_task_ids;
  std::vector<std::size_t> vertex_input_positions;

  void clear() {
    num_vertices = 0;
    xadj.clear();
    adjncy.clear();
    vwgts.clear();
    adjwgt.clear();
    vertex_task_ids.clear();
    vertex_input_positions.clear();
  }

  [[nodiscard]] int32_t num_edges() const {
    return static_cast<int32_t>(adjncy.size() / 2);
  }

  [[nodiscard]] bool empty() const {
    return num_vertices == 0 || adjncy.empty();
  }
};

class METIS_wrapper {
public:
  [[nodiscard]] static bool available() {
#ifdef ENABLE_METIS
    return true;
#else
    return false;
#endif
  }

  [[nodiscard]] static int32_t clamp_weight(uint64_t value) {
    constexpr auto max_weight = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    return static_cast<int32_t>(std::clamp<uint64_t>(value, 1, max_weight));
  }

#ifdef ENABLE_METIS
  bool call_metis_partition(const MetisGraph &graph, int32_t nparts, std::vector<int32_t> &part,
                            uint64_t seed = 0) const {
    if (graph.num_vertices <= 0 || nparts <= 0) {
      return false;
    }
    if (graph.xadj.size() != static_cast<std::size_t>(graph.num_vertices) + 1 ||
        graph.vwgts.size() != static_cast<std::size_t>(graph.num_vertices) ||
        graph.adjncy.size() != graph.adjwgt.size()) {
      return false;
    }

    idx_t nvtxs = static_cast<idx_t>(graph.num_vertices);
    idx_t ncon = 1;
    idx_t nparts_idx = static_cast<idx_t>(nparts);
    idx_t objective = 0;

    std::vector<idx_t> xadj(graph.xadj.begin(), graph.xadj.end());
    std::vector<idx_t> adjncy(graph.adjncy.begin(), graph.adjncy.end());
    std::vector<idx_t> vwgt(graph.vwgts.begin(), graph.vwgts.end());
    std::vector<idx_t> adjwgt(graph.adjwgt.begin(), graph.adjwgt.end());
    std::vector<idx_t> partition(static_cast<std::size_t>(graph.num_vertices), -1);

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED] = static_cast<idx_t>(seed);

    const int status =
        METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), vwgt.data(), nullptr,
                            adjwgt.data(), &nparts_idx, nullptr, nullptr, options, &objective,
                            partition.data());
    if (status != METIS_OK) {
      return false;
    }

    part.resize(partition.size());
    std::transform(partition.begin(), partition.end(), part.begin(),
                   [](idx_t block) { return static_cast<int32_t>(block); });
    return true;
  }
#else
  static bool call_metis_partition(const MetisGraph & /*unused*/, int32_t /*unused*/,
                                   std::vector<int32_t> & /*unused*/,
                                   uint64_t /*unused*/ = 0) {
    std::cerr << "[METIS] error: support was disabled at compile time\n";
    return false;
  }
#endif
};
