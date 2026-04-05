#include "tasks.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {
volatile int64_t BENCHMARK_SINK = 0;

template <typename Fn> double run_ms(Fn &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return static_cast<double>(elapsed.count()) / 1000.0;
}

template <typename Fn> double run_best_of_three_ms(Fn &&fn) {
  double best = run_ms(fn);
  best = std::min(best, run_ms(fn));
  best = std::min(best, run_ms(fn));
  return best;
}

struct BenchmarkCase {
  std::string name;
  taskid_t num_tasks;
  dataid_t num_data;
  int reads_per_task;
  int writes_per_task;
  bool dense_shared_reads;
  dataid_t dense_hot_data;
};

Graph build_graph(const BenchmarkCase &cfg) {
  Graph graph;
  graph.tasks.reserve(static_cast<std::size_t>(cfg.num_tasks));

  for (taskid_t task_id = 0; task_id < cfg.num_tasks; ++task_id) {
    const auto id = graph.add_task("t" + std::to_string(task_id));
    graph.set_tag(id, 0);
    graph.set_type(id, 0);
    graph.set_variant(id, DeviceType::GPU, 1, 1, 1);

    std::vector<dataid_t> read_ids;
    read_ids.reserve(static_cast<std::size_t>(cfg.reads_per_task));
    for (int i = 0; i < cfg.reads_per_task; ++i) {
      dataid_t data_id = 0;
      if (cfg.dense_shared_reads) {
        const auto hot_data = std::max<dataid_t>(cfg.dense_hot_data, 1);
        data_id = static_cast<dataid_t>((task_id + i * 7 + 3) % hot_data);
      } else {
        data_id = static_cast<dataid_t>((task_id * 37 + i * 101 + 7) % cfg.num_data);
      }
      read_ids.push_back(data_id);
    }
    graph.add_read_data(id, read_ids);

    std::vector<dataid_t> write_ids;
    write_ids.reserve(static_cast<std::size_t>(cfg.writes_per_task));
    for (int i = 0; i < cfg.writes_per_task; ++i) {
      write_ids.push_back(static_cast<dataid_t>((task_id * 53 + i * 67 + 11) % cfg.num_data));
    }
    graph.add_write_data(id, write_ids);
  }

  return graph;
}

int run_case(const BenchmarkCase &cfg) {
  constexpr int64_t NUM_QUERIES = 12'000'000;

  Graph graph = build_graph(cfg);

  const auto finalize_ms = run_ms([&]() { graph.finalize(false, false); });
  const auto static_ms = run_ms([&]() {
    StaticTaskInfo tmp(graph);
    (void)tmp.get_n_compute_tasks();
  });

  StaticTaskInfo static_graph(graph);

  int64_t scan_read_hits = 0;
  int64_t roaring_read_hits = 0;
  int64_t scan_unique_hits = 0;
  int64_t roaring_unique_hits = 0;
  int64_t scan_read_index_sum = 0;
  int64_t roaring_read_index_sum = 0;

  const auto scan_read_ms = run_best_of_three_ms([&]() {
    int64_t local = 0;
    for (int64_t q = 0; q < NUM_QUERIES; ++q) {
      const auto task_id = static_cast<taskid_t>((q * 13 + 17) % cfg.num_tasks);
      const auto data_id = static_cast<dataid_t>((q * 29 + 19) % cfg.num_data);
      const auto read = static_graph.get_read(task_id);
      local += (std::find(read.begin(), read.end(), data_id) != read.end()) ? 1 : 0;
    }
    scan_read_hits = local;
    BENCHMARK_SINK ^= local;
  });

  const auto roaring_read_ms = run_best_of_three_ms([&]() {
    int64_t local = 0;
    for (int64_t q = 0; q < NUM_QUERIES; ++q) {
      const auto task_id = static_cast<taskid_t>((q * 13 + 17) % cfg.num_tasks);
      const auto data_id = static_cast<dataid_t>((q * 29 + 19) % cfg.num_data);
      local += static_graph.has_read_data(task_id, data_id) ? 1 : 0;
    }
    roaring_read_hits = local;
    BENCHMARK_SINK ^= local;
  });

  const auto scan_unique_ms = run_best_of_three_ms([&]() {
    int64_t local = 0;
    for (int64_t q = 0; q < NUM_QUERIES; ++q) {
      const auto task_id = static_cast<taskid_t>((q * 7 + 23) % cfg.num_tasks);
      const auto data_id = static_cast<dataid_t>((q * 31 + 5) % cfg.num_data);
      const auto unique = static_graph.get_unique(task_id);
      local += (std::find(unique.begin(), unique.end(), data_id) != unique.end()) ? 1 : 0;
    }
    scan_unique_hits = local;
    BENCHMARK_SINK ^= local;
  });

  const auto roaring_unique_ms = run_best_of_three_ms([&]() {
    int64_t local = 0;
    for (int64_t q = 0; q < NUM_QUERIES; ++q) {
      const auto task_id = static_cast<taskid_t>((q * 7 + 23) % cfg.num_tasks);
      const auto data_id = static_cast<dataid_t>((q * 31 + 5) % cfg.num_data);
      local += static_graph.has_unique_data(task_id, data_id) ? 1 : 0;
    }
    roaring_unique_hits = local;
    BENCHMARK_SINK ^= local;
  });

  const auto scan_read_index_ms = run_best_of_three_ms([&]() {
    int64_t local = 0;
    for (int64_t q = 0; q < NUM_QUERIES; ++q) {
      const auto task_id = static_cast<taskid_t>((q * 17 + 3) % cfg.num_tasks);
      const auto data_id = static_cast<dataid_t>((q * 41 + 2) % cfg.num_data);
      const auto read = static_graph.get_read(task_id);
      const auto it = std::find(read.begin(), read.end(), data_id);
      local += (it == read.end()) ? -1 : static_cast<int64_t>(it - read.begin());
    }
    scan_read_index_sum = local;
    BENCHMARK_SINK ^= local;
  });

  const auto roaring_read_index_ms = run_best_of_three_ms([&]() {
    int64_t local = 0;
    for (int64_t q = 0; q < NUM_QUERIES; ++q) {
      const auto task_id = static_cast<taskid_t>((q * 17 + 3) % cfg.num_tasks);
      const auto data_id = static_cast<dataid_t>((q * 41 + 2) % cfg.num_data);
      local += static_cast<int64_t>(static_graph.get_read_data_index(task_id, data_id));
    }
    roaring_read_index_sum = local;
    BENCHMARK_SINK ^= local;
  });

  if (scan_read_hits != roaring_read_hits || scan_unique_hits != roaring_unique_hits ||
      scan_read_index_sum != roaring_read_index_sum) {
    std::cerr << "Mismatch between scan and bitset benchmark outputs for case '" << cfg.name << "'."
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "case=" << cfg.name << "\n";
  std::cout << "num_tasks=" << cfg.num_tasks << "\n";
  std::cout << "num_data=" << cfg.num_data << "\n";
  std::cout << "reads_per_task=" << cfg.reads_per_task << "\n";
  std::cout << "writes_per_task=" << cfg.writes_per_task << "\n";
  std::cout << "dense_shared_reads=" << cfg.dense_shared_reads << "\n";
  std::cout << "dense_hot_data=" << cfg.dense_hot_data << "\n";
  std::cout << "num_queries=" << NUM_QUERIES << "\n";
  std::cout << "graph_finalize_ms=" << finalize_ms << "\n";
  std::cout << "static_task_info_build_ms=" << static_ms << "\n";
  std::cout << "read_scan_ms=" << scan_read_ms << "\n";
  std::cout << "read_roaring_ms=" << roaring_read_ms << "\n";
  std::cout << "read_speedup=" << (scan_read_ms / roaring_read_ms) << "\n";
  std::cout << "unique_scan_ms=" << scan_unique_ms << "\n";
  std::cout << "unique_roaring_ms=" << roaring_unique_ms << "\n";
  std::cout << "unique_speedup=" << (scan_unique_ms / roaring_unique_ms) << "\n";
  std::cout << "read_index_scan_ms=" << scan_read_index_ms << "\n";
  std::cout << "read_index_roaring_ms=" << roaring_read_index_ms << "\n";
  std::cout << "read_index_speedup=" << (scan_read_index_ms / roaring_read_index_ms) << "\n";
  std::cout << "benchmark_sink=" << BENCHMARK_SINK << "\n";
  std::cout << "---\n";

  return EXIT_SUCCESS;
}

} // namespace

int main() {
  const std::vector<BenchmarkCase> cases{
      {
          .name = "sparse_shared_reads",
          .num_tasks = 6000,
          .num_data = 12000,
          .reads_per_task = 24,
          .writes_per_task = 6,
          .dense_shared_reads = false,
          .dense_hot_data = 0,
      },
      {
          .name = "dense_shared_reads",
          .num_tasks = 6000,
          .num_data = 12000,
          .reads_per_task = 24,
          .writes_per_task = 6,
          .dense_shared_reads = true,
          .dense_hot_data = 256,
      },
  };

  for (const auto &cfg : cases) {
    const int rc = run_case(cfg);
    if (rc != EXIT_SUCCESS) {
      return rc;
    }
  }

  return EXIT_SUCCESS;
}
