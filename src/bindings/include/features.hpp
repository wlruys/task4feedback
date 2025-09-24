#pragma once
#include "devices.hpp"
#include "nbh.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory>
#include <queue>
#include <span>
#include <stack>
#include <sys/types.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using op_t = int32_t;
using f_t = float;

taskid_t max(taskid_t a, taskid_t b) {
  return a > b ? a : b;
}

taskid_t min(taskid_t a, taskid_t b) {
  return a < b ? a : b;
}

enum class NodeType : int8_t {
  ANY = -1,
  TASK = 0,
  DATA_BLOCK = 1,
  DEVICE = 2
};

enum class EdgeType : int8_t {
  ANY = -1,
  TASK_TASK = 0,
  TASK_DATA = 1,
  TASK_DEVICE = 2,
  DATA_DEVICE = 3
};

struct GraphSpec {
public:
  taskid_t max_tasks = 0;
  taskid_t max_data = 0;
  taskid_t max_devices = 5;
  taskid_t max_in_degree = 0;
  taskid_t max_out_degree = 0;
  dataid_t max_data_in_degree = 0;
  dataid_t max_data_out_degree = 0;
  dataid_t max_data_usage = 0;
  taskid_t max_candidates = 3;
  taskid_t max_edges_tasks_tasks = 0;
  taskid_t max_edges_tasks_data = 0;
  taskid_t max_edges_tasks_devices = 0;
  taskid_t max_edges_data_devices = 0;
};

using TaskSet = ankerl::unordered_dense::set<taskid_t>;

class GraphExtractor {
protected:
  TaskSet visited;
  ankerl::unordered_dense::set<dataid_t> data_visited;
  ankerl::unordered_dense::set<taskid_t> local_visited;
  ankerl::unordered_dense::map<taskid_t, int64_t> task_index_map;
  ankerl::unordered_dense::map<dataid_t, int64_t> data_index_map;
  ankerl::unordered_dense::map<devid_t, int64_t> device_index_map;
  std::vector<op_t> source_list;
  std::vector<op_t> target_list;
  std::reference_wrapper<const SchedulerState> state;

public:
  GraphExtractor(const SchedulerState &state) : state(state) {
    source_list.reserve(400);
    target_list.reserve(400);
    visited.reserve(400);
    local_visited.reserve(400);
    task_index_map.reserve(400);
    data_index_map.reserve(400);
    device_index_map.reserve(400);
    data_visited.reserve(400);
  }

  [[nodiscard]] TaskIDList get_active_tasks() const {
    const auto &s = this->state.get();
    return s.counts.get_active_task_list();
  }

  void get_device_selection_mask(taskid_t task_id, TorchArr1D<int8_t> &mask) {
    auto v = mask.view();
    // Loop over each device, check its architecture and fill the mask if it is supported
    const auto &s = this->state.get();
    const auto &device_manager = s.get_device_manager();
    const auto &devices = s.get_devices();
    const auto &static_graph = s.get_tasks();

    for (int i = 0; i < devices.size(); i++) {
      const auto arch = devices.get_type(i);
      bool is_supported = static_graph.is_architecture_supported(task_id, arch);
      v(i) = is_supported ? 1 : 0;
    }
  }

  void _get_k_hop_task_dependents(TaskSet &visited, taskid_t task_id, int k, size_t max_tasks) {

    const auto &s = this->state.get();
    const auto &static_graph = s.get_tasks();

    local_visited.clear();

    std::queue<taskid_t> q;
    q.push(task_id);
    visited.insert(task_id);

    int current_hop = 0;

    while (!q.empty() && current_hop < k) {
      std::size_t level_size = q.size();

      for (std::size_t i = 0; i < level_size; ++i) {
        taskid_t current_task_id = q.front();
        q.pop();

        for (const auto &dep_id : static_graph.get_compute_task_dependents(current_task_id)) {
          if (local_visited.insert(dep_id).second) {
            if (visited.size() >= max_tasks) {
              spdlog::warn("Task count exceeded max tasks: {}", visited.size());
              return;
            }
            q.push(dep_id);
            visited.insert(dep_id);
          }
        }
      }

      current_hop++;
    }
  }

  [[nodiscard]] size_t get_k_hop_task_dependents(TaskSet &visited, TorchInt64Arr1D &initial_tasks,
                                                 int k, TorchInt64Arr1D &output) {
    auto v = output.view();
    size_t max_tasks = output.size();
    static bool has_warned = false;
    std::span<int64_t> initial_tasks_span(initial_tasks.data(), initial_tasks.size());

    for (const auto &task_id_64_bit : initial_tasks_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      _get_k_hop_task_dependents(visited, task_id, k, max_tasks);
      if (visited.size() >= max_tasks) {
        if (!has_warned) {
          spdlog::warn("Task count exceeded max tasks: {}", visited.size());
          has_warned = true;
        }
        break;
      }
    }

    auto count = min(max_tasks, visited.size());

    // Always print the initial tasks first (remove them from visisted)
    // Then fill the rest

    size_t i = 0;
    for (auto task_id_64_bit : initial_tasks_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      visited.erase(task_id);
      v(i) = task_id_64_bit;
      i++;
    }

    for (auto task : visited) {
      if (i >= count) {
        if (!has_warned) {
          spdlog::warn("Task count exceeded max tasks: {}", visited.size());
          has_warned = true;
        }
        break;
      }
      v(i) = static_cast<int64_t>(task);
      i++;
    }

    return count;
  }

  [[nodiscard]] size_t get_k_hop_dependents(TorchInt64Arr1D &initial_tasks, int k,
                                            TorchInt64Arr1D &output) {
    visited.clear();
    return get_k_hop_task_dependents(visited, initial_tasks, k, output);
  }

  void _get_k_hop_task_dependencies(TaskSet &visited, taskid_t task_id, int k, size_t max_tasks) {

    const auto &s = this->state.get();
    const auto &static_graph = s.get_tasks();

    local_visited.clear();

    std::queue<taskid_t> q;
    q.push(task_id);
    visited.insert(task_id);

    int current_hop = 0;

    while (!q.empty() && current_hop < k) {
      std::size_t level_size = q.size();

      for (std::size_t i = 0; i < level_size; ++i) {
        taskid_t current_task_id = q.front();
        q.pop();

        for (const auto &dep_id : static_graph.get_compute_task_dependencies(current_task_id)) {
          if (local_visited.insert(dep_id).second) {
            if (visited.size() >= max_tasks) {
              return;
            }
            q.push(dep_id);
            visited.insert(dep_id);
          }
        }
      }

      current_hop++;
    }
  }

  [[nodiscard]] size_t get_k_hop_task_dependencies(TaskSet &visited, TorchInt64Arr1D &initial_tasks,
                                                   int k, TorchInt64Arr1D &output) {

    std::span<int64_t> initial_tasks_span(initial_tasks.data(), initial_tasks.size());
    static bool has_warned = false;
    auto v = output.view();

    size_t max_tasks = output.size();
    for (const auto &task_id_64_bit : initial_tasks_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      _get_k_hop_task_dependencies(visited, task_id, k, max_tasks);
      if (visited.size() >= max_tasks) {
        if (!has_warned) {
          spdlog::warn("Task count exceeded max tasks, {}", visited.size());
          has_warned = true;
        }
        break;
      }
    }

    auto count = min(max_tasks, visited.size());

    // Always print the initial tasks first (remove them from visisted)
    // Then fill the rest

    size_t i = 0;
    for (auto task_id_64_bit : initial_tasks_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      visited.erase(task_id);
      v(i) = task_id_64_bit;
      i++;
    }

    for (auto task : visited) {
      if (i >= count) {
        if (!has_warned) {
          spdlog::warn("Task count exceeded max tasks, {}", visited.size());
          has_warned = true;
        }
        break;
      }
      v(i) = static_cast<int64_t>(task);
      i++;
    }

    return count;
  }

  size_t get_k_hop_dependencies(TorchInt64Arr1D &initial_tasks, int k, TorchInt64Arr1D &output) {
    visited.clear();
    return get_k_hop_task_dependencies(visited, initial_tasks, k, output);
  }

  [[nodiscard]] size_t get_k_hop_task_bidirectional(TaskSet &visited,
                                                    TorchInt64Arr1D &initial_tasks, int k,
                                                    TorchInt64Arr1D &output) {

    std::span<int64_t> initial_tasks_span(initial_tasks.data(), initial_tasks.size());
    static bool has_warned = false;
    auto v = output.view();

    size_t max_tasks = output.size();
    for (const auto &task_id_64_bit : initial_tasks_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      _get_k_hop_task_dependencies(visited, task_id, k, max_tasks);
      _get_k_hop_task_dependents(visited, task_id, k, max_tasks);
      if (visited.size() >= max_tasks) {
        if (!has_warned) {
          spdlog::warn("Task count exceeded max tasks, {}", visited.size());
          has_warned = true;
        }
        break;
      }
    }

    auto count = min(max_tasks, visited.size());

    // Always print the initial tasks first (remove them from visisted)
    // Then fill the rest

    size_t i = 0;
    for (auto task_id_64_bit : initial_tasks_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      visited.erase(task_id);
      v(i) = task_id_64_bit;
      i++;
    }

    for (auto task : visited) {
      if (i >= count) {
        if (!has_warned) {
          spdlog::warn("Task count exceeded max tasks, {}", visited.size());
          has_warned = true;
        }
        break;
      }
      v(i) = static_cast<int64_t>(task);
      i++;
    }

    return count;
  }

  void _get_k_hop_task_neighborhood(TaskSet &visited, taskid_t task_id, int k, size_t max_tasks) {

    const auto &static_graph = state.get().get_tasks();
    static bool has_warned = false;
    local_visited.clear();

    std::queue<taskid_t> q;
    q.push(task_id);
    visited.insert(task_id);

    int current_hop = 0;

    while (!q.empty() && current_hop < k) {
      std::size_t level_size = q.size();

      for (std::size_t i = 0; i < level_size; ++i) {
        taskid_t current_task_id = q.front();
        q.pop();

        for (const auto &dep_id : static_graph.get_compute_task_dependencies(current_task_id)) {
          if (local_visited.insert(dep_id).second) {
            if (visited.size() >= max_tasks) {
              if (!has_warned) {
                spdlog::warn("Task count exceeded max tasks: {}", visited.size());
                has_warned = true;
              }
              return;
            }
            q.push(dep_id);
            visited.insert(dep_id);
          }
        }

        for (const auto &dep_id : static_graph.get_compute_task_dependents(current_task_id)) {
          if (local_visited.insert(dep_id).second) {
            if (visited.size() >= max_tasks) {
              if (!has_warned) {
                spdlog::warn("Task count exceeded max tasks: {}", visited.size());
                has_warned = true;
              }
              return;
            }
            q.push(dep_id);
            visited.insert(dep_id);
          }
        }
      }

      current_hop++;
    }
  }

  // size_t get_k_hop_dependencies(TorchInt64Arr1D &initial_tasks, int k, TorchInt64Arr1D &output) {
  //   visited.clear();
  //   return get_k_hop_task_dependencies(visited, initial_tasks, k, output);
  // }

  size_t get_k_hop_task_neighborhood(TaskSet &visited, TorchInt64Arr1D &initial_tasks, int k,
                                     TorchInt64Arr1D &output) {
    auto v = output.view();

    size_t max_tasks = output.size();
    std::span<int64_t> initial_tasks_span(initial_tasks.data(), initial_tasks.size());
    static bool has_warned = false;
    for (const auto &task_id_64_bit : initial_tasks_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      _get_k_hop_task_neighborhood(visited, task_id, k, max_tasks);
      if (visited.size() >= max_tasks) {
        if (!has_warned) {
          spdlog::warn("Task count exceeded max tasks, {}", visited.size());
          has_warned = true;
        }
        break;
      }
    }

    auto count = min(max_tasks, visited.size());

    // Always print the initial tasks first (remove them from visisted)
    // Then fill the rest

    size_t i = 0;

    for (auto task_id_64_bit : initial_tasks_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      visited.erase(task_id);
      v(i) = task_id_64_bit;
      i++;
    }

    for (auto task : visited) {
      if (i >= count) {
        if (!has_warned) {
          spdlog::warn("Task count exceeded max tasks");
          has_warned = true;
        }
        break;
      }
      v(i) = static_cast<int64_t>(task);
      i++;
    }

    return count;
  }

  size_t get_k_hop_neighborhood(TorchInt64Arr1D &initial_tasks, int k, TorchInt64Arr1D &output) {
    visited.clear();
    return get_k_hop_task_neighborhood(visited, initial_tasks, k, output);
  }

  size_t get_k_hop_bidirectional(TorchInt64Arr1D &initial_tasks, int k, TorchInt64Arr1D &output) {
    visited.clear();
    return get_k_hop_task_bidirectional(visited, initial_tasks, k, output);
  }

  size_t get_task_task_edges(TorchInt64Arr1D &sources, TorchInt64Arr2D &output,
                             TorchInt64Arr2D &global_output) {

    // Check first dimension is 2
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    auto v = output.view();
    auto gv = global_output.view();

    const auto max_edges = output.shape(1);
    static bool has_warned = false;
    const auto &static_graph = state.get().get_tasks();

    task_index_map.clear();
    task_index_map.reserve(sources.size());

    std::span<int64_t> sources_span(sources.data(), sources.size());

    for (int64_t i = 0; i < sources_span.size(); i++) {
      task_index_map[static_cast<taskid_t>(sources_span[i])] = i;
    }

    std::size_t edge_count = 0;

    for (int64_t source_idx = 0; source_idx < sources_span.size(); source_idx++) {
      const auto source_id = sources_span[source_idx];
      const auto &dependencies = static_graph.get_compute_task_dependencies(source_id);

      // Process each dependency
      for (const auto &dep_id : dependencies) {
        auto it = task_index_map.find(dep_id);
        if (it != task_index_map.end()) {
          v(0, edge_count) = static_cast<int64_t>(source_idx);
          v(1, edge_count) = static_cast<int64_t>(it->second);
          gv(0, edge_count) = static_cast<int64_t>(source_id);
          gv(1, edge_count) = static_cast<int64_t>(dep_id);
          edge_count++;

          if (edge_count >= max_edges) {
            if (!has_warned) {
              spdlog::warn("TaskTask edge count exceeded max edges: {}", edge_count);
              has_warned = true;
            }
            break;
          }
        }
      }
      if (edge_count >= max_edges) {
        if (!has_warned) {
          spdlog::warn("TaskTask edge count exceeded max edges: {}", edge_count);
          has_warned = true;
        }
        break;
      }
    }
    return edge_count;
  }

  size_t get_task_task_edges_reverse(TorchInt64Arr1D &sources, TorchInt64Arr2D &output,
                                     TorchInt64Arr2D &global_output) {
    // Check first dimension is 2
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    auto v = output.view();
    auto gv = global_output.view();
    static bool has_warned = false;
    const auto max_edges = output.shape(1);

    task_index_map.clear();
    task_index_map.reserve(sources.size());

    std::span<int64_t> sources_span(sources.data(), sources.size());

    for (int64_t i = 0; i < sources_span.size(); i++) {
      task_index_map[static_cast<taskid_t>(sources_span[i])] = i;
    }

    const auto &static_graph = state.get().get_tasks();
    std::size_t edge_count = 0;

    for (int64_t source_idx = 0; source_idx < sources_span.size(); source_idx++) {
      const auto source_id = sources_span[source_idx];
      const auto &dependents = static_graph.get_compute_task_data_dependents(source_id);

      // Process each dependency
      for (const auto &dep_id : dependents) {
        auto it = task_index_map.find(dep_id);
        if (it != task_index_map.end()) {
          v(0, edge_count) = static_cast<int64_t>(source_idx);
          v(1, edge_count) = static_cast<int64_t>(it->second);
          gv(0, edge_count) = static_cast<int64_t>(source_id);
          gv(1, edge_count) = static_cast<int64_t>(dep_id);
          edge_count++;
          if (edge_count >= max_edges) {
            if (!has_warned) {
              spdlog::warn("TaskTask edge count exceeded max edges: {}", edge_count);
              has_warned = true;
            }
            break;
          }
        }
      }
      if (edge_count >= max_edges) {
        if (!has_warned) {
          spdlog::warn("TaskTask edge count exceeded max edges: {}", edge_count);
          has_warned = true;
        }
        break;
      }
    }
    return edge_count;
  }

  size_t get_unique_data(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &output) {
    data_visited.clear();
    data_visited.reserve(400);

    const auto max_data = output.size();
    auto v = output.view();
    static bool has_warned = false;
    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    const auto &s = this->state.get();
    const auto &static_graph = s.get_tasks();

    for (const auto &task_id_64_bit : task_ids_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      const auto &unique = static_graph.get_unique(task_id);

      for (int i = 0; i < unique.size(); i++) {

        auto data_id = unique[i];
        data_visited.insert(data_id);
        if (data_visited.size() >= max_data) {
          if (!has_warned) {
            spdlog::warn("Unique data count exceeded max data: {}", data_visited.size());
            has_warned = true;
          }
          break;
        }
      }
      if (data_visited.size() >= max_data) {
        if (!has_warned) {
          spdlog::warn("Unique data count exceeded max data: {}", data_visited.size());
          has_warned = true;
        }
        break;
      }
    }

    size_t count = min(output.size(), data_visited.size());

    // std::cout << "Unique data count: " << count << std::endl;
    size_t i = 0;
    for (auto data_id : data_visited) {
      if (i >= count) {
        if (!has_warned) {
          spdlog::warn("Unique data count exceeded max data: {}", data_visited.size());
          has_warned = true;
        }
        break;
      }
      v(i) = static_cast<int64_t>(data_id);
      i++;
    }
    return count;
  }

  size_t get_read_data(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &output) {
    data_visited.clear();
    data_visited.reserve(400);

    const auto max_data = output.size();
    auto v = output.view();
    static bool has_warned = false;
    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    const auto &s = this->state.get();
    const auto &static_graph = s.get_tasks();

    for (const auto &task_id_64_bit : task_ids_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      const auto read = static_graph.get_read(task_id);

      // std::cout << "Task ID: " << task_id << " Read size: " << read.size() << std::endl;

      for (int i = 0; i < read.size(); i++) {

        auto data_id = read[i];
        data_visited.insert(data_id);
        if (data_visited.size() >= max_data) {
          if (!has_warned) {
            spdlog::warn("Read data count exceeded max data: {}", data_visited.size());
            has_warned = true;
          }
          break;
        }
      }
      if (data_visited.size() >= max_data) {
        if (!has_warned) {
          spdlog::warn("Read data count exceeded max data: {}", data_visited.size());
          has_warned = true;
        }
        break;
      }
    }

    size_t count = min(output.size(), data_visited.size());

    // std::cout << "Unique data count: " << count << std::endl;
    size_t i = 0;
    for (auto data_id : data_visited) {
      if (i >= count) {
        if (!has_warned) {
          spdlog::warn("Read data count exceeded max data: {}", data_visited.size());
          has_warned = true;
        }
        break;
      }
      v(i) = static_cast<int64_t>(data_id);
      i++;
    }
    return count;
  }

  size_t get_write_data(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &output) {
    data_visited.clear();
    data_visited.reserve(400);

    const auto max_data = output.size();
    auto v = output.view();
    static bool has_warned = false;
    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    const auto &s = this->state.get();
    const auto &static_graph = s.get_tasks();

    for (const auto &task_id_64_bit : task_ids_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      const auto write = static_graph.get_write(task_id);

      // std::cout << "Task ID: " << task_id << " Read size: " << read.size() << std::endl;

      for (int i = 0; i < write.size(); i++) {

        auto data_id = write[i];
        data_visited.insert(data_id);
        if (data_visited.size() >= max_data) {
          if (!has_warned) {
            spdlog::warn("Write data count exceeded max data: {}", data_visited.size());
            has_warned = true;
          }
          break;
        }
      }
      if (data_visited.size() >= max_data) {
        if (!has_warned) {
          spdlog::warn("Write data count exceeded max data: {}", data_visited.size());
          has_warned = true;
        }
        break;
      }
    }

    size_t count = min(output.size(), data_visited.size());

    // std::cout << "Unique data count: " << count << std::endl;
    size_t i = 0;
    for (auto data_id : data_visited) {
      if (i >= count) {
        if (!has_warned) {
          spdlog::warn("Unique data count exceeded max data: {}", data_visited.size());
          has_warned = true;
        }
        break;
      }
      v(i) = static_cast<int64_t>(data_id);
      i++;
    }
    return count;
  }

  size_t get_task_data_edges_all(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &data_ids,
                                 TorchInt64Arr2D &output, TorchInt64Arr2D &global_output) {
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    std::span<int64_t> data_ids_span(data_ids.data(), data_ids.size());
    static bool has_warned = false;
    const auto max_edges = output.shape(1);

    data_index_map.clear();
    data_index_map.reserve(data_ids.size());

    auto v = output.view();
    auto gv = global_output.view();

    const auto &static_graph = state.get().get_tasks();

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {
      data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
    }

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);
      for (auto data_id : static_graph.get_unique(task_id)) {
        auto it = data_index_map.find(data_id);
        if (it != data_index_map.end()) {
          v(0, edge_count) = static_cast<int64_t>(i);
          v(1, edge_count) = static_cast<int64_t>(it->second);
          gv(0, edge_count) = static_cast<int64_t>(task_id);
          gv(1, edge_count) = static_cast<int64_t>(data_id);
          edge_count++;
          if (edge_count >= max_edges) {
            if (!has_warned) {
              spdlog::warn("TaskData edge count exceeded max edges: {}", edge_count);
              has_warned = true;
            }
            break;
          }
        }
      }
      if (edge_count >= max_edges) {
        if (!has_warned) {
          spdlog::warn("TaskData edge count exceeded max edges: {}", edge_count);
          has_warned = true;
        }
        break;
      }
    }
    return edge_count;
  }

  size_t get_task_data_edges_read(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &data_ids,
                                  TorchInt64Arr2D &output, TorchInt64Arr2D &global_output) {
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    std::span<int64_t> data_ids_span(data_ids.data(), data_ids.size());
    static bool has_warned = false;
    const auto max_edges = output.shape(1);

    data_index_map.clear();
    data_index_map.reserve(data_ids.size());

    auto v = output.view();
    auto gv = global_output.view();

    const auto &static_graph = state.get().get_tasks();

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {
      // if (data_index_map.find(static_cast<dataid_t>(data_ids_span[i])) == data_index_map.end()) {
      //   data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
      // }
      data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
    }

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);
      for (auto data_id : static_graph.get_read(task_id)) {
        auto it = data_index_map.find(data_id);
        if (it != data_index_map.end()) {
          v(0, edge_count) = static_cast<int64_t>(i);
          v(1, edge_count) = static_cast<int64_t>(it->second);
          gv(0, edge_count) = static_cast<int64_t>(task_id);
          gv(1, edge_count) = static_cast<int64_t>(data_id);
          edge_count++;
          if (edge_count >= max_edges) {
            if (!has_warned) {
              spdlog::warn("TaskData edge count exceeded max edges: {}", edge_count);
              has_warned = true;
            }
            break;
          }
        }
      }
      if (edge_count >= max_edges) {
        if (!has_warned) {
          spdlog::warn("TaskData edge count exceeded max edges: {}", edge_count);
          has_warned = true;
        }
        break;
      }
    }
    return edge_count;
  }

  size_t get_task_data_edges_write(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &data_ids,
                                   TorchInt64Arr2D &output, TorchInt64Arr2D &global_output) {
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    std::span<int64_t> data_ids_span(data_ids.data(), data_ids.size());
    static bool has_warned = false;
    const auto max_edges = output.shape(1);

    data_index_map.clear();
    data_index_map.reserve(data_ids.size());

    auto v = output.view();
    auto gv = global_output.view();

    const auto &static_graph = state.get().get_tasks();

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {
      // if (data_index_map.find(static_cast<dataid_t>(data_ids_span[i])) == data_index_map.end()) {
      //   data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
      // }
      data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
    }

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);
      for (auto data_id : static_graph.get_write(task_id)) {
        auto it = data_index_map.find(data_id);
        if (it != data_index_map.end()) {
          v(0, edge_count) = static_cast<int64_t>(i);
          v(1, edge_count) = static_cast<int64_t>(it->second);
          gv(0, edge_count) = static_cast<int64_t>(task_id);
          gv(1, edge_count) = static_cast<int64_t>(data_id);
          edge_count++;
          if (edge_count >= max_edges) {
            if (!has_warned) {
              spdlog::warn("TaskData edge count exceeded max edges: {}", edge_count);
              has_warned = true;
            }
            break;
          }
        }
      }
      if (edge_count >= max_edges) {
        if (!has_warned) {
          spdlog::warn("TaskData edge count exceeded max edges: {}", edge_count);
          has_warned = true;
        }
        break;
      }
    }
    return edge_count;
  }

  size_t get_task_data_edges_read_mapped(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &data_ids,
                                         TorchInt64Arr2D &output, TorchInt64Arr2D &global_output) {
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    std::span<int64_t> data_ids_span(data_ids.data(), data_ids.size());
    static bool has_warned = false;
    const auto max_edges = output.shape(1);

    data_index_map.clear();
    data_index_map.reserve(data_ids.size());

    auto v = output.view();
    auto gv = global_output.view();

    const auto &static_graph = state.get().get_tasks();
    const auto &task_runtime = state.get().get_task_runtime();

    // // The candidate task id is always the first task id in the list
    // int64_t candidate_task_id = task_ids_span[0];
    // std::cout << "Candidate Task ID: " << candidate_task_id << std::endl;

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {
      data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
    }

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);
      const auto &read = static_graph.get_read(task_id);
      const auto &recent_writers = static_graph.get_most_recent_writers(task_id);

      for (int j = 0; j < read.size(); j++) {
        auto data_id = read[j];
        taskid_t recent_writer_id = recent_writers[j];

        if (recent_writer_id != -1) {
          // We are not the first writer, check to see if it is mapped w.r.t us
          const bool is_mapped = task_runtime.is_compute_mapped(recent_writer_id);
          if (!is_mapped) {
            continue;
          }
          // TODO(wlr): This needs to support retire
        }

        auto it = data_index_map.find(data_id);
        if (it != data_index_map.end()) {
          v(0, edge_count) = static_cast<int64_t>(i);
          v(1, edge_count) = static_cast<int64_t>(it->second);
          gv(0, edge_count) = static_cast<int64_t>(task_id);
          gv(1, edge_count) = static_cast<int64_t>(data_id);
          edge_count++;
          if (edge_count >= max_edges) {
            if (!has_warned) {
              spdlog::warn("Filtered TaskData edge count exceeded max edges: {}", edge_count);
              has_warned = true;
            }
            break;
          }
        }
      }
      if (edge_count >= max_edges) {
        if (!has_warned) {
          spdlog::warn("Filtered TaskData edge count exceeded max edges: {}", edge_count);
          has_warned = true;
        }
        break;
      }
    }
    return edge_count;
  }

  size_t get_task_device_edges(TorchInt64Arr1D &task_ids, TorchInt64Arr2D &output,
                               TorchInt64Arr2D &global_output) {
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    auto v = output.view();
    auto gv = global_output.view();
    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    static bool has_warned = false;

    const auto max_edges = output.shape(1);
    const std::size_t device_count = state.get().get_devices().size();

    const auto &task_runtime = state.get().get_task_runtime();

    std::size_t edge_count = 0;

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);

      devid_t mapped_device = task_runtime.get_compute_task_mapped_device(task_id);

      if (mapped_device != -1) {
        v(0, edge_count) = static_cast<int64_t>(i);
        v(1, edge_count) = static_cast<int64_t>(mapped_device);
        gv(0, edge_count) = static_cast<int64_t>(task_id);
        gv(1, edge_count) = static_cast<int64_t>(mapped_device);
        edge_count++;
        if (edge_count >= max_edges) {
          if (!has_warned) {
            spdlog::warn("TaskDevice edge count exceeded max edges: {}", edge_count);
            has_warned = true;
          }
          break;
        }
      }
    }

    return edge_count;
  }

  size_t get_data_device_edges_filtered(TorchInt64Arr1D &filtered_ids, TorchInt64Arr1D &data_ids,
                                        TorchInt64Arr2D &output, TorchInt64Arr2D &global_output) {

    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    std::span<int64_t> filtered_ids_span(filtered_ids.data(), filtered_ids.size());
    ankerl::unordered_dense::set<int64_t> filtered_ids_set(filtered_ids_span.begin(),
                                                           filtered_ids_span.end());

    auto v = output.view();
    auto gv = global_output.view();
    static bool has_warned = false;

    std::span<int64_t> data_ids_span(data_ids.data(), data_ids.size());

    const auto max_edges = output.shape(1);

    const auto &devices = state.get().get_devices();
    const auto &data_manager = state.get().get_data_manager();

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {

      if (filtered_ids_set.find(data_ids_span[i]) == filtered_ids_set.end()) {
        continue;
      }

      for (int64_t j = 0; j < devices.size(); j++) {

        if (data_manager.check_valid_mapped(static_cast<dataid_t>(data_ids_span[i]), j)) {
          v(0, i) = static_cast<int64_t>(i);
          v(1, i) = static_cast<int64_t>(j);
          gv(0, i) = static_cast<int64_t>(data_ids_span[i]);
          gv(1, i) = static_cast<int64_t>(j);
          edge_count++;
          if (edge_count >= max_edges) {
            if (!has_warned) {
              spdlog::warn("DataDevice edge count exceeded max edges: {}", edge_count);
              has_warned = true;
            }
            break;
          }
        }
      }
      if (edge_count >= max_edges) {
        if (!has_warned) {
          spdlog::warn("DataDevice edge count exceeded max edges: {}", edge_count);
          has_warned = true;
        }
        break;
      }
    }
    return edge_count;
  }

  size_t get_data_device_edges(TorchInt64Arr1D &data_ids, TorchInt64Arr2D &output,
                               TorchInt64Arr2D &global_output) {

    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    auto v = output.view();

    auto gv = global_output.view();
    static bool has_warned = false;
    std::span<int64_t> data_ids_span(data_ids.data(), data_ids.size());

    const auto max_edges = output.shape(1);
    const auto &devices = state.get().get_devices();
    const auto &data_manager = state.get().get_data_manager();

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {
      for (int64_t j = 0; j < devices.size(); j++) {

        if (data_manager.check_valid_mapped(static_cast<dataid_t>(data_ids_span[i]), j)) {
          v(0, i) = static_cast<int64_t>(i);
          v(1, i) = static_cast<int64_t>(j);
          gv(0, i) = static_cast<int64_t>(data_ids_span[i]);
          gv(1, i) = static_cast<int64_t>(j);
          edge_count++;
          if (edge_count >= max_edges) {
            if (!has_warned) {
              spdlog::warn("DataDevice edge count exceeded max edges: {}", edge_count);
              has_warned = true;
            }
            break;
          }
        }
      }
      if (edge_count >= max_edges) {
        if (!has_warned) {
          spdlog::warn("DataDevice edge count exceeded max edges: {}", edge_count);
          has_warned = true;
        }
        break;
      }
    }
    return edge_count;
  }
};

f_t guarded_divide(double a, double b) {
  if (b == 0) {
    return static_cast<f_t>(a);
  }
  return static_cast<f_t>(a / b);
}

void one_hot(int index, std::span<f_t> output) {
  for (std::size_t i = 0; i < output.size(); i++) {
    output[i] = static_cast<f_t>(i == index);
  }
}

// CRTP base for features
template <typename Derived> struct Feature {
  [[nodiscard]] size_t getFeatureDim() const {
    return static_cast<const Derived *>(this)->getFeatureDimImpl();
  }

  void extractFeature(int32_t object_id, TorchArr &output) {

    std::span<float> sp(output.data(), output.size());
    static_cast< Derived *>(this)->extractFeatureImpl(object_id, sp);
  }

  template <typename Span> void extractFeature(int32_t object_id, Span output) {
    static_cast< Derived *>(this)->extractFeatureImpl(object_id, output);
  }
};

template <typename Derived> struct EdgeFeature {
  [[nodiscard]] size_t getFeatureDim() const {
    return static_cast<const Derived *>(this)->getFeatureDimImpl();
  }

  void extractFeature(int32_t source_id, int32_t target_id, TorchArr &output) {
    std::span<float> sp(output.data(), output.size());
    static_cast< Derived *>(this)->extractFeatureImpl(source_id, target_id, sp);
  }

  template <typename Span>
  void extractFeature(int32_t source_id, int32_t target_id, Span output) {
    static_cast< Derived *>(this)->extractFeatureImpl(source_id, target_id, output);
  }
};

// Compile-time FeatureExtractor using variadic templates
template <typename... Features> class FeatureExtractor {
  std::tuple<Features...> features;

  // Helper to compute total feature dimension at compile-time
  template <size_t... Is> size_t computeFeatureDim(std::index_sequence<Is...>) const {
    return (std::get<Is>(features).getFeatureDim() + ...);
  }

public:
  FeatureExtractor(Features... feats) : features(std::move(feats)...) {
  }

  size_t getFeatureDim() const {
    return computeFeatureDim(std::make_index_sequence<sizeof...(Features)>{});
  }

  void getFeatures(int object_id, TorchArr &output) {
    std::span<float> sp(output.data(), output.size());
    getFeatures(object_id, sp);
  }

  template <typename Span> void getFeatures(int object_id, Span output) {
    size_t offset = 0;
    std::apply(
        [&](const auto &...feats) {
          (..., (feats.extractFeature(object_id, output.subspan(offset, feats.getFeatureDim())),
                 offset += feats.getFeatureDim()));
        },
        features);
  }
};

template <typename... Features> class EdgeFeatureExtractor {
  std::tuple<Features...> features;

  // Helper to compute total feature dimension at compile-time
  template <size_t... Is> size_t computeFeatureDim(std::index_sequence<Is...>) const {
    return (std::get<Is>(features).getFeatureDim() + ...);
  }

public:
  EdgeFeatureExtractor(Features... feats) : features(std::move(feats)...) {
  }

  size_t getFeatureDim() const {
    return computeFeatureDim(std::make_index_sequence<sizeof...(Features)>{});
  }

  void getFeatures(int source_id, int target_id, TorchArr &output) const {
    std::span<float> sp(output.data(), output.size());
    getFeatures(source_id, target_id, sp);
  }

  template <typename Span> void getFeatures(int source_id, int target_id, Span output) {
    size_t offset = 0;
    std::apply(
        [&](const auto &...feats) {
          (..., (feats.extractFeature(source_id, target_id,
                                      output.subspan(offset, feats.getFeatureDim())),
                 offset += feats.getFeatureDim()));
        },
        features);
  }
};

template <typename Derived> struct StateFeature : Feature<Derived> {
  const SchedulerState &state;
  const NodeType node_type;

  StateFeature(const SchedulerState &state, const NodeType node_type)
      : state(state), node_type(node_type) {
  }
};

template <typename Derived> struct StateEdgeFeature : EdgeFeature<Derived> {
  const SchedulerState &state;
  const EdgeType edge_type;

  StateEdgeFeature(const SchedulerState &state, const EdgeType edge_type)
      : state(state), edge_type(edge_type) {
  }
};

template <typename Derived> struct IntFeature : StateFeature<Derived> {
  const size_t v;

  IntFeature(SchedulerState &state, size_t v, NodeType node_type)
      : StateFeature<Derived>(state, node_type), v(v) {
  }
};

template <typename Derived> struct IntEdgeFeature : StateEdgeFeature<Derived> {
  const size_t v;

  IntEdgeFeature(SchedulerState &state, size_t v, EdgeType edge_type)
      : StateEdgeFeature<Derived>(state, edge_type), v(v) {
  }
};

struct InDegreeTaskFeature : public StateFeature<InDegreeTaskFeature> {
  InDegreeTaskFeature(const SchedulerState &state)
      : StateFeature<InDegreeTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    output[0] = static_cast<f_t>(static_graph.get_in_degree(task_id));
  }
};

struct OutDegreeTaskFeature : public StateFeature<OutDegreeTaskFeature> {
  OutDegreeTaskFeature(const SchedulerState &state)
      : StateFeature<OutDegreeTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    output[0] = static_cast<f_t>(static_graph.get_out_degree(task_id));
  }
};

struct DepthTaskFeature : public StateFeature<DepthTaskFeature> {
  DepthTaskFeature(const SchedulerState &state)
      : StateFeature<DepthTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &tasks = state.get_graph().tasks;
    output[0] = static_cast<f_t>(tasks[task_id].depth);
  }
};

struct GPUDurationTaskFeature : public StateFeature<GPUDurationTaskFeature> {
  GPUDurationTaskFeature(const SchedulerState &state)
      : StateFeature<GPUDurationTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    output[0] = static_cast<f_t>(static_graph.get_mean_duration(task_id, DeviceType::GPU));
  }
};

struct InputOutputTaskFeature : public StateFeature<InputOutputTaskFeature> {
  InputOutputTaskFeature(const SchedulerState &state)
      : StateFeature<InputOutputTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    const auto &data = state.get_data();
    output[0] = static_cast<f_t>(data.get_total_size(static_graph.get_read(task_id)));
    output[1] = static_cast<f_t>(data.get_total_size(static_graph.get_write(task_id)));
  }
};

struct ReadDataLocationFeature : public StateFeature<ReadDataLocationFeature> {
  ReadDataLocationFeature(const SchedulerState &state)
      : StateFeature<ReadDataLocationFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return devices.size() - 1; // Exclude CPU
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    const auto &data_manager = state.get_data_manager();
    const auto &data = state.get_data();
    auto n_devices = state.get_devices().size();
    for (devid_t i = 1; i < n_devices; i++) {
      for (auto data_id : static_graph.get_read(task_id)) {
        if (data_manager.check_valid_mapped(static_cast<dataid_t>(data_id), i)) {
          output[i - 1] += data.get_size(data_id);
        }
      }
    }
  }
};

struct TaskMeanDurationFeature : public StateFeature<TaskMeanDurationFeature> {
  TaskMeanDurationFeature(const SchedulerState &state)
      : StateFeature<TaskMeanDurationFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    const auto &data_manager = state.get_data_manager();
    const auto &data = state.get_data();
    auto n_devices = state.get_devices().size();
    output[0] = static_graph.get_mean_duration(task_id, DeviceType::GPU);
  }
};

struct PrevReadSizeFeature : public StateFeature<PrevReadSizeFeature> {
  const int stride;
  const int frames;
  const bool add_current;
  PrevReadSizeFeature(const SchedulerState &state, int width, int length, bool add_current, int frames)
      : StateFeature<PrevReadSizeFeature>(state, NodeType::TASK), stride(width * length),
        add_current(add_current), frames(frames) {
  }

  size_t getFeatureDimImpl() const {
    return frames;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    const auto &data = state.get_data();
    const auto &task_runtime = state.get_task_runtime();
    auto n_devices = this->state.get_devices().size() - 1; // Exclude CPU
    if (!add_current) {
      task_id -= stride;
    }
    int i = 0;
    while (task_id >= 0 && i < frames) {
      output[i] =static_cast<f_t>(data.get_total_size(static_graph.get_read(task_id)));
      //output[i] = static_cast<f_t>(std::log(static_cast<double>(1 + output[i])));
      //output[i] = static_cast<f_t>(task_id);
      task_id -= stride;
      ++i;
    }
  }
};

struct PrevMappedSizeFeature : public StateFeature<PrevMappedSizeFeature> {
  const int stride;
  const int frames;
  const bool add_current;
  std::vector<f_t> history;
  PrevMappedSizeFeature(const SchedulerState &state, int width, int length, bool add_current, int frames)
      : StateFeature<PrevMappedSizeFeature>(state, NodeType::TASK), stride(width * length),
        add_current(add_current), frames(frames) {

        const auto &devices = this->state.get_devices();
        auto grid_size = width * length;
        history.resize(grid_size * (devices.size() - 1) * frames, 0.0);

  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return frames * (devices.size() - 1);
  }

  template<typename ID> void shift_history_of_task(ID task_id){
    const auto &devices = this->state.get_devices();
    auto n_devices = devices.size() - 1; // Exclude CPU
    auto grid_size = stride;
    auto offset = (task_id % grid_size) * n_devices * frames;

    std::cout << "Shifting history for task " << task_id << " at offset " << offset << std::endl;
    std:: cout << "History before shift: ";
    for(int f = 0; f < frames; f++){
      for(int d = 0; d < n_devices; d++){
        std::cout << history[offset + f * n_devices + d] << " ";
      }
    }
    std::cout << std::endl;


    for(int f = frames - 1; f > 0; f--){
      for(int d = 0; d < n_devices; d++){
        history[offset + f * n_devices + d] = history[offset + (f - 1) * n_devices + d];
      }
    }
    for(int d = 0; d < n_devices; d++){
      history[offset + d] = 0.0;
    }

    std:: cout << "History after shift: ";
    for(int f = 0; f < frames; f++){
      for(int d = 0; d < n_devices; d++){
        std::cout << history[offset + f * n_devices + d] << " ";
      }
    }
    std::cout << std::endl;
    
  }

  template<typename ID> void update_history_of_task(ID task_id){
    const auto &static_graph = state.get_tasks();
    const auto &data_manager = state.get_data_manager();
    const auto &data = state.get_data();
    const auto &task_runtime = state.get_task_runtime();
    const auto &devices = this->state.get_devices();
    auto n_devices = devices.size() - 1; // Exclude CPU
    auto grid_size = stride;
    auto offset = (task_id % grid_size) * n_devices * frames;

    for (auto data_id : static_graph.get_read(task_id)) {
      auto data_size = static_cast<f_t>(data.get_size(data_id));
      for (devid_t i = 1; i < devices.size(); i++) {
        if (data_manager.check_valid_mapped(static_cast<dataid_t>(data_id), i)) {
          history[offset + i - 1] += data_size;
        }
      }
    }

    std:: cout << "History after update: ";
    for(int f = 0; f < frames; f++){
      for(int d = 0; d < n_devices; d++){
        std::cout << history[offset + f * n_devices + d] << " ";
      }
    }
    std::cout << std::endl;

  }

  template<typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) {
    shift_history_of_task(task_id);
    update_history_of_task(task_id);
    const auto &devices = this->state.get_devices();
    auto n_devices = devices.size() - 1; // Exclude CPU
    auto grid_size = stride;
    auto offset = (task_id % grid_size) * n_devices * frames;
    int i = 0;
    while (i < frames) {
      for(int d = 0; d < n_devices; d++){
        output[i * n_devices + d] = history[offset + i * n_devices + d];
      }
      ++i;
    }
    std::cout << "Extracted feature for task " << task_id << ": ";
    for(int f = 0; f < frames; f++){
      for(int d = 0; d < n_devices; d++){
        std::cout << output[f * n_devices + d] << " ";
      }
    }
    std::cout << std::endl;
  }

};

struct PrevMappedDevice : public StateFeature<PrevMappedDevice> {
  const int stride;
  const int frames;
  const bool add_current;
  PrevMappedDevice(const SchedulerState &state, int width, int length, bool add_current, int frames)
      : StateFeature<PrevMappedDevice>(state, NodeType::TASK), stride(width * length),
        add_current(add_current), frames(frames) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return frames * (devices.size() - 1);
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    const auto &task_runtime = state.get_task_runtime();
    auto n_devices = this->state.get_devices().size() - 1; // Exclude CPU
    if (!add_current) {
      task_id -= stride;
    }
    int i = 0;
    while (task_id >= 0 && i < frames) {
      devid_t mapped_device = task_runtime.get_compute_task_mapped_device(task_id);
      for (devid_t d = 1; d <= n_devices; d++) {
        output[i * n_devices + d - 1] = static_cast<f_t>(d == mapped_device);
      }
      task_id -= stride;
      ++i;
    }
  }
};

struct EmptyTaskFeature : public IntFeature<EmptyTaskFeature> {
  size_t dimension;
  EmptyTaskFeature(SchedulerState &state, size_t dimension)
      : IntFeature<EmptyTaskFeature>(state, dimension, NodeType::TASK), dimension(dimension) {
  }

  size_t getFeatureDimImpl() const {
    return dimension;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
  }
};

struct OneHotMappedDeviceTaskFeature : public StateFeature<OneHotMappedDeviceTaskFeature> {
  OneHotMappedDeviceTaskFeature(const SchedulerState &state)
      : StateFeature<OneHotMappedDeviceTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return devices.size()-1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task_runtime = state.get_task_runtime();
    one_hot(task_runtime.get_compute_task_mapped_device(task_id)-1, output);
  }
};

struct TaskStateFeature : public StateFeature<TaskStateFeature> {
  TaskStateFeature(const SchedulerState &state)
      : StateFeature<TaskStateFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 4;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &s = this->state;
    const auto &task_runtime = s.get_task_runtime();
    const auto state = task_runtime.get_compute_task_state(task_id);
    output[0] = static_cast<f_t>(state >= TaskState::MAPPED);
    output[1] = static_cast<f_t>(state >= TaskState::RESERVED);
    output[2] = static_cast<f_t>(state >= TaskState::LAUNCHED);
    output[3] = static_cast<f_t>(state >= TaskState::COMPLETED);
  }
};

struct CandidateVector : public StateFeature<CandidateVector> {
  CandidateVector(const SchedulerState &state)
      : StateFeature<CandidateVector>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return devices.size() * 4 + 1; // mapped_queue, location, x_center, y_center
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    const auto &task_runtime = state.get_task_runtime();
    const auto &read = static_graph.get_read(task_id);
    const auto &devices = state.get_devices();
    const int n_devices = static_cast<int>(devices.size());
    const auto &data = state.get_data();
    const auto &data_manager = state.get_data_manager();

    f_t input_size = static_cast<f_t>(data.get_total_size(read));

    int64_t sum = 0.0;
    for (int i = 0; i < n_devices; i++) {
      auto mapped_time = static_cast<int64_t>(state.costs.get_mapped_time(i));
      output[i] = static_cast<f_t>(mapped_time);
      sum += mapped_time;
    }

    if (sum > 0.0) {
      for (int i = 0; i < n_devices; i++) {
        output[i] = static_cast<f_t>(output[i] / sum);
      }
    }

    for (int i = 0; i < read.size(); i++) {
      auto data_id = read[i];
      auto data_size = static_cast<f_t>(data.get_size(data_id));
      const f_t x_pos = static_cast<f_t>(data.get_x_pos(data_id));
      const f_t y_pos = static_cast<f_t>(data.get_y_pos(data_id));

      for (int j = 0; j < n_devices; ++j) {
        const bool is_mapped = data_manager.check_valid_mapped(data_id, j);
        const f_t size = is_mapped * data_size;
        const int offset = n_devices + j * 3;
        output[offset] += size;
        output[offset + 1] += (x_pos * size) / input_size;
        output[offset + 2] += (y_pos * size) / input_size;
      }
    }
    output[n_devices * 4] = input_size;
  }
};

struct TaskDataMappedSize : public StateFeature<TaskDataMappedSize> {
  TaskDataMappedSize(const SchedulerState &state)
      : StateFeature<TaskDataMappedSize>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task_runtime = state.get_task_runtime();
    const auto &static_graph = state.get_tasks();
    const auto read = static_graph.get_read(task_id);
    const auto &data = state.get_data();
    const auto &devices = state.get_devices();
    const auto &data_manager = state.get_data_manager();

    for (int32_t i = 0; i < read.size(); i++) {
      auto data_id = read[i];
      const auto data_size = static_cast<double>(data.get_size(data_id));
      for (int32_t j = 0; j < devices.size(); j++) {
        output[j] += data_size * static_cast<f_t>(data_manager.check_valid_mapped(data_id, j));
      }
    }
  }
};


struct TaskCoordinates : public StateFeature<TaskCoordinates> {
  TaskCoordinates(const SchedulerState &state)
      : StateFeature<TaskCoordinates>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 2; // average x, average y
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID task_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    const auto &data = state.get_data();
    const auto read = static_graph.get_read(task_id);

    if (read.empty()) {
      output[0] = static_cast<f_t>(0.0);
      output[1] = static_cast<f_t>(0.0);
      return;
    }

    double sum_x = 0.0;
    double sum_y = 0.0;
    for (int i = 0; i < read.size(); ++i) {
      auto data_id = read[i];
      sum_x += static_cast<double>(data.get_x_pos(data_id));
      sum_y += static_cast<double>(data.get_y_pos(data_id));
    }

    output[0] = static_cast<f_t>(sum_x / read.size());
    output[1] = static_cast<f_t>(sum_y / read.size());
  }
};

struct TaskDataMappedCoordinates : public StateFeature<TaskDataMappedCoordinates> {
  TaskDataMappedCoordinates(const SchedulerState &state)
      : StateFeature<TaskDataMappedCoordinates>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return 2 * devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task_runtime = state.get_task_runtime();
    const auto &static_graph = state.get_tasks();
    const auto read = static_graph.get_read(task_id);
    const auto &data = state.get_data();
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_devices();

    const int n_devices = static_cast<int>(devices.size());

    for (int i = 0; i < read.size(); i++) {
      auto data_id = read[i];
      auto data_size = static_cast<double>(data.get_size(data_id));

      const f_t x_pos = static_cast<f_t>(data.get_x_pos(data_id));
      const f_t y_pos = static_cast<f_t>(data.get_y_pos(data_id));

      for (int j = 0; j < n_devices; j++) {
        const bool is_mapped = data_manager.check_valid_mapped(data_id, j);
        const f_t mapped_v = static_cast<f_t>(is_mapped);
        output[j] += x_pos * mapped_v;
        output[j + n_devices] += y_pos * mapped_v;
      }

      output[2 * n_devices + 1] += data_size * x_pos;
      output[2 * n_devices + 2] += data_size * y_pos;
    }
  }
};

struct TaskDeviceMappedTime : public StateFeature<TaskDeviceMappedTime> {
  TaskDeviceMappedTime(const SchedulerState &state)
      : StateFeature<TaskDeviceMappedTime>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {

    const auto &devices = state.get_devices();

    for (int i = 0; i < devices.size(); i++) {
      auto mapped_time = static_cast<double>(state.costs.get_mapped_time(i));
      output[i] = static_cast<f_t>(mapped_time);
    }

    // Normalize across devices
    double sum = 0.0;
    for (int i = 0; i < devices.size(); i++) {
      sum += output[i];
    }
    if (sum > 0.0) {
      for (int i = 0; i < devices.size(); i++) {
        output[i] /= sum;
      }
    }
  }
};

struct EmptyDataFeature : public IntFeature<EmptyDataFeature> {
  size_t dimension;
  EmptyDataFeature(SchedulerState &state, size_t dimension)
      : IntFeature<EmptyDataFeature>(state, dimension, NodeType::DATA_BLOCK), dimension(dimension) {
  }

  size_t getFeatureDimImpl() const {
    return dimension;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
  }
};

struct DataMappedLocations : public StateFeature<DataMappedLocations> {
  DataMappedLocations(const SchedulerState &state)
      : StateFeature<DataMappedLocations>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_devices();
    for (int i = 0; i < devices.size(); i++) {
      output[i] = static_cast<f_t>(data_manager.check_valid_mapped(data_id, i));
    }
  }
};

struct DataReservedLocations : public StateFeature<DataReservedLocations> {
  DataReservedLocations(const SchedulerState &state)
      : StateFeature<DataReservedLocations>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_devices();
    for (int i = 0; i < devices.size(); i++) {
      output[i] = static_cast<f_t>(data_manager.check_valid_reserved(data_id, i));
    }
  }
};

struct DataLaunchedLocations : public StateFeature<DataLaunchedLocations> {
  DataLaunchedLocations(const SchedulerState &state)
      : StateFeature<DataLaunchedLocations>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_devices();
    for (int i = 0; i < devices.size(); i++) {
      output[i] = static_cast<f_t>(data_manager.check_valid_launched(data_id, i));
    }
  }
};

struct DataSizeFeature : public StateFeature<DataSizeFeature> {
  DataSizeFeature(const SchedulerState &state)
      : StateFeature<DataSizeFeature>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data = state.get_data();
    output[0] = static_cast<f_t>(data.get_size(data_id));
  }
};

struct EmptyDeviceFeature : public IntFeature<EmptyDeviceFeature> {
  size_t dimension;
  EmptyDeviceFeature(SchedulerState &state, size_t dimension)
      : IntFeature<EmptyDeviceFeature>(state, dimension, NodeType::DEVICE), dimension(dimension) {
  }

  size_t getFeatureDimImpl() const {
    return dimension;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
  }
};

struct DeviceMemoryFeature : public StateFeature<DeviceMemoryFeature> {
  DeviceMemoryFeature(const SchedulerState &state)
      : StateFeature<DeviceMemoryFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 3;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    const auto &device_manager = state.get_device_manager();
    auto mapped_mem = static_cast<double>(device_manager.get_mem<TaskState::MAPPED>(device_id));
    auto reserved_mem = static_cast<double>(device_manager.get_mem<TaskState::RESERVED>(device_id));
    auto launched_mem = static_cast<double>(device_manager.get_mem<TaskState::LAUNCHED>(device_id));
    output[0] = mapped_mem;
    output[1] = reserved_mem;
    output[2] = launched_mem;
  }
};

struct DeviceTimeFeature : public StateFeature<DeviceTimeFeature> {
  DeviceTimeFeature(const SchedulerState &state)
      : StateFeature<DeviceTimeFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 3;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    const auto &s = this->state;
    const auto &device_manager = s.get_device_manager();
    auto mapped_time = static_cast<double>(s.costs.get_mapped_time(device_id));
    auto reserved_time = static_cast<double>(s.costs.get_reserved_time(device_id));
    auto launched_time = static_cast<double>(s.costs.get_launched_time(device_id));

    output[0] = mapped_time;
    output[1] = reserved_time;
    output[2] = launched_time;
  }
};

struct DeviceIDFeature : public StateFeature<DeviceIDFeature> {
  DeviceIDFeature(const SchedulerState &state)
      : StateFeature<DeviceIDFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    one_hot(device_id, output);
  }
};

struct EmptyTaskTaskFeature : public IntEdgeFeature<EmptyTaskTaskFeature> {
  size_t dimension;
  EmptyTaskTaskFeature(SchedulerState &state, size_t dimension)
      : IntEdgeFeature<EmptyTaskTaskFeature>(state, dimension, EdgeType::TASK_TASK),
        dimension(dimension) {
  }

  size_t getFeatureDimImpl() const {
    return dimension;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
  }
};

struct TaskTaskDefaultEdgeFeature : public StateEdgeFeature<TaskTaskDefaultEdgeFeature> {
  TaskTaskDefaultEdgeFeature(const SchedulerState &state)
      : StateEdgeFeature<TaskTaskDefaultEdgeFeature>(state, EdgeType::TASK_TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    output[0] = 1.0;
  }
};

struct EmptyTaskDataFeature : public IntEdgeFeature<EmptyTaskDataFeature> {
  size_t dimension;
  EmptyTaskDataFeature(SchedulerState &state, size_t dimension)
      : IntEdgeFeature<EmptyTaskDataFeature>(state, dimension, EdgeType::TASK_DATA),
        dimension(dimension) {
  }

  size_t getFeatureDimImpl() const {
    return dimension;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
  }
};

struct TaskDataUsageFeature : public StateEdgeFeature<TaskDataUsageFeature> {
  TaskDataUsageFeature(const SchedulerState &state)
      : StateEdgeFeature<TaskDataUsageFeature>(state, EdgeType::TASK_DATA) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    const auto &static_graph = state.get_tasks();
    const auto read = static_graph.get_read(source_id);
    const auto write = static_graph.get_write(source_id);

    bool is_read_access = std::find(read.begin(), read.end(), target_id) != read.end();
    bool is_write_access = std::find(write.begin(), write.end(), target_id) != write.end();

    output[0] = static_cast<f_t>(is_read_access);
    output[1] = static_cast<f_t>(is_write_access);
  }
};

struct TaskDataDefaultEdgeFeature : public StateEdgeFeature<TaskDataDefaultEdgeFeature> {
  TaskDataDefaultEdgeFeature(const SchedulerState &state)
      : StateEdgeFeature<TaskDataDefaultEdgeFeature>(state, EdgeType::TASK_DATA) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    output[0] = 1.0;
  }
};

struct TaskDeviceDefaultEdgeFeature : public StateEdgeFeature<TaskDeviceDefaultEdgeFeature> {
  TaskDeviceDefaultEdgeFeature(const SchedulerState &state)
      : StateEdgeFeature<TaskDeviceDefaultEdgeFeature>(state, EdgeType::TASK_DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    output[0] = 1.0;
  }
};

struct DataDeviceDefaultEdgeFeature : public StateEdgeFeature<DataDeviceDefaultEdgeFeature> {
  DataDeviceDefaultEdgeFeature(const SchedulerState &state)
      : StateEdgeFeature<DataDeviceDefaultEdgeFeature>(state, EdgeType::DATA_DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    output[0] = 1.0;
  }
};
