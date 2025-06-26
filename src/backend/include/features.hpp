#pragma once
#include "devices.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <span>
#include <sys/types.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

using op_t = int32_t;
using f_t = float_t;
using namespace nb::literals;

taskid_t max(taskid_t a, taskid_t b) {
  return a > b ? a : b;
}

taskid_t min(taskid_t a, taskid_t b) {
  return a < b ? a : b;
}

enum class NodeType {
  ANY = -1,
  TASK = 0,
  DATA_BLOCK = 1,
  DEVICE = 2
};

enum class EdgeType {
  ANY = -1,
  TASK_TASK = 0,
  TASK_DATA = 1,
  TASK_DEVICE = 2,
  DATA_DEVICE = 3
};

struct GraphSpec {
public:
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

  taskid_t max_tasks = 0;
  taskid_t max_data = 0;
  taskid_t max_devices = 5;

  // taskid_t std_duration = 0;
  // taskid_t std_memcost = 0;
  // taskid_t mean_duration = 0;
  // taskid_t mean_memcost = 0;

  void compute_max_degree(const SchedulerState &state) {
    const auto &tasks = state.get_task_manager().get_tasks();
    for (const auto &task : tasks.get_compute_tasks()) {
      max_in_degree = max(max_in_degree, task.get_dependencies().size());
      max_out_degree = max(max_out_degree, task.get_dependents().size());
      max_data_in_degree = max(max_data_in_degree, task.get_data_dependencies().size());
      max_data_out_degree = max(max_data_out_degree, task.get_data_dependents().size());
      max_data_usage = max(max_data_usage, task.get_unique().size());
    }
  }

  void set_max_task_degree(taskid_t degree) {
    max_in_degree = degree;
    max_out_degree = degree;
  }

  void set_max_data_degree(dataid_t degree) {
    max_data_in_degree = degree;
    max_data_out_degree = degree;
    max_data_usage = degree;
  }

  void set_max_devices(devid_t devices) {
    max_devices = devices;
  }
  void set_max_tasks(taskid_t tasks) {
    max_tasks = tasks;
  }

  void set_max_data(dataid_t data) {
    max_data = data;
  }

  void compute_max_tasks() {
    max_tasks = max_candidates * (max_in_degree + max_out_degree);
  }

  void compute_max_data() {
    max_data = max_tasks * max_data_usage;
  }

  void finalize(const SchedulerState &state, bool use_graph = false) {
    if (use_graph) {
      compute_max_degree(state);
    }
    compute_max_tasks();
    compute_max_data();
  }
};

using TaskSet = std::unordered_set<taskid_t>;

class GraphExtractor {
protected:
  std::reference_wrapper<const SchedulerState> state;
  std::vector<op_t> source_list;
  std::vector<op_t> target_list;
  TaskSet visited;
  std::unordered_set<dataid_t> data_visited;
  std::unordered_set<taskid_t> local_visited;
  std::unordered_map<taskid_t, taskid_t> task_index_map;
  std::unordered_map<dataid_t, dataid_t> data_index_map;
  std::unordered_map<devid_t, devid_t> device_index_map;

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
    const auto &devices = device_manager.get_devices();
    const auto &compute_task = s.get_task_manager().get_tasks().get_compute_task(task_id);

    for (int i = 0; i < devices.size(); i++) {
      const auto &device_type = devices.get_type(i);
      bool is_supported = compute_task.is_supported_architecture(device_type);
      v(i) = is_supported ? 1 : 0;
    }
  }

  void _get_k_hop_task_dependents(TaskSet &visited, taskid_t task_id, int k, size_t max_tasks) {

    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

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

        const auto &task = tasks.get_compute_task(current_task_id);
        for (const auto &dep_id : task.get_dependents()) {
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
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

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

        const auto &task = tasks.get_compute_task(current_task_id);
        for (const auto &dep_id : task.get_dependencies()) {
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

    const auto &tasks = this->state.get().get_task_manager().get_tasks();
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

        const auto &task = tasks.get_compute_task(current_task_id);
        for (const auto &dep_id : task.get_dependencies()) {
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

        for (const auto &dep_id : task.get_dependents()) {
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

    task_index_map.clear();
    task_index_map.reserve(sources.size());

    std::span<int64_t> sources_span(sources.data(), sources.size());

    for (int64_t i = 0; i < sources_span.size(); i++) {
      task_index_map[static_cast<taskid_t>(sources_span[i])] = i;
    }

    const auto &tasks = state.get().get_task_manager().get_tasks();
    std::size_t edge_count = 0;

    for (int64_t source_idx = 0; source_idx < sources_span.size(); source_idx++) {
      const auto source_id = sources_span[source_idx];
      const auto &source_task = tasks.get_compute_task(source_id);
      const auto &dependencies = source_task.get_dependencies();

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

    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    std::size_t edge_count = 0;

    for (int64_t source_idx = 0; source_idx < sources_span.size(); source_idx++) {
      const auto source_id = sources_span[source_idx];
      const auto &source_task = tasks.get_compute_task(source_id);
      const auto &dependents = source_task.get_dependents();

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

  size_t get_unique_filtered_data(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &output) {
    data_visited.clear();
    data_visited.reserve(400);

    const auto max_data = output.size();
    auto v = output.view();
    static bool has_warned = false;
    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    const auto &s = this->state.get();
    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    for (const auto &task_id_64_bit : task_ids_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      const auto &task = tasks.get_compute_task(task_id);
      const auto &read = task.get_read();

      // std::cout << "Task ID: " << task_id << " Read size: " << read.size() << std::endl;

      for (int i = 0; i < read.size(); i++) {

        auto data_id = read[i];
        taskid_t recent_writer_id = task.get_recent_writer(i);
        if (recent_writer_id != -1) {
          // We are not the first writer.
          // Check to see if the most recent writer has been mapped
          const bool is_mapped = s.is_mapped(recent_writer_id);
          // std::cout << "Data ID: " << data_id << " Recent Writer ID: " << recent_writer_id
          //           << " Mapped: " << is_mapped << std::endl;
          if (is_mapped) {
            data_visited.insert(data_id);
          }
        } else {
          data_visited.insert(data_id);
          // std::cout << "Data ID: " << data_id << " No recent writer" << std::endl;
        }
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

  size_t get_unique_data(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &output) {
    data_visited.clear();
    data_visited.reserve(400);

    const auto max_data = output.size();
    auto v = output.view();
    static bool has_warned = false;
    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    const auto &s = this->state.get();
    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    for (const auto &task_id_64_bit : task_ids_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      const auto &task = tasks.get_compute_task(task_id);
      const auto &unique = task.get_unique();

      // std::cout << "Task ID: " << task_id << " Read size: " << read.size() << std::endl;

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
    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    for (const auto &task_id_64_bit : task_ids_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      const auto &task = tasks.get_compute_task(task_id);
      const auto &unique = task.get_read();

      // std::cout << "Task ID: " << task_id << " Read size: " << read.size() << std::endl;

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

  size_t get_write_data(TorchInt64Arr1D &task_ids, TorchInt64Arr1D &output) {
    data_visited.clear();
    data_visited.reserve(400);

    const auto max_data = output.size();
    auto v = output.view();
    static bool has_warned = false;
    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    const auto &s = this->state.get();
    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    for (const auto &task_id_64_bit : task_ids_span) {
      taskid_t task_id = static_cast<taskid_t>(task_id_64_bit);
      const auto &task = tasks.get_compute_task(task_id);
      const auto &unique = task.get_read();

      // std::cout << "Task ID: " << task_id << " Read size: " << read.size() << std::endl;

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

    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {
      // if (data_index_map.find(static_cast<dataid_t>(data_ids_span[i])) == data_index_map.end()) {
      //   data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
      // }
      data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
    }

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);
      const auto &task = tasks.get_compute_task(task_id);
      for (auto data_id : task.get_unique()) {
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

    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {
      // if (data_index_map.find(static_cast<dataid_t>(data_ids_span[i])) == data_index_map.end()) {
      //   data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
      // }
      data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
    }

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);
      const auto &task = tasks.get_compute_task(task_id);
      for (auto data_id : task.get_read()) {
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

    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {
      // if (data_index_map.find(static_cast<dataid_t>(data_ids_span[i])) == data_index_map.end()) {
      //   data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
      // }
      data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
    }

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);
      const auto &task = tasks.get_compute_task(task_id);
      for (auto data_id : task.get_read()) {
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

    const auto &task_manager = state.get().get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    // // The candidate task id is always the first task id in the list
    // int64_t candidate_task_id = task_ids_span[0];
    // std::cout << "Candidate Task ID: " << candidate_task_id << std::endl;

    std::size_t edge_count = 0;
    for (int64_t i = 0; i < data_ids_span.size(); i++) {
      data_index_map[static_cast<dataid_t>(data_ids_span[i])] = i;
    }

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);
      const auto &task = tasks.get_compute_task(task_id);
      const auto &read = task.get_read();

      for (int j = 0; j < read.size(); j++) {
        auto data_id = read[j];
        taskid_t recent_writer_id = task.get_recent_writer(j);

        // std::cout << "Task ID: " << task_id << " Data ID: " << data_id
        //           << " Recent Writer ID: " << recent_writer_id << std::endl;

        if (recent_writer_id != -1) {
          // We are not the first writer, check to see if it is mapped w.r.t us
          const bool is_mapped = state.get().is_mapped(recent_writer_id);
          // std::cout << "Recent Writer ID: " << recent_writer_id << " Mapped: " << is_mapped
          //           << std::endl;
          if (!is_mapped) {
            continue;
          }
          // TODO: When we merge with retire, we should ignore the data id if it has no valid
          // location even if we are the first writer (it can be uninitialized)
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

  size_t get_task_device_edges_mapped(TorchInt64Arr1D &task_ids, TorchInt64Arr2D &output,
                                      TorchInt64Arr2D &global_output) {
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    auto v = output.view();
    auto gv = global_output.view();
    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());
    static bool has_warned = false;

    const auto max_edges = output.shape(1);

    const auto &device_manager = state.get().get_device_manager();
    const std::size_t device_count = state.get().get_device_manager().get_devices().size();

    const auto &task_manager = state.get().get_task_manager();

    std::size_t edge_count = 0;

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      const auto &task_id = static_cast<taskid_t>(task_ids_span[i]);

      devid_t mapped_device = task_manager.get_mapping(task_id);

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

  size_t get_task_device_edges(TorchInt64Arr1D &task_ids, TorchInt64Arr2D &output,
                               TorchInt64Arr2D &global_output) {
    if (output.shape(0) != 2) {
      throw std::runtime_error("Edge output shape must be 2 x N");
    }

    auto v = output.view();
    auto gv = global_output.view();
    static bool has_warned = false;
    std::span<int64_t> task_ids_span(task_ids.data(), task_ids.size());

    // All tasks use all devices
    const auto max_edges = output.shape(1);

    const auto &device_manager = state.get().get_device_manager();
    const std::size_t device_count = device_manager.get_devices().size();
    const auto &tasks = state.get().get_task_manager().get_tasks();

    std::size_t edge_count = 0;

    for (int64_t i = 0; i < task_ids_span.size(); i++) {
      for (int64_t j = 0; j < device_count; j++) {
        v(0, edge_count) = static_cast<int64_t>(i);
        v(1, edge_count) = static_cast<int64_t>(j);
        gv(0, edge_count) = static_cast<int64_t>(task_ids_span[i]);
        gv(1, edge_count) = static_cast<int64_t>(j);
        edge_count++;
        if (edge_count >= max_edges) {
          if (!has_warned) {
            spdlog::warn("TaskDevice edge count exceeded max edges: {}", edge_count);
            has_warned = true;
          }
          break;
        }
      }
      if (edge_count >= max_edges) {
        if (!has_warned) {
          spdlog::warn("TaskDevice edge count exceeded max edges: {}", edge_count);
          has_warned = true;
        }
        break;
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
    std::unordered_set<int64_t> filtered_ids_set(filtered_ids_span.begin(),
                                                 filtered_ids_span.end());

    auto v = output.view();
    auto gv = global_output.view();
    static bool has_warned = false;
    std::span<int64_t> data_ids_span(data_ids.data(), data_ids.size());

    const auto max_edges = output.shape(1);

    const auto &device_manager = state.get().get_device_manager();
    const auto &devices = device_manager.get_devices();
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

    const auto &device_manager = state.get().get_device_manager();
    const auto &devices = device_manager.get_devices();
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

  void extractFeature(int32_t object_id, TorchArr &output) const {
    std::span<float> sp(output.data(), output.size());
    static_cast<const Derived *>(this)->extractFeatureImpl(object_id, sp);
  }

  template <typename Span> void extractFeature(int32_t object_id, Span output) const {
    static_cast<const Derived *>(this)->extractFeatureImpl(object_id, output);
  }
};

template <typename Derived> struct EdgeFeature {
  [[nodiscard]] size_t getFeatureDim() const {
    return static_cast<const Derived *>(this)->getFeatureDimImpl();
  }

  void extractFeature(int32_t source_id, int32_t target_id, TorchArr &output) const {
    std::span<float> sp(output.data(), output.size());
    static_cast<const Derived *>(this)->extractFeatureImpl(source_id, target_id, sp);
  }

  template <typename Span>
  void extractFeature(int32_t source_id, int32_t target_id, Span output) const {
    static_cast<const Derived *>(this)->extractFeatureImpl(source_id, target_id, output);
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

  void getFeatures(int object_id, TorchArr &output) const {
    std::span<float> sp(output.data(), output.size());
    getFeatures(object_id, sp);
  }

  template <typename Span> void getFeatures(int object_id, Span output) const {
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

  template <typename Span> void getFeatures(int source_id, int target_id, Span output) const {
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

  static f_t get_in_degree(const ComputeTask &task) {
    return static_cast<f_t>(task.get_dependencies().size());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    double in_degree = static_cast<double>(get_in_degree(task));
    double mean_in_degree = state.stats.indegree_stats.mean;
    double std_in_degree = state.stats.indegree_stats.stddev;
    in_degree = guarded_divide(in_degree - mean_in_degree, std_in_degree);
  }
};

struct OutDegreeTaskFeature : public StateFeature<OutDegreeTaskFeature> {
  OutDegreeTaskFeature(const SchedulerState &state)
      : StateFeature<OutDegreeTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  static f_t get_out_degree(const ComputeTask &task) {
    return static_cast<f_t>(task.get_dependents().size());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    double out_degree = static_cast<double>(get_out_degree(task));
    double mean_out_degree = state.stats.outdegree_stats.mean;
    double std_out_degree = state.stats.outdegree_stats.stddev;
    out_degree = guarded_divide(out_degree - mean_out_degree, std_out_degree);
  }
};

struct DepthTaskFeature : public StateFeature<DepthTaskFeature> {
  DepthTaskFeature(const SchedulerState &state)
      : StateFeature<DepthTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  static f_t get_depth(const ComputeTask &task) {
    return static_cast<f_t>(task.get_depth());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    double depth = static_cast<double>(get_depth(task));
    // double mean_depth = state.stats.depth_stats.mean;
    // double std_depth = state.stats.depth_stats.stddev;
    // depth = guarded_divide(depth - mean_depth, std_depth);
    output[0] = depth;
  }
};

struct ReadDataLocationFeature : public StateFeature<ReadDataLocationFeature> {
  ReadDataLocationFeature(const SchedulerState &state)
      : StateFeature<ReadDataLocationFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size() - 1; // Exclude CPU
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const ComputeTask &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    const auto &data_manager = state.get_data_manager();
    const auto &data = data_manager.get_data();
    const auto &devices = state.get_device_manager().get_devices();
    double max_size = static_cast<double>(state.stats.total_read_stats.max);
    double total_read = static_cast<double>(data.get_total_size(task.get_read()));
    for (devid_t i = 1; i < devices.size(); i++) {
      // output[i - 1] = -guarded_divide(total_read, max_size);
      for (auto data_id : task.get_read()) {
        if (data_manager.check_valid_mapped(static_cast<dataid_t>(data_id), i)) {
          auto data_size = static_cast<double>(data.get_size(data_id));
          output[i - 1] += guarded_divide(data_size, max_size);
        }
      }
    }
  }
};

struct PrevReadSizeFeature : public StateFeature<PrevReadSizeFeature> {
  PrevReadSizeFeature(const SchedulerState &state)
      : StateFeature<PrevReadSizeFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 3;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &data = data_manager.get_data();
    double max_size = static_cast<double>(state.stats.total_read_stats.max);

    task_id -= 16;
    int i = 0;
    while (task_id >= 0 && i < 3) {
      const ComputeTask &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
      double total_read = static_cast<double>(data.get_total_size(task.get_read()));
      output[i] = guarded_divide(total_read, max_size);
      task_id -= 16;
      i++;
    }
  }
};

struct TagCandidateFeature : public StateFeature<TagCandidateFeature> {
  TagCandidateFeature(const SchedulerState &state)
      : StateFeature<TagCandidateFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1; // Exclude CPU
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    if (task_id == state.current_candidate) {
      output[0] = 1.0; // Current candidate
    } else {
      output[0] = 0.0; // Not current candidate
    }
  }
};

struct TagTaskFeature : public StateFeature<TagTaskFeature> {
  // Generally task tags are categorical, but currently this is a standin for location within the
  // subgraph
  TagTaskFeature(const SchedulerState &state)
      : StateFeature<TagTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    double tag = static_cast<double>(task.get_tag());
    double mean_tag = state.stats.tag_stats.mean;
    double std_tag = state.stats.tag_stats.stddev;
    tag = guarded_divide(tag - mean_tag, std_tag);
    output[0] = tag;
  }
};

struct DurationTaskFeature : public StateFeature<DurationTaskFeature> {
  DurationTaskFeature(const SchedulerState &state)
      : StateFeature<DurationTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  static f_t get_duration(const ComputeTask &task, DeviceType arch) {
    return static_cast<f_t>(task.get_variant(arch).get_observed_time());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    output[0] = log(get_duration(task, DeviceType::CPU) + 1);
    output[1] = log(get_duration(task, DeviceType::GPU) + 1);
  }
};

struct StandardizedGPUDurationTaskFeature
    : public StateFeature<StandardizedGPUDurationTaskFeature> {
  StandardizedGPUDurationTaskFeature(const SchedulerState &state)
      : StateFeature<StandardizedGPUDurationTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  static f_t get_duration(const ComputeTask &task, DeviceType arch) {
    return static_cast<f_t>(task.get_variant(arch).get_observed_time());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    double mean_duration = state.stats.duration_stats.mean;
    double std_duration = state.stats.duration_stats.stddev;
    // output[0] = guarded_divide(get_duration(task, DeviceType::GPU) - mean_duration,
    // std_duration);
    output[0] = guarded_divide(get_duration(task, DeviceType::GPU), mean_duration);
  }
};

struct StandardizedInputOutputTaskFeature
    : public StateFeature<StandardizedInputOutputTaskFeature> {
  StandardizedInputOutputTaskFeature(const SchedulerState &state)
      : StateFeature<StandardizedInputOutputTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  static f_t get_input_size(const ComputeTask &task, const Data &data) {
    const auto &read = task.get_read();
    if (read.empty()) {
      return 0;
    }
    return static_cast<f_t>(data.get_total_size(read));
  }

  static f_t get_output_size(const ComputeTask &task, const Data &data) {
    const auto &write = task.get_write();
    if (write.empty()) {
      return 0;
    }

    return static_cast<f_t>(data.get_total_size(write));
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    const auto &data = state.get_data_manager().get_data();
    double mean_io = state.stats.io_stats.mean;
    double std_io = state.stats.io_stats.stddev;
    double mean_block_size = state.stats.block_size_stats.mean;

    auto input_size = get_input_size(task, data);
    auto output_size = get_output_size(task, data);

    // Put it in terms of block size scale (only matters for unstandardized case)
    // input_size = guarded_divide(input_size, mean_block_size);
    // output_size = guarded_divide(output_size, mean_block_size);

    // mean_io = guarded_divide(mean_io, mean_block_size);
    // std_io = guarded_divide(std_io, mean_block_size);

    // output[0] = guarded_divide(input_size - mean_io, std_io);
    // output[1] = guarded_divide(output_size - mean_io, std_io);

    output[0] = guarded_divide(input_size, mean_io);
    output[1] = guarded_divide(output_size, mean_io);
  }
};

struct MemoryTaskFeature : public StateFeature<MemoryTaskFeature> {
  MemoryTaskFeature(const SchedulerState &state)
      : StateFeature<MemoryTaskFeature>(state, NodeType::TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  static f_t get_task_memcost(const ComputeTask &task, DeviceType arch) {
    return static_cast<f_t>(task.get_variant(arch).get_mem());
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task = state.get_task_manager().get_tasks().get_compute_task(task_id);
    output[0] = log(get_task_memcost(task, DeviceType::CPU));
    output[1] = log(get_task_memcost(task, DeviceType::GPU));
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
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID task_id, Span output) const {
    const auto &task_manager = state.get_task_manager();
    one_hot(task_manager.state.get_mapping(task_id), output);
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
    const auto &task_manager = s.get_task_manager();
    const auto state = task_manager.state.get_state(task_id);
    output[0] = static_cast<f_t>(state == TaskState::MAPPED);
    output[1] = static_cast<f_t>(state == TaskState::RESERVED);
    output[2] = static_cast<f_t>(state == TaskState::LAUNCHED);
    output[3] = static_cast<f_t>(state == TaskState::COMPLETED);
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
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_device_manager().get_devices();
    for (std::size_t i = 0; i < devices.size(); i++) {
      output[i] = static_cast<f_t>(data_manager.check_valid_mapped(data_id, i));
    }
  }
};

struct ScaledDataMappedLocations : public StateFeature<ScaledDataMappedLocations> {
  ScaledDataMappedLocations(const SchedulerState &state)
      : StateFeature<ScaledDataMappedLocations>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data = state.get_data_manager().get_data();
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_device_manager().get_devices();

    // Standardize the size
    double mean_size = state.stats.block_size_stats.mean;
    double std_size = state.stats.block_size_stats.stddev;

    // Get the size of the data
    auto data_size = static_cast<double>(data.get_size(data_id));
    // Scale the size
    data_size = guarded_divide(data_size, mean_size);

    for (std::size_t i = 0; i < devices.size(); i++) {
      output[i] = static_cast<f_t>(data_manager.check_valid_mapped(data_id, i)) * data_size;
    }
  }
};

struct DataReservedLocations : public StateFeature<DataReservedLocations> {
  DataReservedLocations(const SchedulerState &state)
      : StateFeature<DataReservedLocations>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_device_manager().get_devices();
    for (std::size_t i = 0; i < devices.size(); i++) {
      output[i] = static_cast<f_t>(data_manager.check_valid_reserved(data_id, i));
    }
  }
};

struct DataLaunchedLocations : public StateFeature<DataLaunchedLocations> {
  DataLaunchedLocations(const SchedulerState &state)
      : StateFeature<DataLaunchedLocations>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data_manager = state.get_data_manager();
    const auto &devices = state.get_device_manager().get_devices();
    for (std::size_t i = 0; i < devices.size(); i++) {
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
    const auto &data = state.get_data_manager().get_data();
    output[0] = log(static_cast<f_t>(data.get_size(data_id)) + 1);
  }
};

struct StandardizedDataSizeFeature : public StateFeature<StandardizedDataSizeFeature> {
  StandardizedDataSizeFeature(const SchedulerState &state)
      : StateFeature<StandardizedDataSizeFeature>(state, NodeType::DATA_BLOCK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID data_id, Span output) const {
    const auto &data = state.get_data_manager().get_data();
    double mean_size = state.stats.block_size_stats.mean;
    double std_size = state.stats.block_size_stats.stddev;
    output[0] = guarded_divide(static_cast<f_t>(data.get_size(data_id)) - mean_size, std_size);
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
    output[0] = log(mapped_mem + 1);
    output[1] = log(reserved_mem + 1);
    output[2] = log(launched_mem + 1);
  }
};

struct DeviceTimeFeature : public StateFeature<DeviceTimeFeature> {
  DeviceTimeFeature(const SchedulerState &state)
      : StateFeature<DeviceTimeFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    const auto &s = this->state;
    const auto &device_manager = s.get_device_manager();
    auto mapped_time = static_cast<double>(s.costs.get_mapped_time(device_id));
    // auto reserved_time = static_cast<double>(s.costs.get_reserved_time(device_id));
    // auto launched_time = static_cast<double>(s.costs.get_launched_time(device_id));

    // Normalize by mean duration
    auto mean_duration = static_cast<double>(s.stats.duration_stats.mean);
    // auto std_duration = static_cast<double>(s.stats.duration_stats.stddev);
    mapped_time = guarded_divide(mapped_time, mean_duration);
    // reserved_time = guarded_divide(reserved_time, mean_duration);
    // launched_time = guarded_divide(launched_time, mean_duration);

    output[0] = mapped_time;
    // output[1] = reserved_time;
    // output[2] = launched_time;
  }
};

struct DeviceReadDataFeature : public StateFeature<DeviceReadDataFeature> {
  DeviceReadDataFeature(const SchedulerState &state)
      : StateFeature<DeviceReadDataFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    const ComputeTask &task =
        state.get_task_manager().get_tasks().get_compute_task(state.current_candidate);
    const auto &data_manager = state.get_data_manager();
    const auto &data = data_manager.get_data();
    const auto &devices = state.get_device_manager().get_devices();
    double max_size = static_cast<double>(state.stats.total_read_stats.max);
    double total_read = static_cast<double>(data.get_total_size(task.get_read()));
    output[0] = guarded_divide(total_read, max_size);
    // for (devid_t i = 1; i < devices.size(); i++) {
    //   // output[i - 1] = -guarded_divide(total_read, max_size);
    //   for (auto data_id : task.get_read()) {
    //     if (data_manager.check_valid_mapped(static_cast<dataid_t>(data_id), i)) {
    //       auto data_size = static_cast<double>(data.get_size(data_id));
    //       output[i - 1] += guarded_divide(data_size, max_size);
    //     }
    //   }
    // }
  }
};

struct DeviceIDFeature : public StateFeature<DeviceIDFeature> {
  DeviceIDFeature(const SchedulerState &state)
      : StateFeature<DeviceIDFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    const auto &devices = this->state.get_device_manager().get_devices();
    return devices.size();
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    one_hot(device_id, output);
  }
};

struct DeviceArchitectureFeature : public StateFeature<DeviceArchitectureFeature> {
  DeviceArchitectureFeature(const SchedulerState &state)
      : StateFeature<DeviceArchitectureFeature>(state, NodeType::DEVICE) {
  }

  size_t getFeatureDimImpl() const {
    return 2;
  }

  template <typename ID, typename Span> void extractFeatureImpl(ID device_id, Span output) const {
    const auto &devices = state.get_device_manager().get_devices();
    output[0] = static_cast<f_t>(devices.get_type(device_id) == DeviceType::CPU);
    output[1] = static_cast<f_t>(devices.get_type(device_id) == DeviceType::GPU);
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

struct TaskTaskSharedDataFeature : public StateEdgeFeature<TaskTaskSharedDataFeature> {
  TaskTaskSharedDataFeature(const SchedulerState &state)
      : StateEdgeFeature<TaskTaskSharedDataFeature>(state, EdgeType::TASK_TASK) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    const auto &tasks = state.get_task_manager().get_tasks();

    const auto &source_task = tasks.get_compute_task(source_id);
    const auto &target_task = tasks.get_compute_task(target_id);
    const auto &data_manager = state.get_data_manager();
    const auto &data = data_manager.get_data();

    double total_memory_cost_source = 0;
    for (auto data_id : source_task.get_unique()) {
      total_memory_cost_source += data.get_size(data_id);
    }

    double total_memory_cost_target = 0;
    for (auto data_id : target_task.get_unique()) {
      total_memory_cost_target += data.get_size(data_id);
    }

    auto shared_memory_cost = static_cast<double>(
        data_manager.shared_size(source_task.get_unique(), target_task.get_unique()));
    output[0] = guarded_divide(shared_memory_cost, total_memory_cost_source);
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

struct TaskDataRelativeSizeFeature : public StateEdgeFeature<TaskDataRelativeSizeFeature> {
  TaskDataRelativeSizeFeature(const SchedulerState &state)
      : StateEdgeFeature<TaskDataRelativeSizeFeature>(state, EdgeType::TASK_DATA) {
  }

  size_t getFeatureDimImpl() const {
    return 1;
  }

  template <typename ID, typename Span>
  void extractFeatureImpl(ID source_id, ID target_id, Span output) const {
    const auto &tasks = state.get_task_manager().get_tasks();
    const auto &source_task = tasks.get_compute_task(source_id);
    const auto &data_manager = state.get_data_manager();
    const auto &data = data_manager.get_data();

    double total_memory_cost_source = 0;
    for (auto data_id : source_task.get_unique()) {
      total_memory_cost_source += data.get_size(data_id);
    }

    double data_size = static_cast<double>(data.get_size(target_id));
    output[0] = guarded_divide(data_size, total_memory_cost_source);
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
    const auto &tasks = state.get_task_manager().get_tasks();
    const auto &source_task = tasks.get_compute_task(source_id);

    bool is_read_access = std::find(source_task.get_read().begin(), source_task.get_read().end(),
                                    target_id) != source_task.get_read().end();
    bool is_write_access = std::find(source_task.get_write().begin(), source_task.get_write().end(),
                                     target_id) != source_task.get_write().end();

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
