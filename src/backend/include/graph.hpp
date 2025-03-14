#pragma once

#include "data_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct Writer {
  bool found = false;
  taskid_t task_id = 0;
};

class GraphTemplate {
public:
  std::vector<taskid_t> task_tags;
  std::vector<taskid_t> task_types;
  std::vector<std::string> task_names;
  std::vector<std::vector<taskid_t>> read_data;
  std::vector<std::vector<taskid_t>> write_data;
  std::vector<std::array<std::array<int64_t, 4>, num_device_types>> variant_info;
  std::vector<std::vector<taskid_t>> dependencies;

  std::unordered_map<std::string, taskid_t> task_name_to_id;

  GraphTemplate() {
    task_tags.reserve(100);
    task_names.reserve(100);
    read_data.reserve(100);
    write_data.reserve(100);
    variant_info.reserve(100);
    dependencies.reserve(100);
    task_types.reserve(100);
  }

  taskid_t size() {
    return task_names.size();
  }

  void resize(int new_size) {
    task_types.resize(new_size);
    task_tags.resize(new_size);
    task_names.resize(new_size);
    read_data.resize(new_size);
    write_data.resize(new_size);
    variant_info.resize(new_size);
    dependencies.resize(new_size);
  }

  taskid_t add_task(const std::string &task_name, taskid_t tag) {
    taskid_t task_id = size();
    task_tags.push_back(tag);
    task_names.push_back(task_name);
    task_types.push_back(0);
    task_name_to_id[task_name] = task_id;
    read_data.push_back(std::vector<taskid_t>());
    write_data.push_back(std::vector<taskid_t>());

    std::array<std::array<int64_t, 4>, num_device_types> zero_variant_array{};
    // Disable all variants initially
    for (int64_t i = 0; i < num_device_types; i++) {
      zero_variant_array[i][0] = static_cast<int64_t>(DeviceType::NONE);
      zero_variant_array[i][1] = 0;
      zero_variant_array[i][2] = 0;
      zero_variant_array[i][3] = 0;
    }
    variant_info.push_back(zero_variant_array);
    dependencies.push_back(std::vector<taskid_t>());
    return task_id;
  }

  void add_read_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    read_data[task_id].insert(read_data[task_id].end(), data_ids.begin(), data_ids.end());
  }

  void add_write_data(taskid_t task_id, std::vector<dataid_t> &data_ids) {
    write_data[task_id].insert(write_data[task_id].end(), data_ids.begin(), data_ids.end());
  }

  int get_tag(taskid_t task_id) {
    return task_tags[task_id];
  }

  std::string get_name(taskid_t task_id) {
    return task_names[task_id];
  }

  void set_tag(taskid_t task_id, taskid_t tag) {
    task_tags[task_id] = tag;
  }

  void set_type(taskid_t task_id, taskid_t type) {
    task_types[task_id] = type;
  }

  int get_type(taskid_t task_id) {
    return task_types[task_id];
  }

  taskid_t get_id(const std::string &task_name) {
    return task_name_to_id[task_name];
  }

  std::vector<dataid_t> &get_read_data(taskid_t task_id) {
    return read_data[task_id];
  }

  std::vector<dataid_t> &get_write_data(taskid_t task_id) {
    return write_data[task_id];
  }

  void add_variant_info(taskid_t task_id, DeviceType device_type, vcu_t vcus, mem_t memory,
                        timecount_t time) {
    int64_t arch_idx = static_cast<int64_t>(device_type);
    variant_info[task_id].at(arch_idx) = {arch_idx, vcus, memory, time};
  }

  vcu_t get_vcu(taskid_t task_id, DeviceType device_type) {
    int64_t arch_idx = static_cast<int64_t>(device_type);
    return variant_info[task_id].at(arch_idx)[1];
  }

  mem_t get_memory(taskid_t task_id, DeviceType device_type) {
    int64_t arch_idx = static_cast<int64_t>(device_type);
    return variant_info[task_id].at(arch_idx)[2];
  }

  timecount_t get_time(taskid_t task_id, DeviceType device_type) {
    int64_t arch_idx = static_cast<int64_t>(device_type);
    return variant_info[task_id].at(arch_idx)[3];
  }

  void set_vcu(taskid_t task_id, DeviceType device_type, int64_t vcus) {
    int64_t arch_idx = static_cast<int64_t>(device_type);
    variant_info[task_id].at(arch_idx)[1] = vcus;
  }

  void set_memory(taskid_t task_id, DeviceType device_type, int64_t memory) {
    int64_t arch_idx = static_cast<int64_t>(device_type);
    variant_info[task_id].at(arch_idx)[2] = memory;
  }

  void set_time(taskid_t task_id, DeviceType device_type, int64_t time) {
    int64_t arch_idx = static_cast<int64_t>(device_type);
    variant_info[task_id].at(arch_idx)[3] = time;
  }

  std::vector<int64_t> get_variant_info(taskid_t task_id, DeviceType device_type) {
    int64_t arch_idx = static_cast<int64_t>(device_type);
    std::vector<int64_t> info(4);
    for (int64_t i = 0; i < 4; i++) {
      info[i] = variant_info[task_id].at(arch_idx)[i];
    }
    return info;
  }

  void add_dependency(taskid_t task_id, taskid_t dependency_id) {
    dependencies[task_id].push_back(dependency_id);
  }

  void add_dependencies(taskid_t task_id, std::vector<taskid_t> &dependency_ids) {
    dependencies[task_id].insert(dependencies[task_id].end(), dependency_ids.begin(),
                                 dependency_ids.end());
  }

  std::vector<taskid_t> &get_dependencies(taskid_t task_id) {
    return dependencies[task_id];
  }

  void remove_dependency(taskid_t task_id, taskid_t dependency_id) {
    std::vector<taskid_t> &deps = dependencies[task_id];
    deps.erase(std::remove(deps.begin(), deps.end(), dependency_id), deps.end());
  }

  void clear_dependencies(taskid_t task_id) {
    dependencies[task_id].clear();
  }

  void remove_task(taskid_t task_id) {
    // Remove task from list and shift all tasks after it
    task_tags.erase(task_tags.begin() + task_id);
    task_names.erase(task_names.begin() + task_id);
    read_data.erase(read_data.begin() + task_id);
    write_data.erase(write_data.begin() + task_id);
    variant_info.erase(variant_info.begin() + task_id);
    dependencies.erase(dependencies.begin() + task_id);

    // Update task_name_to_id
    for (auto it = task_name_to_id.begin(); it != task_name_to_id.end(); it++) {
      if (it->second == task_id) {
        task_name_to_id.erase(it);
        break;
      }
    }

    for (auto &it : task_name_to_id) {
      if (it.second > task_id) {
        it.second--;
      }
    }

    // Update dependencies
    for (taskid_t i = 0; i < size(); i++) {
      for (taskid_t j = 0; j < dependencies[i].size(); j++) {
        if (dependencies[i][j] == task_id) {
          dependencies[i].erase(dependencies[i].begin() + j);
          j--;
        } else if (dependencies[i][j] > task_id) {
          dependencies[i][j]--;
        }
      }
    }
  }

  GraphTemplate reindex(std::unordered_map<std::string, taskid_t> &task_name_to_new_id) {

    GraphTemplate new_graph;
    new_graph.resize(size());

    for (taskid_t i = 0; i < size(); i++) {
      taskid_t new_id = task_name_to_new_id[task_names[i]];
      new_graph.task_tags[new_id] = task_tags[i];
      new_graph.task_names[new_id] = task_names[i];
      new_graph.read_data[new_id] = read_data[i];
      new_graph.write_data[new_id] = write_data[i];
      new_graph.variant_info[new_id] = variant_info[i];
      new_graph.dependencies[new_id] = dependencies[i];
    }

    for (taskid_t i = 0; i < size(); i++) {
      for (taskid_t dep : dependencies[i]) {
        taskid_t new_id = task_name_to_new_id[task_names[i]];
        taskid_t new_dep = task_name_to_new_id[task_names[dep]];
        new_graph.add_dependency(new_id, new_dep);
      }
    }

    return new_graph;
  }

  void replicate(int replications) {
    taskid_t total_new = size() * replications;
    taskid_t offset = size();
    for (int i = 1; i < replications; i++) {
      for (taskid_t j = 0; j < size(); j++) {
        add_task(task_names[j], task_tags[j]);
        for (taskid_t dep : dependencies[j]) {
          add_dependency(j + offset, dep + offset);
        }
        add_read_data(j + offset, read_data[j]);
        add_write_data(j + offset, write_data[j]);

        for (std::size_t k = 0; k < num_device_types; k++) {
          std::array<int64_t, 4> info = variant_info[j][k];
          variant_info[j + offset][k] = info;
        }
      }
    }
  }

  void stack(std::vector<GraphTemplate> graphs) {
    taskid_t total_new = 0;
    for (GraphTemplate &graph : graphs) {
      total_new += graph.size();
    }

    taskid_t offset = size();
    for (GraphTemplate &graph : graphs) {
      for (taskid_t i = 0; i < graph.size(); i++) {
        add_task(graph.task_names[i], graph.task_tags[i]);
        for (taskid_t dep : graph.dependencies[i]) {
          add_dependency(i + offset, dep + offset);
        }
        add_read_data(i + offset, graph.read_data[i]);
        add_write_data(i + offset, graph.write_data[i]);

        for (std::size_t j = 0; j < num_device_types; j++) {
          std::array<int64_t, 4> info = graph.variant_info[i][j];
          variant_info[i + offset][j] = info;
        }
      }
      offset += graph.size();
    }
  }

  Tasks to_tasks() {
    Tasks tasks(size());

    for (taskid_t i = 0; i < size(); i++) {
      tasks.create_compute_task(i, get_name(i), get_dependencies(i));
      tasks.set_tag(i, get_tag(i));
      tasks.set_read(i, get_read_data(i));
      tasks.set_write(i, get_write_data(i));
      tasks.set_type(i, get_type(i));
      for (std::size_t j = 0; j < num_device_types; j++) {
        std::vector<int64_t> info = get_variant_info(i, static_cast<DeviceType>(j));
        // GraphTemplate::add_task initializes all variants to DeviceType::NONE
        // setting info[0] to -1
        if (info[0] != -1) {
          tasks.add_variant(i, static_cast<DeviceType>(j), info[1], info[2], info[3]);
        }
      }
    }

    return tasks;
  }

  void fill_dependencies_from_data_usage() {
    // Assumes that the graph is in a valid topological ordering and that the data dependencies are
    // correct

    std::unordered_map<dataid_t, taskid_t> last_writer;
    for (taskid_t i = 0; i < size(); i++) {
      for (dataid_t data_id : read_data[i]) {
        if (last_writer.find(data_id) != last_writer.end()) {
          // only add if its not already in the dependencies
          if (std::find(dependencies[i].begin(), dependencies[i].end(), last_writer[data_id]) ==
              dependencies[i].end()) {
            add_dependency(i, last_writer[data_id]);
          }
        }
      }

      for (dataid_t data_id : write_data[i]) {
        last_writer[data_id] = i;
      }
    }
  }
};

class GraphManager {

private:
  static std::unordered_map<taskid_t, MinimalTask> create_minimal_tasks(const TaskIDList &task_ids,
                                                                        Tasks &tasks);

  static std::unordered_map<taskid_t, MinimalTask>
  create_minimal_tasks(const ComputeTaskList &tasks);

  static void find_recent_writers(TaskIDList &sorted, Tasks &tasks,
                                  std::unordered_map<taskid_t, dataid_t> &writers);

public:
  static void populate_data_dependencies(TaskIDList &sorted, Tasks &tasks, bool create_data_tasks,
                                         bool add_missing_writers);
  static void populate_dependents(Tasks &tasks);
  static void add_missing_writer_dependencies(std::unordered_map<dataid_t, taskid_t> &writers,
                                              ComputeTask &task, Tasks &tasks);

  static void create_data_tasks(std::unordered_map<dataid_t, taskid_t> &writers, ComputeTask &task,
                                Tasks &tasks);

  static Writer find_writer(std::unordered_map<dataid_t, taskid_t> &writers, dataid_t data_id);

  static void update_writers(std::unordered_map<dataid_t, taskid_t> &writers,
                             const DataIDList &write, taskid_t task_id);

  static void calculate_depth(TaskIDList &sorted, Tasks &tasks);
  static void finalize(Tasks &tasks, bool create_data_tasks, bool add_missing_writers);

  static TaskIDList initial_tasks(const ComputeTaskList &tasks);
  static TaskIDList initial_tasks(const TaskIDList &task_ids, Tasks &tasks);

  static TaskIDList random_topological_sort(Tasks &tasks, unsigned long seed = 0);
  static TaskIDList random_topological_sort(const TaskIDList &task_ids, Tasks &tasks,
                                            unsigned long seed = 0);

  static TaskIDList breadth_first_sort(Tasks &tasks);
  static TaskIDList breadth_first_sort(const TaskIDList &task_ids, Tasks &tasks);

  static TaskIDList depth_first_sort(Tasks &tasks);
  static TaskIDList depth_first_sort(const TaskIDList &task_ids, Tasks &tasks);
};