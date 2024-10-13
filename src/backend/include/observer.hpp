#pragma once
#include "devices.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include <functional>
#include <limits>
#include <unordered_map>

struct NormalizationInfo {
  depcount_t max_in_degree;
  double average_in_degree;
  double stddev_in_degree;

  depcount_t max_out_degre;
  double average_out_degree;
  double stddev_out_degree;

  timecount_t serial_execution_time;
  std::array<double, num_device_types> average_duration;
  std::array<double, num_device_types> stddev_duration;

  std::array<double, num_device_types> average_task_memcost;
  std::array<double, num_device_types> stddev_task_memcost;

  double average_data_size;
  double stddev_data_size;
};

struct TaskDataEdges {
  std::vector<taskid_t> tasks;
  std::vector<dataid_t> data;
};

struct TaskDeviceEdges {
  std::vector<taskid_t> tasks;
  std::vector<devid_t> devices;
};

struct DataDeviceEdges {
  std::vector<dataid_t> data;
  std::vector<devid_t> devices;
};

class Observer {
public:
  std::reference_wrapper<SchedulerState> state;
  NormalizationInfo graph_info;

  Observer(SchedulerState &state) : state(state) {}

  double get_in_degree(const ComputeTask &task) {
    return static_cast<double>(task.get_dependencies().size());
  }

  double get_out_degree(const ComputeTask &task) {
    return static_cast<double>(task.get_dependents().size());
  }

  double get_duration(const ComputeTask &task, DeviceType arch) {
    return static_cast<double>(task.get_variant(arch).get_observed_time());
  }

  double get_task_memcost(const ComputeTask &task, DeviceType arch) {
    return static_cast<double>(task.get_variant(arch).get_mem());
  }

  double get_task_data_memcost(const ComputeTask &task, const Data &data) {
    double memcost = 0;
    for (const auto &data_id : task.get_unique()) {
      memcost += static_cast<double>(data.get_size(data_id));
    }
    return memcost;
  }

  void preprocess_global(const Tasks &tasks, const Data &data) {

    // NOTE(wlr): This is AN AWFUL estimator of stddev
    //            I just want something running for now
    //            TODO: Implement this better

    double in_degree_sum = 0;
    double in_degree_sum2 = 0;
    depcount_t max_in_degree = 0;

    double out_degree_sum = 0;
    double out_degree_sum2 = 0;
    depcount_t max_out_degree = 0;

    std::array<double, num_device_types> duration_sum = {0, 0};
    std::array<double, num_device_types> duration_sum2 = {0, 0};
    timecount_t duration_worst_sum = 0;

    std::array<double, num_device_types> task_memcost_sum = {0, 0};
    std::array<double, num_device_types> task_memcost_sum2 = {0, 0};

    double data_size_sum = 0;
    double data_size_sum2 = 0;

    for (const auto &task : tasks.get_compute_tasks()) {
      in_degree_sum += get_in_degree(task);
      in_degree_sum2 += get_in_degree(task) * get_in_degree(task);
      max_in_degree = std::max(max_in_degree, task.get_dependencies().size());

      out_degree_sum += get_out_degree(task);
      out_degree_sum2 += get_out_degree(task) * get_out_degree(task);
      max_out_degree = std::max(max_out_degree, task.get_dependents().size());

      double task_data_memcost = get_task_data_memcost(task, data);

      timecount_t duration_worst = 0;
      for (std::size_t i = 0; i < num_device_types; i++) {
        const auto d = get_duration(task, static_cast<DeviceType>(i));
        duration_sum[i] += d;
        duration_sum2[i] += d * d;

        const auto m = get_task_memcost(task, static_cast<DeviceType>(i)) +
                       task_data_memcost;
        task_memcost_sum[i] += m;
        task_memcost_sum2[i] += m * m;

        if (static_cast<timecount_t>(d) > duration_worst) {
          duration_worst = static_cast<timecount_t>(d);
        }
      }

      duration_worst_sum += duration_worst;
    }

    for (auto size : data.get_sizes()) {
      data_size_sum += static_cast<double>(size);
      data_size_sum2 += static_cast<double>(size) * static_cast<double>(size);
    }

    const auto num_tasks = static_cast<double>(tasks.compute_size());
    const auto num_data = static_cast<double>(data.size());

    graph_info.max_in_degree = max_in_degree;
    graph_info.average_in_degree = in_degree_sum / num_tasks;
    graph_info.stddev_in_degree =
        std::sqrt(in_degree_sum2 / num_tasks -
                  graph_info.average_in_degree * graph_info.average_in_degree);

    graph_info.max_out_degre = max_out_degree;
    graph_info.average_out_degree = out_degree_sum / num_tasks;
    graph_info.stddev_out_degree = std::sqrt(out_degree_sum2 / num_tasks -
                                             graph_info.average_out_degree *
                                                 graph_info.average_out_degree);

    graph_info.serial_execution_time = duration_worst_sum;
    for (std::size_t i = 0; i < num_device_types; i++) {
      graph_info.average_duration[i] = duration_sum[i] / num_tasks;
      graph_info.stddev_duration[i] = std::sqrt(
          duration_sum2[i] / num_tasks -
          graph_info.average_duration[i] * graph_info.average_duration[i]);

      graph_info.average_task_memcost[i] = task_memcost_sum[i] / num_tasks;
      graph_info.stddev_task_memcost[i] =
          std::sqrt(task_memcost_sum2[i] / num_tasks -
                    graph_info.average_task_memcost[i] *
                        graph_info.average_task_memcost[i]);
    }

    graph_info.average_data_size = data_size_sum / num_data;
    graph_info.stddev_data_size =
        std::sqrt(data_size_sum2 / num_data -
                  graph_info.average_data_size * graph_info.average_data_size);

    SPDLOG_DEBUG("Graph properties:");
    SPDLOG_DEBUG("Max in-degree: {}", graph_info.max_in_degree);
    SPDLOG_DEBUG("Average in-degree: {}", graph_info.average_in_degree);
    SPDLOG_DEBUG("Stddev in-degree: {}", graph_info.stddev_in_degree);

    SPDLOG_DEBUG("Max out-degree: {}", graph_info.max_out_degre);
    SPDLOG_DEBUG("Average out-degree: {}", graph_info.average_out_degree);
    SPDLOG_DEBUG("Stddev out-degree: {}", graph_info.stddev_out_degree);

    SPDLOG_DEBUG("Serial execution time: {}", graph_info.serial_execution_time);
    for (std::size_t i = 0; i < num_device_types; i++) {
      SPDLOG_DEBUG("Average duration on device type {}: {}",
                   to_string(static_cast<DeviceType>(i)),
                   graph_info.average_duration[i]);
      SPDLOG_DEBUG("Stddev duration on device type {}: {}",
                   to_string(static_cast<DeviceType>(i)),
                   graph_info.stddev_duration[i]);

      SPDLOG_DEBUG("Average task memcost on device type {}: {}",
                   to_string(static_cast<DeviceType>(i)),
                   graph_info.average_task_memcost[i]);
      SPDLOG_DEBUG("Stddev task memcost on device type {}: {}",
                   to_string(static_cast<DeviceType>(i)),
                   graph_info.stddev_task_memcost[i]);
    }

    SPDLOG_DEBUG("Average data size: {}", graph_info.average_data_size);
    SPDLOG_DEBUG("Stddev data size: {}", graph_info.stddev_data_size);
  }

  void preprocess_global() {
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    const auto &device_manager = s.get_device_manager();
    const auto &devices = s.get_device_manager().get_devices();
    const auto &data = s.get_data_manager().get_data();
    preprocess_global(tasks, data);
  }

  TaskIDList get_active_tasks() {
    const auto &s = this->state.get();
    return s.counts.get_active_task_list();
  }

  TaskIDList get_k_hop_tasks(const TaskIDList &initial, int k) {
    // Follow task dependents k times to build graph
    // TODO(wlr): Test impl is horribly inefficient sorry

    TaskIDList result;
    std::unordered_set<taskid_t> to_visit;

    const auto &s = this->state.get();
    const auto &tasks = s.get_task_manager().get_tasks();

    for (auto task_id : initial) {
      to_visit.insert(task_id);
    }

    while (to_visit.size() > 0 && k > 0) {
      std::unordered_set<taskid_t> next_to_visit;
      for (auto task_id : to_visit) {
        result.push_back(task_id);
        const auto &task = tasks.get_compute_task(task_id);
        for (auto dep_id : task.get_dependencies()) {
          next_to_visit.insert(dep_id);
        }
      }
      to_visit = next_to_visit;
      k--;
    }

    return result;
  }

  std::vector<double> get_task_features(taskid_t task_id) {
    constexpr std::size_t FEATURE_LENGTH = 5;
    std::vector<double> features(FEATURE_LENGTH);

    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    const auto &task = tasks.get_compute_task(task_id);

    features[0] = get_in_degree(task) / graph_info.average_in_degree;
    features[1] = get_out_degree(task) / graph_info.average_out_degree;

    features[2] = static_cast<double>(task_manager.state.is_mapped(task_id));
    features[3] = static_cast<double>(task_manager.state.is_reserved(task_id));
    features[4] = static_cast<double>(task_manager.state.is_launched(task_id));
    return features;
  }

  std::vector<double> get_variant_features(taskid_t task_id, DeviceType arch) {
    constexpr std::size_t FEATURE_LENGTH = 2;
    std::vector<double> features(FEATURE_LENGTH);
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();

    const auto &task = tasks.get_compute_task(task_id);

    features[0] =
        get_duration(task, arch) /
        graph_info.average_duration.at(static_cast<std::size_t>(arch));
    features[1] =
        get_task_memcost(task, arch) /
        graph_info.average_task_memcost.at(static_cast<std::size_t>(arch));

    return features;
  }

  std::vector<double> get_task_device_features(taskid_t task_id,
                                               devid_t device_id) {
    const auto &s = this->state.get();
    const auto &devices = s.get_device_manager().get_devices();
    const auto &device = devices.get_device(device_id);

    DeviceType arch = device.arch;
    return get_variant_features(task_id, arch);
  }

  std::vector<double> get_data_features(dataid_t data_id) {
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    const auto &device_manager = s.get_device_manager();
    const auto &data_manager = s.get_data_manager();
    const auto &devices = s.get_device_manager().get_devices();
    const auto &data = s.get_data_manager().get_data();

    const auto &data_size = data.get_size(data_id);

    const std::size_t FEATURE_LENGTH = 1 + devices.size();
    std::vector<double> features(FEATURE_LENGTH);

    features[0] = static_cast<double>(data_size) / graph_info.average_data_size;
    for (std::size_t i = 0; i < devices.size(); i++) {
      features[i + 1] = static_cast<double>(
          data_manager.get_mapped_locations().is_valid(data_id, i));
    }
    return features
  }

  std::vector<double> get_task_data_features(taskid_t task_id,
                                             dataid_t data_id) {
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    const auto &data = s.get_data_manager().get_data();

    const auto &task = tasks.get_compute_task(task_id);
    const auto &data_size = data.get_size(data_id);

    const std::size_t FEATURE_LENGTH = 3;
    std::vector<double> features(FEATURE_LENGTH);

    double task_data_memcost = get_task_data_memcost(task, data);
    bool is_read = std::find(task.get_read().begin(), task.get_read().end(),
                             data_id) != task.get_read().end();
    bool is_write = std::find(task.get_write().begin(), task.get_write().end(),
                              data_id) != task.get_write().end();

    features[0] = static_cast<double>(data_size) / task_data_memcost;
    features[1] = static_cast<double>(is_read);
    features[2] = static_cast<double>(is_write);

    return features;
  }

  std::vector<double> get_device_features(devid_t device_id) {
    constexpr std::size_t FEATURE_LENGTH = 8;
    std::vector<double> features(FEATURE_LENGTH);

    const auto &s = this->state.get();
    const auto &device_manager = s.get_device_manager();
    const auto &devices = s.get_device_manager().get_devices();

    const auto &device = devices.get_device(device_id);

    features[0] = static_cast<double>(device.arch == DeviceType::CPU);
    features[1] = static_cast<double>(device.arch == DeviceType::GPU);

    double total_mapped_mem = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_mapped_mem += static_cast<double>(
          device_manager.get_mem<TaskState::MAPPED>(device_id));
    }

    double total_reserved_mem = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_reserved_mem += static_cast<double>(
          device_manager.get_mem<TaskState::RESERVED>(device_id));
    }

    double total_launched_mem = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_launched_mem += static_cast<double>(
          device_manager.get_mem<TaskState::LAUNCHED>(device_id));
    }

    features[2] = static_cast<double>(
                      device_manager.get_mem<TaskState::MAPPED>(device_id)) /
                  total_mapped_mem;
    features[3] = static_cast<double>(
                      device_manager.get_mem<TaskState::RESERVED>(device_id)) /
                  total_reserved_mem;
    features[4] = static_cast<double>(
                      device_manager.get_mem<TaskState::LAUNCHED>(device_id)) /
                  total_launched_mem;

    double total_mapped_time = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_mapped_time += static_cast<double>(s.costs.get_mapped_time(i));
    }
    double total_reserved_time = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_reserved_time += static_cast<double>(s.costs.get_reserved_time(i));
    }

    double total_launched_time = 0;
    for (std::size_t i = 0; i < devices.size(); i++) {
      total_launched_time += static_cast<double>(s.costs.get_launched_time(i));
    }

    features[5] = static_cast<double>(s.costs.get_mapped_time(device_id)) /
                  total_mapped_time;
    features[6] = static_cast<double>(s.costs.get_reserved_time(device_id)) /
                  total_reserved_time;
    features[7] = static_cast<double>(s.costs.get_launched_time(device_id)) /
                  total_launched_time;

    return features;
  }

  std::size_t get_number_of_unique_data(taskid_t *ids, std::size_t n) {
    std::set<dataid_t> unique_set;
    for (std::size_t i = 0; i < n; i++) {
      const auto &task =
          state.get().get_task_manager().get_tasks().get_compute_task(ids[i]);
      for (auto data_id : task.get_unique()) {
        unique_set.insert(data_id);
      }
    }
    return unique_set.size();
  }

  auto get_unique_datamap(const TaskIDList &task_ids) {
    std::map<dataid_t, std::size_t> unique_map;

    std::size_t counter = 0;
    for (auto task_id : task_ids) {
      const auto &task =
          state.get().get_task_manager().get_tasks().get_compute_task(task_id);
      for (auto data_id : task.get_unique()) {
        unique_map[data_id] = counter++;
      }
    }
    return unique_map;
  }

  TaskDataEdges get_task_data_edges(const TaskIDList &task_ids) {
    TaskDataEdges edges;

    for (const auto &task_id : task_ids) {
      const auto &task =
          state.get().get_task_manager().get_tasks().get_compute_task(task_id);
      for (const auto &data_id : task.get_unique()) {
        edges.tasks.push_back(task_id);
        edges.data.push_back(data_id);
      }
    }
    return edges;
  }

  TaskDeviceEdges get_task_device_edges(const TaskIDList &task_ids) {
    TaskDeviceEdges edges;

    for (const auto &task_id : task_ids) {
      const auto &task =
          state.get().get_task_manager().get_tasks().get_compute_task(task_id);
      const auto &supported_architectures = task.get_supported_architectures();

      for (auto arch : supported_architectures) {
        const auto &device_ids =
            state.get().get_device_manager().get_devices().get_devices(arch);
        for (auto device_id : device_ids) {
          edges.tasks.push_back(task_id);
          edges.devices.push_back(device_id);
        }
      }
    }
    return edges;
  }

  DataDeviceEdges get_data_device_edges(const TaskIDList &task_ids) {
    DataDeviceEdges edges;

    auto unique_map = get_unique_datamap(task_ids);

    for (const auto &data_id : unique_map) {
      const auto &valid_sources =
          state.get().get_data_manager().get_valid_mapped_locations(
              data_id.first);
      for (const auto &device_id : valid_sources) {
        edges.data.push_back(data_id.first);
        edges.devices.push_back(device_id);
      }
    }
    return edges;
  }
};

/*

  void get_task_adjacency_matrix(taskid_t *ids, std::size_t n, double *matrix)
  {
    // Assume matrix is zeroed out
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    const auto &device_manager = s.get_device_manager();
    const auto &devices = s.get_device_manager().get_devices();
    const auto &data = s.get_data_manager().get_data();

    std::unordered_map<taskid_t, std::size_t> task_index;
    for (std::size_t i = 0; i < n; i++) {
      task_index[ids[i]] = i;
    }

    for (std::size_t i = 0; i < n; i++) {
      const auto &task = tasks.get_compute_task(ids[i]);
      for (const auto &dep_id : task.get_dependencies()) {
        if (task_index.find(dep_id) != task_index.end()) {
          matrix[i * n + task_index[dep_id]] = 1;
        }
      }
    }
  }


  void get_task_data_bipartite(taskid_t *ids, std::size_t n, double *matrix) {
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    const auto &device_manager = s.get_device_manager();
    const auto &devices = s.get_device_manager().get_devices();
    const auto &data = s.get_data_manager().get_data();

    auto unique_map = get_unique_datamap(ids, n);

    for (const auto &task : tasks.get_compute_tasks()) {
      for (auto data_id : task.get_unique()) {
        if (unique_map.find(data_id) != unique_map.end()) {
          matrix[task.id * n + unique_map[data_id]] =
              static_cast<double>(data.get_size(data_id));
        }
      }
    }
  }

  void get_data_device_bipartite(taskid_t *task_ids, std::size_t n,
                                 double *matrix) {
    const auto &s = this->state.get();
    const auto &task_manager = s.get_task_manager();
    const auto &tasks = task_manager.get_tasks();
    const auto &device_manager = s.get_device_manager();
    const auto &data_manager = s.get_data_manager();
    const auto &devices = s.get_device_manager().get_devices();
    const auto &data = s.get_data_manager().get_data();

    auto unique_map = get_unique_datamap(task_ids, n);

    // Loop over all data_ids in unique_map and get valid mapped locations
  from
    // device_manager for each data_id
    for (const auto &data_id : unique_map) {
      const auto &valid_sources =
          data_manager.get_valid_mapped_locations(data_id.first);
      for (const auto &device_id : valid_sources) {
        matrix[data_id.second * device_manager.size() + device_id] = 1;
      }
    }
  }


*/