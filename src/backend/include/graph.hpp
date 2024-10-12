#pragma once

#include "settings.hpp"
#include "tasks.hpp"
#include <unordered_map>

struct Writer {
  bool found = false;
  taskid_t task_id = 0;
};

class GraphManager {

private:
  static std::unordered_map<taskid_t, MinimalTask>
  create_minimal_tasks(const TaskIDList &task_ids, Tasks &tasks);

  static std::unordered_map<taskid_t, MinimalTask>
  create_minimal_tasks(const ComputeTaskList &tasks);

  static void
  find_recent_writers(TaskIDList &sorted, Tasks &tasks,
                      std::unordered_map<taskid_t, dataid_t> &writers);

public:
  static void populate_data_dependencies(TaskIDList &sorted, Tasks &tasks);
  static void populate_dependents(Tasks &tasks);
  static void add_missing_writer_dependencies(
      std::unordered_map<dataid_t, taskid_t> &writers, ComputeTask &task,
      Tasks &tasks);

  static void create_data_tasks(std::unordered_map<dataid_t, taskid_t> &writers,
                                ComputeTask &task, Tasks &tasks);

  static Writer find_writer(std::unordered_map<dataid_t, taskid_t> &writers,
                            dataid_t data_id);

  static void update_writers(std::unordered_map<dataid_t, taskid_t> &writers,
                             const DataIDList &write, taskid_t task_id);

  static void calculate_depth(TaskIDList &sorted, Tasks &tasks);
  static void finalize(Tasks &tasks, bool create_data_tasks = false);

  static TaskIDList initial_tasks(const ComputeTaskList &tasks);
  static TaskIDList initial_tasks(const TaskIDList &task_ids, Tasks &tasks);

  static TaskIDList random_topological_sort(Tasks &tasks,
                                            unsigned long seed = 0);
  static TaskIDList random_topological_sort(const TaskIDList &task_ids,
                                            Tasks &tasks,
                                            unsigned long seed = 0);

  static TaskIDList breadth_first_sort(Tasks &tasks);
  static TaskIDList breadth_first_sort(const TaskIDList &task_ids,
                                       Tasks &tasks);

  static TaskIDList depth_first_sort(Tasks &tasks);
  static TaskIDList depth_first_sort(const TaskIDList &task_ids, Tasks &tasks);
};