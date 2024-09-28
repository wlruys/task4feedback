#pragma once

#include "task_manager.hpp"
#include <unordered_map>

class GraphManager {

private:
  static std::unordered_map<taskid_t, MinimalTask>
  create_minimal_tasks(const TaskIDList &task_ids, Tasks &tasks);

  static std::unordered_map<taskid_t, MinimalTask>
  create_minimal_tasks(const ComputeTaskList &tasks);

public:
  static void populate_dependents(Tasks &tasks);
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