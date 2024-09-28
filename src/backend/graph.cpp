#include "include/graph.hpp"
#include "include/queues.hpp"
#include "include/settings.hpp"
#include "include/tasks.hpp"
#include "task_manager.hpp"
#include <queue>
#include <random>
#include <stack>

void GraphManager::populate_dependents(Tasks &tasks) {
  for (auto &task : tasks.get_compute_tasks()) {
    for (auto dependency_id : task.get_dependencies()) {
      auto &dependency_task = tasks.get_compute_task(dependency_id);
      dependency_task.add_dependent(task.id);
    }
  }
}

TaskIDList GraphManager::initial_tasks(const ComputeTaskList &tasks) {
  TaskIDList tasks_without_dependencies;
  for (const auto &task : tasks) {
    if (task.get_dependencies().empty()) {
      tasks_without_dependencies.push_back(task.id);
    }
  }
  return tasks_without_dependencies;
}

TaskIDList GraphManager::initial_tasks(const TaskIDList &task_ids,
                                       Tasks &tasks) {
  TaskIDList tasks_without_dependencies;
  for (auto task_id : task_ids) {
    const auto &task = tasks.get_task(task_id);
    if (task.get_dependencies().empty()) {
      tasks_without_dependencies.push_back(task.id);
    }
  }
  return tasks_without_dependencies;
}

std::unordered_map<taskid_t, MinimalTask>
GraphManager::create_minimal_tasks(const TaskIDList &task_ids, Tasks &tasks) {
  std::unordered_map<taskid_t, MinimalTask> minimal_tasks;
  for (auto task_id : task_ids) {
    const auto &task = tasks.get_task(task_id);
    minimal_tasks.emplace(task.id, MinimalTask(task));
  }
  return minimal_tasks;
}

std::unordered_map<taskid_t, MinimalTask>
GraphManager::create_minimal_tasks(const ComputeTaskList &tasks) {
  std::unordered_map<taskid_t, MinimalTask> minimal_tasks;
  for (const auto &task : tasks) {
    minimal_tasks.emplace(task.id, MinimalTask(task));
  }
  return minimal_tasks;
}

TaskIDList
depth_first_sort_(TaskIDList &starting_tasks,
                  std::unordered_map<taskid_t, MinimalTask> &minimal_task_map) {
  TaskIDList sorted_tasks;
  std::stack<taskid_t> stack;
  for (auto taskid : starting_tasks) {
    stack.push(taskid);
  }

  while (!stack.empty()) {
    auto taskid = stack.top();
    stack.pop();
    auto &task = minimal_task_map[taskid];
    sorted_tasks.push_back(task.id);

    for (auto dependent_id : task.dependents) {
      auto &dependent_task = minimal_task_map[dependent_id];
      dependent_task.dependencies.erase(taskid);
      if (dependent_task.dependencies.empty()) {
        stack.push(dependent_id);
      }
    }
  }

  return sorted_tasks;
}

TaskIDList breadth_first_sort_(
    TaskIDList &starting_tasks,
    std::unordered_map<taskid_t, MinimalTask> &minimal_task_map) {

  TaskIDList sorted_tasks;
  std::queue<taskid_t> queue;
  for (auto taskid : starting_tasks) {
    queue.push(taskid);
  }

  while (!queue.empty()) {
    auto taskid = queue.front();
    queue.pop();
    auto &task = minimal_task_map[taskid];
    sorted_tasks.push_back(task.id);

    for (auto dependent_id : task.dependents) {
      auto &dependent_task = minimal_task_map[dependent_id];
      dependent_task.dependencies.erase(taskid);
      if (dependent_task.dependencies.empty()) {
        queue.push(dependent_id);
      }
    }
  }

  return sorted_tasks;
}

TaskIDList random_topological_sort_(
    TaskIDList &starting_tasks,
    std::unordered_map<taskid_t, MinimalTask> &minimal_task_map,
    unsigned long seed) {

  TaskIDList sorted_tasks;
  auto r = ContainerQueue<taskid_t, std::priority_queue>(seed);

  for (auto taskid : starting_tasks) {
    r.push_random(taskid);
  }

  while (!r.empty()) {
    auto taskid = r.top();
    r.pop();
    auto &task = minimal_task_map[taskid];
    sorted_tasks.push_back(task.id);

    for (auto dependent_id : task.dependents) {
      auto &dependent_task = minimal_task_map[dependent_id];
      dependent_task.dependencies.erase(taskid);
      if (dependent_task.dependencies.empty()) {
        r.push(dependent_id);
      }
    }
  }

  return sorted_tasks;
}

TaskIDList GraphManager::breadth_first_sort(const TaskIDList &task_ids,
                                            Tasks &tasks) {
  std::unordered_map<taskid_t, MinimalTask> minimal_task_map =
      create_minimal_tasks(task_ids, tasks);

  TaskIDList starting_tasks = initial_tasks(task_ids, tasks);

  return breadth_first_sort_(starting_tasks, minimal_task_map);
}

TaskIDList GraphManager::depth_first_sort(const TaskIDList &task_ids,
                                          Tasks &tasks) {
  TaskIDList sorted_tasks;
  std::unordered_map<taskid_t, MinimalTask> minimal_task_map =
      create_minimal_tasks(task_ids, tasks);

  TaskIDList starting_tasks = initial_tasks(task_ids, tasks);

  return depth_first_sort_(starting_tasks, minimal_task_map);
}

TaskIDList GraphManager::random_topological_sort(const TaskIDList &task_ids,
                                                 Tasks &tasks,
                                                 unsigned long seed) {
  TaskIDList sorted_tasks;

  std::unordered_map<taskid_t, MinimalTask> minimal_task_map =
      create_minimal_tasks(task_ids, tasks);

  TaskIDList starting_tasks = initial_tasks(task_ids, tasks);

  return random_topological_sort_(starting_tasks, minimal_task_map, seed);
}

TaskIDList GraphManager::breadth_first_sort(Tasks &tasks) {
  const auto &task_list = tasks.get_compute_tasks();
  std::unordered_map<taskid_t, MinimalTask> minimal_task_map =
      create_minimal_tasks(task_list);

  TaskIDList starting_tasks = initial_tasks(task_list);

  return breadth_first_sort_(starting_tasks, minimal_task_map);
}

TaskIDList GraphManager::depth_first_sort(Tasks &tasks) {
  TaskIDList sorted_tasks;
  const auto &task_list = tasks.get_compute_tasks();
  std::unordered_map<taskid_t, MinimalTask> minimal_task_map =
      create_minimal_tasks(task_list);

  TaskIDList starting_tasks = initial_tasks(task_list);

  return depth_first_sort_(starting_tasks, minimal_task_map);
}

TaskIDList GraphManager::random_topological_sort(Tasks &tasks,
                                                 unsigned long seed) {
  TaskIDList sorted_tasks;
  const auto &task_list = tasks.get_compute_tasks();
  std::unordered_map<taskid_t, MinimalTask> minimal_task_map =
      create_minimal_tasks(task_list);

  TaskIDList starting_tasks = initial_tasks(task_list);

  return random_topological_sort_(starting_tasks, minimal_task_map, seed);
}