#include "include/graph.hpp"
#include "include/queues.hpp"
#include "include/settings.hpp"
#include "include/tasks.hpp"
#include <queue>
#include <random>

namespace Graph {

void populate_dependents(TaskManager &tm) {
  for (taskid_t i = 0; i < tm.size(); ++i) {
    for (auto &dependency : tm.tasks[i].dependencies) {
      tm.tasks[dependency].dependents.push_back(i);
    }
  }
}

TaskIDList get_tasks_without_dependencies(TaskManager &tm) {
  TaskIDList tasks_without_dependencies;
  for (taskid_t i = 0; i < tm.tasks.size(); ++i) {
    if (tm.tasks[i].dependencies.size() == 0) {
      tasks_without_dependencies.push_back(i);
    }
  }
  return tasks_without_dependencies;
}

template <typename Obj>
std::unordered_map<Obj, Obj> convert_vector_to_map(std::vector<Obj> &vec) {
  std::unordered_map<Obj, Obj> map;
  for (auto &obj : vec) {
    map[obj] = obj;
  }
  return map;
}

template <typename Key, typename Obj>
std::unordered_map<Key, Obj> convert_vector_to_id_map(std::vector<Obj> &vec) {
  std::unordered_map<Key, Obj> map;
  for (auto &obj : vec) {
    map[obj.id] = obj;
  }
  return map;
}

std::unordered_map<taskid_t, MinimalTask>
create_minimal_tasks(TaskManager &tm) {
  std::unordered_map<taskid_t, MinimalTask> minimal_tasks;
  for (taskid_t i = 0; i < tm.tasks.size(); ++i) {
    auto &task = tm.tasks[i];
    minimal_tasks.emplace(i, MinimalTask(tm.tasks[i]));
  }
  return minimal_tasks;
}

std::vector<taskid_t> get_tasks_without_dependencies(
    std::unordered_map<taskid_t, MinimalTask> &minimal_tasks) {
  std::vector<taskid_t> tasks_without_dependencies;

  for (auto &id_task_pair : minimal_tasks) {
    auto &minimal_task = id_task_pair.second;
    if (minimal_task.dependencies.size() == 0) {
      tasks_without_dependencies.push_back(minimal_task.id);
    }
  }
  return tasks_without_dependencies;
}

/**
 * The function `random_topological_sort` returns a list of task IDs that are
 * randomly sorted in a topological order.
 *
 * @param tm TaskManager object that contains a list of tasks, where each task
 * has a unique identifier (taskid_t) and a list of dependencies that are also
 * task identifiers.
 *
 * @return The function `random_topological_sort` returns a list of task IDs
 * that are randomly sorted in a topological order.
 */
TaskIDList random_topological_sort(TaskManager &tm, int seed) {
  TaskIDList sorted_tasks;

  std::unordered_map<taskid_t, MinimalTask> minimal_task_map =
      create_minimal_tasks(tm);

  std::vector<taskid_t> tasks_without_dependencies =
      get_tasks_without_dependencies(minimal_task_map);

  auto r = Randomizer<taskid_t, std::priority_queue>(seed);

  for (auto &task : tasks_without_dependencies) {
    r.push(task);
  }

  while (r.size() > 0) {
    auto taskid = r.top();
    r.pop();
    auto &task = minimal_task_map[taskid];
    sorted_tasks.push_back(task.id);

    std::cout << "Selected task dependents: ";
    print(task.dependents);

    for (auto dependent : task.dependents) {
      auto &dependent_task = minimal_task_map[dependent];
      dependent_task.dependencies.erase(taskid);
      if (dependent_task.dependencies.size() == 0) {
        r.push(dependent);
      }
    }
  }

  return sorted_tasks;
}

} // namespace Graph