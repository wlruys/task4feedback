#pragma once

#include "tasks.hpp"
#include <list>
#include <random>
#include <unordered_map>
#include <vector>

namespace Graph {
TaskIDList get_tasks_without_dependencies(TaskManager &tm);
TaskIDList random_topological_sort(TaskManager &tm, unsigned long seed = 0);
void populate_dependents(TaskManager &tm);
void populate_data_dependencies(TaskManager &tm);
void populate_data_dependents(TaskManager &tm);
void initalize_counts(TaskManager &tm);
} // namespace Graph
