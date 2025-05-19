#include "include/task_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"

// TaskStateInfo

TaskStateInfo::TaskStateInfo(const Tasks &tasks) {
  auto n = tasks.size();
  n_compute_tasks = tasks.compute_size();
  n_data_tasks = tasks.data_size();
  state.resize(n, TaskState::SPAWNED);
  counts.resize(n, DepCount());
  mapping.resize(n_compute_tasks, -1);
  // mapping_priority.resize(n, 0);
  reserving_priority.resize(n, 0);
  launching_priority.resize(n, 0);
  is_virtual.resize(n, false);
  sources.resize(n, 0);
}

TaskStatus TaskStateInfo::get_status(taskid_t id) const {
  if (is_launchable(id)) {
    return TaskStatus::LAUNCHABLE;
  }
  if (is_reservable(id)) {
    return TaskStatus::RESERVABLE;
  }
  if (is_mappable(id)) {
    return TaskStatus::MAPPABLE;
  }
  return TaskStatus::NONE;
}

bool TaskStateInfo::is_mappable(taskid_t id) const {
  return counts[id].unmapped == 0 && this->get_state(id) == TaskState::SPAWNED;
}

bool TaskStateInfo::is_reservable(taskid_t id) const {
  return counts[id].unreserved == 0 && this->get_state(id) == TaskState::MAPPED;
}

bool TaskStateInfo::is_launchable(taskid_t id) const {
  return counts[id].incomplete == 0 && this->get_state(id) == TaskState::RESERVED;
}

bool TaskStateInfo::is_mapped(taskid_t id) const {
  return this->get_state(id) >= TaskState::MAPPED;
}

bool TaskStateInfo::is_reserved(taskid_t id) const {
  return this->get_state(id) >= TaskState::RESERVED;
}

bool TaskStateInfo::is_launched(taskid_t id) const {
  return this->get_state(id) >= TaskState::LAUNCHED;
}

bool TaskStateInfo::is_completed(taskid_t id) const {
  return this->get_state(id) >= TaskState::COMPLETED;
}

bool TaskStateInfo::decrement_unmapped(taskid_t id) {
  counts[id].unmapped--;
  assert(counts[id].unmapped >= 0);
  // return if the task just became mappable
  return (counts[id].unmapped == 0) && (state[id] == TaskState::SPAWNED);
}

bool TaskStateInfo::decrement_unreserved(taskid_t id) {
  counts[id].unreserved--;
  assert(counts[id].unreserved >= 0);
  // return if the task just became reservable
  return (counts[id].unreserved == 0) && (state[id] == TaskState::MAPPED);
}

bool TaskStateInfo::decrement_incomplete(taskid_t id) {
  counts[id].incomplete--;
  assert(counts[id].incomplete >= 0);
  // return if the task just became launchable
  return (counts[id].incomplete == 0) && (state[id] == TaskState::RESERVED);
}

// void TaskStateInfo::set_mapping_priority(taskid_t id, priority_t priority) {
//   mapping_priority[id] = priority;
// }

void TaskStateInfo::set_reserving_priority(taskid_t id, priority_t priority) {
  reserving_priority[id] = priority;
}

void TaskStateInfo::set_launching_priority(taskid_t id, priority_t priority) {
  launching_priority[id] = priority;
}

void TaskStateInfo::set_unmapped(taskid_t id, depcount_t count) {
  counts[id].unmapped = count;
}

void TaskStateInfo::set_unreserved(taskid_t id, depcount_t count) {
  counts[id].unreserved = count;
}

void TaskStateInfo::set_incomplete(taskid_t id, depcount_t count) {
  counts[id].incomplete = count;
}

// TaskRecords

void TaskRecords::record_mapped(taskid_t id, timecount_t time) {
  auto index = task_to_index(id, mapped_idx);
  state_times[index] = time;
}

void TaskRecords::record_reserved(taskid_t id, timecount_t time) {
  auto index = task_to_index(id, reserved_idx);
  state_times[index] = time;
}

void TaskRecords::record_launched(taskid_t id, timecount_t time) {
  auto index = task_to_index(id, launched_idx);
  state_times[index] = time;
}

void TaskRecords::record_completed(taskid_t id, timecount_t time) {
  auto index = task_to_index(id, completed_idx);
  state_times[index] = time;
}

timecount_t TaskRecords::get_mapped_time(taskid_t id) const {
  auto index = task_to_index(id, mapped_idx);
  return state_times[index];
}

timecount_t TaskRecords::get_reserved_time(taskid_t id) const {
  auto index = task_to_index(id, reserved_idx);
  return state_times[index];
}

timecount_t TaskRecords::get_launched_time(taskid_t id) const {
  auto index = task_to_index(id, launched_idx);
  return state_times[index];
}

timecount_t TaskRecords::get_completed_time(taskid_t id) const {
  auto index = task_to_index(id, completed_idx);
  return state_times[index];
}

TaskState TaskRecords::get_state_at_time(taskid_t id, timecount_t query_time) const {
  if (query_time < get_mapped_time(id)) {
    return TaskState::SPAWNED;
  }
  if (query_time < get_reserved_time(id)) {
    return TaskState::MAPPED;
  }
  if (query_time < get_launched_time(id)) {
    return TaskState::RESERVED;
  }
  if (query_time < get_completed_time(id)) {
    return TaskState::LAUNCHED;
  }
  return TaskState::COMPLETED;
}

// Task Manager
void TaskManager::initialize_state() {
  const auto &const_tasks = this->tasks.get();
  state = TaskStateInfo(tasks);
  records = TaskRecords(tasks);

  const auto &compute_tasks = const_tasks.get_compute_tasks();
  for (const auto &task : compute_tasks) {
    taskid_t id = task.id;
    auto n_deps = static_cast<depcount_t>(task.get_dependencies().size());
    state.set_state(id, TaskState::SPAWNED);
    state.set_unmapped(id, n_deps);
    state.set_unreserved(id, n_deps);

    auto n_data_deps = static_cast<depcount_t>(task.get_data_dependencies().size());
    auto n_total_deps = n_deps + n_data_deps;

    state.set_incomplete(id, n_total_deps);
  }

  for (const auto &task : const_tasks.get_data_tasks()) {
    taskid_t id = task.id;
    auto n_deps = static_cast<depcount_t>(task.get_dependencies().size());
    state.set_state(id, TaskState::SPAWNED);
    state.set_unmapped(id, 0);
    state.set_unreserved(id, 0);
    state.set_incomplete(id, n_deps);
  }
}

void TaskManager::set_mapping(taskid_t id, devid_t devid) {
  state.set_mapping(id, devid);
}

const TaskIDList &TaskManager::notify_mapped(taskid_t id, timecount_t time) {
  const auto &task_objects = get_tasks();

  records.record_mapped(id, time);
  state.set_state(id, TaskState::MAPPED);

  // clear exising task buffer
  task_buffer.clear();

  for (auto dependent_id : task_objects.get_dependents(id)) {
    if (state.decrement_unmapped(dependent_id)) {
      task_buffer.push_back(dependent_id);
      assert(state.get_state(dependent_id) == TaskState::SPAWNED);
      assert(task_objects.is_compute(dependent_id));
    }
  }

  return task_buffer;
}

const TaskIDList &TaskManager::notify_reserved(taskid_t id, timecount_t time) {
  const auto &task_objects = get_tasks();

  records.record_reserved(id, time);
  state.set_state(id, TaskState::RESERVED);

  // clear existing task buffer
  task_buffer.clear();

  for (auto dependent_id : task_objects.get_dependents(id)) {
    if (state.decrement_unreserved(dependent_id)) {
      task_buffer.push_back(dependent_id);
      assert(state.get_state(dependent_id) == TaskState::MAPPED);
      assert(task_objects.is_compute(dependent_id));
    }
  }

  return task_buffer;
}

void TaskManager::notify_launched(taskid_t id, timecount_t time) {
  state.set_state(id, TaskState::LAUNCHED);
  records.record_launched(id, time);
}

const TaskIDList &TaskManager::notify_completed(taskid_t id, timecount_t time) {
  const auto &task_objects = get_tasks();

  records.record_completed(id, time);
  state.set_state(id, TaskState::COMPLETED);
  // clear task_buffer
  task_buffer.clear();
  if (!task_objects.is_eviction(id)) {
    for (auto dependent_id : task_objects.get_dependents(id)) {
      if (state.decrement_incomplete(dependent_id)) {
        task_buffer.push_back(dependent_id);
        assert(state.get_state(dependent_id) == TaskState::RESERVED);
        assert(task_objects.is_compute(dependent_id));
      }
    }
  }

  return task_buffer;
}

const TaskIDList &TaskManager::notify_data_completed(taskid_t id, timecount_t time) {
  MONUnusedParameter(time);
  const auto &task_objects = get_tasks();
  // clear task_buffer
  task_buffer.clear();

  for (auto dependent_id : task_objects.get_data_dependents(id)) {
    if (state.decrement_incomplete(dependent_id)) {
      task_buffer.push_back(dependent_id);
      assert(state.get_state(dependent_id) == TaskState::RESERVED);
      assert(task_objects.is_data(dependent_id));
    }
  }

  return task_buffer;
}

// TaskPrinter

Color TaskPrinter::get_task_color(taskid_t id) const {
  const auto _tm = tm.get();
  auto task_state = _tm.state.get_state(id);
  if (task_state == TaskState::SPAWNED && !_tm.state.is_mappable(id)) {
    return Color::white;
  }
  if (task_state == TaskState::SPAWNED) {
    return Color::blue;
  }

  if (task_state == TaskState::MAPPED && !_tm.state.is_reservable(id)) {
    return Color::cyan;
  }
  if (task_state == TaskState::MAPPED) {
    return Color::magenta;
  }

  if (task_state == TaskState::RESERVED && !_tm.state.is_launchable(id)) {
    return Color::yellow;
  }
  if (task_state == TaskState::RESERVED) {
    return Color::green;
  }

  if (task_state == TaskState::LAUNCHED) {
    return Color::red;
  }

  if (task_state == TaskState::COMPLETED) {
    return Color::grey;
  }
  return Color::white;
}

template <typename DependencyList>
Table TaskPrinter::make_list_table(DependencyList &dependencies) {
  Table dep_table;
  Table::Row_t deps;
  for (auto dep : dependencies) {
    std::cout << dep << std::endl;
    deps.emplace_back(std::to_string(dep));
  }
  dep_table.add_row(deps);

  std::cout << dep_table << std::endl;
  return dep_table;
}
template <typename DependencyList>
Table TaskPrinter::make_list_table(DependencyList &dependencies, std::string name) {
  Table dep_table;
  Table::Row_t deps;
  std::vector<Color> colors;
  deps.emplace_back(name);
  for (auto dep : dependencies) {
    deps.emplace_back(std::to_string(dep));
    colors.push_back(get_task_color(dep));
  }
  dep_table.add_row(deps);

  for (std::size_t i = 0; i < colors.size(); ++i) {
    dep_table[0][i + 1].format().font_color(colors[i]);
  }

  return dep_table;
}

template <typename DependencyList>
Table TaskPrinter::make_list_table_named(DependencyList &dependencies, std::string name) {
  Table dep_table;
  Table::Row_t deps;
  std::vector<Color> colors;
  deps.emplace_back(name);
  for (auto dep : dependencies) {
    std::string dep_name = tm.get().tasks.get().get_name(dep);
    deps.emplace_back(dep_name);
    colors.push_back(get_task_color(dep));
  }
  dep_table.add_row(deps);

  for (std::size_t i = 0; i < colors.size(); ++i) {
    dep_table[0][i + 1].format().font_color(colors[i]);
  }

  return dep_table;
}

template <typename DataList> Table TaskPrinter::make_data_table(DataList &read, DataList &write) {
  Table data_table;
  Table::Row_t read_row;
  Table::Row_t write_row;

  read_row.emplace_back("Read");
  write_row.emplace_back("Write");

  for (auto data : read) {
    read_row.emplace_back(std::to_string(data));
  }
  for (auto data : write) {
    write_row.emplace_back(std::to_string(data));
  }
  data_table.add_row(read_row);
  data_table.add_row(write_row);
  return data_table;
}

Table TaskPrinter::make_variant_table(Variant v) {

  Table device_table;
  Table::Row_t device_row;
  Table::Row_t variant_row;

  device_row.emplace_back(to_string(v.arch));

  Table variant_table;
  Table::Row_t header_row;
  Table::Row_t value_row;

  header_row.emplace_back("VCU");
  header_row.emplace_back("MEM");
  header_row.emplace_back("TIME");

  value_row.emplace_back(std::to_string(v.get_vcus()));
  value_row.emplace_back(std::to_string(v.get_mem()));
  value_row.emplace_back(std::to_string(v.get_observed_time()));

  variant_table.add_row(header_row);
  variant_table.add_row(value_row);

  variant_row.emplace_back(variant_table);

  device_table.add_row(device_row);
  device_table.add_row(variant_row);

  return device_table;
}

template <typename VariantList> Table TaskPrinter::make_variant_tables(VariantList vlist) {
  Table variant_table;
  Table::Row_t variants;

  for (auto &v : vlist) {
    if (v.arch == DeviceType::NONE) {
      continue;
    }
    auto table = make_variant_table(v);
    variants.push_back(table);
  }
  variant_table.add_row(variants);

  return variant_table;
}

Table TaskPrinter::make_status_table(taskid_t id) {
  Table status_table;

  Table::Row_t header_row = {"ID",       "Name",       "State",     "Status",
                             "Unmapped", "Unreserved", "Incomplete"};

  const auto &_tm = this->tm.get();

  Table::Row_t value_row = {std::to_string(id),
                            _tm.tasks.get().get_name(id),
                            to_string(_tm.state.get_state(id)),
                            to_string(_tm.state.get_status(id)),
                            std::to_string(_tm.state.get_unmapped(id)),
                            std::to_string(_tm.state.get_unreserved(id)),
                            std::to_string(_tm.state.get_incomplete(id))};

  status_table.add_row(header_row);
  status_table.add_row(value_row);

  return status_table;
}

Table TaskPrinter::wrap_tables(
    const std::vector<std::function<tabulate::Table(taskid_t)>> &generators, taskid_t id) const {
  Table table;

  Table::Row_t task_name;
  task_name.emplace_back("Task " + tm.get().get_tasks().get_name(id));
  table.add_row(task_name);

  for (const auto &generator : generators) {
    Table::Row_t inner_row;
    auto inner_table = generator(id);
    inner_row.emplace_back(inner_table);
    table.add_row(inner_row);
  }

  return table;
}

void TaskPrinter::print_tables(
    const std::vector<std::function<tabulate::Table(taskid_t)>> &generators, taskid_t id) {
  Table table;

  Table::Row_t task_name;
  task_name.emplace_back("Task " + tm.get().get_tasks().get_name(id));
  table.add_row(task_name);
  table.format()
      .font_color(get_task_color(id))
      .border_top("+")
      .border_bottom("+")
      .border_left("+")
      .border_right("+");

  std::cout << table << std::endl;

  for (const auto &generator : generators) {
    Table::Row_t inner_row;
    auto inner_table = generator(id);
    if (inner_table.size() > 0) {
      if (inner_table[0].size() == 1) {
        continue;
      }
      std::cout << inner_table << std::endl;
    }
  }
}

Table TaskPrinter::wrap_in_task_table(taskid_t id, tabulate::Table table) {
  Table task_table;
  Table::Row_t task_row;
  Table::Row_t table_row;
  task_row.emplace_back("Task " + std::to_string(id));
  table_row.emplace_back(table);

  task_table.add_row(task_row);
  task_table.add_row(table_row);

  return task_table;
}

void TaskManager::print_task(taskid_t id) {
  TaskPrinter printer(*this);

  auto status_table_generator = [&](taskid_t id) { return printer.make_status_table(id); };

  auto dependency_table_generator = [&](taskid_t id) {
    return printer.make_list_table_named(tasks.get().get_dependencies(id), "Dependencies");
  };

  auto dependent_table_generator = [&](taskid_t id) {
    return printer.make_list_table_named(tasks.get().get_dependents(id), "Dependents");
  };

  auto data_table_generator = [&](taskid_t id) {
    return printer.make_data_table(tasks.get().get_read(id), tasks.get().get_write(id));
  };

  auto variant_table_generator = [&](taskid_t id) {
    return printer.make_variant_tables(tasks.get().get_variants(id));
  };

  printer.print_tables({status_table_generator, dependency_table_generator,
                        dependent_table_generator, data_table_generator, variant_table_generator},
                       id);
}
