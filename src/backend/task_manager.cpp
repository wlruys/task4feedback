#include "include/task_manager.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"

// Tasks

Tasks::Tasks(taskid_t num_compute_tasks)
    : num_compute_tasks(num_compute_tasks) {
  compute_tasks.resize(num_compute_tasks);
  task_names.resize(num_compute_tasks);
}

std::size_t Tasks::size() const {
  return compute_tasks.size() + data_tasks.size();
}
std::size_t Tasks::compute_size() const { return num_compute_tasks; }

std::size_t Tasks::data_size() const { return data_tasks.size(); }

bool Tasks::empty() const { return size() == 0; }

bool Tasks::is_compute(taskid_t id) const { return id < compute_size(); }

bool Tasks::is_data(taskid_t id) const { return id >= compute_size(); }

void Tasks::add_compute_task(ComputeTask task) {
  compute_tasks[task.id] = std::move(task);
}

void Tasks::add_data_task(DataTask task) {
  data_tasks.emplace_back(std::move(task));
}

void Tasks::create_compute_task(taskid_t id, std::string name,
                                TaskIDList dependencies) {
  task_names[id] = std::move(name);
  ComputeTask task(id);
  task.set_dependencies(std::move(dependencies));
  add_compute_task(task);
}

void Tasks::add_variant(taskid_t id, DeviceType arch, vcu_t vcu, mem_t mem,
                        timecount_t time) {
  compute_tasks[id].add_variant(arch, vcu, mem, time);
}

void Tasks::set_read(taskid_t id, DataIDList read) {
  compute_tasks[id].set_read(std::move(read));
}

void Tasks::set_write(taskid_t id, DataIDList write) {
  compute_tasks[id].set_write(std::move(write));
}

const TaskIDList &Tasks::get_dependencies(taskid_t id) const {
  if (is_compute(id)) {
    return compute_tasks[id].get_dependencies();
  }
  return data_tasks[id - compute_size()].get_dependencies();
}

const TaskIDList &Tasks::get_dependents(taskid_t id) const {
  if (is_compute(id)) {
    return compute_tasks[id].get_dependents();
  }
  return data_tasks[id - num_compute_tasks].get_dependents();
}

const VariantList &Tasks::get_variants(taskid_t id) const {
  return compute_tasks[id].get_variants();
}

const Variant &Tasks::get_variant(taskid_t id, DeviceType arch) const {
  return compute_tasks[id].get_variants()[static_cast<std::size_t>(arch)];
}

const DataIDList &Tasks::get_read(taskid_t id) const {
  return compute_tasks[id].get_read();
}

const DataIDList &Tasks::get_write(taskid_t id) const {
  return compute_tasks[id].get_write();
}

const Resources &Tasks::get_task_resources(taskid_t id, DeviceType arch) const {
  return get_variant(id, arch).resources;
}

std::vector<DeviceType> Tasks::get_supported_architectures(taskid_t id) const {
  return compute_tasks[id].get_supported_architectures();
}

Task &Tasks::get_task(taskid_t id) {
  if (id < num_compute_tasks) {
    return get_compute_task(id);
  }
  return get_data_task(id);
}

const Task &Tasks::get_task(taskid_t id) const {
  if (id < num_compute_tasks) {
    return get_compute_task(id);
  }
  return get_data_task(id);
}

// TaskStateInfo

TaskStateInfo::TaskStateInfo(std::size_t n) {
  state.resize(n, TaskState::SPAWNED);
  counts.resize(n, DepCount());
  mapping.resize(n);
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
  return counts[id].incomplete == 0 &&
         this->get_state(id) == TaskState::RESERVED;
}

bool TaskStateInfo::is_mapped(taskid_t id) const {
  return this->get_state(id) == TaskState::MAPPED;
}

bool TaskStateInfo::is_reserved(taskid_t id) const {
  return this->get_state(id) == TaskState::RESERVED;
}

bool TaskStateInfo::is_launched(taskid_t id) const {
  return this->get_state(id) == TaskState::LAUNCHED;
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

void TaskStateInfo::set_mapping_priority(taskid_t id, priority_t priority) {
  mapping_priority[id] = priority;
}

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

// Task Manager
void TaskManager::initialize_state() {
  const auto &const_tasks = this->tasks;

  state = TaskStateInfo(tasks.size());

  const auto &compute_tasks = const_tasks.get_compute_tasks();
  for (const auto &task : compute_tasks) {
    taskid_t id = task.id;
    auto n_deps = static_cast<depcount_t>(task.get_dependencies().size());
    state.set_state(id, TaskState::SPAWNED);
    state.set_unmapped(id, n_deps);
    state.set_unreserved(id, n_deps);
    state.set_incomplete(id, n_deps);
  }
}

void TaskManager::set_mapping(taskid_t id, devid_t devid) {
  state.set_mapping(id, devid);
}

const TaskIDList &TaskManager::notify_mapped(taskid_t id, timecount_t time) {
  const auto &task_objects = get_tasks();

  records.record_mapped(id, time);

  // clear exising task buffer
  task_buffer.clear();

  for (auto dependent_id : task_objects.get_dependents(id)) {
    if (state.decrement_unmapped(dependent_id)) {
      task_buffer.push_back(dependent_id);
    }
  }

  return task_buffer;
}

const TaskIDList &TaskManager::notify_reserved(taskid_t id, timecount_t time) {
  const auto &task_objects = get_tasks();

  records.record_reserved(id, time);

  // clear existing task buffer
  task_buffer.clear();

  for (auto dependent_id : task_objects.get_dependents(id)) {
    if (state.decrement_unreserved(dependent_id)) {
      task_buffer.push_back(dependent_id);
    }
  }

  return task_buffer;
}

void TaskManager::notify_launched(taskid_t id, timecount_t time) {
  records.record_launched(id, time);
}

const TaskIDList &TaskManager::notify_completed(taskid_t id, timecount_t time) {
  const auto &task_objects = get_tasks();

  records.record_completed(id, time);

  // clear task_buffer
  task_buffer.clear();

  for (auto dependent_id : task_objects.get_dependents(id)) {
    if (state.decrement_incomplete(dependent_id)) {
      task_buffer.push_back(dependent_id);
    }
  }

  return task_buffer;
}

// TaskPrinter

Color TaskPrinter::get_task_color(taskid_t id) {
  auto task_state = tm.state.get_state(id);
  if (task_state == TaskState::SPAWNED && !tm.state.is_mappable(id)) {
    return Color::white;
  }
  if (task_state == TaskState::SPAWNED) {
    return Color::blue;
  }

  if (task_state == TaskState::MAPPED && !tm.state.is_reservable(id)) {
    return Color::cyan;
  }
  if (task_state == TaskState::MAPPED) {
    return Color::magenta;
  }

  if (task_state == TaskState::RESERVED && !tm.state.is_launchable(id)) {
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
Table TaskPrinter::make_list_table(DependencyList &dependencies,
                                   std::string name) {
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
Table TaskPrinter::make_list_table_named(DependencyList &dependencies,
                                         std::string name) {
  Table dep_table;
  Table::Row_t deps;
  std::vector<Color> colors;
  deps.emplace_back(name);
  for (auto dep : dependencies) {
    std::string dep_name = tm.tasks.get_name(dep);
    deps.emplace_back(dep_name);
    colors.push_back(get_task_color(dep));
  }
  dep_table.add_row(deps);

  for (std::size_t i = 0; i < colors.size(); ++i) {
    dep_table[0][i + 1].format().font_color(colors[i]);
  }

  return dep_table;
}

template <typename DataList>
Table TaskPrinter::make_data_table(DataList &read, DataList &write) {
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
  value_row.emplace_back(std::to_string(v.get_execution_time()));

  variant_table.add_row(header_row);
  variant_table.add_row(value_row);

  variant_row.emplace_back(variant_table);

  device_table.add_row(device_row);
  device_table.add_row(variant_row);

  return device_table;
}

template <typename VariantList>
Table TaskPrinter::make_variant_tables(VariantList vlist) {
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

  Table::Row_t value_row = {std::to_string(id),
                            tm.tasks.get_name(id),
                            to_string(tm.state.get_state(id)),
                            to_string(tm.state.get_status(id)),
                            std::to_string(tm.state.get_unmapped(id)),
                            std::to_string(tm.state.get_unreserved(id)),
                            std::to_string(tm.state.get_incomplete(id))};

  status_table.add_row(header_row);
  status_table.add_row(value_row);

  return status_table;
}

Table TaskPrinter::wrap_tables(
    const std::vector<std::function<tabulate::Table(taskid_t)>> &generators,
    taskid_t id) const {
  Table table;

  Table::Row_t task_name;
  task_name.emplace_back("Task " + tm.tasks.get_name(id));
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
    const std::vector<std::function<tabulate::Table(taskid_t)>> &generators,
    taskid_t id) {
  Table table;

  Table::Row_t task_name;
  task_name.emplace_back("Task " + tm.tasks.get_name(id));
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

  auto status_table_generator = [&](taskid_t id) {
    return printer.make_status_table(id);
  };

  auto dependency_table_generator = [&](taskid_t id) {
    return printer.make_list_table_named(tasks.get_dependencies(id),
                                         "Dependencies");
  };

  auto dependent_table_generator = [&](taskid_t id) {
    return printer.make_list_table_named(tasks.get_dependents(id),
                                         "Dependents");
  };

  auto data_table_generator = [&](taskid_t id) {
    return printer.make_data_table(tasks.get_read(id), tasks.get_write(id));
  };

  auto variant_table_generator = [&](taskid_t id) {
    return printer.make_variant_tables(tasks.get_variants(id));
  };

  printer.print_tables({status_table_generator, dependency_table_generator,
                        dependent_table_generator, data_table_generator,
                        variant_table_generator},
                       id);
}
