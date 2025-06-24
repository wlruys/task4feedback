#include "include/tasks.hpp"
#include "include/settings.hpp"

std::vector<DeviceType> ComputeTask::get_supported_architectures() const {
  std::vector<DeviceType> supported_architectures;

  SPDLOG_DEBUG("Getting supported architectures for task {}", id);
  SPDLOG_DEBUG("Task has {} variants", variants.size());

  for (const auto &variant : variants) {
    auto arch = variant.get_arch();

    if (arch != DeviceType::NONE) {
      supported_architectures.push_back(arch);
    }
  }
  return supported_architectures;
}

bool ComputeTask::is_supported_architecture(DeviceType arch) const {
  const auto &variant = variants[static_cast<std::size_t>(arch)];
  return variant.get_arch() == arch;
}

std::vector<Variant> ComputeTask::get_variant_vector() const {
  std::vector<Variant> variant_vector;
  for (const auto &variant : variants) {
    auto arch = variant.get_arch();
    if (arch != DeviceType::NONE) {
      variant_vector.push_back(variant);
    }
  }
  return variant_vector;
}

// Tasks

Tasks::Tasks(taskid_t num_compute_tasks) : num_compute_tasks(num_compute_tasks) {
  current_task_id = num_compute_tasks;
  compute_tasks.resize(num_compute_tasks);
  task_names.resize(num_compute_tasks);
}

std::size_t Tasks::size() const {
  return compute_tasks.size() + data_tasks.size();
}
std::size_t Tasks::compute_size() const {
  return num_compute_tasks;
}

std::size_t Tasks::data_size() const {
  return data_tasks.size();
}

bool Tasks::empty() const {
  return size() == 0;
}

bool Tasks::is_compute(taskid_t id) const {
  return id < compute_size();
}

bool Tasks::is_data(taskid_t id) const {
  // Note eviction tasks return as data tasks (because they are a form of data task)
  return id >= compute_size();
}

bool Tasks::is_eviction(taskid_t id) const {
  // Any tasks outside the range of compute and data tasks are considered eviction tasks
  return id >= data_size() + compute_size();
}

void Tasks::add_compute_task(ComputeTask task) {
  compute_tasks[task.id] = std::move(task);
}

void Tasks::add_data_task(DataTask task) {
  data_tasks.emplace_back(std::move(task));
}

StatsBundle<timecount_t>
Tasks::get_duration_statistics(std::vector<DeviceType> &device_types) const {
  std::vector<timecount_t> durations;
  durations.reserve(compute_tasks.size());

  StatsBundle<timecount_t> stats;
  for (const auto &task : compute_tasks) {
    const auto &variants = task.get_variants();
    for (const auto &device_type : device_types) {
      const auto &variant = variants[static_cast<std::size_t>(device_type)];
      assert(variant.get_arch() == device_type);
      durations.push_back(variant.get_observed_time());
    }
  }

  return StatsBundle(durations);
}

void Tasks::create_data_task(ComputeTask &task, bool has_writer, taskid_t writer_id,
                             dataid_t data_id) {
  taskid_t data_task_id = current_task_id++;
  data_tasks.emplace_back(data_task_id);
  DataTask &data_task = data_tasks.back();
  auto &compute_task_name = task_names[task.id];
  // task_names.push_back(compute_task_name + "_data[" + std::to_string(data_id) + "]");
  task_names.push_back("_data[" + std::to_string(data_id) + "]");
  data_task.set_data_id(data_id);
  data_task.set_compute_task(task.id);

  if (has_writer) {
    data_task.add_dependency(writer_id);
    auto &writer_task = get_compute_task(writer_id);
    writer_task.add_data_dependent(data_task_id);
  }

  data_task.add_dependent(task.id);
  task.add_data_dependency(data_task_id);
}

void Tasks::create_compute_task(taskid_t id, std::string name, TaskIDList dependencies) {
  assert(id < num_compute_tasks);
  task_names[id] = std::move(name);
  ComputeTask task(id);
  task.set_dependencies(std::move(dependencies));
  add_compute_task(task);
}

void Tasks::add_variant(taskid_t id, DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time) {
  compute_tasks[id].add_variant(arch, vcu, mem, time);
}

void Tasks::set_read(taskid_t id, DataIDList read) {
  compute_tasks[id].set_read(std::move(read));
}

void Tasks::set_write(taskid_t id, DataIDList write) {
  compute_tasks[id].set_write(std::move(write));
}

void Tasks::set_retire(taskid_t id, DataIDList retire) {
  compute_tasks[id].set_retire(std::move(retire));
}

void Tasks::set_tag(taskid_t id, int tag) {
  compute_tasks[id].set_tag(tag);
}

void Tasks::set_type(taskid_t id, int type) {
  compute_tasks[id].set_type(type);
}

int Tasks::get_type(taskid_t id) const {
  return compute_tasks[id].get_type();
}

int Tasks::get_tag(taskid_t id) const {
  return compute_tasks[id].get_tag();
}

const TaskIDList &Tasks::get_dependencies(taskid_t id) const {
  if (is_compute(id)) {
    return get_compute_task(id).get_dependencies();
  }
  return get_data_task(id).get_dependencies();
}

const TaskIDList &Tasks::get_dependents(taskid_t id) const {
  if (is_compute(id)) {
    return get_compute_task(id).get_dependents();
  }
  return get_data_task(id).get_dependents();
}

const VariantList &Tasks::get_variants(taskid_t id) const {
  return get_compute_task(id).get_variants();
}

const Variant &Tasks::get_variant(taskid_t id, DeviceType arch) const {
  return get_compute_task(id).get_variants()[static_cast<std::size_t>(arch)];
}

const DataIDList &Tasks::get_read(taskid_t id) const {
  return get_compute_task(id).get_read();
}

const DataIDList &Tasks::get_write(taskid_t id) const {
  return get_compute_task(id).get_write();
}

const DataIDList &Tasks::get_retire(taskid_t id) const {
  return get_compute_task(id).get_retire();
}

const Resources &Tasks::get_task_resources(taskid_t id, DeviceType arch) const {
  return get_variant(id, arch).resources;
}

std::vector<DeviceType> Tasks::get_supported_architectures(taskid_t id) const {
  return get_compute_task(id).get_supported_architectures();
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

const TaskIDList &Tasks::get_data_dependencies(taskid_t id) const {
  return get_compute_task(id).get_data_dependencies();
}

const TaskIDList &Tasks::get_data_dependents(taskid_t id) const {
  return get_compute_task(id).get_data_dependents();
}

std::size_t Tasks::get_depth(taskid_t id) const {
  return get_compute_task(id).get_depth();
}

dataid_t Tasks::get_data_id(taskid_t id) const {
  return get_data_task(id).get_data_id();
}