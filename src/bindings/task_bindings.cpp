#include "devices.hpp"
#include "resources.hpp"
#include "tasks.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

void init_task_ext(nb::module_ &m) {

  nb::enum_<TaskState>(m, "TaskState", nb::is_arithmetic())
      .value("SPAWNED", TaskState::SPAWNED)
      .value("MAPPED", TaskState::MAPPED)
      .value("RESERVED", TaskState::RESERVED)
      .value("LAUNCHED", TaskState::LAUNCHED)
      .value("COMPLETED", TaskState::COMPLETED)
      .export_values();

  nb::enum_<TaskStatus>(m, "TaskStatus", nb::is_arithmetic())
      .value("NON", TaskStatus::NONE)
      .value("MAPPABLE", TaskStatus::MAPPABLE)
      .value("RESERVABLE", TaskStatus::RESERVABLE)
      .value("LAUNCHABLE", TaskStatus::LAUNCHABLE)
      .export_values();

  nb::bind_vector<std::vector<Variant>>(m, "VariantVector");
  nb::bind_vector<std::vector<DeviceType>>(m, "DeviceTypeVector");
  nb::class_<Variant>(m, "Variant")
      .def(nb::init<>()) // default constructor
      .def(nb::init<DeviceType, vcu_t, mem_t, timecount_t>(), "arch"_a, "vcu"_a, "mem"_a, "time"_a)
      .def("get_arch", &Variant::get_arch)
      .def("get_vcus", &Variant::get_vcus)
      .def("get_mem", &Variant::get_mem)
      .def("get_observed_time", &Variant::get_observed_time)
      .def("__str__",
           [](const Variant &v) {
             return "Variant(arch=" + std::to_string(static_cast<int>(v.get_arch())) +
                    ", vcu=" + std::to_string(v.get_vcus()) +
                    ", mem=" + std::to_string(v.get_mem()) +
                    ", time=" + std::to_string(v.get_observed_time()) + ")";
           })
      .def("__repr__", [](const Variant &v) {
        return "Variant(arch=" + std::to_string(static_cast<int>(v.get_arch())) +
               ", vcu=" + std::to_string(v.get_vcus()) + ", mem=" + std::to_string(v.get_mem()) +
               ", time=" + std::to_string(v.get_observed_time()) + ")";
      });

  nb::class_<Task>(m, "Task")
      .def(nb::init<taskid_t>(), "id"_a)
      .def_prop_ro("id", &Task::get_id)
      .def_prop_ro("depth", &Task::get_depth)
      .def_prop_ro("dependencies", &Task::get_dependencies, nb::rv_policy::reference_internal)
      .def_prop_ro("dependents", &Task::get_dependents, nb::rv_policy::reference_internal)
      .def("get_id", &Task::get_id)
      .def("set_depth", &Task::set_depth, "depth"_a)
      .def("get_depth", &Task::get_depth)
      .def("get_dependencies", &Task::get_dependencies, nb::rv_policy::reference_internal)
      .def("get_dependents", &Task::get_dependents, nb::rv_policy::reference_internal);

  nb::class_<ComputeTask, Task>(m, "ComputeTask")
      .def(nb::init<taskid_t>(), "id"_a)
      .def_prop_ro("read", nb::overload_cast<>(&ComputeTask::get_read, nb::const_),
                   nb::rv_policy::reference_internal)
      .def_prop_ro("write", nb::overload_cast<>(&ComputeTask::get_write, nb::const_),
                   nb::rv_policy::reference_internal)
      .def_prop_ro("data_dependencies",
                   nb::overload_cast<>(&ComputeTask::get_data_dependencies, nb::const_),
                   nb::rv_policy::reference_internal)
      .def_prop_ro("data_dependents",
                   nb::overload_cast<>(&ComputeTask::get_data_dependents, nb::const_),
                   nb::rv_policy::reference_internal)
      .def_prop_ro("supported_architectures",
                   nb::overload_cast<>(&ComputeTask::get_supported_architectures, nb::const_),
                   nb::rv_policy::reference_internal)
      .def_prop_ro("tag", &ComputeTask::get_tag)
      .def_prop_ro("type", &ComputeTask::get_type)
      .def("get_read", nb::overload_cast<>(&ComputeTask::get_read, nb::const_),
           nb::rv_policy::reference_internal)
      .def("get_write", nb::overload_cast<>(&ComputeTask::get_write, nb::const_),
           nb::rv_policy::reference_internal)
      .def("get_data_dependencies",
           nb::overload_cast<>(&ComputeTask::get_data_dependencies, nb::const_),
           nb::rv_policy::reference_internal)
      .def("get_data_dependents",
           nb::overload_cast<>(&ComputeTask::get_data_dependents, nb::const_),
           nb::rv_policy::reference_internal)
      .def("get_supported_architectures", &ComputeTask::get_supported_architectures,
           nb::rv_policy::reference_internal)
      .def("get_tag", &ComputeTask::get_tag)
      .def("get_type", &ComputeTask::get_type)
      .def("get_variant", nb::overload_cast<DeviceType>(&ComputeTask::get_variant, nb::const_),
           "arch"_a, nb::rv_policy::reference_internal)
      .def("get_variants", nb::overload_cast<>(&ComputeTask::get_variant_vector, nb::const_),
           nb::rv_policy::copy)
      .def(
          "__str__",
          [](const ComputeTask &t) { return "ComputeTask(id=" + std::to_string(t.get_id()) + ")"; })
      .def("__repr__", [](const ComputeTask &t) {
        return "ComputeTask(id=" + std::to_string(t.get_id()) + ")";
      });

  nb::class_<DataTask, Task>(m, "DataTask")
      .def(nb::init<taskid_t>(), "id"_a)
      .def("get_compute_task", &DataTask::get_compute_task, nb::rv_policy::reference_internal)
      .def("get_data_id", &DataTask::get_data_id)
      .def("__str__",
           [](const DataTask &t) {
             return "DataTask(id=" + std::to_string(t.get_id()) +
                    ", block=" + std::to_string(t.get_data_id()) + ")";
           })
      .def("__repr__", [](const DataTask &t) {
        return "DataTask(id=" + std::to_string(t.get_id()) +
               ", block=" + std::to_string(t.get_data_id()) + ")";
      });

  nb::class_<Tasks>(m, "Tasks")
      .def(nb::init<taskid_t>(), "num_tasks"_a)
      .def("create_compute_task", &Tasks::create_compute_task, "id"_a, "name"_a, "dependencies"_a)
      .def("set_read", &Tasks::set_read, "id"_a, "dataids"_a)
      .def("set_write", &Tasks::set_write, "id"_a, "dataids"_a)
      .def("set_retire", &Tasks::set_retire, "id"_a, "dataids"_a)
      .def("add_variant",
           nb::overload_cast<taskid_t, DeviceType, vcu_t, mem_t, timecount_t>(&Tasks::add_variant),
           "id"_a, "arch"_a, "vcu"_a, "mem"_a, "time"_a)
      .def("get_name", nb::overload_cast<taskid_t>(&Tasks::get_name, nb::const_), "id"_a,
           nb::rv_policy::reference_internal)
      .def("get_depth", nb::overload_cast<taskid_t>(&Tasks::get_depth, nb::const_), "id"_a)
      .def("get_data_id", nb::overload_cast<taskid_t>(&Tasks::get_data_id, nb::const_), "id"_a)
      .def("get_data_dependencies",
           nb::overload_cast<taskid_t>(&Tasks::get_data_dependencies, nb::const_), "id"_a,
           nb::rv_policy::reference_internal)
      .def("get_data_dependents",
           nb::overload_cast<taskid_t>(&Tasks::get_data_dependents, nb::const_), "id"_a,
           nb::rv_policy::reference_internal)
      .def("get_dependencies", nb::overload_cast<taskid_t>(&Tasks::get_dependencies, nb::const_),
           "id"_a, nb::rv_policy::reference_internal)
      .def("get_dependents", nb::overload_cast<taskid_t>(&Tasks::get_dependents, nb::const_),
           "id"_a, nb::rv_policy::reference_internal)
      .def("get_read", nb::overload_cast<taskid_t>(&Tasks::get_read, nb::const_), "id"_a,
           nb::rv_policy::reference_internal)
      .def("get_write", nb::overload_cast<taskid_t>(&Tasks::get_write, nb::const_), "id"_a,
           nb::rv_policy::reference_internal)
      .def("is_compute", &Tasks::is_compute, "id"_a)
      .def("is_data", &Tasks::is_data, "id"_a)
      .def("size", &Tasks::size)
      .def("compute_size", &Tasks::compute_size)
      .def("data_size", &Tasks::data_size)
      .def("empty", &Tasks::empty)
      .def("get_compute_task", nb::overload_cast<taskid_t>(&Tasks::get_compute_task, nb::const_),
           "id"_a, nb::rv_policy::reference_internal)
      .def("get_data_task", nb::overload_cast<taskid_t>(&Tasks::get_data_task, nb::const_), "id"_a,
           nb::rv_policy::reference_internal)
      .def("get_compute_tasks", nb::overload_cast<>(&Tasks::get_compute_tasks, nb::const_),
           nb::rv_policy::reference_internal)
      .def("get_data_tasks", nb::overload_cast<>(&Tasks::get_data_tasks, nb::const_),
           nb::rv_policy::reference_internal)
      // .def("get_task_resources", nb::overload_cast<taskid_t>(&Tasks::get_task_resources,
      // nb::const_), "id"_a, nb::rv_policy::reference_internal) .def("get_task_resources",
      // nb::overload_cast<taskid_t, DeviceType>(&Tasks::get_task_resources, nb::const_), "id"_a,
      // "arch"_a, nb::rv_policy::reference_internal)
      .def("get_task", nb::overload_cast<taskid_t>(&Tasks::get_task, nb::const_), "id"_a,
           nb::rv_policy::reference_internal);
}
