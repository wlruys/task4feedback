#include "devices.hpp"
#include "nbh.hpp"
#include "resources.hpp"
#include "tasks.hpp"
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
      .def("get_mean_duration", &Variant::get_mean_duration)
      .def("__str__",
           [](const Variant &v) {
             return "Variant(arch=" + std::to_string(static_cast<int>(v.get_arch())) +
                    ", vcu=" + std::to_string(v.get_vcus()) +
                    ", mem=" + std::to_string(v.get_mem()) +
                    ", time=" + std::to_string(v.get_mean_duration()) + ")";
           })
      .def("__repr__", [](const Variant &v) {
        return "Variant(arch=" + std::to_string(static_cast<int>(v.get_arch())) +
               ", vcu=" + std::to_string(v.get_vcus()) + ", mem=" + std::to_string(v.get_mem()) +
               ", time=" + std::to_string(v.get_mean_duration()) + ")";
      });

  nb::class_<Graph>(m, "Graph")
      .def(nb::init<>())
      .def("add_task", &Graph::add_task, "name"_a)
      .def("add_read_data", &Graph::add_read_data, "id"_a, "read_data"_a)
      .def("add_write_data", &Graph::add_write_data, "id"_a, "write_data"_a)
      .def("add_retire_data", &Graph::add_retire_data, "id"_a, "retire_data"_a)
      .def("set_tag", &Graph::set_tag, "id"_a, "tag"_a)
      .def("set_type", &Graph::set_type, "id"_a, "type"_a)
      .def("add_dependency", &Graph::add_dependency, "from_task"_a, "to_task"_a)
      .def("add_dependencies", &Graph::add_dependencies, "from_task"_a, "to_tasks"_a)
      .def("set_variant", &Graph::set_variant, "id"_a, "arch"_a, "vcu"_a, "mem"_a, "time"_a)
      .def("get_n_compute_tasks", &Graph::get_n_compute_tasks)
      .def("get_n_data_tasks", &Graph::get_n_data_tasks)
      .def("finalize", &Graph::finalize, "ensure_dependencies"_a = false,
           "create_data_tasks"_a = true);

  nb::class_<StaticTaskInfo>(m, "StaticTaskInfo")
      .def(nb::init<Graph &>(), "graph"_a, nb::keep_alive<1, 2>());

  nb::class_<RuntimeTaskInfo>(m, "RuntimeTaskInfo");
}
