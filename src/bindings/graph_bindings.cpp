#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/bind_map.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "graph.hpp"
#include "tasks.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_graph_ext(nb::module_& m) {
    nb::bind_vector<std::vector<GraphTemplate>>(m, "GraphTemplateVector");
    nb::class_<GraphManager>(m, "GraphManager")
        .def_static("finalize", &GraphManager::finalize, "tasks"_a, "create_data_tasks"_a= true, "add_missing_writers"_a= false);

    nb::class_<GraphTemplate>(m, "GraphTemplate")
        .def(nb::init<>())
        .def("add_task", &GraphTemplate::add_task)
        .def("get_name", &GraphTemplate::get_name, nb::rv_policy::copy)
        .def("add_dependency", &GraphTemplate::add_dependency)
        .def("add_dependencies", &GraphTemplate::add_dependencies)
        .def("get_dependencies", &GraphTemplate::get_dependencies, nb::rv_policy::reference_internal)
        .def("get_read_data", &GraphTemplate::get_read_data, nb::rv_policy::reference_internal)
        .def("get_write_data", &GraphTemplate::get_write_data, nb::rv_policy::reference_internal)
        .def("add_read_data", &GraphTemplate::add_read_data)
        .def("add_write_data", &GraphTemplate::add_write_data)
        .def("add_variant", &GraphTemplate::add_variant_info)
        .def("set_vcu", &GraphTemplate::set_vcu)
        .def("set_memory", &GraphTemplate::set_memory)
        .def("set_time", &GraphTemplate::set_time)
        .def("get_vcu", &GraphTemplate::get_vcu)
        .def("get_memory", &GraphTemplate::get_memory)
        .def("get_time", &GraphTemplate::get_time)
        .def("get_tag", &GraphTemplate::get_tag)
        .def("set_tag", &GraphTemplate::set_tag)
        .def("get_id", &GraphTemplate::get_id)
        .def("get_type", &GraphTemplate::get_type)
        .def("set_type", &GraphTemplate::set_type)
        .def("size", &GraphTemplate::size)
        .def("to_tasks", &GraphTemplate::to_tasks);

}


