#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "graph.hpp"
#include "tasks.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_graph_ext(nb::module_& m) {
    nb::class_<GraphManager>(m, "GraphManager")
        .def_static("populate_dependents", &GraphManager::populate_dependents, "tasks"_a)
        .def_static("random_topological_sort", nb::overload_cast<Tasks&, long unsigned int>(&GraphManager::random_topological_sort), "tasks"_a, "seed"_a);

}


