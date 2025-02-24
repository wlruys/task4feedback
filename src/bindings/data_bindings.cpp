#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "data_manager.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_data_ext(nb::module_& m) {
    nb::class_<Data>(m, "Data")
        .def(nb::init<>())
        .def(nb::init<size_t>(), "num_blocks"_a)
        .def("create_block", &Data::create_block, "id"_a, "size"_a, "location"_a, "name"_a)
        .def("set_size", &Data::set_size, "id"_a, "size"_a)
        .def("set_location", &Data::set_location, "id"_a, "location"_a)
        .def("set_name", &Data::set_name, "id"_a, "name"_a)
        .def("get_size", &Data::get_size, "id"_a)
        .def("get_location", &Data::get_location, "id"_a)
        .def("get_name", &Data::get_name, "id"_a);

    nb::class_<ValidEventArray>(m, "ValidEventArray")
        .def_ro("size", &ValidEventArray::size)
        .def_ro("starts", &ValidEventArray::starts)
        .def_ro("stops", &ValidEventArray::stops);
}
