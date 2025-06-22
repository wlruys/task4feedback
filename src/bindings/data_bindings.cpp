#include "data_manager.hpp"
#include "nbh.hpp"
namespace nb = nanobind;
using namespace nb::literals;
 
void init_data_ext(nb::module_ &m) {
  nb::class_<Data>(m, "Data")
      .def(nb::init<>())
      .def(nb::init<size_t>(), "num_blocks"_a)
      .def("size", &Data::size)
      .def("create_block", &Data::create_block, "id"_a, "size"_a, "location"_a, "name"_a)
      .def("append_block", &Data::append_block, "size"_a, "location"_a, "name"_a)
      .def("set_size", &Data::set_size, "id"_a, "size"_a)
      .def("set_location", &Data::set_location, "id"_a, "location"_a)
      .def("set_name", &Data::set_name, "id"_a, "name"_a)
      .def("set_x_pos", &Data::set_x_pos, "id"_a, "x"_a)
      .def("set_y_pos", &Data::set_y_pos, "id"_a, "y"_a)
      .def("get_x_pos", &Data::get_x_pos, "id"_a)
      .def("get_y_pos", &Data::get_y_pos, "id"_a)
      .def("get_x_pos_vec", &Data::get_x_pos, "id"_a)
      .def("get_y_pos_vec", &Data::get_y_pos, "id"_a)
      .def("get_size", &Data::get_size, "id"_a)
      .def("get_location", &Data::get_location, "id"_a)
      .def("get_name", &Data::get_name, "id"_a)
      .def("set_type", &Data::set_type, "id"_a, "type"_a)
      .def("get_type", &Data::get_type, "id"_a)
      .def("set_tag", &Data::set_tag, "id"_a, "tag"_a)
      .def("get_tag", &Data::get_tag, "id"_a)
      .def("get_id", &Data::get_id, "name"_a);

  nb::class_<ValidEventArray>(m, "ValidEventArray")
      .def_ro("size", &ValidEventArray::size)
      .def_ro("starts", &ValidEventArray::starts)
      .def_ro("stops", &ValidEventArray::stops);
}
