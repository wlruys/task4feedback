#include "communication.hpp"
#include "nbh.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_topology_ext(nb::module_ &m) {

  nb::class_<Topology>(m, "Topology")
      .def(nb::init<devid_t>(), "num_devices"_a)
      .def("set_bandwidth", &Topology::set_bandwidth, "src"_a, "dst"_a, "bandwidth"_a)
      .def("set_max_connections", &Topology::set_max_connections, "src"_a, "dst"_a, "max_links"_a)
      .def("set_latency", &Topology::set_latency, "src"_a, "dst"_a, "latency"_a)
      .def("get_latency", &Topology::get_latency, "src"_a, "dst"_a)
      .def("get_bandwidth", &Topology::get_bandwidth, "src"_a, "dst"_a)
      .def("get_max_connections", &Topology::get_max_connections, "src"_a, "dst"_a);
}