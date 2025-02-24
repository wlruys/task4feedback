#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h> 
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/bind_map.h>

#include "settings.hpp"
#include "communication_manager.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_topology_ext(nb::module_& m) {

    nb::class_<Topology>(m, "Topology")
        .def(nb::init<devid_t>(), "num_devices"_a)
        .def("set_bandwidth", &Topology::set_bandwidth, "src"_a, "dst"_a, "bandwidth"_a)
        .def("set_max_connections", &Topology::set_max_connections, "src"_a, "dst"_a, "max_links"_a)
        .def("set_latency", &Topology::set_latency, "src"_a, "dst"_a, "latency"_a)
        .def("get_latency", &Topology::get_latency, "src"_a, "dst"_a)
        .def("get_bandwidth", &Topology::get_bandwidth, "src"_a, "dst"_a)
        .def("get_max_connections", &Topology::get_max_connections, "src"_a, "dst"_a);

   nb::class_<CommunicationStats>(m, "CommunicationStats")
        .def(nb::init<>())  // Default constructor
        .def(nb::init<timecount_t, mem_t>(), "latency"_a, "bandwidth"_a) // Constructor with latency and bandwidth
        .def_rw("latency", &CommunicationStats::latency)
        .def_rw("bandwidth", &CommunicationStats::bandwidth);

    nb::class_<CommunicationRequest>(m, "CommunicationRequest")
        .def(nb::init<>()) // Add default constructor
        .def(nb::init<taskid_t, devid_t, devid_t, mem_t>(), "data_task_id"_a, "source"_a, "destination"_a, "size"_a)
        .def_rw("data_task_id", &CommunicationRequest::data_task_id)
        .def_rw("source", &CommunicationRequest::source)
        .def_rw("destination", &CommunicationRequest::destination)
        .def_rw("size", &CommunicationRequest::size);
        
    nb::class_<CommunicationNoise>(m, "CommunicationNoise")
        .def(nb::init<Topology &, unsigned int>(), "topology"_a, "seed"_a = 0)
        .def("get", &CommunicationNoise::get, "request"_a, nb::rv_policy::reference_internal)
        .def("set",
            nb::overload_cast<const CommunicationRequest &, const CommunicationStats &>(
                &CommunicationNoise::set
            ), "request"_a, "stats"_a
        )
        .def("dump_to_binary", &CommunicationNoise::dump_to_binary, "filename"_a)
        .def("load_from_binary", &CommunicationNoise::load_from_binary, "filename"_a);

    nb::class_<UniformCommunicationNoise, CommunicationNoise>(
        m, "UniformCommunicationNoise"
    )
        .def(nb::init<Topology &, unsigned int>(), "topology"_a, "seed"_a = 0);


}