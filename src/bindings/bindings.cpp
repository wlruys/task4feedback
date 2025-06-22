
#include "nbh.hpp"

#include "data_bindings.cpp"
#include "device_bindings.cpp"
#include "event_bindings.cpp"
#include "graph_bindings.cpp"
#include "mapper_bindings.cpp"
#include "observer_bindings.cpp"
#include "scheduler_bindings.cpp"
#include "simulator_bindings.cpp"
#include "task_bindings.cpp"
#include "task_noise_bindings.cpp"
#include "topology_bindings.cpp"
#include <cstdint>

namespace nb = nanobind;
using namespace nb::literals;

// void init_event_ext(nb::module_ &);
// void init_device_ext(nb::module_ &);
// void init_task_ext(nb::module_ &);
// void init_data_ext(nb::module_ &);
// void init_graph_ext(nb::module_ &);
// void init_task_noise_ext(nb::module_ &);
// void init_topology_ext(nb::module_ &);
// void init_scheduler_ext(nb::module_ &);
// void init_mapper_ext(nb::module_ &);
// void init_observer_ext(nb::module_ &);
// void init_simulator_ext(nb::module_ &);

NB_MODULE(fastsim2, m) {
  // nb::set_leak_warnings(false);
  nb::bind_vector<std::vector<uint32_t>>(m, "UInt32Vector");
  nb::bind_vector<std::vector<uint64_t>>(m, "UInt64Vector");
  nb::bind_vector<std::vector<int32_t>>(m, "Int32Vector");
  nb::bind_vector<std::vector<int64_t>>(m, "Int64Vector");
  nb::bind_vector<std::vector<float>>(m, "FloatVector");
  nb::bind_vector<std::vector<double>>(m, "DoubleVector");
  nb::bind_vector<std::vector<std::string>>(m, "StringVector");
  // init_event_ext(m);
  // init_device_ext(m);
  // init_task_ext(m);
  // init_data_ext(m);
  // init_graph_ext(m);
  // init_task_noise_ext(m);
  // init_topology_ext(m);
  // init_scheduler_ext(m);
  // init_mapper_ext(m);
  // //init_observer_ext(m);
  // init_simulator_ext(m);
}
