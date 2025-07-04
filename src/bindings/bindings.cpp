#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "devices.hpp"
#include "resources.hpp"
#include "settings.hpp"
#include "tasks.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_event_ext(nb::module_ &);
void init_device_ext(nb::module_ &);
void init_task_ext(nb::module_ &);
void init_data_ext(nb::module_ &);
void init_graph_ext(nb::module_ &);
void init_task_noise_ext(nb::module_ &);
void init_topology_ext(nb::module_ &);
void init_scheduler_ext(nb::module_ &);
void init_mapper_ext(nb::module_ &);
void init_observer_ext(nb::module_ &);
void init_simulator_ext(nb::module_ &);

NB_MODULE(fastsim2, m) {
  nb::set_leak_warnings(false);
  nb::bind_vector<std::vector<uint32_t>>(m, "UInt32Vector");
  nb::bind_vector<std::vector<uint64_t>>(m, "UInt64Vector");
  nb::bind_vector<std::vector<int32_t>>(m, "Int32Vector");
  nb::bind_vector<std::vector<int64_t>>(m, "Int64Vector");
  nb::bind_vector<std::vector<float>>(m, "FloatVector");
  nb::bind_vector<std::vector<Task>>(m, "TaskVector");
  nb::bind_vector<std::vector<ComputeTask>>(m, "ComputeTaskVector");
  nb::bind_vector<std::vector<DataTask>>(m, "DataTaskVector");
  nb::bind_vector<std::vector<DeviceType>>(m, "DeviceTypeVector");
  init_event_ext(m);
  init_device_ext(m);
  init_task_ext(m);
  init_data_ext(m);
  init_graph_ext(m);
  init_task_noise_ext(m);
  init_topology_ext(m);
  init_scheduler_ext(m);
  init_mapper_ext(m);
  init_observer_ext(m);
  init_simulator_ext(m);
}
