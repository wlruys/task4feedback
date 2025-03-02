#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>

#include "devices.hpp"
#include "noise.hpp"
#include "resources.hpp"
#include "tasks.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_task_noise_ext(nb::module_ &m) {
  nb::class_<TaskNoise>(m, "TaskNoise")
      .def(nb::init<Tasks &, unsigned int, unsigned int>(), "tasks"_a, "seed"_a = 0, "pseed"_a = 0)
      .def("set_seed", &TaskNoise::set_seed, "seed"_a)
      .def("set_pseed", &TaskNoise::set_pseed, "pseed"_a)
      .def("get", &TaskNoise::get, "task_id"_a, "arch"_a)
      .def("set", nb::overload_cast<taskid_t, DeviceType, timecount_t>(&TaskNoise::set),
           "task_id"_a, "arch"_a, "noise"_a)
      .def("set", nb::overload_cast<std::vector<timecount_t>>(&TaskNoise::set), "noise"_a)
      .def("set_priority", nb::overload_cast<taskid_t, priority_t>(&TaskNoise::set_priority),
           "task_id"_a, "p"_a)
      .def("set_priority", nb::overload_cast<std::vector<priority_t>>(&TaskNoise::set_priority),
           "noise"_a)
      .def("randomize_duration", &TaskNoise::generate_duration)
      .def("randomize_priority", &TaskNoise::generate_priority)
      .def("dump_to_binary", &TaskNoise::dump_to_binary, "filename"_a)
      .def("load_from_binary", &TaskNoise::load_from_binary, "filename"_a)
      .def("dump_priorities_to_binary", &TaskNoise::dump_priorities_to_binary, "filename"_a)
      .def("load_priorities_from_binary", &TaskNoise::load_priorities_from_binary, "filename"_a);

  nb::class_<LognormalTaskNoise>(m, "LognormalTaskNoise")
      .def(nb::init<Tasks &, unsigned int, unsigned int>(), "tasks"_a, "seed"_a = 0, "pseed"_a = 0);
}
