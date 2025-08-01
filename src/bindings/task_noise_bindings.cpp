#include "devices.hpp"
#include "nbh.hpp"
#include "noise.hpp"
#include "resources.hpp"
#include "tasks.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_task_noise_ext(nb::module_ &m) {
  nb::class_<TaskNoise>(m, "TaskNoise")
      .def(nb::init<StaticTaskInfo &, unsigned int, unsigned int>(), "tasks"_a, "seed"_a = 0,
           "pseed"_a = 0, nb::keep_alive<1, 2>()) // Keep tasks object alive as long as noise object exists
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
      .def("get_priorities", nb::overload_cast<>(&TaskNoise::get_priorities, nb::const_),
           nb::rv_policy::reference_internal)
      .def("get_durations", nb::overload_cast<>(&TaskNoise::get_durations, nb::const_),
           nb::rv_policy::reference_internal)
      .def("randomize_duration", &TaskNoise::generate_duration)
      .def("randomize_priority", &TaskNoise::generate_priority);

  nb::class_<LognormalTaskNoise>(m, "LognormalTaskNoise")
      .def(nb::init<StaticTaskInfo &, unsigned int, unsigned int>(), "tasks"_a, "seed"_a = 0,
           "pseed"_a = 0, nb::keep_alive<1, 2>());
}
