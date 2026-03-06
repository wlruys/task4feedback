#include "action.hpp"
#include "nbh.hpp"
#include "scheduler.hpp"
#include "tasks.hpp"
#include <cstdint>

namespace nb = nanobind;
using namespace nb::literals;

void init_scheduler_ext(nb::module_ &m) {

  nb::class_<SchedulerState>(m, "SchedulerState")
      .def("get_global_time", &SchedulerState::get_global_time)
      .def("get_mapping_priority", &SchedulerState::get_mapping_priority, "task_id"_a,
           nb::rv_policy::reference_internal)
      .def("get_reserving_priority", &SchedulerState::get_reserving_priority, "task_id"_a,
           nb::rv_policy::reference_internal)
      .def("get_launching_priority", &SchedulerState::get_launching_priority, "task_id"_a,
           nb::rv_policy::reference_internal)
      .def(
          "get_task_runtime",
          [](const SchedulerState &self) -> const auto & { return self.get_task_runtime(); },
          nb::rv_policy::reference_internal)
      .def(
          "get_tasks", [](const SchedulerState &self) -> const auto & { return self.get_tasks(); },
          nb::rv_policy::reference_internal);

  nb::class_<HysteresisTransitionConditions>(m, "HysteresisTransitionConditions")
      .def(nb::init<>())
      .def(nb::init<int32_t, int32_t, int32_t>(), "open_in_flight"_a, "close_in_flight"_a,
           "starvation_threshold"_a)
      .def_ro("open_in_flight", &HysteresisTransitionConditions::open_in_flight)
      .def_ro("close_in_flight", &HysteresisTransitionConditions::close_in_flight)
      .def_ro("starvation_threshold", &HysteresisTransitionConditions::starvation_threshold)
      .def_ro("last_window_opened", &HysteresisTransitionConditions::last_window_opened)
      .def_ro("window_open", &HysteresisTransitionConditions::window_open);
}
