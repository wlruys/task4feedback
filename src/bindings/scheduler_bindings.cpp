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

  nb::class_<BatchTransitionConditions>(m, "BatchTransitionConditions")
      .def(nb::init<>())
      .def(nb::init<int32_t, int32_t, int32_t>(), "batch_size"_a, "queue_threshold"_a,
           "max_in_flight"_a)
      .def_ro("batch_size", &BatchTransitionConditions::batch_size)
      .def_ro("queue_threshold", &BatchTransitionConditions::queue_threshold)
      .def_ro("max_in_flight", &BatchTransitionConditions::max_in_flight)
      .def_ro("last_accessed", &BatchTransitionConditions::last_accessed)
      .def_ro("active_batch", &BatchTransitionConditions::active_batch);
}
