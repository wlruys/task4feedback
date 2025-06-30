#include "action.hpp"
#include "nbh.hpp"
#include "scheduler.hpp"
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
           nb::rv_policy::reference_internal);

  nb::class_<TransitionConditions>(m, "TransitionConditions")
      .def("should_map", &TransitionConditions::should_map)
      .def("should_reserve", &TransitionConditions::should_reserve)
      .def("should_launch", &TransitionConditions::should_launch);

  nb::class_<DefaultTransitionConditions, TransitionConditions>(m, "DefaultTransitionConditions")
      .def(nb::init<>());

  nb::class_<RangeTransitionConditions, TransitionConditions>(m, "RangeTransitionConditions")
      .def(nb::init<std::size_t, std::size_t, std::size_t>(), "mapped_reserved_gap_"_a,
           "reserved_launched_gap"_a, "total_in_flight_"_a)
      .def_ro("mapped_reserved_gap", &RangeTransitionConditions::mapped_reserved_gap)
      .def_ro("reserved_launched_gap", &RangeTransitionConditions::reserved_launched_gap)
      .def_ro("total_in_flight", &RangeTransitionConditions::total_in_flight);

  nb::class_<BatchTransitionConditions, TransitionConditions>(m, "BatchTransitionConditions")
      .def(nb::init<std::size_t, std::size_t, std::size_t>(), "batch_size"_a, "queue_threshold"_a,
           "max_in_flight"_a)
      .def_ro("batch_size", &BatchTransitionConditions::batch_size)
      .def_ro("queue_threshold", &BatchTransitionConditions::queue_threshold)
      .def_ro("max_in_flight", &BatchTransitionConditions::max_in_flight)
      .def_ro("last_accessed", &BatchTransitionConditions::last_accessed)
      .def_ro("active_batch", &BatchTransitionConditions::active_batch);
}