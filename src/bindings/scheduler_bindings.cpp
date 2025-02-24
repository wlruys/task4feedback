#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h> 
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/bind_map.h>
#include <vector>
#include <string>
#include <cstdint>
#include "settings.hpp"
#include "scheduler.hpp"
#include "action.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_scheduler_ext(nb::module_& m) {
    
    nb::class_<SchedulerState>(m, "SchedulerState")
        .def("get_global_time", &SchedulerState::get_global_time)
        .def("is_mapped", &SchedulerState::is_mapped, "task_id"_a)
        .def("is_reservable", &SchedulerState::is_reservable, "task_id"_a)
        .def("is_launched", &SchedulerState::is_launched, "task_id"_a)
        .def("is_mappable", &SchedulerState::is_mappable, "task_id"_a)
        .def("is_reservable", &SchedulerState::is_reservable, "task_id"_a)
        .def("is_launchable", &SchedulerState::is_launchable, "task_id"_a)
        .def("get_mapping_priority", &SchedulerState::get_mapping_priority, "task_id"_a, nb::rv_policy::reference_internal)
        .def("get_reserving_priority", &SchedulerState::get_reserving_priority, "task_id"_a, nb::rv_policy::reference_internal)
        .def("get_launching_priority", &SchedulerState::get_launching_priority, "task_id"_a, nb::rv_policy::reference_internal)
        .def("get_mapping", &SchedulerState::get_mapping, "task_id"_a)
        .def("get_execution_time", &SchedulerState::get_execution_time, "task_id"_a)
        .def("get_task_name", &SchedulerState::get_task_name, "task_id"_a, nb::rv_policy::copy);

    nb::class_<TransitionConditions>(m, "TransitionConditions")
        .def("should_map", &TransitionConditions::should_map)
        .def("should_reserve", &TransitionConditions::should_reserve)
        .def("should_launch", &TransitionConditions::should_launch);
    

    nb::class_<DefaultTransitionConditions, TransitionConditions>(m, "DefaultTransitionConditions")
        .def(nb::init<> ());
    
    nb::class_<RangeTransitionConditions, TransitionConditions>(m, "RangeTransitionConditions")
        .def(nb::init<std::size_t, std::size_t, std::size_t>(), "mapped_reserved_gap_"_a, "reserved_launched_gap"_a, "total_in_flight_"_a)
        .def_ro("mapped_reserved_gap", &RangeTransitionConditions::mapped_reserved_gap)
        .def_ro("reserved_launched_gap", &RangeTransitionConditions::reserved_launched_gap)
        .def_ro("total_in_flight", &RangeTransitionConditions::total_in_flight);

    
}