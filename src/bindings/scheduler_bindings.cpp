#include "action.hpp"
#include "scheduler.hpp"
#include "settings.hpp"
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

void init_scheduler_ext(nb::module_ &m) {

  nb::class_<SchedulerState>(m, "SchedulerState")
      .def("get_global_time", &SchedulerState::get_global_time)
      .def("is_mapped", &SchedulerState::is_mapped, "task_id"_a)
      .def("is_reservable", &SchedulerState::is_reservable, "task_id"_a)
      .def("is_launched", &SchedulerState::is_launched, "task_id"_a)
      .def("is_mappable", &SchedulerState::is_mappable, "task_id"_a)
      .def("is_reservable", &SchedulerState::is_reservable, "task_id"_a)
      .def("is_launchable", &SchedulerState::is_launchable, "task_id"_a)
      .def("gather_graph_statistics", &SchedulerState::gather_graph_statistics, "device_types"_a)
      .def("get_mappings", &SchedulerState::get_mappings, nb::rv_policy::reference_internal)
      .def("get_mapping_priority", &SchedulerState::get_mapping_priority, "task_id"_a,
           nb::rv_policy::reference_internal)
      .def("get_reserving_priority", &SchedulerState::get_reserving_priority, "task_id"_a,
           nb::rv_policy::reference_internal)
      .def("get_launching_priority", &SchedulerState::get_launching_priority, "task_id"_a,
           nb::rv_policy::reference_internal)
      .def("get_mapping", &SchedulerState::get_mapping, "task_id"_a)
      .def("get_execution_time", &SchedulerState::get_execution_time, "task_id"_a)
      .def("get_task_name", &SchedulerState::get_task_name, "task_id"_a, nb::rv_policy::copy)
      .def("get_mapped_time", &SchedulerState::get_mapped_time, "task_id"_a)
      .def("get_reserved_time", &SchedulerState::get_reserved_time, "task_id"_a)
      .def("get_launched_time", &SchedulerState::get_launched_time, "task_id"_a)
      .def("get_completed_time", &SchedulerState::get_completed_time, "task_id"_a)
      .def("track_resource_guard", &SchedulerState::track_resource_guard)
      .def("track_location_guard", &SchedulerState::track_location_guard)
      .def("get_mapped_vcu_at", &SchedulerState::get_mapped_vcu_at, "device_id"_a, "time"_a)
      .def("get_reserved_vcu_at", &SchedulerState::get_reserved_vcu_at, "device_id"_a, "time"_a)
      .def("get_launched_vcu_at", &SchedulerState::get_launched_vcu_at, "device_id"_a, "time"_a)
      .def("get_mapped_mem_at", &SchedulerState::get_mapped_mem_at, "device_id"_a, "time"_a)
      .def("get_reserved_mem_at", &SchedulerState::get_reserved_mem_at, "device_id"_a, "time"_a)
      .def("get_launched_mem_at", &SchedulerState::get_launched_mem_at, "device_id"_a, "time"_a)
      .def("get_mapped_vcu_events", &SchedulerState::get_mapped_vcu_events, "device_id"_a)
      .def("get_reserved_vcu_events", &SchedulerState::get_reserved_vcu_events, "device_id"_a)
      .def("get_launched_vcu_events", &SchedulerState::get_launched_vcu_events, "device_id"_a)
      .def("get_mapped_mem_events", &SchedulerState::get_mapped_mem_events, "device_id"_a)
      .def("get_reserved_mem_events", &SchedulerState::get_reserved_mem_events, "device_id"_a)
      .def("get_launched_mem_events", &SchedulerState::get_launched_mem_events, "device_id"_a)
      .def("get_state_at", &SchedulerState::get_state_at, "task_id"_a, "time"_a)
      .def("is_data_task_virtual", &SchedulerState::is_data_task_virtual, "task_id"_a)
      .def("get_data_task_source", &SchedulerState::get_data_task_source, "task_id"_a)
      .def("get_valid_intervals_mapped", &SchedulerState::get_valid_intervals_mapped, "data_id"_a,
           "device_id"_a)
      .def("get_valid_intervals_reserved", &SchedulerState::get_valid_intervals_reserved,
           "data_id"_a, "device_id"_a)
      .def("get_valid_intervals_launched", &SchedulerState::get_valid_intervals_launched,
           "data_id"_a, "device_id"_a)
      .def("check_valid_mapped_at", &SchedulerState::check_valid_mapped_at, "data_id"_a,
           "device_id"_a, "query_time"_a)
      .def("check_valid_reserved_at", &SchedulerState::check_valid_reserved_at, "data_id"_a,
           "device_id"_a, "query_time"_a)
      .def("check_valid_launched_at", &SchedulerState::check_valid_launched_at, "data_id"_a,
           "device_id"_a, "query_time"_a);

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