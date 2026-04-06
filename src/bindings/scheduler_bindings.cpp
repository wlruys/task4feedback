#include "action.hpp"
#include "nbh.hpp"
#include "scheduler.hpp"
#include "tasks.hpp"
#include <cstdint>

namespace nb = nanobind;
using namespace nb::literals;

void init_scheduler_ext(nb::module_ &m) {

  nb::class_<TaskDevicePhaseInfo>(m, "TaskDevicePhaseInfo")
      .def("get_mapped_tasks",
           [](const TaskDevicePhaseInfo &self, devid_t device_id) {
             return as_sorted_vector(self.get_mapped_tasks(device_id));
           },
           "device_id"_a)
      .def("get_reserved_tasks",
           [](const TaskDevicePhaseInfo &self, devid_t device_id) {
             return as_sorted_vector(self.get_reserved_tasks(device_id));
           },
           "device_id"_a)
      .def("has_mapped", &TaskDevicePhaseInfo::has_mapped, "task_id"_a, "device_id"_a)
      .def("has_reserved", &TaskDevicePhaseInfo::has_reserved, "task_id"_a, "device_id"_a)
      .def("size", &TaskDevicePhaseInfo::size);

  nb::class_<TaskDataUsageInfo>(m, "TaskDataUsageInfo")
      .def("size", &TaskDataUsageInfo::size)
      .def("get_mapped_usage", &TaskDataUsageInfo::get_mapped_usage, "data_id"_a)
      .def("get_reserved_usage", &TaskDataUsageInfo::get_reserved_usage, "data_id"_a);

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
          nb::rv_policy::reference_internal)
      .def("enable_task_device_phase_info", &SchedulerState::enable_task_device_phase_info,
           "expected_tasks_per_device"_a = 0)
      .def("disable_task_device_phase_info", &SchedulerState::disable_task_device_phase_info)
      .def("has_task_device_phase_info", &SchedulerState::has_task_device_phase_info)
      .def(
          "get_task_device_phase_info",
          [](const SchedulerState &self) -> const TaskDevicePhaseInfo * {
            return self.get_task_device_phase_info();
          },
          nb::rv_policy::reference_internal)
      .def("enable_task_data_usage_info", &SchedulerState::enable_task_data_usage_info)
      .def("disable_task_data_usage_info", &SchedulerState::disable_task_data_usage_info)
      .def("has_task_data_usage_info", &SchedulerState::has_task_data_usage_info)
      .def(
          "get_task_data_usage_info",
          [](const SchedulerState &self) -> const TaskDataUsageInfo * {
            return self.get_task_data_usage_info();
          },
          nb::rv_policy::reference_internal);

  nb::class_<TransitionConditions>(m, "TransitionConditions")
      .def("should_map", &TransitionConditions::should_map)
      .def("update_map", &TransitionConditions::update_map)
      .def("should_reserve", &TransitionConditions::should_reserve)
      .def("should_launch", &TransitionConditions::should_launch)
      .def("should_launch_data", &TransitionConditions::should_launch_data);

  nb::class_<DefaultTransitionConditions, TransitionConditions>(m, "DefaultTransitionConditions")
      .def(nb::init<>());

  nb::class_<RangeTransitionConditions, TransitionConditions>(m, "RangeTransitionConditions")
        .def(nb::init<int32_t, int32_t, int32_t>(), "mapped_reserved_gap"_a = 1,
              "reserved_launched_gap"_a = 1, "total_in_flight"_a = 1)
      .def_rw("mapped_reserved_gap", &RangeTransitionConditions::mapped_reserved_gap)
      .def_rw("reserved_launched_gap", &RangeTransitionConditions::reserved_launched_gap)
      .def_rw("total_in_flight", &RangeTransitionConditions::total_in_flight);

  nb::class_<BatchTransitionConditions, TransitionConditions>(m, "BatchTransitionConditions")
        .def(nb::init<int32_t, int32_t, int32_t>(), "batch_size"_a = 20,
              "queue_threshold"_a = 2, "max_in_flight"_a = 16)
      .def_rw("batch_size", &BatchTransitionConditions::batch_size)
      .def_rw("queue_threshold", &BatchTransitionConditions::queue_threshold)
      .def_rw("max_in_flight", &BatchTransitionConditions::max_in_flight)
      .def_ro("last_accessed", &BatchTransitionConditions::last_accessed)
      .def_ro("active_batch", &BatchTransitionConditions::active_batch);

  nb::class_<PlannedThresholdTransitionConditions, TransitionConditions>(
      m, "PlannedThresholdTransitionConditions")
      .def(nb::init<int32_t, int64_t>(), "planned_threshold"_a = 1,
           "max_reserved_threshold"_a = 16)
      .def_rw("planned_threshold", &PlannedThresholdTransitionConditions::planned_threshold)
      .def_rw("max_reserved_threshold",
              &PlannedThresholdTransitionConditions::max_reserved_threshold);

  nb::class_<HysteresisTransitionConditions, TransitionConditions>(
      m, "HysteresisTransitionConditions")
      .def(nb::init<>())
      .def(nb::init<int32_t, int32_t, int32_t>(), "open_in_flight"_a, "close_in_flight"_a,
           "starvation_threshold"_a)
      .def_rw("open_in_flight", &HysteresisTransitionConditions::open_in_flight)
      .def_rw("close_in_flight", &HysteresisTransitionConditions::close_in_flight)
      .def_rw("starvation_threshold", &HysteresisTransitionConditions::starvation_threshold)
      .def_ro("last_window_opened", &HysteresisTransitionConditions::last_window_opened)
      .def_ro("window_open", &HysteresisTransitionConditions::window_open);
}
