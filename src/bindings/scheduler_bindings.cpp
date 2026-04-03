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
      .def("should_reserve", &TransitionConditions::should_reserve)
      .def("should_launch", &TransitionConditions::should_launch);

  nb::class_<DefaultTransitionConditions, TransitionConditions>(m, "DefaultTransitionConditions")
      .def(nb::init<>());

  nb::class_<RangeTransitionConditions, TransitionConditions>(m, "RangeTransitionConditions")
      .def(nb::init<int32_t, int32_t, int32_t>(), "mapped_reserved_gap"_a,
           "reserved_launched_gap"_a, "total_in_flight"_a)
      .def_ro("mapped_reserved_gap", &RangeTransitionConditions::mapped_reserved_gap)
      .def_ro("reserved_launched_gap", &RangeTransitionConditions::reserved_launched_gap)
      .def_ro("total_in_flight", &RangeTransitionConditions::total_in_flight);

  nb::class_<BatchTransitionConditions, TransitionConditions>(m, "BatchTransitionConditions")
      .def(nb::init<int32_t, int32_t, int32_t>(), "batch_size"_a, "queue_threshold"_a,
           "max_in_flight"_a)
      .def_ro("batch_size", &BatchTransitionConditions::batch_size)
      .def_ro("queue_threshold", &BatchTransitionConditions::queue_threshold)
      .def_ro("max_in_flight", &BatchTransitionConditions::max_in_flight)
      .def_ro("last_accessed", &BatchTransitionConditions::last_accessed)
      .def_ro("active_batch", &BatchTransitionConditions::active_batch);

  nb::class_<DeviceThresholdTransitionConditions, TransitionConditions>(
      m, "DeviceThresholdTransitionConditions")
      .def(nb::init<>())
      .def(nb::init<int32_t, int32_t>(), "mapped_threshold"_a, "reserved_threshold"_a)
      .def_prop_ro("mapped_threshold", &DeviceThresholdTransitionConditions::get_mapped_threshold)
      .def_prop_ro("reserved_threshold",
                   &DeviceThresholdTransitionConditions::get_reserved_threshold);
}
