#include "action.hpp"
#include "nbh.hpp"
#include "scheduler.hpp"
#include <cstdint>
#include <span>

namespace nb = nanobind;
using namespace nb::literals;

void init_mapper_ext(nb::module_ &m) {
  nb::bind_vector<std::vector<Action>>(m, "ActionVector");
  nb::class_<Action>(m, "Action")
      .def(nb::init<std::size_t, devid_t, priority_t, priority_t>())
      .def_rw("pos", &Action::pos)
      .def_rw("device", &Action::device)
      .def_rw("reservable_priority", &Action::reservable_priority)
      .def_rw("launchable_priority", &Action::launchable_priority)

      .def("__str__", [](const Action &a) {
        return "Action(pos=" + std::to_string(a.pos) + ", device=" + std::to_string(a.device) +
               ", res_pri=" + std::to_string(a.reservable_priority) +
               ", launch_pri=" + std::to_string(a.launchable_priority) + ")";
      });

  nb::class_<Mapper>(m, "Mapper")
      .def("map_task", &Mapper::map_task, "task_id"_a, "state"_a)
      .def("map_tasks",
           [](Mapper &m, const TaskIDList &tasks, const SchedulerState &state) -> ActionList & {
             return m.map_tasks(std::span<const taskid_t>(tasks), state);
           },
           "tasks"_a, "state"_a, nb::rv_policy::reference_internal);

  nb::class_<RandomMapper, Mapper>(m, "RandomMapper")
      .def(nb::init<>())
      .def(nb::init<RandomMapper &>(), "other"_a)
      .def(nb::init<unsigned int>(), "seed"_a);

  nb::class_<RoundRobinMapper, Mapper>(m, "RoundRobinMapper")
      .def(nb::init<>())
      .def(nb::init<RoundRobinMapper>(), "other"_a);

  nb::class_<StaticMapper, Mapper>(m, "StaticMapper")
      .def(nb::init<DeviceIDList>(), "device_ids_"_a)
      .def(nb::init<DeviceIDList, PriorityList, PriorityList>(), "device_ids"_a,
           "reserving_priorities"_a, "launching_priorities"_a)
      .def(nb::init<StaticMapper &>(), "other"_a)
      .def("set_mapping", &StaticMapper::set_mapping, "device_ids_"_a)
      .def("set_reserving_priorities", &StaticMapper::set_reserving_priorities,
           "reserving_priorites_"_a)
      .def("set_launching_priorities", &StaticMapper::set_launching_priorities,
           "launching_priorites_"_a);

  nb::class_<StaticActionMapper, Mapper>(m, "StaticActionMapper")
      .def(nb::init<ActionList>(), "actions"_a)
      .def(nb::init<StaticActionMapper &>(), "other"_a);

  nb::class_<DeviceTime>(m, "DeviceTime")
      .def(nb::init<devid_t, timecount_t>(), "device_id"_a, "time"_a)
      .def_ro("device_id", &DeviceTime::device_id)
      .def_ro("time", &DeviceTime::time);

  nb::class_<EFTMapper, Mapper>(m, "EFTMapper")
      .def(nb::init<>())
      .def(nb::init<std::size_t, std::size_t>(), "num_tasks"_a, "num_devices"_a)
      .def(nb::init<EFTMapper &>(), "other"_a)
      .def("get_best_device", &EFTMapper::get_best_device, "task_id"_a, "state"_a)
      .def("get_dependency_finish_time", &EFTMapper::get_dependency_finish_time, "task_id"_a,
           "state"_a)
      .def("get_device_available_time", &EFTMapper::get_device_available_time, "device_id"_a,
           "state"_a)
      .def("get_finish_time", &EFTMapper::get_finish_time, "task_id"_a, "device_id"_a, "start_t"_a,
           "state"_a)
      .def("time_for_transfer", &EFTMapper::time_for_transfer, "task_id"_a, "device_id"_a,
           "state"_a);

  nb::class_<MemoryAwareEFTMapper, EFTMapper>(m, "MemoryAwareEFTMapper")
      .def(nb::init<>())
      .def(nb::init<std::size_t, std::size_t, double>(), "num_tasks"_a, "num_devices"_a,
           "alpha"_a = 1.0)
      .def(nb::init<MemoryAwareEFTMapper &>(), "other"_a)
      .def_rw("alpha", &MemoryAwareEFTMapper::alpha);

  nb::class_<DequeueEFTMapper, EFTMapper>(m, "DequeueEFTMapper")
      .def(nb::init<>())
      .def(nb::init<std::size_t, std::size_t>(), "num_tasks"_a, "num_devices"_a)
      .def(nb::init<DequeueEFTMapper &>(), "other"_a);

  nb::class_<DataAwareMapper, Mapper>(m, "DataAwareMapper")
      .def(nb::init<>())
      .def(nb::init<std::size_t, std::size_t>(), "num_tasks"_a, "num_devices"_a)
      .def(nb::init<DataAwareMapper &>(), "other"_a)
      .def("map_task", &DataAwareMapper::map_task, "task_id"_a, "state"_a)
      .def(
          "map_tasks",
          [](DataAwareMapper &mapper, const TaskIDList &tasks,
             const SchedulerState &state) -> ActionList & {
            return mapper.map_tasks(std::span<const taskid_t>(tasks), state);
          },
          "tasks"_a, "state"_a, nb::rv_policy::reference_internal);

  nb::class_<KaHyParMapper, EFTMapper>(m, "KaHyParMapper")
      .def(nb::init<>())
      .def(nb::init<std::size_t, std::size_t>(), "num_tasks"_a, "num_devices"_a)
      .def(nb::init<KaHyParMapper &>(), "other"_a)
      .def_rw("mapped_threshold", &KaHyParMapper::mapped_threshold)
      .def_rw("reserved_threshold", &KaHyParMapper::reserved_threshold)
      .def("map_task", &KaHyParMapper::map_task, "task_id"_a, "state"_a)
      .def(
          "map_tasks",
          [](KaHyParMapper &mapper, const TaskIDList &tasks,
             const SchedulerState &state) -> ActionList & {
            return mapper.map_tasks(std::span<const taskid_t>(tasks), state);
          },
          "tasks"_a, "state"_a, nb::rv_policy::reference_internal);

  nb::class_<DARTSMapper, Mapper>(m, "DARTSMapper")
      .def(nb::init<>())
      .def(nb::init<std::size_t, std::size_t>(), "num_tasks"_a, "num_devices"_a)
      .def(nb::init<DARTSMapper &>(), "other"_a)
      .def_prop_rw("mapped_threshold", &DARTSMapper::get_mapped_threshold,
                   &DARTSMapper::set_mapped_threshold)
      .def_prop_rw("reserved_threshold", &DARTSMapper::get_reserved_threshold,
                   &DARTSMapper::set_reserved_threshold)
      .def_rw("extended_frontier_enabled", &DARTSMapper::extended_frontier_enabled)
      .def_rw("trace_decisions", &DARTSMapper::trace_decisions)
      .def("set_thresholds", &DARTSMapper::set_thresholds, "mapped_threshold"_a,
           "reserved_threshold"_a)
      .def("use_mapped_threshold", &DARTSMapper::use_mapped_threshold, "mapped_threshold"_a)
      .def("use_reserved_threshold", &DARTSMapper::use_reserved_threshold,
           "reserved_threshold"_a)
      .def("disable_thresholds", &DARTSMapper::disable_thresholds)
      .def("map_task", &DARTSMapper::map_task, "task_id"_a, "state"_a)
      .def(
          "map_tasks",
          [](DARTSMapper &mapper, const TaskIDList &tasks,
             const SchedulerState &state) -> ActionList & {
            return mapper.map_tasks(std::span<const taskid_t>(tasks), state);
          },
          "tasks"_a, "state"_a, nb::rv_policy::reference_internal);
}
