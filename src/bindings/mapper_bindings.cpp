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

void init_mapper_ext(nb::module_& m) {
    nb::bind_vector<std::vector<Action>>(m, "ActionVector");
    nb::class_<Action>(m, "Action")
        .def(nb::init<taskid_t, std::size_t, devid_t, priority_t, priority_t>(), "task_id"_a, "pos"_a, "device"_a, "reservable_priority"_a, "launchable_priority"_a);

    nb::class_<Mapper>(m, "Mapper")
        .def("map_task", &Mapper::map_task, "task_id"_a, "state"_a)
        .def("map_tasks", &Mapper::map_tasks, "tasks"_a, "state"_a);

    nb::class_<RandomMapper, Mapper>(m, "RandomMapper")
        .def(nb::init<unsigned int>(), "seed"_a);

    nb::class_<RoundRobinMapper, Mapper>(m, "RoundRobinMapper")
        .def(nb::init<>());

    nb::class_<StaticMapper, Mapper>(m, "StaticMapper")
        .def(nb::init<DeviceIDList>(), "device_ids_"_a)
        .def(nb::init<DeviceIDList, PriorityList, PriorityList>(), "device_ids"_a, "reserving_priorities"_a, "launching_priorities"_a)
        .def("set_mapping", &StaticMapper::set_mapping, "device_ids_"_a)
        .def("set_reserving_priorities", &StaticMapper::set_reserving_priorities, "reserving_priorites_"_a)
        .def("set_launching_priorities", &StaticMapper::set_launching_priorities, "launching_priorites_"_a);

    nb::class_<StaticActionMapper, Mapper>(m, "StaticActionMapper")
        .def(nb::init<ActionList>(), "actions"_a);

    nb::class_<DeviceTime>(m, "DeviceTime")
        .def(nb::init<devid_t, timecount_t>(), "device_id"_a, "time"_a)
        .def_ro("device_id", &DeviceTime::device_id)
        .def_ro("time", &DeviceTime::time);

    nb::class_<EFTMapper, Mapper>(m, "EFTMapper")
        .def(nb::init<std::size_t, std::size_t>(), "num_tasks"_a, "num_devices"_a)
        .def("get_best_device", &EFTMapper::get_best_device, "task_id"_a, "state"_a)
        .def("get_dependency_finish_time", &EFTMapper::get_dependency_finish_time, "task_id"_a, "state"_a)
        .def("get_device_available_time", &EFTMapper::get_device_available_time, "device_id"_a, "state"_a)
        .def("get_finish_time", &EFTMapper::get_finish_time, "task_id"_a, "device_id"_a, "start_t"_a, "state"_a)
        .def("time_for_transfer", &EFTMapper::time_for_transfer, "task_id"_a, "device_id"_a, "state"_a);

    nb::class_<DequeueEFTMapper, EFTMapper>(m, "DequeueEFTMapper")
        .def(nb::init<std::size_t, std::size_t>(), "num_tasks"_a, "num_devices"_a);
    
    
}