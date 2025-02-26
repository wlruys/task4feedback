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
#include "simulator.hpp"
#include "tasks.hpp"
#include "device_manager.hpp"
#include "devices.hpp"
#include "data_manager.hpp"
#include "communication_manager.hpp"
#include "noise.hpp"
#include "action.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_simulator_ext(nb::module_& m) {

    m.def("start_logger", &logger_setup);

    nb::enum_<ExecutionState>(m, "ExecutionState", nb::is_arithmetic())
        .value("NONE", ExecutionState::NONE)
        .value("RUNNING", ExecutionState::RUNNING)
        .value("COMPLETE", ExecutionState::COMPLETE)
        .value("BREAKPOINT", ExecutionState::BREAKPOINT)
        .value("EXTERNAL_MAPPING", ExecutionState::EXTERNAL_MAPPING)
        .value("ERROR", ExecutionState::ERROR)
        .export_values();

    
    
    nb::class_<SchedulerInput>(m, "SchedulerInput")
        .def(nb::init<Tasks&, Data&, Devices&, Topology&, TaskNoise&, CommunicationNoise&, TransitionConditions&>(), "tasks"_a, "data"_a, "devices"_a, "topology"_a, "task_noise"_a, "communication_noise"_a, "transition_conditions"_a)
        .def(nb::init<SchedulerInput&>(), "other"_a);

    nb::class_<Simulator>(m, "Simulator")
        .def_ro("initialized", &Simulator::initialized)
        .def_ro("use_python_mapper", &Simulator::use_python_mapper)
        .def_ro("last_state", &Simulator::last_state)
        .def_ro("last_event", &Simulator::last_event)
        .def_ro("data_initialized", &Simulator::data_initialized)
        .def(nb::init<SchedulerInput&, Mapper&>(), "input"_a, "mapper"_a)
        .def(nb::init<Simulator&>(), "other"_a)
        .def("initialize", &Simulator::initialize, "create_data_tasks"_a = true, "initialize_data_manager"_a = true)
        .def("initialize_data", &Simulator::initialize_data_manager)
        .def("enable_python_mapper", [](Simulator& s) { s.set_use_python_mapper(true); })
        .def("disable_python_mapper", [](Simulator& s) { s.set_use_python_mapper(false); })
        .def("set_mapper", &Simulator::set_mapper, "mapper"_a)
        .def("get_state", nb::overload_cast<>(&Simulator::get_state, nb::const_))
        .def("run", &Simulator::run)
        .def("get_current_time", &Simulator::get_current_time)
        .def("get_mappable_candidates", &Simulator::get_mappable_candidates)
        .def("map_tasks", &Simulator::map_tasks, "action_list"_a);



    
}