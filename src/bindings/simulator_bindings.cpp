#include "action.hpp"
#include "communication.hpp"
#include "data.hpp"
#include "devices.hpp"
#include "nbh.hpp"
#include "noise.hpp"
#include "resources.hpp"
#include "scheduler.hpp"
#include "simulator.hpp"
#include "tasks.hpp"
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <span>

namespace nb = nanobind;
using namespace nb::literals;

void init_simulator_ext(nb::module_ &m) {

  m.def("start_logger", &init_simulator_logger);

  nb::enum_<ExecutionState>(m, "ExecutionState", nb::is_arithmetic())
      .value("NONE", ExecutionState::NONE)
      .value("RUNNING", ExecutionState::RUNNING)
      .value("COMPLETE", ExecutionState::COMPLETE)
      .value("BREAKPOINT", ExecutionState::BREAKPOINT)
      .value("EXTERNAL_MAPPING", ExecutionState::EXTERNAL_MAPPING)
      .value("ERROR", ExecutionState::ERROR)
      .export_values();

  nb::class_<SchedulerInput>(m, "SchedulerInput")
      .def(nb::init<Graph &, StaticTaskInfo &, Data &, Devices &, Topology &, TaskNoise &,
                    TransitionConditions &>(),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(),
           nb::keep_alive<1, 5>(), nb::keep_alive<1, 6>(), nb::keep_alive<1, 7>(),
           nb::keep_alive<1, 8>())
      .def(nb::init<SchedulerInput &>(), nb::keep_alive<1, 2>());

  nb::class_<Simulator>(m, "Simulator")
      .def_ro("initialized", &Simulator::initialized)
      .def_ro("use_python_mapper", &Simulator::use_python_mapper)
      .def_ro("last_execution_state", &Simulator::last_state)
      .def_ro("data_initialized", &Simulator::data_initialized)
      .def_ro("events_processed", &Simulator::events_processed)
      .def(nb::init<SchedulerInput &, Mapper &>(), nb::keep_alive<1, 2>(),
           nb::keep_alive<1, 3>()) // Keep input and mapper alive
      .def(nb::init<Simulator &>(), "other"_a)
      .def("initialize", &Simulator::initialize, "create_data_tasks"_a = true,
           "initialize_data_manager"_a = false)
      .def("set_steps", &Simulator::set_steps, "steps"_a)
      .def("start_drain", &Simulator::start_drain)
      .def("stop_drain", &Simulator::stop_drain)
      .def("initialize_data", &Simulator::initialize_data_manager)
      .def("enable_python_mapper", [](Simulator &s) { s.set_use_python_mapper(true); })
      .def("disable_python_mapper", [](Simulator &s) { s.set_use_python_mapper(false); })
      .def("skip_external_mapping", &Simulator::skip_external_mapping,
           "enqueue_mapping_event"_a = true)
      .def("set_mapper", &Simulator::set_mapper, nb::keep_alive<1, 2>()) // Keep mapper alive
      .def("get_state", nb::overload_cast<>(&Simulator::get_state, nb::const_),
           nb::rv_policy::reference_internal)
      .def("run", &Simulator::run)
      .def("get_current_time", &Simulator::get_current_time)
      .def("get_evicted_memory_size", &Simulator::get_evicted_memory_size)
      .def("get_max_memory_usage", &Simulator::get_max_memory_usage)
      .def("get_mappable_candidates",
           [](Simulator &s, TorchInt64Arr1D &arr) {
             std::span<int64_t> span(arr.data(), arr.size());
             return s.get_mappable_candidates(span);
           })
      .def("get_mappable_candidates", &Simulator::get_mappable_candidates)
      .def("map_tasks", &Simulator::map_tasks)
      .def("add_task_breakpoint", &Simulator::add_task_breakpoint)
      .def("clear_breakpoints", &Simulator::clear_breakpoints);
}