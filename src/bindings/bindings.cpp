#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h> 
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/bind_map.h>

#include "settings.hpp"
#include "tasks.hpp"
#include "devices.hpp"
#include "resources.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_event_ext(nb::module_ &);
void init_device_ext(nb::module_ &);
void init_task_ext(nb::module_ &);
void init_data_ext(nb::module_ &);
void init_graph_ext(nb::module_ &);
void init_task_noise_ext(nb::module_ &);
void init_topology_ext(nb::module_ &);
void init_scheduler_ext(nb::module_ &);
void init_mapper_ext(nb::module_ &);
void init_observer_ext(nb::module_ &);
void init_simulator_ext(nb::module_ &);


NB_MODULE(fastsim2, m){
    nb::bind_vector<std::vector<uint32_t>>(m, "UInt32Vector");
    nb::bind_vector<std::vector<uint64_t>>(m, "UInt64Vector");
    nb::bind_vector<std::vector<uint32_t>>(m, "Int32Vector");
    nb::bind_vector<std::vector<uint64_t>>(m, "Int64Vector");
    nb::bind_vector<std::vector<float>>(m, "FloatVector");
    nb::bind_vector<std::vector<Task>>(m, "TaskVector");
    nb::bind_vector<std::vector<ComputeTask>>(m, "ComputeTaskVector");
    nb::bind_vector<std::vector<DataTask>>(m, "DataTaskVector");
    init_event_ext(m);
    init_device_ext(m);
    init_task_ext(m);
    init_data_ext(m);
    init_graph_ext(m);
    init_task_noise_ext(m);
    init_topology_ext(m);
    init_scheduler_ext(m);
    init_mapper_ext(m);
    init_simulator_ext(m);
}

// NB_MODULE(fastsim2, m) {
//     m.doc() = "This is a \"hello world\" example with nanobind";
//     m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);

//         // Enums (using nb::enum_ for clarity and consistency)
//     nb::enum_<EventType>(m, "EventType", nb::is_arithmetic())
//         .value("MAPPER", EventType::MAPPER)
//         .value("RESERVER", EventType::RESERVER)
//         .value("LAUNCHER", EventType::LAUNCHER)
//         .value("EVICTOR", EventType::EVICTOR)
//         .value("COMPLETER", EventType::COMPLETER)
//         .export_values(); // Export enum values to the module scope

//     nb::enum_<DeviceType>(m, "DeviceType", nb::is_arithmetic()) // Add nb::is_arithmetic()
//       .value("NONE", DeviceType::NONE)
//       .value("CPU", DeviceType::CPU)
//       .value("GPU", DeviceType::GPU)
//       .export_values();

//     nb::enum_<TaskState>(m, "TaskState", nb::is_arithmetic()) // Add nb::is_arithmetic()
//         .value("SPAWNED", TaskState::SPAWNED)
//         .value("MAPPED", TaskState::MAPPED)
//         .value("RESERVED", TaskState::RESERVED)
//         .value("LAUNCHED", TaskState::LAUNCHED)
//         .value("COMPLETED", TaskState::COMPLETED)
//         .export_values();
        
//     nb::enum_<ExecutionState>(m, "ExecutionState", nb::is_arithmetic()) //Add nb::is_arithmetic()
//       .value("NONE", ExecutionState::NONE)
//       .value("RUNNING", ExecutionState::RUNNING)
//       .value("COMPLETE", ExecutionState::COMPLETE)
//       .value("BREAKPOINT", ExecutionState::BREAKPOINT)
//       .value("EXTERNAL_MAPPING", ExecutionState::EXTERNAL_MAPPING)
//       .value("ERROR", ExecutionState::ERROR)
//       .export_values();

//     nb::class_<Topology>(m, "Topology")
//         .def(nb::init<std::size_t>(), "num_devices"_a)
//         .def("set_bandwidth", &Topology::set_bandwidth, "src"_a, "dst"_a, "bandwidth"_a)
//         .def("set_max_connections", &Topology::set_max_connections, "src"_a, "dst"_a, "max_links"_a)
//         .def("set_latency", &Topology::set_latency, "src"_a, "dst"_a, "latency"_a)
//         .def("get_latency", &Topology::get_latency, "src"_a, "dst"_a)
//         .def("get_bandwidth", &Topology::get_bandwidth, "src"_a, "dst"_a)
//         .def("get_max_connections", &Topology::get_max_connections, "src"_a, "dst"_a);

//    nb::class_<CommunicationStats>(m, "CommunicationStats")
//         .def(nb::init<>())  // Default constructor
//         .def(nb::init<timecount_t, mem_t>(), "latency"_a, "bandwidth"_a) // Constructor with latency and bandwidth
//         .def_rw("latency", &CommunicationStats::latency)
//         .def_rw("bandwidth", &CommunicationStats::bandwidth);

//     nb::class_<CommunicationRequest>(m, "CommunicationRequest")
//         .def(nb::init<>()) // Add default constructor
//         .def(nb::init<taskid_t, devid_t, devid_t, mem_t>(), "data_task_id"_a, "source"_a, "destination"_a, "size"_a)
//         .def_rw("data_task_id", &CommunicationRequest::data_task_id)
//         .def_rw("source", &CommunicationRequest::source)
//         .def_rw("destination", &CommunicationRequest::destination)
//         .def_rw("size", &CommunicationRequest::size);
        
//     nb::class_<CommunicationNoise>(m, "CommunicationNoise")
//         .def(nb::init<Topology &, unsigned int>(), "topology"_a, "seed"_a = 0)
//         .def("get", &CommunicationNoise::get, "request"_a)
//         .def("set",
//             nb::overload_cast<const CommunicationRequest &, const CommunicationStats &>(
//                 &CommunicationNoise::set
//             ), "request"_a, "stats"_a
//         ) // Explicit overload for clarity
//         .def("dump_to_binary", &CommunicationNoise::dump_to_binary, "filename"_a)
//         .def("load_from_binary", &CommunicationNoise::load_from_binary, "filename"_a);

//     nb::class_<UniformCommunicationNoise, CommunicationNoise>(
//         m, "UniformCommunicationNoise"
//     )
//         .def(nb::init<Topology &, unsigned int>(), "topology"_a, "seed"_a = 0);
// }