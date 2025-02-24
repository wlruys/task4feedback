#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/bind_vector.h>

#include "tasks.hpp"
#include "noise.hpp"
#include "devices.hpp"
#include "resources.hpp"

namespace nb = nanobind;
using namespace nb::literals;


    // cdef cppclass TaskNoise:
    //     TaskNoise(Tasks& tasks, unsigned int seed, unsigned int pseed)
    //     timecount_t get(taskid_t task_id, DeviceType arch)
    //     void set(taskid_t task_id, DeviceType arch, timecount_t noise)
    //     void set(vector[timecount_t] noise)
    //     void set_priority(taskid_t task_id, priority_t p)
    //     void set_priority(vector[priority_t] noise)
    //     void lock()
    //     void generate()
    //     void generate_duration()
    //     void generate_priority()
    //     void dump_to_binary(const string filename)
    //     void load_from_binary(const string filename)
    //     void dump_priorities_to_binary(const string filename)
    //     void load_priorities_from_binary(const string filename)

        

    // cdef cppclass ExternalTaskNoise(TaskNoise):
    //     ExternalTaskNoise(Tasks& tasks, unsigned int seed, unsigned int pseed)
    //     void set_function(esf_t function)

    // cdef cppclass LognormalTaskNoise(TaskNoise):
    //     LognormalTaskNoise(Tasks& tasks, unsigned int seed, unsigned int pseed)

void init_task_noise_ext(nb::module_& m) {
    nb::class_<TaskNoise>(m, "TaskNoise").
        def(nb::init<Tasks&, unsigned int, unsigned int>(), "tasks"_a, "seed"_a, "pseed"_a)
        .def("get", &TaskNoise::get, "task_id"_a, "arch"_a)
        .def("set", nb::overload_cast<taskid_t, DeviceType, timecount_t>(&TaskNoise::set), "task_id"_a, "arch"_a, "noise"_a)
        .def("set", nb::overload_cast<std::vector<timecount_t>>(&TaskNoise::set), "noise"_a)
        .def("set_priority", nb::overload_cast<taskid_t, priority_t>(&TaskNoise::set_priority), "task_id"_a, "p"_a)
        .def("set_priority", nb::overload_cast<std::vector<priority_t>>(&TaskNoise::set_priority), "noise"_a)
        .def("generate_duration", &TaskNoise::generate_duration)
        .def("generate_priority", &TaskNoise::generate_priority)
        .def("dump_to_binary", &TaskNoise::dump_to_binary, "filename"_a)
        .def("load_from_binary", &TaskNoise::load_from_binary, "filename"_a)
        .def("dump_priorities_to_binary", &TaskNoise::dump_priorities_to_binary, "filename"_a)
        .def("load_priorities_from_binary", &TaskNoise::load_priorities_from_binary, "filename"_a);

    nb::class_<LognormalTaskNoise>(m, "LognormalTaskNoise").
        def(nb::init<Tasks&, unsigned int, unsigned int>(), "tasks"_a, "seed"_a, "pseed"_a);
        
}

