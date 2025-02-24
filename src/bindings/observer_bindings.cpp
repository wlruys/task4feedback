#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/ndarray.h>

#include "observer.hpp"
#include "simulator.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_observer(nb::module_& m) {

    nb::class_<Observer>(m, "Observer")
        .def(nb::init<const Simulator&>(), "simulator"_a) //Use const Simulator&
        .def("global_features", &Observer::global_features)
        .def("get_active_tasks", &Observer::get_active_tasks)
        .def("get_k_hop_dependents", &Observer::get_k_hop_dependents, "initial_tasks"_a.noconvert(), "n"_a, "k"_a)
        .def("get_k_hop_dependencies", &Observer::get_k_hop_dependencies, "initial_tasks"_a.noconvert(), "n"_a, "k"_a)
        .def("get_device_mask_int8", [](Observer& o, taskid_t task_id, nb::ndarray_t<int8_t, nb::shape<nb::any>> valid_devices) {
                if (valid_devices.ndim() != 1) {
                    throw std::runtime_error("device_mask must be a 1D array.");
                }            
                o.get_device_mask_int8(task_id, valid_devices.data(), valid_devices.shape(0));
            }, "task_id"_a, "valid_devices"_a.noconvert()) // Use .noconvert()

        .def("get_task_features", [](Observer &o, nb::ndarray_t<const taskid_t, nb::shape<nb::any>> task_ids) {
                if (task_ids.ndim() != 1) {
                    throw std::runtime_error("task_ids must be a 1D array.");
                }            
                return o.get_task_features(task_ids.data(), task_ids.shape(0));
        }, "task_ids"_a.noconvert())

        .def("get_device_features", [](Observer &o, nb::ndarray_t<const devid_t, nb::shape<nb::any>> device_ids) {
            if (device_ids.ndim() != 1) {
                    throw std::runtime_error("device_ids must be a 1D array.");
                }      
            return o.get_device_features(device_ids.data(), device_ids.shape(0));
        }, "device_ids"_a.noconvert())

        .def("get_data_features", [](Observer &o, nb::ndarray_t<const dataid_t, nb::shape<nb::any>> data_ids) {
                if (data_ids.ndim() != 1) {
                    throw std::runtime_error("data_ids must be a 1D array.");
                }            
                return o.get_data_features(data_ids.data(), data_ids.shape(0));
        }, "data_ids"_a.noconvert())

        .def("get_task_task_edges", [](Observer &o, nb::ndarray_t<const taskid_t, nb::shape<nb::any>> source_tasks, nb::ndarray_t<const taskid_t, nb::shape<nb::any>> target_tasks) {
            if (source_tasks.ndim() != 1 || target_tasks.ndim() != 1) {
                throw std::runtime_error("source_tasks and target_tasks must be 1D arrays");
            }
                return o.get_task_task_edges(source_tasks.data(), source_tasks.size(), target_tasks.data(), target_tasks.size());
        }, "source_tasks"_a.noconvert(), "target_tasks"_a.noconvert())
        .def("get_task_data_edges", [](Observer &o, nb::ndarray_t<const taskid_t, nb::shape<nb::any>> task_ids) {
            if (task_ids.ndim() != 1) {
                throw std::runtime_error("task_ids must be a 1D array");
            }                
            return o.get_task_data_edges(task_ids.data(), task_ids.size());
        }, "task_ids"_a.noconvert())
        .def("get_task_device_edges", [](Observer &o, nb::ndarray_t<const taskid_t, nb::shape<nb::any>> task_ids) {
            if (task_ids.ndim() != 1) {
                throw std::runtime_error("task_ids must be a 1D array");
            }        
            return o.get_task_device_edges(task_ids.data(), task_ids.size());
        }, "task_ids"_a.noconvert())
        .def("get_data_device_edges", [](Observer &o, nb::ndarray_t<const taskid_t, nb::shape<nb::any>> task_ids) {
            if (task_ids.ndim() != 1) {
                throw std::runtime_error("task_ids must be a 1D array");
            }
            return o.get_data_device_edges(task_ids.data(), task_ids.size());
        }, "task_ids"_a.noconvert())  // Keep as reference if possible
        .def("get_number_of_unique_data", &Observer::get_number_of_unique_data, "ids"_a.noconvert(), "n"_a)  // Keep as reference if possible
        .def("get_n_tasks", &Observer::get_n_tasks);
}