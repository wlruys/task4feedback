# cython: language_level=3
# cython: embedsignature=True
# cython: language=c++

import cython 

from task_state cimport random_topological_sort, TaskManager, TaskIDList, DeviceType, DataIDList, DeviceIDList, populate_dependents, taskid_t, devid_t, depcount_t, vcu_t, mem_t, timecount_t, copy_t
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.utility cimport move
from libcpp.string cimport string

import numpy as np
cimport numpy as np

cdef convert_to_taskid_list(list taskid_list):
    cdef TaskIDList result
    result.reserve(len(taskid_list))
    for taskid in taskid_list:
        result.push_back(taskid)
    return result

cdef convert_to_dataid_list(list dataid_list):
    cdef DataIDList result
    result.reserve(len(dataid_list))
    for dataid in dataid_list:
        result.push_back(dataid)
    return result

cdef convert_to_devid_list(list devid_list):
    cdef DeviceIDList result
    result.reserve(len(devid_list))
    for devid in devid_list:
        result.push_back(devid)
    return result


cdef convert_taskid_list_to_numpy(TaskIDList taskid_list, copy: bool = False):
    cdef np.npy_intp n = taskid_list.size()
    cdef np.ndarray[np.uint64_t, ndim=1] result

    if not copy:
        result = np.PyArray_SimpleNewFromData(1, &n, np.NPY_UINT64, <void*>taskid_list.data())
    else:
        result = np.PyArray_SimpleNew(1, &n, np.NPY_UINT64)
        for i in range(n):
            result[i] = taskid_list[i]

    return result

cdef convert_dataid_list_to_numpy(DataIDList dataid_list, copy: bool = False):
    cdef np.npy_intp n = dataid_list.size()
    cdef np.ndarray[np.uint64_t, ndim=1] result

    if not copy:
        result = np.PyArray_SimpleNewFromData(1, &n, np.NPY_UINT64, <void*>dataid_list.data())
    else:
        result = np.PyArray_SimpleNew(1, &n, np.NPY_UINT64)
        for i in range(n):
            result[i] = dataid_list[i]

    return result

cdef convert_devid_list_to_numpy(DeviceIDList devid_list, copy: bool = False):
    cdef np.npy_intp n = devid_list.size()
    cdef np.ndarray[np.uint32_t, ndim=1] result

    if not copy:
        result = np.PyArray_SimpleNewFromData(1, &n, np.NPY_UINT32, <void*>devid_list.data())
    else:
        result = np.PyArray_SimpleNew(1, &n, np.NPY_UINT32)
        for i in range(n):
            result[i] = devid_list[i]

    return result


cdef class Simulator:
    cdef TaskManager* task_manager

    def __cinit__(self, n: int):
        self.task_manager = new TaskManager(n)

    def add_task(self, taskid_t taskid, str pyname, list py_dependencies):
        cdef TaskIDList dependencies = convert_to_taskid_list(py_dependencies)
        cname = pyname.encode('utf-8')
        self.task_manager.add_task(taskid, cname, move(dependencies))
        print("Added task", taskid, pyname, py_dependencies)

    def add_read_set(self, taskid_t taskid, list py_dataids):
        cdef DataIDList dataids = convert_to_dataid_list(py_dataids)
        self.task_manager.set_read(taskid, move(dataids))

    def add_write_set(self, taskid_t taskid, list py_dataids):
        cdef DataIDList dataids = convert_to_dataid_list(py_dataids)
        self.task_manager.set_write(taskid, move(dataids))
    
    def add_variant(self, taskid_t id, DeviceType arch, vcu_t vcus, mem_t mem, timecount_t time):
        self.task_manager.add_variant(id, arch, vcus, mem, time)

    def initialize_dependents(self):
        populate_dependents(deref(self.task_manager))

    def random_topological_sort(self):
        cdef TaskIDList result = random_topological_sort(deref(self.task_manager), 0)
        return convert_taskid_list_to_numpy(result, copy=True)

    def print_task(self, taskid_t taskid):
        self.task_manager.print_task(taskid)