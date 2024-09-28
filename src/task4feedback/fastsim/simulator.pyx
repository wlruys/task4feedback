# cython: language_level=3
# cython: embedsignature=True
# cython: language=c++

import cython 

from graph cimport GraphManager
from settings cimport taskid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t, copy_t
from tasks cimport Tasks
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


cdef class TaskGraph:
    cdef Tasks* tasks

    def __cinit__(self, n: int):
        self.tasks = new Tasks(n)

    def create_task(self, taskid_t tid, str pyname, list py_dependencies):
        print(pyname, py_dependencies)
        cdef TaskIDList dependencies = convert_to_taskid_list(py_dependencies)
        cname = pyname.encode('utf-8')
        self.tasks.create_compute_task(tid, cname, move(dependencies))
        print("Added task", tid, pyname, py_dependencies)   

    def add_read_set(self, taskid_t taskid, list py_dataids):
        cdef DataIDList dataids = convert_to_dataid_list(py_dataids)
        self.tasks.set_read(taskid, move(dataids))

    def add_write_set(self, taskid_t taskid, list py_dataids):
        cdef DataIDList dataids = convert_to_dataid_list(py_dataids)
        self.tasks.set_write(taskid, move(dataids))

    def add_variant(self, taskid_t id, DeviceType arch, vcu_t vcus, mem_t mem, timecount_t time):
        self.tasks.add_variant(id, arch, vcus, mem, time)

    def get_dependencies(self, taskid_t taskid):
        cdef TaskIDList dependencies = deref(self.tasks).get_dependencies(taskid)
        return convert_taskid_list_to_numpy(dependencies, copy=True)

    def get_dependents(self, taskid_t taskid):
        cdef TaskIDList dependents = deref(self.tasks).get_dependents(taskid)
        return convert_taskid_list_to_numpy(dependents, copy=True)

    def get_read(self, taskid_t taskid):
        cdef DataIDList dataids = deref(self.tasks).get_read(taskid)
        return convert_dataid_list_to_numpy(dataids, copy=True)

    def get_write(self, taskid_t taskid):
        cdef DataIDList dataids = deref(self.tasks).get_write(taskid)
        return convert_dataid_list_to_numpy(dataids, copy=True)
    
    def initialize_dependents(self):
        GraphManager.populate_dependents(deref(self.tasks))