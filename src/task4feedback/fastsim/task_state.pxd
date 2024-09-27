#cython: language_level=3
#cython: embedsignature=True
#cython: language=c++

from settings cimport taskid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t, copy_t

import cython
cimport cython

from libcpp.vector cimport vector 
from cython.operator cimport dereference as deref, preincrement as inc


cdef extern from "include/tasks.hpp":

    cdef cppclass TaskManager:
        TaskManager(int n)
        void add_task(taskid_t id, TaskIDList dependencies)
        void set_read(taskid_t id, DataIDList dataids)
        void set_write(taskid_t id, DataIDList dataids)
        void add_variant(taskid_t id, DeviceType arch, vcu_t vcus, mem_t mem, timecount_t time)
        void print_task(taskid_t id)


cdef extern from "include/graph.hpp" namespace "Graph":
    void populate_dependents(TaskManager &task_manager)
    TaskIDList random_topological_sort(TaskManager &task_manager, int seed)

