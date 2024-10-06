#cython: language_level=3
#cython: embedsignature=True
#cython: language=c++

from settings cimport taskid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t

import cython
cimport cython

from libcpp.vector cimport vector 
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t, int32_t, uint32_t, uint64_t 
from libcpp.string cimport string

cdef extern from "include/tasks.hpp":
    cpdef enum class TaskState(int):
        SPAWNED,
        MAPPED,
        RESERVED,
        LAUNCHED,
        COMPLETE

    cpdef enum class TaskStatus(int):
        NONE,
        MAPPABLE,
        RESERVABLE,
        LAUNCHABLE

    cdef cppclass Variant:
        Variant(DeviceType arch, vcu_t vcus, mem_t mem, timecount_t time)
        DeviceType get_arch() const
        vcu_t get_vcus() const
        mem_t get_mem() const
        timecount_t get_execution_time() const




cdef extern from "include/task_manager.hpp":

    cdef cppclass Tasks:
        Tasks(uint64_t n)
        void create_compute_task(taskid_t id, string name, TaskIDList dependencies)
        void set_read(taskid_t id, DataIDList dataids)
        void set_write(taskid_t id, DataIDList dataids)
        void add_variant(taskid_t id, DeviceType arch, vcu_t vcus, mem_t mem, timecount_t time)
        
        const TaskIDList& get_dependencies(taskid_t id) const
        const TaskIDList& get_dependents(taskid_t id) const 
        const DataIDList& get_read(taskid_t id) const
        const DataIDList& get_write(taskid_t id) const
        const vector[Variant]& get_variants(taskid_t id) const
        TaskState get_state(taskid_t id) const
        TaskStatus get_status(taskid_t id) const


    

