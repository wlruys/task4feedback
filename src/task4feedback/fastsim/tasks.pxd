#cython: language_level=3
#cython: embedsignature=True
#cython: language=c++

from settings cimport taskid_t, dataid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t

import cython
cimport cython

from libcpp.vector cimport vector 
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t, int32_t, uint32_t, uint64_t 
from libcpp.string cimport string
from libcpp.utility cimport move
from libc.stddef cimport size_t
from libcpp cimport bool

cdef extern from "include/tasks.hpp":
    cpdef enum class TaskState(int):
        SPAWNED,
        MAPPED,
        RESERVED,
        LAUNCHED,
        COMPLETED

    cpdef enum class TaskStatus(int):
        NONE,
        MAPPABLE,
        RESERVABLE,
        LAUNCHABLE

    cdef cppclass Variant:
        Variant()
        Variant(DeviceType arch, vcu_t vcus, mem_t mem, timecount_t time)
        DeviceType get_arch() const
        vcu_t get_vcus() const
        mem_t get_mem() const
        timecount_t get_observed_time() const


cdef extern from "include/task_manager.hpp":

    cdef cppclass Tasks:
        Tasks(uint64_t n)
        void create_compute_task(taskid_t id, string name, TaskIDList dependencies)
        void set_read(taskid_t id, DataIDList dataids)
        void set_write(taskid_t id, DataIDList dataids)
        void add_variant(taskid_t id, DeviceType arch, vcu_t vcus, mem_t mem, timecount_t time)
        
        const TaskIDList& get_dependencies(taskid_t id) const
        const TaskIDList& get_dependents(taskid_t id) const 
        const TaskIDList& get_data_dependencies(taskid_t id) const
        const TaskIDList& get_data_dependents(taskid_t id) const
        const DataIDList& get_read(taskid_t id) const
        const DataIDList& get_write(taskid_t id) const
        const vector[Variant] get_variant_vector(taskid_t id) const
        const Variant& get_variant(taskid_t id, DeviceType arch) const
        
        const string get_name(taskid_t id) const
        size_t get_depth(taskid_t id) const
        dataid_t get_data_id(taskid_t id) const

        size_t size() const
        size_t compute_size() const 
        size_t data_size() const
        bool empty() const
        bool is_compute(taskid_t id) const 
        bool is_data(taskid_t id) const

