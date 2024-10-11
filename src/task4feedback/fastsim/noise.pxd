#cython: language_level=3
#cython: embedsignature=True
#cython: language=c++

from settings cimport taskid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t

import cython
cimport cython
from tasks cimport Tasks

from libcpp.vector cimport vector 
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t, int32_t, uint32_t, uint64_t 
from libcpp.string cimport string

cdef extern from "include/noise.hpp":

    ctypedef double (*esf_t)(uint64_t, uint64_t) noexcept nogil

    cdef cppclass TaskNoise:
        TaskNoise(Tasks& tasks, unsigned int seed)
        timecount_t get(taskid_t task_id, DeviceType arch)
        void set(taskid_t task_id, DeviceType arch, timecount_t noise)
        void generate()
        void dump_to_binary(const string filename)
        void load_from_binary(const string filename)

        

    cdef cppclass ExternalTaskNoise(TaskNoise):
        ExternalTaskNoise(Tasks& tasks, unsigned int seed)
        timecount_t get(taskid_t task_id, DeviceType arch)
        void set(taskid_t task_id, DeviceType arch, timecount_t noise)
        void generate()
        void dump_to_binary(const string filename)
        void load_from_binary(const string filename)
        void set_function(esf_t function)


    cdef cppclass LognormalTaskNoise(TaskNoise):
        LognormalTaskNoise(Tasks& tasks, unsigned int seed)
        timecount_t get(taskid_t task_id, DeviceType arch)
        void set(taskid_t task_id, DeviceType arch, timecount_t noise)
        void generate()
        void dump_to_binary(const string filename)
        void load_from_binary(const string filename)