#cython: language_level=3
#cython: embedsignature=True
#cython: language=c++

from settings cimport taskid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t

from tasks cimport Tasks
from devices cimport Devices

import cython
cimport cython

from libcpp.vector cimport vector 
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t, int32_t, uint32_t, uint64_t 
from libcpp.string cimport string

cdef extern from "include/simulator.hpp":

    cpdef enum class StopReason(int):
        COMPLETE,
        BREAKPOINT,
        MAPPING,
        ERROR
        
    cdef cppclass Simulator:
        Simulator(Tasks& tasks, Devices& devices)
        void initialize(unsigned int seed)
        StopReason run()
