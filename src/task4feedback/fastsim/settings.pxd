#cython: language_level=3
#cython: embedsignature=True
#cython: language=c++

import cython
cimport cython

from libcpp.vector cimport vector 
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t, uint16_t, int32_t, uint32_t, uint64_t

cdef extern from "include/settings.hpp":

    ctypedef uint64_t priority_t
    ctypedef uint64_t taskid_t 
    ctypedef uint64_t dataid_t
    ctypedef uint64_t devid_t
    ctypedef uint32_t copy_t


    ctypedef uint64_t depcount_t

    ctypedef vector[taskid_t] TaskIDList
    ctypedef vector[dataid_t] DataIDList
    ctypedef vector[devid_t] DeviceIDList

cdef extern from "include/devices.hpp":
    cpdef enum class DeviceType(int):
        CPU,
        GPU

cdef extern from "include/resources.hpp":
    ctypedef uint16_t vcu_t
    ctypedef uint64_t mem_t
    ctypedef uint64_t timecount_t

    cdef cppclass Resources:
        Resources(vcu_t vcus, mem_t mem)
        vcu_t get_vcus() const
        mem_t get_mem() const

