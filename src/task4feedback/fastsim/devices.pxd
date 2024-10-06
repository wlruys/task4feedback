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

cdef extern from "include/device_manager.hpp":

    cdef cppclass Devices:
        Devices(uint64_t n)
        void create_device(devid_t id, string name, DeviceType arch, vcu_t vcus, mem_t mem)
        DeviceIDList& get_devices(DeviceType arch)
        DeviceType get_type(devid_t id)
        const string get_name(devid_t id)



