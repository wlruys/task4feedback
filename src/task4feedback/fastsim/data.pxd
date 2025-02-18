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

cdef extern from "include/data_manager.hpp":

    cdef cppclass Data:
        Data()
        Data(size_t num_blocks)

        void create_block(dataid_t id, mem_t size, devid_t location, string name)
        void set_size(dataid_t id, mem_t size)
        void set_location(dataid_t id, devid_t location)
        void set_name(dataid_t id, string name)

        mem_t get_size(dataid_t id) const
        devid_t get_location(dataid_t id) const
        string get_name(dataid_t id) const

    cdef cppclass ValidEventArray:
        timecount_t* starts 
        timecount_t* stops 
