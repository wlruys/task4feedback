#cython: language_level=3
#cython: embedsignature=True
#cython: language=c++

from settings cimport copy_t, taskid_t, timecount_t, dataid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t

import cython
cimport cython

from libcpp.vector cimport vector 
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t, int32_t, uint8_t, uint32_t, uint64_t 
from libcpp.string cimport string
from libcpp.utility cimport move
from libc.stddef cimport size_t
from libcpp cimport bool

cdef extern from "include/communication_manager.hpp":

    cdef cppclass Topology:
        Topology(size_t num_devices)
        void set_bandwidth(devid_t src, devid_t dst, mem_t bandwidth)
        void set_max_connections(devid_t src, devid_t dst, uint8_t max_links)
        void set_latency(devid_t src, devid_t dst, timecount_t latency)
        timecount_t get_latency(devid_t src, devid_t dst) const
        mem_t get_bandwidth(devid_t src, devid_t dst) const
        copy_t get_max_connections(devid_t src, devid_t dst) const
        
    cdef cppclass CommunicationStats:
        timecount_t latency
        mem_t bandwidth
        CommunicationStats()
        CommunicationStats(timecount_t latency, mem_t bandwidth)

    cdef cppclass CommunicationRequest:
        taskid_t data_task_id 
        devid_t source 
        devid_t destination
        mem_t size
        CommunicationRequest()
        CommunicationRequest(taskid_t data_task_id, devid_t source, devid_t destination, mem_t size)

    cdef cppclass CommunicationNoise:
        CommunicationNoise(Topology& topology, unsigned int seed)
        CommunicationStats get(const CommunicationRequest& request)
        void set(const CommunicationRequest& request, const CommunicationStats& stats)
        void dump_to_binary(const string filename)
        void load_from_binary(const string filename)

    cdef cppclass UniformCommunicationNoise(CommunicationNoise):
        UniformCommunicationNoise(Topology& topology, unsigned int seed)

