#cython: language_level=3
#cython: embedsignature=True
#cython: language=c++

from settings cimport dataid_t, taskid_t, TaskIDList, DataIDList, DeviceIDList, PriorityList, DeviceType, devid_t, priority_t, depcount_t, vcu_t, mem_t, timecount_t

from tasks cimport Tasks
from devices cimport Devices
from communication cimport Topology, CommunicationNoise
from data cimport Data
from noise cimport TaskNoise 

import cython
cimport cython

from libcpp.vector cimport vector 
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport uint8_t, int64_t, int32_t, uint32_t, uint64_t 
from libcpp.string cimport string
from libcpp.utility cimport move
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from libcpp cimport bool 

cdef extern from "include/action.hpp":
    cdef cppclass Action:
        Action(taskid_t task_id, size_t pos, devid_t device, priority_t reservable_priority, priority_t launchable_priority)
        taskid_t task_id
        size_t pos
        devid_t device
        priority_t reservable_priority
        priority_t launchable_priority

    ctypedef vector[Action] ActionList

cdef extern from "include/events.hpp":

    cpdef enum class EventType(int):
        MAPPER,
        RESERVER,
        LAUNCHER,
        EVICTOR,
        COMPLETER

cdef extern from "include/scheduler.hpp":
    cdef cppclass Mapper:
        pass 

    cdef cppclass SchedulerInput:
        SchedulerInput(Tasks &tasks, Data &data, Devices &devices, Topology &topology,
                 Mapper &mapper, TaskNoise &task_noise,
                 CommunicationNoise &comm_noise)

    

    cdef cppclass StaticMapper(Mapper):
        void set_mapping(DeviceIDList& devices)
        void set_launching_priorities(PriorityList& priorities)
        void set_reserving_priorities(PriorityList& priorities)

cdef extern from "include/simulator.hpp":

    cdef void logger_setup()

    cpdef enum class ExecutionState(int):
        NONE,
        RUNNING,
        COMPLETE,
        BREAKPOINT,
        EXTERNAL_MAPPING,
        ERROR
    cdef cppclass Simulator:
        Simulator(Simulator&)
        Simulator(SchedulerInput& input)
        void initialize(bool create_data_tasks)
        ExecutionState run()
        timecount_t get_current_time()
        TaskIDList get_mappable_candidates()
        void map_tasks(ActionList& actions)
        void add_task_breakpoint(EventType event_type, taskid_t task_id)
        void add_time_breakpoint(timecount_t time)
        void set_use_python_mapper(bool use_python_mapper)
        void set_mapper(Mapper& mapper)



cdef extern from "include/observer.hpp":

    cdef cppclass TaskDataEdges:
        TaskIDList task_id
        DataIDList data_ids

    cdef cppclass TaskDeviceEdges:
        TaskIDList task_id
        DeviceIDList device_ids

    cdef cppclass DataDeviceEdges:
        DataIDList data_id
        DeviceIDList device_ids
    

    cdef cppclass Observer:
        Observer(Simulator& simulator)
        preprocess_global()
        TaskIDList get_active_tasks()
        TaskIDList get_k_hop_tasks()
        vector[double] get_task_features(taskid_t task_id)
        vector[double] get_data_features(dataid_t data_id)
        vector[double] get_device_features(devid_t device_id)
        vector[double] get_task_data_features(taskid_t task_id, dataid_t data_id)
        vector[double] get_task_device_features(taskid_t task_id, devid_t device_id)
        TaskDataEdges get_task_data_edges(taskid_t task_id)
        TaskDeviceEdges get_task_device_edges(taskid_t task_id)
        DataDeviceEdges get_data_device_edges(dataid_t data_id)

    

