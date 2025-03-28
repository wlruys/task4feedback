#cython: language_level=3
#cython: embedsignature=True
#cython: language=c++

from settings cimport dataid_t, taskid_t, TaskIDList, DataIDList, DeviceIDList, PriorityList, DeviceType, devid_t, priority_t, depcount_t, vcu_t, mem_t, timecount_t

from tasks cimport Tasks, TaskState 
from devices cimport Devices, ResourceEventArray
from communication cimport Topology, CommunicationNoise
from data cimport Data, ValidEventArray
from noise cimport TaskNoise 

import cython
cimport cython

from libcpp.vector cimport vector 
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int8_t, uint8_t, int64_t, int32_t, uint32_t, uint64_t 
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

    cdef cppclass SchedulerState:
        pass 

    cdef cppclass StaticMapper(Mapper):
        void set_mapping(DeviceIDList& devices)
        void set_launching_priorities(PriorityList& priorities)
        void set_reserving_priorities(PriorityList& priorities)

    cdef cppclass EFTMapper(Mapper):
        EFTMapper(size_t n_tasks, size_t n_devices)

    cdef cppclass DequeueEFTMapper(Mapper):
        DequeueEFTMapper(size_t n_tasks, size_t n_devices)

    cdef cppclass Deque

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
        void initialize(bool create_data_tasks, bool use_transition_conditions, bool initialize_data_manager)
        void initialize_data_manager()
        ExecutionState run()
        timecount_t get_current_time()
        TaskIDList get_mappable_candidates()
        void map_tasks(ActionList& actions)
        void add_task_breakpoint(EventType event_type, taskid_t task_id, bool collective)
        void add_time_breakpoint(timecount_t time)
        void set_use_python_mapper(bool use_python_mapper)
        void set_mapper(Mapper& mapper)
        SchedulerState& get_state()
        devid_t get_mapping(taskid_t task_id)
        priority_t get_mapping_priority(taskid_t task_id)
        priority_t get_reserving_priority(taskid_t task_id)
        priority_t get_launching_priority(taskid_t task_id)

        timecount_t get_mapped_time(taskid_t task_id)
        timecount_t get_reserved_time(taskid_t task_id)
        timecount_t get_launched_time(taskid_t task_id)
        timecount_t get_completed_time(taskid_t task_id)

        TaskState get_state_at_time(taskid_t task_id, timecount_t time)

        vcu_t get_mapped_vcu_at_time(devid_t device_id, timecount_t time)
        vcu_t get_reserved_vcu_at_time(devid_t device_id, timecount_t time)
        vcu_t get_launched_vcu_at_time(devid_t device_id, timecount_t time)

        mem_t get_mapped_mem_at_time(devid_t device_id, timecount_t time)
        mem_t get_reserved_mem_at_time(devid_t device_id, timecount_t time)
        mem_t get_launched_mem_at_time(devid_t device_id, timecount_t time)

        ResourceEventArray[vcu_t] get_vcu_events_mapped(devid_t device_id)
        ResourceEventArray[vcu_t] get_vcu_events_reserved(devid_t device_id)
        ResourceEventArray[vcu_t] get_vcu_events_launched(devid_t device_id)

        ResourceEventArray[mem_t] get_mem_events_mapped(devid_t device_id)
        ResourceEventArray[mem_t] get_mem_events_reserved(devid_t device_id)
        ResourceEventArray[mem_t] get_mem_events_launched(devid_t device_id)

        bool check_valid_mapped(dataid_t data_id, devid_t device_id, timecount_t query_time)
        bool check_valid_reserved(dataid_t data_id, devid_t device_id, timecount_t query_time)
        bool check_valid_launched(dataid_t data_id, devid_t device_id, timecount_t query_time)

        ValidEventArray get_valid_intervals_mapped(dataid_t data_id, devid_t device_id)
        ValidEventArray get_valid_intervals_reserved(dataid_t data_id, devid_t device_id)
        ValidEventArray get_valid_intervals_launched(dataid_t data_id, devid_t device_id)






cdef extern from "include/observer.hpp":

    ctypedef uint32_t op_t 

    cdef cppclass Features:
        float* features
        size_t feature_dim
        size_t feature_len 

    cdef cppclass DataFeatures(Features):
        pass

    cdef cppclass TaskFeatures(Features):
        pass

    cdef cppclass DeviceFeatures(Features):
        pass

    cdef cppclass TaskDataEdges(Features):
        op_t* data2id
        size_t data2id_len
        op_t* edges 

    cdef cppclass TaskDeviceEdges(Features):
        op_t* device2id
        size_t device2id_len
        op_t* edges

    cdef cppclass DataDeviceEdges(Features):
        op_t* data2id
        size_t data2id_len
        op_t* device2id
        size_t device2id_len
        op_t* edges

    cdef cppclass TaskTaskEdges(Features):
        op_t* edges

    cdef cppclass Observer:
        Observer(Simulator& simulator)
        void global_features()
        TaskIDList get_active_tasks()
        TaskIDList get_k_hop_dependents(taskid_t* initial_tasks, size_t ntasks, int k)
        TaskIDList get_k_hop_dependencies(taskid_t* initial_tasks, size_t ntasks, int k)
        void get_device_mask_int8(taskid_t task_id, int8_t* devices, size_t max_devices)
        TaskFeatures get_task_features(taskid_t* tasks, size_t ntasks)
        DataFeatures get_data_features(dataid_t* data, size_t ndata)
        DeviceFeatures get_device_features(devid_t* devices, size_t ndevices)
        TaskTaskEdges get_task_task_edges(taskid_t* sources, size_t nsource, taskid_t* targets, size_t ntarget)
        TaskDataEdges get_task_data_edges(taskid_t* task_ids, size_t ntasks)
        TaskDeviceEdges get_task_device_edges(taskid_t* task_ids, size_t ntasks)
        DataDeviceEdges get_data_device_edges(dataid_t* data_ids, size_t ndata)
        int get_n_tasks()

    

