# cython: language_level=3
# cython: embedsignature=True
# cython: language=c++

import cython 

from graph cimport GraphManager
from settings cimport taskid_t, dataid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t, copy_t
from tasks cimport Tasks, Variant, TaskState, TaskStatus
from data cimport Data 
from noise cimport TaskNoise, ExternalTaskNoise, LognormalTaskNoise, esf_t
from communication cimport Topology, CommunicationNoise, CommunicationRequest, CommunicationStats
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.utility cimport move
from libcpp.string cimport string
from enum import IntEnum 
import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp cimport bool
from numba.core.ccallback import CFunc 
from cpython cimport Py_INCREF, Py_DECREF

class PyTaskState(IntEnum):
    SPAWNED = <int>TaskState.SPAWNED,
    MAPPED = <int>TaskState.MAPPED,
    RESERVED = <int>TaskState.RESERVED,
    LAUNCHED = <int>TaskState.LAUNCHED,
    COMPLETED = <int>TaskState.COMPLETED

    def __str__(self):
        return self.name

class PyTaskStatus(IntEnum):
    NONE = <int>TaskStatus.NONE,
    MAPPABLE = <int>TaskStatus.MAPPABLE,
    RESERVABLE = <int>TaskStatus.RESERVABLE,
    LAUNCHABLE = <int>TaskStatus.LAUNCHABLE

    def __str__(self):
        return self.name

class PyEventType(IntEnum):
    MAPPER = <int>EventType.MAPPER,
    RESERVER = <int>EventType.RESERVER,
    LAUNCHER = <int>EventType.LAUNCHER,
    EVICTOR = <int>EventType.EVICTOR,
    COMPLETER = <int>EventType.COMPLETER

    def __str__(self):
        return self.name

class PyExecutionState(IntEnum):
    NONE = <int>ExecutionState.NONE,
    RUNNING = <int>ExecutionState.RUNNING,
    COMPLETE = <int>ExecutionState.COMPLETE,
    BREAKPOINT = <int>ExecutionState.BREAKPOINT,
    EXTERNAL_MAPPING = <int>ExecutionState.EXTERNAL_MAPPING,
    ERROR = <int>ExecutionState.ERROR 

    def __str__(self):
        return self.name

class PyDeviceType(IntEnum):
    CPU = <int>DeviceType.CPU,
    GPU = <int>DeviceType.GPU

    def __str__(self):
        return self.name

cdef TaskState convert_py_task_state(py_task_state):
    return TaskState(<int>py_task_state)

def convert_cpp_task_state(TaskState task_state):
    return PyTaskState(<int>task_state)


cdef TaskStatus convert_py_task_status(py_task_status):
    return TaskStatus(<int>py_task_status)

def convert_cpp_task_status(TaskStatus task_status):
    return PyTaskStatus(<int>task_status)

cdef EventType convert_py_event_type(py_event_type):
    return EventType(<int>py_event_type)

def convert_cpp_event_type(EventType event_type):
    return PyEventType(<int>event_type)

cdef DeviceType convert_py_device_type(py_device_type):
    return DeviceType(<int>py_device_type)

def convert_cpp_device_type(DeviceType device_type):
    return PyDeviceType(<int>device_type)

cdef ExecutionState convert_py_execution_state(py_execution_state):
    return ExecutionState(<int>py_execution_state)

def convert_cpp_execution_state(ExecutionState execution_state):
    return PyExecutionState(<int>execution_state)

cdef convert_to_taskid_list(list taskid_list):
    cdef TaskIDList result
    result.reserve(len(taskid_list))
    for taskid in taskid_list:
        result.push_back(taskid)
    return result

cdef convert_to_dataid_list(list dataid_list):
    cdef DataIDList result
    result.reserve(len(dataid_list))
    for dataid in dataid_list:
        result.push_back(dataid)
    return result

cdef convert_to_devid_list(list devid_list):
    cdef DeviceIDList result
    result.reserve(len(devid_list))
    for devid in devid_list:
        result.push_back(devid)
    return result


cdef convert_taskid_list_to_numpy(TaskIDList taskid_list, copy: bool = False):
    cdef size_t n = taskid_list.size()
    if n == 0:
        return np.array([], dtype=np.uint64)

    cdef cnp.uint64_t[:] result = <cnp.uint64_t[:n]>taskid_list.data()
    if copy:
        return np.asarray(result, copy=True)
    else:
        return np.asarray(result)

cdef convert_dataid_list_to_numpy(DataIDList dataid_list, copy: bool = False):
    cdef size_t n = dataid_list.size()
    if n == 0:
        return np.array([], dtype=np.uint64)

    cdef cnp.uint64_t[:] result = <cnp.uint64_t[:n]>dataid_list.data()
    if copy:
        return np.array(result, copy=True)
    else:
        return np.asarray(result)

cdef convert_devid_list_to_numpy(DeviceIDList devid_list, copy: bool = False):
    cdef size_t n = devid_list.size()
    if n == 0:
        return np.array([], dtype=np.uint64)

    cdef cnp.uint64_t[:] result = <cnp.uint64_t[:n]>devid_list.data()
    if copy:
        return np.asarray(result, copy=True)
    else:
        return np.asarray(result)   

cpdef start_logger():
    logger_setup()

cdef class PyAction:
    cdef Action* action

    def __cinit__(self, taskid_t task_id, size_t pos, devid_t device, priority_t reservable_priority, priority_t launchable_priority):
        self.action = new Action(task_id, pos, device, reservable_priority, launchable_priority)

    def __dealloc__(self):
        del self.action

cdef class PyDevices:
    cdef Devices* devices
    cdef uint64_t n

    def __cinit__(self, n: int):
        self.devices = new Devices(n)
        self.n = n
        print("Created Devices")

    def size(self):
        return self.n

    def create_device(self, devid_t id, str pyname, DeviceType arch, vcu_t vcus, mem_t mem):
        cname = pyname.encode('utf-8')
        self.devices.create_device(id, cname, arch, vcus, mem)

    def get_devices(self, DeviceType arch):
        cdef DeviceIDList devices = deref(self.devices).get_devices(arch)
        return convert_devid_list_to_numpy(devices, copy=True)

    def get_type(self, devid_t id):
        return deref(self.devices).get_type(id)

    def get_name(self, devid_t id):
        cdef string s = deref(self.devices).get_name(id)
        py_s = s.decode('utf-8')
        return py_s

    def __dealloc__(self):
        del self.devices
        print("Deleted Devices")

cdef class PyTopology:
    cdef Topology* topology

    def __cinit__(self, n: int):
        self.topology = new Topology(n)
        print("Created Topology")

    def set_bandwidth(self, devid_t src, devid_t dst, mem_t bandwidth):
        self.topology.set_bandwidth(src, dst, bandwidth)

    def set_max_connections(self, devid_t src, devid_t dst, copy_t max_links):
        self.topology.set_max_connections(src, dst, max_links)

    def set_latency(self, devid_t src, devid_t dst, timecount_t latency):
        self.topology.set_latency(src, dst, latency)

    def get_latency(self, devid_t src, devid_t dst):
        return self.topology.get_latency(src, dst)

    def get_bandwidth(self, devid_t src, devid_t dst):
        return self.topology.get_bandwidth(src, dst)

    def get_max_connections(self, devid_t src, devid_t dst):
        return self.topology.get_max_connections(src, dst)

    def __dealloc__(self):
        del self.topology
        print(("Deleted Topology"))




cdef class PyData:
    cdef Data* data

    def __cinit__(self, num_blocks: int):
        self.data = new Data(num_blocks)
        print("Created Data")

    def create_block(self, dataid_t id, mem_t size, devid_t location, str pyname):
        cname = pyname.encode('utf-8')
        self.data.create_block(id, size, location, cname)

    def set_size(self, dataid_t id, mem_t size):
        self.data.set_size(id, size)

    def set_location(self, dataid_t id, devid_t location):
        self.data.set_location(id, location)

    def set_name(self, dataid_t id, str pyname):
        cname = pyname.encode('utf-8')
        self.data.set_name(id, cname)

    def get_size(self, dataid_t id):
        return self.data.get_size(id)

    def get_location(self, dataid_t id):
        return self.data.get_location(id)

    def get_name(self, dataid_t id):
        cdef string s = self.data.get_name(id)
        py_s = s.decode('utf-8')
        return py_s

    def __dealloc__(self):
        del self.data
        print("Deleted Data")


cdef class PyTasks:
    cdef Tasks* tasks

    def __cinit__(self, n: int):
        self.tasks = new Tasks(n)
        print("Created Tasks")

    def n_compute_tasks(self):
        return deref(self.tasks).compute_size()

    def n_data_tasks(self):
        return deref(self.tasks).data_size()

    def is_compute(self, taskid_t taskid):
        return deref(self.tasks).is_compute(taskid)

    def n_tasks(self):
        return deref(self.tasks).size()

    def is_data(self, taskid_t taskid):
        return deref(self.tasks).is_data(taskid)

    def create_task(self, taskid_t tid, str pyname, list py_dependencies):
        cdef TaskIDList dependencies = convert_to_taskid_list(py_dependencies)
        cname = pyname.encode('utf-8')
        self.tasks.create_compute_task(tid, cname, move(dependencies))

    def add_read_set(self, taskid_t taskid, list py_dataids):
        cdef DataIDList dataids = convert_to_dataid_list(py_dataids)
        self.tasks.set_read(taskid, move(dataids))

    def add_write_set(self, taskid_t taskid, list py_dataids):
        cdef DataIDList dataids = convert_to_dataid_list(py_dataids)
        self.tasks.set_write(taskid, move(dataids))

    def add_variant(self, taskid_t id, DeviceType arch, vcu_t vcus, mem_t mem, timecount_t time):
        self.tasks.add_variant(id, arch, vcus, mem, time)

    def get_dependencies(self, taskid_t taskid):
        cdef TaskIDList dependencies = deref(self.tasks).get_dependencies(taskid)
        return convert_taskid_list_to_numpy(dependencies, copy=True)

    def get_dependents(self, taskid_t taskid):
        cdef TaskIDList dependents = deref(self.tasks).get_dependents(taskid)
        return convert_taskid_list_to_numpy(dependents, copy=True)

    def get_data_dependencies(self, taskid_t taskid):
        cdef TaskIDList taskids = deref(self.tasks).get_data_dependencies(taskid)
        return convert_taskid_list_to_numpy(taskids, copy=True)

    def get_data_dependents(self, taskid_t taskid):
        cdef TaskIDList taskids = deref(self.tasks).get_data_dependents(taskid)
        return convert_taskid_list_to_numpy(taskids, copy=True)

    def get_variants(self, taskid_t taskid):
        cdef vector[Variant] variants = deref(self.tasks).get_variant_vector(taskid)
        cdef size_t n = variants.size()
        cdef size_t m = 4
        cdef uint64_t[:, :] result = np.empty((n, m), dtype=np.uint64)
        for i in range(n):
            result[i, 0] = <uint64_t>variants[i].get_arch()
            result[i, 1] = <uint64_t>variants[i].get_vcus()
            result[i, 2] = <uint64_t>variants[i].get_mem()
            result[i, 3] = <uint64_t>variants[i].get_observed_time()
        return np.asarray(result)

    def get_variant(self, taskid_t taskid, DeviceType arch):
        cdef Variant variant = deref(self.tasks).get_variant(taskid, arch)
        return variant.get_vcus(), variant.get_mem(), variant.get_observed_time()

    def get_name(self, taskid_t taskid):
        cdef string s = deref(self.tasks).get_name(taskid)
        py_s = s.decode('utf-8')
        return py_s

    def get_depth(self, taskid_t taskid):
        return deref(self.tasks).get_depth(taskid)

    def get_data_id(self, taskid_t taskid):
        return deref(self.tasks).get_data_id(taskid)
    

    def get_read_set(self, taskid_t taskid):
        cdef DataIDList dataids = deref(self.tasks).get_read(taskid)
        return convert_dataid_list_to_numpy(dataids, copy=True)

    def get_write_set(self, taskid_t taskid):
        cdef DataIDList dataids = deref(self.tasks).get_write(taskid)
        return convert_dataid_list_to_numpy(dataids, copy=True)

    def __dealloc__(self):
        del self.tasks
        print("Deleted Tasks")


cdef class PyTaskNoise:
    cdef PyTasks tasks
    cdef TaskNoise* noise

    def __cinit__(self, PyTasks tasks, unsigned int seed = 0, unsigned int pseed = 1000):
        self.tasks = tasks
        self.noise = new TaskNoise(deref(tasks.tasks), seed, pseed)

    def __dealloc__(self):
        del self.noise

    def get(self, taskid_t task_id, arch):
        #cdef DeviceType carch = convert_py_device_type(arch)
        return self.noise.get(task_id, arch)

    def set(self, taskid_t task_id, arch, timecount_t noise):
        #cdef DeviceType carch = convert_py_device_type(arch)
        self.noise.set(task_id, arch, noise)

    def set(self, taskid_t task_id, priority_t p):
        self.noise.set_priority(task_id, p)


    def set_vector(self, timecount_t[:, :] noise):
        cdef int n = noise.shape[0]
        cdef int m = noise.shape[1]
        cdef vector[timecount_t] noise_vector
        for i in range(n):
            for j in range(m):
                noise_vector.push_back(noise[i, j])
        self.noise.set(noise_vector)

    def set_priority_vector(self, priority_t[:] noise):
        cdef PriorityList priority_list
        for p in noise:
            priority_list.push_back(p)
        self.noise.set_priority(priority_list)

    def lock(self):
        self.noise.lock()

    def generate(self):
        self.noise.generate()

    def generate_durations(self):
        self.noise.generate_duration()

    def generate_priorities(self):
        self.noise.generate_priority()

    def dump_to_binary(self, str filename):
        cname = filename.encode('utf-8')
        self.noise.dump_to_binary(cname)

    def load_from_binary(self, str filename):
        cname = filename.encode('utf-8')
        self.noise.load_from_binary(cname)

    def dump_priority_to_binary(self, str filename):
        cname = filename.encode('utf-8')
        self.noise.dump_priorities_to_binary(cname)

    def load_priority_from_binary(self, str filename):
        cname = filename.encode('utf-8')
        self.noise.load_priorities_from_binary(cname)
 
cdef class PyExternalTaskNoise(PyTaskNoise):
    cdef object cfunc 

    def __cinit__(self, PyTasks tasks, unsigned int seed = 0, unsigned int pseed = 1000):
        self.tasks = tasks
        self.noise = new ExternalTaskNoise(deref(tasks.tasks), seed, pseed)
        self.cfunc = None 


    def __dealloc__(self):
        if self.cfunc is not None:
            Py_DECREF(self._sample_func_keeper)
        del self.noise

    def set_function(self, cf):
        cdef unsigned long long f_temp = cf.address 
        cdef esf_t f = <esf_t> f_temp
        self.cfunc = cf
        if self.cfunc is not None:
            Py_DECREF(self.cfunc)
        Py_INCREF(self.cfunc)
        (<ExternalTaskNoise*>self.noise).set_function(f)



cdef class PyLognormalTaskNoise(PyTaskNoise):

    def __cinit__(self, PyTasks tasks, unsigned int seed = 0, unsigned int pseed = 1000):
        self.tasks = tasks
        self.noise = new LognormalTaskNoise(deref(tasks.tasks), seed, pseed)


cdef class PyCommunicationNoise:
    cdef CommunicationNoise* noise
    cdef PyTopology topology

    def __cinit__(self, PyTopology topology, unsigned int seed = 0):
        self.topology = topology
        self.noise = new CommunicationNoise(deref(topology.topology), seed)

    def __dealloc__(self):
        del self.noise

    def get(self, taskid_t data_task_id, devid_t source, devid_t destination):
        cdef CommunicationRequest request = CommunicationRequest(data_task_id, source, destination, 0)
        cdef CommunicationStats stats = self.noise.get(request)
        return stats.latency, stats.bandwidth

    def set(self, taskid_t data_task_id, devid_t source, devid_t destination, timecount_t latency, mem_t bandwidth):
        cdef CommunicationRequest request = CommunicationRequest(data_task_id, source, destination, 0)
        cdef CommunicationStats stats = CommunicationStats(latency, bandwidth)
        self.noise.set(request, stats)

    def dump_to_binary(self, str filename):
        cname = filename.encode('utf-8')
        self.noise.dump_to_binary(cname)

    def load_from_binary(self, str filename):
        cname = filename.encode('utf-8')
        self.noise.load_from_binary(cname)

cdef class PyMapper:
    cdef Mapper* mapper

    def __cinit__(self):
        self.mapper = new Mapper()
        print("Created Mapper")

    def __dealloc__(self):
        del self.mapper
        print("Deleted Mapper")


cdef class PyStaticMapper(PyMapper):

    def __cinit__(self):
        self.mapper = new StaticMapper()

    def set_mapping(self, devid_t[:] device_list):
        cdef DeviceIDList devices
        for device in device_list:
            devices.push_back(device)
        (<StaticMapper*>self.mapper).set_mapping(devices)

    def set_launching_priorities(self, priority_t[:] priorities):
        cdef PriorityList priority_list
        for priority in priorities:
            priority_list.push_back(priority)
        (<StaticMapper*>self.mapper).set_launching_priorities(priority_list)

    def set_reserving_priorities(self, priority_t[:] priorities):
        cdef PriorityList priority_list
        for priority in priorities:
            priority_list.push_back(priority)
        (<StaticMapper*>self.mapper).set_reserving_priorities(priority_list)


cdef class PySchedulerInput:
    cdef SchedulerInput* input 

    def __cinit__(self, PyTasks tasks, PyData data, PyDevices devices, PyTopology topology, PyMapper mapper, PyTaskNoise task_noise, PyCommunicationNoise comm_noise):
        self.input = new SchedulerInput(deref(tasks.tasks), deref(data.data), deref(devices.devices), deref(topology.topology), deref(mapper.mapper), deref(task_noise.noise), deref(comm_noise.noise))

    def __dealloc__(self):
        del self.input

cdef class PySimulator:
    cdef PySchedulerInput input 
    cdef Simulator* simulator

    def __cinit__(self, input):
        if isinstance(input, PySchedulerInput):
            self.input = input 
            self.simulator = new Simulator(deref((<PySchedulerInput>input).input))
        elif isinstance(input, PySimulator):
            self.simulator = new Simulator(deref((<PySimulator>input).simulator))
        else:
            raise ValueError("Input must be a PySchedulerInput or PySimulator")

    def copy(self):
        cdef PySimulator new_simulator = PySimulator(self)
        return new_simulator

    cdef create(self):
        self.simulator = new Simulator(deref(self.input.input))

    def initialize(self, bool create_data_tasks = 0):
        self.simulator.initialize(create_data_tasks)

    def run(self):
        cdef ExecutionState stop_reason = self.simulator.run()
        return convert_cpp_execution_state(stop_reason)

    def get_current_time(self):
        return self.simulator.get_current_time()

    def set_mapper(self, PyMapper mapper):
        self.simulator.set_mapper(deref(mapper.mapper))

    def get_mappable_candidates(self):
        cdef TaskIDList candidates = deref(self.simulator).get_mappable_candidates()
        if candidates.empty():
            return np.array([], dtype=np.uint64)
        else:
            return convert_taskid_list_to_numpy(candidates, copy=True)

    def map_tasks(self, list[PyAction] actions):
        cdef ActionList action_list
        cdef PyAction action
        cdef Action* action_ptr

        for action in actions:
            action_ptr = action.action
            action_list.push_back(deref(action_ptr))
        self.simulator.map_tasks(action_list)

    def add_task_breakpoint(self, event_type, taskid_t task_id):
        self.simulator.add_task_breakpoint(convert_py_event_type(event_type), task_id)

    def add_time_breakpoint(self, timecount_t time):
        self.simulator.add_time_breakpoint(time)

    def use_python_mapper(self, bool use_python_mapper):
        self.simulator.set_use_python_mapper(use_python_mapper)

    def __dealloc__(self):
        del self.simulator  


cdef class PyObserver:
    cdef Observer* observer

    def __cinit__(self, PySimulator simulator):
        self.observer = new Observer(deref(simulator.simulator))

    def global_features(self):
        self.observer.global_features()

    def get_active_tasks(self):
        cdef TaskIDList tasks = self.observer.get_active_tasks()
        return convert_taskid_list_to_numpy(tasks, copy=True)

    def get_k_hop_tasks(self, list initial, int k):
        cdef TaskIDList initial_tasks = convert_to_taskid_list(initial)
        cdef TaskIDList tasks = self.observer.get_k_hop_tasks(initial_tasks, k)
        return convert_taskid_list_to_numpy(tasks, copy=True)

    def get_task_features(self, list[taskid_t] task_ids):
        cdef TaskFeatures task_features = self.observer.get_task_features(convert_to_taskid_list(task_ids))
        cdef size_t d = task_features.feature_dim
        cdef size_t n = task_features.features.size() // d

        if n == 0:
            return np.array([])

        cdef double[:, :] f = <double[:n, :d]>task_features.features.data()
        return np.asarray(f, dtype=np.float64)

    def get_data_features(self, list[dataid_t] data_ids):
        cdef DataFeatures data_features = self.observer.get_data_features(convert_to_dataid_list(data_ids))
        cdef size_t d = data_features.feature_dim
        cdef size_t n = data_features.features.size() // d

        if n == 0:
            return np.array([])

        cdef double[:, :] f = <double[:n, :d]>data_features.features.data()
        return np.asarray(f, dtype=np.float64)

    def get_device_features(self, list[devid_t] device_ids):
        cdef DeviceFeatures device_features = self.observer.get_device_features(convert_to_devid_list(device_ids))
        cdef size_t d = device_features.feature_dim
        cdef size_t n = device_features.features.size() // d

        if n == 0:
            return np.array([])

        cdef double[:, :] f = <double[:n, :d]>device_features.features.data()
        return np.asarray(f, dtype=np.float64)

    def get_task_task_edges(self, list[taskid_t] source_tasks, list[taskid_t] target_tasks):
        cdef TaskTaskEdges edges = self.observer.get_task_task_edges(convert_to_taskid_list(source_tasks), convert_to_taskid_list(target_tasks))
        cdef size_t n_source = len(source_tasks)
        cdef size_t n_target = len(target_tasks)
        cdef size_t n_edges = edges.tasks.size()
        cdef size_t d = edges.feature_dim

        if n_edges == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        cdef cnp.uint64_t[:] dep2id = <cnp.uint64_t[:n_target]>edges.dep2id.data()
        cdef cnp.uint64_t[:] tasks = <cnp.uint64_t[:n_edges]>edges.tasks.data()
        cdef cnp.uint64_t[:] deps = <cnp.uint64_t[:n_edges]>edges.deos.data() 
        cdef double[:, :] features_array = <double[:n_edges, :d]>edges.features.data()
        return np.asarray(dep2id), np.asarray(tasks), np.asarray(deps), np.asarray(features_array)


    def get_task_data_edges(self, list[taskid_t] task_ids):
        cdef TaskDataEdges edges = self.observer.get_task_data_edges(convert_to_taskid_list(task_ids))
        cdef size_t n = edges.tasks.size()
        cdef size_t d = edges.feature_dim
        cdef size_t n_data = edges.data2id.size()

        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        cdef cnp.uint64_t[:] data2id = <cnp.uint64_t[:n_data]>edges.data2id.data()
        cdef cnp.uint64_t[:] tasks = <cnp.uint64_t[:n]>edges.tasks.data()
        cdef cnp.uint64_t[:] data = <cnp.uint64_t[:n]>edges.data.data()
        cdef double[:, :] features_array = <double[:n, :d]>edges.features.data()
        return np.asarray(data2id), np.asarray(tasks), np.asarray(data), np.asarray(features_array)

    def get_task_device_edges(self, list[taskid_t] task_ids):
        cdef TaskDeviceEdges edges = self.observer.get_task_device_edges(convert_to_taskid_list(task_ids))
        cdef size_t n = edges.tasks.size()
        cdef size_t d = edges.feature_dim
        cdef size_t n_devices = edges.device.size()

        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        cdef cnp.uint64_t[:] device = <cnp.uint64_t[:n_devices]>edges.device2id.data()
        cdef cnp.uint64_t[:] tasks = <cnp.uint64_t[:n]>edges.tasks.data()
        cdef cnp.uint64_t[:] devices = <cnp.uint64_t[:n]>edges.devices.data()
        cdef double[:, :] features_array = <double[:n, :d]>edges.features.data()
        return np.asarray(device2id), np.asarray(tasks), np.asarray(devices), np.asarray(features_array)


    def get_data_device_edges(self, list[taskid_t] task_ids):
        cdef DataDeviceEdges edges = self.observer.get_data_device_edges(convert_to_taskid_list(task_ids))
        cdef size_t n = edges.data.size()
        cdef size_t d = edges.feature_dim
        cdef size_t n_data = edges.data2id.size()
        cdef size_t n_devices = edges.device2id.size()

        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        cdef cnp.uint64_t[:] data2id = <cnp.uint64_t[:n_data]>edges.data2id.data()
        cdef cnp.uint64_t[:] device2id = <cnp.uint64_t[:n_devices]>edges.device2id.data()
        cdef cnp.uint64_t[:] data = <cnp.uint64_t[:n]>edges.data.data()
        cdef cnp.uint64_t[:] devices = <cnp.uint64_t[:n]>edges.devices.data()
        cdef double[:, :] features_array = <double[:n, :d]>edges.features.data()
        return np.asarray(data2id), np.asarray(device2id), np.asarray(data), np.asarray(devices), np.asarray(features_array)

    def __dealloc__(self):
        del self.observer