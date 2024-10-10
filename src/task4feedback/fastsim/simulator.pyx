# cython: language_level=3
# cython: embedsignature=True
# cython: language=c++

import cython 

from graph cimport GraphManager
from settings cimport taskid_t, dataid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t, copy_t
from tasks cimport Tasks, Variant, TaskState, TaskStatus
from data cimport Data 
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.utility cimport move
from libcpp.string cimport string
from enum import IntEnum 
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
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
    PYTHON_MAPPING = <int>ExecutionState.PYTHON_MAPPING,
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
    cdef np.npy_intp n = taskid_list.size()
    cdef np.ndarray[np.uint64_t, ndim=1] result

    if not copy:
        result = np.PyArray_SimpleNewFromData(1, &n, np.NPY_UINT64, <uint64_t*>taskid_list.data())
    else:
        result = np.PyArray_SimpleNew(1, &n, np.NPY_UINT64)
        for i in range(n):
            result[i] = taskid_list[i]

    return result

cdef convert_dataid_list_to_numpy(DataIDList dataid_list, copy: bool = False):
    cdef np.npy_intp n = dataid_list.size()
    cdef np.ndarray[np.uint64_t, ndim=1] result

    if not copy:
        result = np.PyArray_SimpleNewFromData(1, &n, np.NPY_UINT64, <void*>dataid_list.data())
    else:
        result = np.PyArray_SimpleNew(1, &n, np.NPY_UINT64)
        for i in range(n):
            result[i] = dataid_list[i]

    return result

cdef convert_devid_list_to_numpy(DeviceIDList devid_list, copy: bool = False):
    cdef np.npy_intp n = devid_list.size()
    cdef np.ndarray[np.uint32_t, ndim=1] result

    if not copy:
        result = np.PyArray_SimpleNewFromData(1, &n, np.NPY_UINT64, <void*>devid_list.data())
    else:
        result = np.PyArray_SimpleNew(1, &n, np.NPY_UINT64)
        for i in range(n):
            result[i] = devid_list[i]

    return result


cdef class PyAction:
    cdef Action* action

    def __cinit__(self, taskid_t task_id, size_t pos, devid_t device, priority_t reservable_priority, priority_t launchable_priority):
        self.action = new Action(task_id, pos, device, reservable_priority, launchable_priority)

    def __dealloc__(self):
        del self.action

cdef class PyDevices:
    cdef Devices* devices

    def __cinit__(self, n: int):
        self.devices = new Devices(n)

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


cdef class PyData:
    cdef Data* data

    def __cinit__(self, num_blocks: int):
        self.data = new Data(num_blocks)

    def add_block(self, dataid_t id, mem_t size, devid_t location, str pyname):
        cname = pyname.encode('utf-8')
        self.data.add_block(id, size, location, cname)

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


cdef class PyTasks:
    cdef Tasks* tasks

    def __cinit__(self, n: int):
        self.tasks = new Tasks(n)

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
        cdef np.ndarray[np.uint64_t, ndim=2] result
        cdef np.npy_intp n = variants.size()
        cdef np.npy_intp m = 4
        result = np.PyArray_SimpleNew(2, [n, m], np.NPY_UINT64)
        for i in range(n):
            result[i, 0] = <uint64_t>variants[i].get_arch()
            result[i, 1] = <uint64_t>variants[i].get_vcus()
            result[i, 2] = <uint64_t>variants[i].get_mem()
            result[i, 3] = <uint64_t>variants[i].get_execution_time()
        return result

    def get_variant(self, taskid_t taskid, DeviceType arch):
        cdef Variant variant = deref(self.tasks).get_variant(taskid, arch)
        return variant.get_vcus(), variant.get_mem(), variant.get_execution_time()

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


cdef class PyMapper:
    cdef Mapper* mapper

    def __cinit__(self):
        self.mapper = new Mapper()

    def __dealloc__(self):
        del self.mapper


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

cdef class PySimulator:
    cdef PyTasks pytasks
    cdef PyDevices pydevices
    cdef PyMapper pymapper 
    cdef Simulator* simulator

    def __cinit__(self, PyTasks tasks, PyDevices devices, PyMapper mapper):
        self.pytasks = tasks
        self.pydevices = devices
        self.pymapper = mapper

        self.simulator = new Simulator(deref(tasks.tasks), deref(devices.devices), deref(mapper.mapper))

    def initialize(self, unsigned int seed, bool create_data_tasks = 0):
        self.simulator.initialize(seed, create_data_tasks)

    def run(self):
        cdef ExecutionState stop_reason = self.simulator.run()
        return convert_cpp_execution_state(stop_reason)

    def get_current_time(self):
        return self.simulator.get_current_time()

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

    def __dealloc__(self):
        del self.simulator  
