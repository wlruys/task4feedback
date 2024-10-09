# cython: language_level=3
# cython: embedsignature=True
# cython: language=c++

import cython 

from graph cimport GraphManager
from settings cimport taskid_t, TaskIDList, DataIDList, DeviceIDList, DeviceType, devid_t, depcount_t, vcu_t, mem_t, timecount_t, copy_t
from tasks cimport Tasks
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.utility cimport move
from libcpp.string cimport string
from enum import IntEnum 
import numpy as np
cimport numpy as np

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
    

cdef class PyTasks:
    cdef Tasks* tasks

    def __cinit__(self, n: int):
        self.tasks = new Tasks(n)

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

    def get_read(self, taskid_t taskid):
        cdef DataIDList dataids = deref(self.tasks).get_read(taskid)
        return convert_dataid_list_to_numpy(dataids, copy=True)

    def get_write(self, taskid_t taskid):
        cdef DataIDList dataids = deref(self.tasks).get_write(taskid)
        return convert_dataid_list_to_numpy(dataids, copy=True)
    
    def initialize_dependents(self):
        GraphManager.populate_dependents(deref(self.tasks))

    def __dealloc__(self):
        del self.tasks


cdef class PySimulator:
    cdef Simulator* simulator

    def __cinit__(self, PyTasks tasks, PyDevices devices):
        self.simulator = new Simulator(deref(tasks.tasks), deref(devices.devices))

    def initialize(self, unsigned int seed):
        self.simulator.initialize(seed)

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
