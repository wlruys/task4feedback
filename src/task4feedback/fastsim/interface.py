from .simulator import (
    PyTasks,
    PyData,
    PyDevices,
    PyTopology,
    PySimulator,
    PyAction,
    PyExecutionState,
    PyStaticMapper,
    PyEventType,
    PyTaskNoise,
    PyCommunicationNoise,
    PySchedulerInput,
    start_logger,
)
from dataclasses import dataclass



@dataclass
def DataHandle:
    data: PyData
    ids_to_data: dict[int, int]
    data_to_ids: dict[int, int]
    
