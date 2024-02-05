from .recorder import *
from typing import List, Dict, Any, Tuple, Type
from dataclasses import dataclass, field, InitVar
import json


def device_to_tid(device: Devices) -> int:
    return hash(device)


def export_task_record(record: TaskRecord) -> Dict:
    if isinstance(record, ComputeTaskRecord):
        cat = "compute"
        source = "None"
    elif isinstance(record, DataTaskRecord):
        cat = "data"
        source = str(record.source)
    else:
        cat = "unknown"
        source = "None"

    return {
        "name": str(record.name),
        "ts": record.start_time.duration,
        "dur": (record.end_time - record.start_time).duration,
        "cat": cat,
        "ph": "X",
        "pid": 0,
        "tid": device_to_tid(record.devices),
        "args": {
            "device": str(record.devices),
            "source": source,
        },
    }


def export_task_records(
    records: ComputeTaskRecorder,
    data_records: Optional[DataTaskRecorder] = None,
    filename: str = "task_profile.json",
):
    task_profile = {}
    task_profile["traceEvents"] = [
        export_task_record(record) for record in records.tasks.values()
    ]
    if data_records is not None:
        task_profile["traceEvents"] += [
            export_task_record(record) for record in data_records.tasks.values()
        ]

    task_profile["displayTimeUnit"] = "ms"

    json.dump(task_profile, open(filename, "w"), indent=2)
