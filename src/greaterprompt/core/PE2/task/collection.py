from .direct import DirectTask
from .zeroshotcot import ZeroshotCoTTask

TASK2CLASS = {
    "ii": DirectTask,
    "math": ZeroshotCoTTask,
    "bbh": ZeroshotCoTTask,
    "cf": DirectTask,
    "custom": ZeroshotCoTTask,
}

def Task2Class(task):
    if task not in TASK2CLASS:
        raise NotImplementedError
    else:
        return TASK2CLASS[task]
