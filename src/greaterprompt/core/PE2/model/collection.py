from .direct import DirectModel
from .zeroshotcot import ZeroshotCoTModel

MODEL2CLASS = {
    "direct": DirectModel,
    "zeroshotcot": ZeroshotCoTModel
}

def Model2Class(task):
    if task not in MODEL2CLASS:
        raise NotImplementedError
    else:
        return MODEL2CLASS[task]