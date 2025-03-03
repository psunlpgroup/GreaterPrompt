from .ape_trainer import APETrainer
from .apo_trainer import APOTrainer
from .default_trainer import DefaultTrainer
from .pe2_trainer import PE2Trainer

TRAINER2CLASS = {
    "ape": APETrainer,
    "apo": APOTrainer,
    "default": DefaultTrainer,
    "pe2": PE2Trainer
}

def Trainer2Class(trainer):
    if trainer not in TRAINER2CLASS:
        raise NotImplementedError
    else:
        return TRAINER2CLASS[trainer]