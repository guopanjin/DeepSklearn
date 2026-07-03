from deepsklearn.optimizers import get_constantLR,get_cosineAnnealingLR,get_linear_scheduler,StreamingScheduler
from deepsklearn.config.constants import Constant
_SCHEDULER_REGISTRY={
    Constant.Scheduler.constant_scheduler:get_constantLR,
    Constant.Scheduler.linear_scheduler:get_linear_scheduler,
    Constant.Scheduler.consin_scheduler:get_cosineAnnealingLR,
    Constant.Scheduler.streaming_scheduler:StreamingScheduler
}


def get_scheduler(name:str,**kwargs):
    if name not in _SCHEDULER_REGISTRY:
        raise ValueError(f"there is no such scheduler functions:{name},available:{sorted(_SCHEDULER_REGISTRY.keys())}")
    cls = _SCHEDULER_REGISTRY[name]
    return cls(**kwargs)