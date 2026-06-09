from .feature_processors import ContinuousProcessor,CategoricalProcessor

_REGISTRY={
     "continuous":ContinuousProcessor,
     "catagorical":CategoricalProcessor
}

class BuildProcessor:
    @staticmethod
    def get_feature_processor(cfg):
        if cfg['type'] not in _REGISTRY:
            raise ValueError(f"there is no such type:{cfg['type']},available:{list(_REGISTRY.keys())}")
        cls=_REGISTRY[cfg['type']]
        args=cfg['args'] # this is a dict type
        return cls(**args) # initialize class here
