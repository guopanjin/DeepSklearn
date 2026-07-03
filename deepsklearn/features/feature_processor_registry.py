from deepsklearn.features.feature_processors import ContinuousProcessor,CategoricalProcessor

_Feature_processor_REGISTRY={
     "continuous":ContinuousProcessor,
     "catagorical":CategoricalProcessor
}

def get_feature_processor(cfg):
    if cfg['type'] not in _Feature_processor_REGISTRY:
        raise ValueError(f"there is no such type:{cfg['type']},available:{sorted(_Feature_processor_REGISTRY.keys())}")
    cls=_Feature_processor_REGISTRY[cfg['type']]
    args=cfg['args'] # this is a dict type
    return cls(**args) # initialize class here
