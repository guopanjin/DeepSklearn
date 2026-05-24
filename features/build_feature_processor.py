from features.registry import REGISTRY

class BuildProcessor:
    @staticmethod
    def registry_processor(cfg):
        cls=REGISTRY[cfg['type']]
        args=cfg['args'] # this is a dict type
        return cls(**args) # initialize class here

