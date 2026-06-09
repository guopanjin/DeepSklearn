import torch.nn as nn
from deepsklearn.activations.swiglu import SwiGlu
from deepsklearn.config.constants import Constant

_ACTIVATION_REGISTRY={
    Constant.Activation.sigmoid:(nn.Sigmoid,{}),
    Constant.Activation.tanh:(nn.Tanh,{}),
    Constant.Activation.relu:(nn.ReLU,{}),
    Constant.Activation.leaky_relu:(nn.LeakyReLU,{"negative_slope":1e-2}),
    Constant.Activation.silu:(nn.SiLU,{}), #x*sigmod(x)
    Constant.Activation.gelu:(nn.GELU,{}), #x*cdf(x)
    Constant.Activation.swiglu:(SwiGlu,{})
}


def get_activation(name:str,**kwargs)->nn.Module:
    if name not in _ACTIVATION_REGISTRY:
        raise ValueError(f"there is no such activation functions:{name},avalable:{list(_ACTIVATION_REGISTRY.keys())}")
    cls,default_parameter = _ACTIVATION_REGISTRY[name]
    merge_parameter={**default_parameter,**kwargs}
    return cls(**merge_parameter)

