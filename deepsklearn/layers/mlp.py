import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.normalizations import RMSNormal
from deepsklearn.initializations import init_embedding
# defined a general MLP block
class MLPBlock(nn.Module):
    def __init__(self,
                 *,
                 input_dim,
                 hidden_layers=(128, 64, 32),
                 norm=False,
                 dropout:float=0.0
                 ):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_layers=hidden_layers
        self.norm=norm
        self.dropout=dropout
        module_list=[]
        pre_layer=self.input_dim
        for hidden_layer in self.hidden_layers:
            module_list.append(nn.Linear(pre_layer,hidden_layer))
            module_list.append(nn.ReLU())
            if self.norm:
                module_list.append(RMSNormal(hidden_layer))
            if self.dropout>0:
                module_list.append(nn.Dropout(self.dropout))
            pre_layer=hidden_layer
        self.layers=nn.ModuleList(module_list)
    def forward(self,x):
        for layer  in  self.layers:
            x=layer(x)
        return x