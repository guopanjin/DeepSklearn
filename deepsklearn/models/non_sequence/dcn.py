import numpy as np
import torch
import torch.nn as nn
from deepsklearn.initializations import init_embedding
from deepsklearn.layers import MLPBlock
'''
input: dict {
    feature_name:batch_value.
}
Deep & Cross Network:
DCN paper hyper parameter:
 1: concat(embedding)--->mlp(2*1024)-->(batch_size,1024)
 2: concat(embedding)--->corss layer(6)--->(batch_size,concat_embedding)
 linear(1+2)-->logits
 
cross_layer:
  w_l=(feature_size,1)
  x_{l+1}=(x_l@w_l)*x0 + bias(feature_size) + x_l
dcn_v2:
  cross_layer:
  w_l=(feature_size,feature_size) 
  x_{l+1}=(x_l@w_l + bias(feature_size))*x0  + x_l  
   
'''
class CrossNetwork(nn.Module):
    def __init__(self,
                 feature_size,
                 num_layer: int = 6,
                 ):
        super().__init__()
        self.feature_size=feature_size
        self.num_layer=num_layer
        layer_list=[]
        bias_list=[]
        for i in range(num_layer):
            layer_list.append(nn.Linear(self.feature_size,1,bias=False))
            bias_list.append(nn.Parameter(torch.zeros(self.feature_size)))
        self.layers=nn.ModuleList(layer_list)
        self.bias=nn.ParameterList(bias_list)
    '''
    x=(batch_size,feature_size)
    return (batch_size,feature_size)
    '''
    def forward(self,x):
        xl=x
        x0=x
        for i in range(self.num_layer):
            xl=self.layers[i](xl)*x0+self.bias[i]+xl
        return xl

class DCN(nn.Module):
    def __init__(self,
                 *,
                 feature_columns:dict,
                 hidden_layers=(128,64,32),
                 embedding_size=32,
                 dropout=0.0,
                 norm=False,
                 customize_init_embedding=True,
                 num_cross_layer:int=6
                 ):
        super().__init__()
        self.num_cross_layer=num_cross_layer
        self.feature_columns=feature_columns
        self.hidden_layers=hidden_layers
        self.embedding_size=embedding_size
        self.dropout = dropout
        self.norm=norm
        self.customize_init_embedding=customize_init_embedding
        self.embedding_dict=nn.ModuleDict({feature_name:nn.Embedding(embedding_number,embedding_dim=self.embedding_size) for feature_name,embedding_number in self.feature_columns.items() })
        self.feature_size=len(self.feature_columns)
        self.mlp_block=MLPBlock(
            input_dim=self.feature_size*self.embedding_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            norm=self.norm
        )
        self.output_layer=nn.Linear(self.hidden_layers[-1]+self.feature_size*self.embedding_size,1)
        self.cross_layer=CrossNetwork(feature_size=self.feature_size*self.embedding_size,
                                    num_layer=self.num_cross_layer
                                    )
        if customize_init_embedding:
            self.apply(lambda m:init_embedding(m))
    '''
    x:dict ==>{feature_name,batch_value}
    return:logits
    '''
    def forward(self,x):
        embedding_list=[]
        for feature_name,batch_value in x.items():
            embedding_list.append(self.embedding_dict[feature_name](batch_value))
        #[(batch_size,embedding_size),]
        dnn_input=torch.concat(embedding_list,dim=-1) # (batch_size,embedding_size*feature_size)
        dnn_output=self.mlp_block(dnn_input)
        cross_output=self.cross_layer(dnn_input)
        logits=self.output_layer(torch.concat([dnn_output,cross_output],dim=-1))
        return logits
