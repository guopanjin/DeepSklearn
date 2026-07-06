import torch
import torch.nn as nn
import torch.nn.functional as F

import deepsklearn.layers
from deepsklearn.normalizations import RMSNormal
from deepsklearn.initializations import init_embedding
from deepsklearn.layers import MLPBlock
'''
build a general DNN.
input: dict {
    feature_name:batch_value.
}
--->embedding--->sigmod---> bce_loss
optimizer based on different + lr_scheduler + RmsNormal + AMP
'''
class DNN(nn.Module):
    def __init__(self,
                 *,
                 feature_columns:dict,
                 hidden_layers=(128,64,32),
                 embedding_size=32,
                 dropout=0.0,
                 norm=False,
                 customize_init_embedding=True
                 ):
        super().__init__()
        self.feature_columns=feature_columns
        self.hidden_layers=hidden_layers
        self.embedding_size=embedding_size
        self.dropout = dropout
        self.norm=norm
        self.dropout=dropout
        self.customize_init_embedding=customize_init_embedding
        self.embedding_dict=nn.ModuleDict({feature_name:nn.Embedding(embedding_number,embedding_dim=self.embedding_size) for feature_name,embedding_number in self.feature_columns.items() })
        self.mlp_block=MLPBlock(
            input_dim=len(self.feature_columns)*self.embedding_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            norm=self.norm
        )
        self.output_layer=nn.Linear(self.hidden_layers[-1],1)
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
        x=torch.concat(embedding_list,dim=-1) # (batch_size,embedding_size*feature_size)
        x=self.mlp_block(x)
        x=self.output_layer(x)# (batch_size,1)
        return x



