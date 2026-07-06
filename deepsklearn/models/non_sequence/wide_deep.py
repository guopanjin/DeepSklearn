import torch
import torch.nn as nn
from deepsklearn.initializations import init_embedding
from deepsklearn.layers import MLPBlock
'''
input: dict {
    feature_name:batch_value.
}
logits=linear+bais + deep 
'''
class WideDeep(nn.Module):
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
        self.customize_init_embedding=customize_init_embedding
        self.embedding_dict=nn.ModuleDict({feature_name:nn.Embedding(embedding_number,embedding_dim=self.embedding_size) for feature_name,embedding_number in self.feature_columns.items() })
        self.mlp_block=MLPBlock(
            input_dim=len(self.feature_columns)*self.embedding_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            norm=self.norm
        )
        self.output_layer=nn.Linear(self.hidden_layers[-1],1)
        self.linear_embedding_dict=nn.ModuleDict({feature_name:nn.Embedding(embedding_number,1) for feature_name,embedding_number in self.feature_columns.items() })
        self.bias=nn.Parameter(torch.zeros(1))
        if customize_init_embedding:
            self.apply(lambda m:init_embedding(m))
    '''
    x:dict ==>{feature_name,batch_value}
    return:logits
    '''
    def forward(self,x):
        embedding_list=[]
        linear_embedding_list=[]
        for feature_name,batch_value in x.items():
            embedding_list.append(self.embedding_dict[feature_name](batch_value))
            linear_embedding_list.append(self.linear_embedding_dict[feature_name](batch_value))
        #[(batch_size,embedding_size),]
        dnn_input=torch.concat(embedding_list,dim=-1) # (batch_size,embedding_size*feature_size)
        dnn_hidden=self.mlp_block(dnn_input)
        dnn_output=self.output_layer(dnn_hidden)# (batch_size,1)
        linear_output=torch.sum(torch.concat(linear_embedding_list,dim=1),dim=1,keepdim=True) #(batch_size,1)
        output=linear_output+self.bias+dnn_output #(batch_size,1)
        return output



