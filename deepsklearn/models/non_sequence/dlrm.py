import numpy as np
import torch
import torch.nn as nn
from deepsklearn.initializations import init_embedding
from deepsklearn.layers import MLPBlock,FMInteraction
'''
input: dict {
    feature_name:batch_value.
}
Deep Learning Recommendation Model:
 Embedding_dim=16
 paper structure:
  dense feature --->bottle MLP(512-256-64-16)-->dense_output (batch_sze,embedding_dim)
  sparse feature--->embedding_list
  cross_part=pair wise inner product([embedding_list,dense_output])
  concat(cross_part,dense_output)--->MLP(512-256-1)--->linear---> logits
 
'''
class DLRM(nn.Module):
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
        self.feature_size=len(self.feature_columns)
        self.mlp_block=MLPBlock(
            input_dim= (self.feature_size**2-self.feature_size)//2,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            norm=self.norm
        )
        self.register_buffer("mask",torch.triu(torch.ones(self.feature_size,self.feature_size),diagonal=1).to(torch.bool))
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
        stacked_embeddings=torch.stack(embedding_list,dim=1)#(batch_size,feature_size,embedding_dim)
        stacked_embeddings_T=torch.transpose(stacked_embeddings,1,2) #(batch_size,embedding_dim,feature_size)
        inner_product_output=stacked_embeddings@stacked_embeddings_T #(batch_size,feature_size,feature_size)

        inner_product_output=inner_product_output[:, self.mask] #(batch_size,feature_size*)

        mlp_input=torch.concat([inner_product_output],dim=-1)
        mlp_output=self.mlp_block(mlp_input)
        logits=self.output_layer(mlp_output)
        return logits