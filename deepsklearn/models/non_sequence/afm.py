import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.initializations import init_embedding
from deepsklearn.layers import MLPBlock
'''
Attention Factorization Machine
cross_logtis=[batch_size,feature_size,embedding_dim]  -- [batch_size,cross_feature_size,embedding_dim] -->MLP(embedding_dim,embedding_dim)-->liner(embedding_dim,1)
           --->sequeen(batch_size,cross_feature_size) --->a_ai=(batch_size,cross_feature_size)--->
           softmax(aij)*[batch_size,cross_feature_size,embedding_dim]--->weight_sum(batch_size,embedding_dim)
           --->linear(embedding_dim,1)-->(batch_size,1)
AFM=sum(wi*xi) + bias +cross_logits
'''
class AFM(nn.Module):
    def __init__(self,
                 feature_columns:dict,
                 k=8,
                 customize_init_embedding=True):
        super().__init__()
        self.feature_columns=feature_columns
        self.feature_size=len(feature_columns)
        self.interaction_dim=((self.feature_size)**2 -(self.feature_size))//2
        self.k=k
        # to register by model.to(device)
        self.register_buffer("mask",torch.triu(torch.ones((self.feature_size, self.feature_size)), diagonal=1).to(torch.bool))
        #lr part
        feature_embedding_raw_dict={feature_name:nn.Embedding(int(embedding_size),1) for feature_name,embedding_size in feature_columns.items()}
        self.feature_embedding_dict =nn.ModuleDict(feature_embedding_raw_dict)
        self.bias=nn.Parameter(torch.zeros(1))
        #fm part
        fm_embedding_raw_dict={feature_name:nn.Embedding(int(embedding_size),self.k) for feature_name,embedding_size in feature_columns.items()}
        self.fm_embedding_dict=nn.ModuleDict(fm_embedding_raw_dict)

        self.mlp_block=MLPBlock(input_dim=self.k,hidden_layers=(self.k,))
        self.attention_input_layer=nn.Linear(self.k,1)
        self.cross_output_layer=nn.Linear(self.k,1)
        if customize_init_embedding:
            self.apply(lambda m:init_embedding(m))

    def forward(self,x):
        feature_embeddings=[self.feature_embedding_dict[feature_name](feature_values)  for feature_name, feature_values in x.items()]
        tensor0=torch.cat(feature_embeddings,dim=1)#(batch_size,embedding_dim*len(feature_columns))
        # concat feature_embeddings
        linear_output= torch.sum(tensor0,dim=-1,keepdim=True) + self.bias # (batch_size,1)

        #fm cross part
        fm_embeddings=[self.fm_embedding_dict[feature_name](feature_values)  for feature_name, feature_values in x.items()]
        tensor1=torch.stack(fm_embeddings,dim=1) #(batch_size,feature_size,embedding_size)
        tensor2=torch.unsqueeze(tensor1,dim=1)
        tensor3=tensor2.expand(-1,tensor2.shape[2],-1,-1)#(batch_size,feature_size,feature_size,embedding_size)
        tensor4=torch.transpose(tensor3,1,2)
        tensor5=tensor4*tensor3
        vector_interaction_part=tensor5[:,self.mask]#(batch_size,interaction_dim,embedding_dim)
        mlp_output=self.mlp_block(vector_interaction_part) #(batch_size,interaction_dim,embedding_size)
        attention_input=self.attention_input_layer(mlp_output)#(batch_size,interaction_dim,1)
        attention_input=torch.squeeze(attention_input,dim=-1)#(batch_size,interaction_dim)
        weight=F.softmax(attention_input,dim=-1)#(batch_size,interaction_dim)
        #(batch_size,embedding_dim,interaction_dim)@(batch_size,interaction_dim)
        attention_out=torch.transpose(vector_interaction_part,1,2)@torch.unsqueeze(weight,dim=-1) #(batch_size,embedding_size,1)
        attention_out=torch.squeeze(attention_out,-1)
        cross_output=self.cross_output_layer(attention_out)

        logits=linear_output+cross_output
        return logits