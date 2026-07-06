import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.initializations import init_embedding
from deepsklearn.layers import MLPBlock
'''
NFM:Neural Factorization Machines:
nfm = linear + bias +MLP(vector_cross)
'''
class NFM(nn.Module):
    def __init__(self,
                 feature_columns:dict,
                 k=8,
                 hidden_layers=(128,64,32),
                 customize_init_embedding=True):
        super().__init__()
        self.feature_columns=feature_columns
        self.k=k
        self.hidden_layers=hidden_layers
        #lr part
        feature_embedding_raw_dict={feature_name:nn.Embedding(int(embedding_size),1) for feature_name,embedding_size in feature_columns.items()}
        self.feature_embedding_dict =nn.ModuleDict(feature_embedding_raw_dict)
        self.bias=nn.Parameter(torch.zeros(1))
        #fm part
        fm_embedding_raw_dict={feature_name:nn.Embedding(int(embedding_size),self.k) for feature_name,embedding_size in feature_columns.items()}
        self.fm_embedding_dict=nn.ModuleDict(fm_embedding_raw_dict)
        self.mlp_block=MLPBlock(input_dim=self.k,hidden_layers=self.hidden_layers)
        self.fm_mlp_linear=nn.Linear(hidden_layers[-1],1)
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
        first_part=torch.pow(torch.sum(tensor1,dim=1),2) #(batch_size,embedding_size)
        sencond_part=torch.sum(torch.pow(tensor1,2),dim=1) #(batch_size,embedding_size)
        cross_part=0.5*(first_part-sencond_part) #(batch_size,embedding_size)
        mlp_output=self.mlp_block(cross_part)
        fm_mlp_out=self.fm_mlp_linear(mlp_output)
        logits=linear_output+fm_mlp_out
        return logits