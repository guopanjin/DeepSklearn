import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.initializations import init_embedding
'''
Factorization Machines:
cross_part:
1/2* (sum(embedding)^2- sum(embedding^2))
'''
class FM(nn.Module):
    def __init__(self,
                 feature_columns:dict,
                 k=8,
                 customize_init_embedding=True):
        super().__init__()
        self.feature_columns=feature_columns
        self.k=k
        #lr part
        feature_embedding_raw_dict={feature_name:nn.Embedding(int(embedding_size),1) for feature_name,embedding_size in feature_columns.items()}
        self.feature_embedding_dict =nn.ModuleDict(feature_embedding_raw_dict)
        self.bias=nn.Parameter(torch.zeros(1))
        #fm part
        fm_embedding_raw_dict={feature_name:nn.Embedding(int(embedding_size),self.k) for feature_name,embedding_size in feature_columns.items()}
        self.fm_embedding_dict=nn.ModuleDict(fm_embedding_raw_dict)
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
        cross_part=0.5*(torch.sum(first_part-sencond_part,dim=1,keepdim=True)) #(batch_size,1)
        logits=linear_output+cross_part
        return logits