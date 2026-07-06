import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.initializations import init_embedding

class LR(nn.Module):
    def __init__(self,feature_columns:dict, customize_init_embedding=True):
        super().__init__()
        self.feature_columns=feature_columns
        #actually nn.embedding is one-hot and if the embedding_dim=1 is LR
        feature_embedding_raw_dict={feature_name:nn.Embedding(int(embedding_size),1) for feature_name,embedding_size in feature_columns.items()}
        self.bias=nn.Parameter(torch.zeros(1))
        self.feature_embedding_dict =nn.ModuleDict(feature_embedding_raw_dict)
        if customize_init_embedding:
            self.apply(lambda m:init_embedding(m))

    def forward(self,x):
        feature_embeddings=[self.feature_embedding_dict[feature_name](feature_values)  for feature_name, feature_values in x.items()]
        x=torch.cat(feature_embeddings,dim=1)#(batch_size,embedding_dim*len(feature_columns))
        # concat feature_embeddings
        logits=torch.sum(x,dim=-1,keepdim=True) +self.bias #(batch_size,1)
        return logits