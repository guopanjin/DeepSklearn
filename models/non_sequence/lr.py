import torch
import torch.nn as nn
import torch.nn.functional as F

class LR(nn.Module):
    def __init__(self,feature_columns:dict,embedding_dim=32):
        super().__init__()
        self.feature_columns=feature_columns
        self.embeddig_dim=embedding_dim
        self.feature_embedding_dict={feature_name:nn.Embedding(int(embedding_size),self.embeddig_dim) for feature_name,embedding_size in feature_columns.items()}
        self.layer1=nn.Linear(self.embeddig_dim*len(feature_columns),1)

    def forward(self,x):
        feature_embeddings=[self.feature_embedding_dict[feature_name](feature_values)  for feature_name, feature_values in x.items()]
        x=torch.cat(feature_embeddings,dim=1)#(batch_size,embedding_dim*len(feature_columns))
        # concat feature_embeddings
        logits=self.layer1(x)
        return logits