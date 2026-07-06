import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.initializations import init_embedding
from deepsklearn.layers import MultiHeadAttention
'''
Automatic Feature Interaction Learning

multi-head attention (3 MHA ): 
  n_heads=2, layers=3

e=[batch_size,feature_size,embedding_dim]

[batch_size,feature_size,embedding_dim] --->relu( MHA(2 heads) + e)*3 -->[batch_size,feature_size,embedding_dim] 
--->flattern -->[batch_size,feature_size*embedding_dim]--->linear() [batch_size,1] --->logtis.

'''
class AutoInt(nn.Module):
    def __init__(self,
                 *,
                 feature_columns:dict,
                 hidden_layers=(128,64,32),
                 embedding_size=32,
                 dropout=0.0,
                 norm=False,
                 num_mhas=3,
                 num_heads=2,
                 customize_init_embedding=True
                 ):
        super().__init__()
        self.feature_columns=feature_columns
        self.feature_size=len(self.feature_columns)
        self.hidden_layers=hidden_layers
        self.embedding_size=embedding_size
        self.dropout = dropout
        self.norm=norm
        self.num_heads=num_heads
        self.num_mhas=num_mhas
        self.customize_init_embedding=customize_init_embedding
        self.embedding_dict=nn.ModuleDict({feature_name:nn.Embedding(embedding_number,embedding_dim=self.embedding_size) for feature_name,embedding_number in self.feature_columns.items() })

        mha_layers=[]
        for i in range(self.num_mhas):
            mha_layers.append(MultiHeadAttention(embed_dim=self.embedding_size,
                                                 num_heads=self.num_heads
                                                 ))
        self.mha_module_layers=nn.ModuleList(mha_layers)
        self.out_layer=nn.Linear(self.feature_size*self.embedding_size,1)

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
        embedding_stack=torch.stack(embedding_list,dim=1) # (batch_size,feature_size,embedding_size)

        for mha_layer in self.mha_module_layers:
            embedding_stack=F.relu(mha_layer(embedding_stack)+embedding_stack)

        mha_out=torch.flatten(embedding_stack,start_dim=1,end_dim=-1) #(batch_size,feature_size*embedding_size)
        logits=self.out_layer(mha_out)
        return logits

