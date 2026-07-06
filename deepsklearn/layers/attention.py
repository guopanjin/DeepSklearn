import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 num_heads:int
                 ):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        if self.num_heads > self.embed_dim:
            raise ValueError("num_heads must less len embed_dim")
        if embed_dim % num_heads!=0:
            raise ValueError(f"{embed_dim} must be divisible by {num_heads}")
        self.head_dim=embed_dim // num_heads
        self.scale=self.head_dim**(0.5)
        self.q_w=nn.Linear(self.embed_dim,self.embed_dim,bias=False)
        self.k_w=nn.Linear(self.embed_dim,self.embed_dim,bias=False)
        self.v_w=nn.Linear(self.embed_dim,self.embed_dim,bias=False)
        self.out_w=nn.Linear(self.embed_dim,self.embed_dim,bias=False)
    #x=(batch_size,feature_size,embed_dim)
    def forward(self,x):
        q=self.q_w(x) #(batch_size,feature_size,embed_dim)
        k=self.k_w(x) #(batch_size,feature_size,embed_dim)
        v=self.v_w(x) #(batch_size,feature_size,embed_dim)
        batch_size,feature_size,_=x.shape
        q=torch.reshape(q, (batch_size,feature_size,self.num_heads,self.head_dim)) # (batch_size,feature_size,num_heads,head_dim)
        k=torch.reshape(k,(batch_size,feature_size,self.num_heads,self.head_dim)) #(batch_size,feature_size,num_heads,head_dim)
        v=torch.reshape(v,(batch_size,feature_size,self.num_heads,self.head_dim)) #(batch_size,feature_size,num_heads,head_dim)

        q=torch.transpose(q,1,2) # (batch_size,num_heads,feature_size,head_dim)
        k=torch.permute(k,(0,2,3,1)) # (batch_size,num_heads,head_dim,feature_size)
        v = torch.transpose(v,1,2) #(batch_size,num_heads,feature_size,head_dim)

        attention_score=(q@k)/self.scale # (batch_size,num_heads,feature_size,feature_size)
        attention_weight=F.softmax(attention_score,dim=-1) #(batch_size,num_heads,feature_size,feature_size)

        attention_out=attention_weight@v#(batch_size,num_heads,feature_size,head_dim)
        attention_out=torch.reshape(torch.transpose(attention_out,1,2),(batch_size,feature_size,self.embed_dim))
        out=self.out_w(attention_out)
        return out






