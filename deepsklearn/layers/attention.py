import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 num_heads:int,
                 use_causal_mask:bool=False
                 ):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.use_causal_mask=use_causal_mask
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
    '''
    x=(batch_size,feature_size,embed_dim)
    key_padding_mask=(batch_size,feature_size)
    '''
    def forward(self,x,key_padding_mask=None):
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
        if key_padding_mask is not None:
            # for the attention_score,the last dimension is feature_weight for each feature
            # dim=-2 is feature_size.so we need to copy mask for each feature
            mask=torch.unsqueeze(key_padding_mask,dim=-2).expand((batch_size,feature_size,feature_size))
            mask=torch.unsqueeze(mask,dim=1).expand((batch_size,self.num_heads,feature_size,feature_size))
            #For nan,one number is nan,all number is nan,so need to change float('-inf') to 1e-9,in case,some row is all padding.
            attention_score=torch.masked_fill(attention_score,mask,1e-9)
        if self.use_causal_mask:
            #diagonal=1  token can attention to itself
            causal_mask=torch.triu(torch.ones(feature_size,feature_size),diagonal=1).to(torch.bool)#(feature_size,feature_size)
            causal_mask=causal_mask.to(q.device)
            attention_score=torch.masked_fill(attention_score,causal_mask,1e-9)

        attention_weight=F.softmax(attention_score,dim=-1) #(batch_size,num_heads,feature_size,feature_size)

        attention_out=attention_weight@v#(batch_size,num_heads,feature_size,head_dim)
        attention_out=torch.reshape(torch.transpose(attention_out,1,2),(batch_size,feature_size,self.embed_dim))
        out=self.out_w(attention_out)
        return out
'''
first normal:
hidden_input=x+self.mha(normal1(x))
output=ffn(normal2(hidden_intput))+hidden_input

after_normal:
hidden_input=normal1(x+self.mha((x)))

output=normal2(ffn((hidden_intput))+hidden_input)
'''
class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 num_heads:int,
                 ffn_dim:int,
                 norm_first:bool=False,
                 norm_layer:callable=None,
                 use_causal_mask:bool=False
                 ):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.ffn_dim=ffn_dim
        self.norm_first=norm_first
        self.norm_layer=norm_layer
        self.use_causal_mask=use_causal_mask
        self.mha=MultiHeadAttention(embed_dim=self.embed_dim,
                                    num_heads=self.num_heads,
                                    use_causal_mask=self.use_causal_mask
                                    )
        if self.norm_layer is None:
            self.normal1=nn.LayerNorm(self.embed_dim)
            self.normal2=nn.LayerNorm(self.embed_dim)
        else:
            self.normal1=self.norm_layer(self.embed_dim)
            self.normal2=self.norm_layer(self.embed_dim)


        self.ffn_layer1=nn.Linear(self.embed_dim,self.ffn_dim)
        self.ffn_layer2=nn.Linear(self.ffn_dim,self.embed_dim)

    def forward(self,x,key_padding_mask=None):
        if self.norm_first:
            hidden_input=self.mha(self.normal1(x),key_padding_mask) + x
            hidden_output1=F.relu(self.ffn_layer1(self.normal2(hidden_input)))
            output=self.ffn_layer2(hidden_output1) +hidden_input
            return output
        else:
            hidden_input = self.normal1(self.mha(x, key_padding_mask) + x)
            hidden_output1 = F.relu(self.ffn_layer1(hidden_input))
            output = self.normal2(self.ffn_layer2(hidden_output1) + hidden_input)
            return  output







